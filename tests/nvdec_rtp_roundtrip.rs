mod support;

use std::{collections::HashMap, collections::VecDeque, sync::Arc, thread, time::Duration};

use anyhow::{bail, Context, Result};
use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_H264_GUID,
        NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
    },
    Bitstream, Buffer, DecodeCodec, DecodeError, DecodeOptions, Decoder, EncodePictureParams,
    Encoder, EncoderInitParams, ErrorKind,
};

use crate::support::{
    quality_metrics::calculate_quality_metrics,
    rtp_pipeline::{packetize_access_unit, AccessUnit, RtpCodec, RtpToAccessUnit},
    rtp_reorder::RtpOrderingMode,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: usize = 16;
const CLOCK_RATE: u32 = 90_000;
const FPS: u32 = 30;
const TIMESTAMP_STEP: i64 = (CLOCK_RATE / FPS) as i64;
const QUALITY_REPRODUCTION_RATE_MIN: f64 = 0.80;
const QUALITY_PSNR_MIN_DB: f64 = 20.0;

#[derive(Debug, Clone)]
struct ReferenceFrame {
    timestamp_90k: i64,
    rgb: Vec<u8>,
    argb: Vec<u8>,
}

#[derive(Debug, Clone)]
struct EncodedAccessUnit {
    timestamp_90k: i64,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
enum RoundtripCodec {
    H264,
    H265,
    Av1,
}

impl RoundtripCodec {
    fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "h265",
            Self::Av1 => "av1",
        }
    }

    fn encode_guid(self) -> GUID {
        match self {
            Self::H264 => NV_ENC_CODEC_H264_GUID,
            Self::H265 => NV_ENC_CODEC_HEVC_GUID,
            Self::Av1 => NV_ENC_CODEC_AV1_GUID,
        }
    }

    fn decode_codec(self) -> DecodeCodec {
        match self {
            Self::H264 => DecodeCodec::H264,
            Self::H265 => DecodeCodec::H265,
            Self::Av1 => DecodeCodec::Av1,
        }
    }

    fn rtp_codec(self) -> RtpCodec {
        match self {
            Self::H264 => RtpCodec::H264,
            Self::H265 => RtpCodec::H265,
            Self::Av1 => RtpCodec::Av1,
        }
    }
}

#[test]
fn h264_roundtrip_quality_gate() -> Result<()> {
    run_roundtrip(RoundtripCodec::H264, false)
}

#[test]
fn h265_roundtrip_quality_gate() -> Result<()> {
    run_roundtrip(RoundtripCodec::H265, false)
}

#[test]
fn av1_roundtrip_quality_gate_or_skip() -> Result<()> {
    run_roundtrip(RoundtripCodec::Av1, true)
}

fn run_roundtrip(codec: RoundtripCodec, allow_skip: bool) -> Result<()> {
    let cuda_ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("skip {} test: CUDA init failed: {err:?}", codec.name());
            return Ok(());
        }
    };

    let probe_encoder = Encoder::initialize_with_cuda(cuda_ctx.clone())
        .with_context(|| format!("initialize encoder for {}", codec.name()))?;
    let encode_supported = probe_encoder
        .get_encode_guids()
        .with_context(|| format!("query encode guids for {}", codec.name()))?
        .contains(&codec.encode_guid());
    if !encode_supported {
        if allow_skip {
            eprintln!("skip {} test: NVENC codec is not supported", codec.name());
            return Ok(());
        }
        bail!("{} codec is not supported by NVENC", codec.name());
    }
    drop(probe_encoder);

    let references = generate_reference_frames(FRAMES, WIDTH, HEIGHT);
    let encoded_access_units =
        encode_frames(cuda_ctx.clone(), codec, &references).with_context(|| {
            format!(
                "encode reference frames for {} ({WIDTH}x{HEIGHT}, {FRAMES} frames)",
                codec.name(),
            )
        })?;

    let decode_options = DecodeOptions::default();
    let mut decoder = match Decoder::new(cuda_ctx, codec.decode_codec(), decode_options) {
        Ok(decoder) => decoder,
        Err(DecodeError::Unsupported(msg)) if allow_skip => {
            eprintln!("skip {} test: NVDEC unsupported: {msg}", codec.name());
            return Ok(());
        }
        Err(err) => {
            return Err(err).with_context(|| format!("create decoder for {}", codec.name()))
        }
    };

    let mut assembler = RtpToAccessUnit::new(codec.rtp_codec(), RtpOrderingMode::default());
    let mut sequence = 1_u16;
    let mut decoded_frames = Vec::new();

    for (index, encoded) in encoded_access_units.iter().enumerate() {
        let mut packets = packetize_access_unit(
            codec.rtp_codec(),
            &encoded.data,
            encoded.timestamp_90k as u32,
            sequence,
            1200,
        )
        .with_context(|| format!("packetize {} access unit {index}", codec.name()))?;
        sequence = sequence.wrapping_add(packets.len() as u16);
        maybe_shuffle_for_reorder_test(&mut packets, index);

        for packet in packets {
            let access_units = assembler
                .push_packet(packet)
                .with_context(|| format!("assemble RTP access units for {}", codec.name()))?;
            decode_access_units(&mut decoder, access_units, &mut decoded_frames)
                .with_context(|| format!("decode RTP access unit for {}", codec.name()))?;
        }
    }

    decode_access_units(&mut decoder, assembler.flush(), &mut decoded_frames)
        .with_context(|| format!("decode flushed RTP access units for {}", codec.name()))?;
    decoded_frames.extend(
        decoder
            .flush()
            .with_context(|| format!("flush decoder for {}", codec.name()))?,
    );

    evaluate_quality(codec, &references, &decoded_frames)?;
    Ok(())
}

fn decode_access_units(
    decoder: &mut Decoder,
    access_units: Vec<AccessUnit>,
    decoded_frames: &mut Vec<nvidia_video_codec_sdk::DecodedRgbFrame>,
) -> Result<()> {
    for access_unit in access_units {
        let mut frames = decoder
            .push_access_unit(&access_unit.data, access_unit.timestamp_90k)
            .context("push access unit into decoder")?;
        decoded_frames.append(&mut frames);
    }
    Ok(())
}

fn encode_frames(
    cuda_ctx: Arc<CudaContext>,
    codec: RoundtripCodec,
    references: &[ReferenceFrame],
) -> Result<Vec<EncodedAccessUnit>> {
    let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;
    let encode_guid = codec.encode_guid();

    let input_formats = encoder
        .get_supported_input_formats(encode_guid)
        .with_context(|| format!("query input format for {}", codec.name()))?;
    let buffer_format = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
    if !input_formats.contains(&buffer_format) {
        bail!("{} does not support ARGB input", codec.name());
    }

    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    let mut preset = encoder
        .get_preset_config(encode_guid, NV_ENC_PRESET_P1_GUID, tuning_info)
        .with_context(|| format!("query preset for {}", codec.name()))?;
    preset.presetCfg.frameIntervalP = 1;
    preset.presetCfg.rcParams.lookaheadDepth = 0;

    let mut init = EncoderInitParams::new(encode_guid, WIDTH, HEIGHT);
    init.preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(tuning_info)
        .framerate(FPS, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset.presetCfg);
    let session = encoder
        .start_session(buffer_format, init)
        .with_context(|| format!("start encoder session for {}", codec.name()))?;

    let buffer_count = 4_usize;
    let mut input_buffers = (0..buffer_count)
        .map(|_| session.create_input_buffer())
        .collect::<Result<Vec<_>, _>>()
        .context("create input buffer pool")?;
    let mut output_buffers = (0..buffer_count)
        .map(|_| session.create_output_bitstream())
        .collect::<Result<Vec<_>, _>>()
        .context("create output buffer pool")?;

    let mut in_use = VecDeque::with_capacity(buffer_count);
    let mut encoded = Vec::new();

    for (frame_index, frame) in references.iter().enumerate() {
        let mut input = input_buffers
            .pop()
            .ok_or_else(|| anyhow::anyhow!("input buffer pool exhausted"))?;
        let mut output = output_buffers
            .pop()
            .ok_or_else(|| anyhow::anyhow!("output buffer pool exhausted"))?;

        {
            let mut lock = input.lock().context("lock input buffer for write")?;
            // SAFETY: data is generated for the exact frame dimensions and ARGB layout.
            unsafe {
                lock.write_pitched(&frame.argb, (WIDTH as usize) * 4, HEIGHT as usize);
            }
        }

        loop {
            let params = EncodePictureParams {
                input_timestamp: frame.timestamp_90k as u64,
                encode_frame_idx: frame_index as u64,
                ..Default::default()
            };
            match session.encode_picture(&mut input, &mut output, params) {
                Ok(()) => {
                    in_use.push_back((input, output));
                    break;
                }
                Err(err) if err.kind() == ErrorKind::EncoderBusy => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(err) if err.kind() == ErrorKind::NeedMoreInput => {
                    in_use.push_back((input, output));
                    break;
                }
                Err(err) => return Err(err).context("encode picture failed"),
            }
        }

        if in_use.len() < buffer_count {
            continue;
        }
        pop_and_collect_one(
            &mut in_use,
            &mut input_buffers,
            &mut output_buffers,
            &mut encoded,
        )?;
    }

    session
        .end_of_stream()
        .context("send encoder end-of-stream")?;
    while !in_use.is_empty() {
        pop_and_collect_one(
            &mut in_use,
            &mut input_buffers,
            &mut output_buffers,
            &mut encoded,
        )?;
    }

    Ok(encoded)
}

fn pop_and_collect_one<'a>(
    in_use: &mut VecDeque<(Buffer<'a>, Bitstream<'a>)>,
    input_buffers: &mut Vec<Buffer<'a>>,
    output_buffers: &mut Vec<Bitstream<'a>>,
    encoded: &mut Vec<EncodedAccessUnit>,
) -> Result<()> {
    let (input, mut output) = in_use
        .pop_front()
        .ok_or_else(|| anyhow::anyhow!("in_use queue was unexpectedly empty"))?;
    {
        let lock = output.lock().context("lock output bitstream")?;
        if !lock.data().is_empty() {
            let timestamp_90k = i64::try_from(lock.timestamp()).unwrap_or(i64::MAX);
            encoded.push(EncodedAccessUnit {
                timestamp_90k,
                data: lock.data().to_vec(),
            });
        }
    }
    input_buffers.push(input);
    output_buffers.push(output);
    Ok(())
}

fn evaluate_quality(
    codec: RoundtripCodec,
    references: &[ReferenceFrame],
    decoded_frames: &[nvidia_video_codec_sdk::DecodedRgbFrame],
) -> Result<()> {
    if decoded_frames.is_empty() {
        bail!("{} decoded no frames", codec.name());
    }

    let reference_by_timestamp: HashMap<i64, &ReferenceFrame> = references
        .iter()
        .map(|frame| (frame.timestamp_90k, frame))
        .collect();

    for (index, decoded) in decoded_frames.iter().enumerate() {
        if decoded.width != WIDTH || decoded.height != HEIGHT {
            bail!(
                "{} decoded unexpected frame size {}x{}",
                codec.name(),
                decoded.width,
                decoded.height,
            );
        }

        let reference = reference_by_timestamp
            .get(&decoded.timestamp_90k)
            .copied()
            .or_else(|| references.get(index))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} could not map decoded frame index={} timestamp={}",
                    codec.name(),
                    index,
                    decoded.timestamp_90k,
                )
            })?;

        let metrics =
            calculate_quality_metrics(&reference.rgb, &decoded.data).with_context(|| {
                format!(
                    "{} metric calculation failed for timestamp={}",
                    codec.name(),
                    decoded.timestamp_90k,
                )
            })?;
        if metrics.reproduction_rate < QUALITY_REPRODUCTION_RATE_MIN {
            bail!(
                "{} reproduction_rate too low: {:.4} < {:.4} (timestamp={})",
                codec.name(),
                metrics.reproduction_rate,
                QUALITY_REPRODUCTION_RATE_MIN,
                decoded.timestamp_90k,
            );
        }
        if metrics.psnr < QUALITY_PSNR_MIN_DB {
            bail!(
                "{} PSNR too low: {:.4} dB < {:.4} dB (timestamp={})",
                codec.name(),
                metrics.psnr,
                QUALITY_PSNR_MIN_DB,
                decoded.timestamp_90k,
            );
        }
    }

    Ok(())
}

fn generate_reference_frames(count: usize, width: u32, height: u32) -> Vec<ReferenceFrame> {
    let mut frames = Vec::with_capacity(count);
    for index in 0..count {
        let timestamp_90k = index as i64 * TIMESTAMP_STEP;
        let rgb = generate_rgb_pattern(width, height, index);
        let argb = rgb_to_argb(&rgb);
        frames.push(ReferenceFrame {
            timestamp_90k,
            rgb,
            argb,
        });
    }
    frames
}

fn generate_rgb_pattern(width: u32, height: u32, frame_index: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    let phase = frame_index as f32 * 0.1;
    for y in 0..height {
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let yf = y as f32 / height as f32;
            let r = (255.0 * (xf * 0.7 + (phase * 0.5).sin().abs() * 0.3)) as u8;
            let g = (255.0 * (yf * 0.8 + (phase * 0.3).cos().abs() * 0.2)) as u8;
            let b = (255.0 * (0.5 + 0.5 * (phase + xf * 2.0 + yf).sin())) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }
    rgb
}

fn rgb_to_argb(rgb: &[u8]) -> Vec<u8> {
    let mut argb = Vec::with_capacity((rgb.len() / 3) * 4);
    for pixel in rgb.chunks_exact(3) {
        argb.push(pixel[2]);
        argb.push(pixel[1]);
        argb.push(pixel[0]);
        argb.push(255);
    }
    argb
}

fn maybe_shuffle_for_reorder_test(
    packets: &mut [rtc_rtp::packet::Packet],
    access_unit_index: usize,
) {
    if access_unit_index % 2 == 1 && packets.len() >= 3 {
        packets.swap(0, 1);
    }
}
