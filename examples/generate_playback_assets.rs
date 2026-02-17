use std::{
    collections::VecDeque,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_H264_GUID,
        NV_ENC_CODEC_HEVC_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
    },
    Bitstream, Buffer, EncodePictureParams, Encoder, EncoderInitParams, ErrorKind,
};
use serde::Serialize;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FPS: u32 = 30;
const FRAMES: usize = 120;
const CLOCK_RATE: i64 = 90_000;
const TIMESTAMP_STEP: i64 = CLOCK_RATE / FPS as i64;

#[derive(Debug, Clone, Copy)]
enum SampleCodec {
    H264,
    Hevc,
    Av1,
}

impl SampleCodec {
    fn all() -> [Self; 3] {
        [Self::H264, Self::Hevc, Self::Av1]
    }

    fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Av1 => "av1",
        }
    }

    fn encode_guid(self) -> GUID {
        match self {
            Self::H264 => NV_ENC_CODEC_H264_GUID,
            Self::Hevc => NV_ENC_CODEC_HEVC_GUID,
            Self::Av1 => NV_ENC_CODEC_AV1_GUID,
        }
    }

    fn output_basename(self) -> &'static str {
        match self {
            Self::H264 => "sample_h264.bin",
            Self::Hevc => "sample_hevc.bin",
            Self::Av1 => "sample_av1.bin",
        }
    }
}

#[derive(Debug)]
struct EncodedAccessUnit {
    timestamp_90k: i64,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct AccessUnitEntry {
    offset: u64,
    len: u64,
    timestamp_90k: i64,
}

#[derive(Debug, Serialize)]
struct PlaybackIndex {
    codec: String,
    width: u32,
    height: u32,
    fps: u32,
    access_units: Vec<AccessUnitEntry>,
}

fn main() -> Result<()> {
    let out_dir = PathBuf::from("output/playback_assets");
    fs::create_dir_all(&out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let cuda_ctx = CudaContext::new(0).context("failed to create CUDA context")?;
    let probe_encoder =
        Encoder::initialize_with_cuda(cuda_ctx.clone()).context("failed to initialize encoder")?;
    let supported_guids = probe_encoder
        .get_encode_guids()
        .context("failed to query encode GUIDs")?;
    drop(probe_encoder);

    for codec in SampleCodec::all() {
        if !supported_guids.contains(&codec.encode_guid()) {
            eprintln!("skip {}: codec is not supported by NVENC", codec.name());
            continue;
        }
        let access_units = encode_synthetic_sequence(cuda_ctx.clone(), codec)
            .with_context(|| format!("failed to encode {}", codec.name()))?;
        write_assets(&out_dir, codec, &access_units)
            .with_context(|| format!("failed to write assets for {}", codec.name()))?;
        println!(
            "generated {}: {} access units",
            codec.name(),
            access_units.len()
        );
    }

    println!("output dir: {}", out_dir.display());
    Ok(())
}

fn encode_synthetic_sequence(
    cuda_ctx: Arc<CudaContext>,
    codec: SampleCodec,
) -> Result<Vec<EncodedAccessUnit>> {
    let encoder = Encoder::initialize_with_cuda(cuda_ctx).context("initialize encoder")?;
    let encode_guid = codec.encode_guid();

    let buffer_format = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
    let supported_formats = encoder
        .get_supported_input_formats(encode_guid)
        .with_context(|| format!("query input formats for {}", codec.name()))?;
    if !supported_formats.contains(&buffer_format) {
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
        .with_context(|| format!("start session for {}", codec.name()))?;

    let buffer_count = 4usize;
    let mut input_buffers = (0..buffer_count)
        .map(|_| session.create_input_buffer())
        .collect::<Result<Vec<_>, _>>()
        .context("create input buffers")?;
    let mut output_buffers = (0..buffer_count)
        .map(|_| session.create_output_bitstream())
        .collect::<Result<Vec<_>, _>>()
        .context("create output buffers")?;

    let mut in_use = VecDeque::with_capacity(buffer_count);
    let mut encoded = Vec::new();

    for frame_index in 0..FRAMES {
        let mut input = input_buffers
            .pop()
            .ok_or_else(|| anyhow::anyhow!("input pool exhausted"))?;
        let mut output = output_buffers
            .pop()
            .ok_or_else(|| anyhow::anyhow!("output pool exhausted"))?;

        let argb = generate_argb_pattern(WIDTH, HEIGHT, frame_index);
        {
            let mut lock = input.lock().context("lock input buffer")?;
            // SAFETY: generated ARGB buffer matches dimensions exactly.
            unsafe {
                lock.write_pitched(&argb, WIDTH as usize * 4, HEIGHT as usize);
            }
        }

        loop {
            let params = EncodePictureParams {
                input_timestamp: (frame_index as i64 * TIMESTAMP_STEP) as u64,
                encode_frame_idx: frame_index as u64,
                ..Default::default()
            };
            match session.encode_picture(&mut input, &mut output, params) {
                Ok(()) => {
                    in_use.push_back((input, output));
                    break;
                }
                Err(err) if err.kind() == ErrorKind::EncoderBusy => {
                    thread::sleep(Duration::from_millis(2));
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

    session.end_of_stream().context("send end of stream")?;
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
        .ok_or_else(|| anyhow::anyhow!("in_use queue unexpectedly empty"))?;
    {
        let lock = output.lock().context("lock output bitstream")?;
        if !lock.data().is_empty() {
            encoded.push(EncodedAccessUnit {
                timestamp_90k: i64::try_from(lock.timestamp()).unwrap_or(i64::MAX),
                bytes: lock.data().to_vec(),
            });
        }
    }
    input_buffers.push(input);
    output_buffers.push(output);
    Ok(())
}

fn write_assets(
    output_dir: &Path,
    codec: SampleCodec,
    access_units: &[EncodedAccessUnit],
) -> Result<()> {
    if access_units.is_empty() {
        bail!("{} produced no access units", codec.name());
    }

    let bitstream_path = output_dir.join(codec.output_basename());
    let index_path = bitstream_path.with_extension("json");

    let mut output = File::create(&bitstream_path)
        .with_context(|| format!("create {}", bitstream_path.display()))?;

    let mut offset = 0u64;
    let mut entries = Vec::with_capacity(access_units.len());
    for au in access_units {
        output.write_all(&au.bytes)?;
        entries.push(AccessUnitEntry {
            offset,
            len: au.bytes.len() as u64,
            timestamp_90k: au.timestamp_90k,
        });
        offset += au.bytes.len() as u64;
    }

    let index = PlaybackIndex {
        codec: codec.name().to_string(),
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        access_units: entries,
    };
    let mut index_file =
        File::create(&index_path).with_context(|| format!("create {}", index_path.display()))?;
    serde_json::to_writer_pretty(&mut index_file, &index).context("write index json")?;
    index_file.write_all(b"\n")?;
    Ok(())
}

fn generate_argb_pattern(width: u32, height: u32, frame_index: usize) -> Vec<u8> {
    let mut argb = Vec::with_capacity((width * height * 4) as usize);
    let phase = frame_index as f32 * 0.08;
    for y in 0..height {
        for x in 0..width {
            let xf = x as f32 / width as f32;
            let yf = y as f32 / height as f32;
            let r = (255.0 * (0.5 + 0.5 * (phase + xf * 5.0).sin())) as u8;
            let g = (255.0 * (0.5 + 0.5 * (phase * 0.6 + yf * 6.0).cos())) as u8;
            let b = (255.0 * (0.3 + 0.7 * ((xf - yf + phase * 0.3).abs().fract()))) as u8;
            argb.push(b);
            argb.push(g);
            argb.push(r);
            argb.push(255);
        }
    }
    argb
}
