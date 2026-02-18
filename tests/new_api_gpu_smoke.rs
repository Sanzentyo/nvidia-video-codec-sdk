use std::{path::PathBuf, ptr, sync::Arc, thread, time::Duration};

use anyhow::{Context, Result};
use cudarc::driver::{sys::CUstream, CudaContext};
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT, NV_ENC_CAPS, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID,
        NV_ENC_TUNING_INFO,
    },
    DecodeCodec, DecodeOptions, DecodeRect, Decoder, Encoder, EncoderInitParams, ErrorKind,
    ReconfigureParams,
};

fn maybe_cuda() -> Option<Arc<CudaContext>> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            eprintln!("skip: CUDA init failed: {err:?}");
            None
        }
    }
}

fn encode_one_black_frame(
    session: &nvidia_video_codec_sdk::Session,
    width: u32,
    height: u32,
) -> Result<()> {
    let mut input = session
        .create_input_buffer()
        .context("create input buffer")?;
    let mut output = session
        .create_output_bitstream()
        .context("create output bitstream")?;

    let frame = vec![0_u8; (width * height * 4) as usize];
    unsafe {
        input.lock().context("lock input")?.write(&frame);
    }

    loop {
        match session.encode_picture(&mut input, &mut output, Default::default()) {
            Ok(()) => break,
            Err(err) if err.kind() == ErrorKind::EncoderBusy => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(err) if err.kind() == ErrorKind::NeedMoreInput => break,
            Err(err) => return Err(err).context("encode_picture failed"),
        }
    }

    Ok(())
}

#[test]
fn encoder_capability_and_reconfigure_smoke() -> Result<()> {
    let Some(cuda) = maybe_cuda() else {
        return Ok(());
    };

    let encoder = Encoder::initialize_with_cuda(cuda.clone()).context("initialize encoder")?;
    let supported = encoder
        .get_encode_guids()
        .context("get_encode_guids")?
        .contains(&NV_ENC_CODEC_H264_GUID);
    if !supported {
        eprintln!("skip: h264 encode not supported");
        return Ok(());
    }

    let width_max = encoder
        .get_capability(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS::NV_ENC_CAPS_WIDTH_MAX)
        .context("get capability WIDTH_MAX")?;
    assert!(width_max > 0);

    let tuning = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    let mut preset_start = encoder
        .get_preset_config(NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, tuning)
        .context("get_preset_config start")?;
    preset_start.presetCfg.rcParams.averageBitRate = 2_000_000;

    let mut init = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, 640, 360);
    init.preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(tuning)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_start.presetCfg);

    let mut session = encoder
        .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, init)
        .context("start session")?;

    encode_one_black_frame(&session, 640, 360)?;

    let mut preset_bitrate = session
        .get_encoder()
        .get_preset_config(NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, tuning)
        .context("get_preset_config bitrate")?;
    preset_bitrate.presetCfg.rcParams.averageBitRate = 1_000_000;

    let mut init_bitrate = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, 640, 360);
    init_bitrate
        .preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(tuning)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_bitrate.presetCfg);

    session
        .reconfigure(ReconfigureParams::new(init_bitrate))
        .context("reconfigure bitrate")?;

    encode_one_black_frame(&session, 640, 360)?;

    let mut preset_resize = session
        .get_encoder()
        .get_preset_config(NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, tuning)
        .context("get_preset_config resize")?;
    preset_resize.presetCfg.rcParams.averageBitRate = 1_000_000;

    let mut init_resize_invalid = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, 320, 180);
    init_resize_invalid
        .preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(tuning)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_resize.presetCfg);

    let err = session
        .reconfigure(ReconfigureParams::new(init_resize_invalid))
        .expect_err("resolution change without force_idr should fail");
    assert_eq!(err.kind(), ErrorKind::InvalidParam);

    let mut preset_resize_ok = session
        .get_encoder()
        .get_preset_config(NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, tuning)
        .context("get_preset_config resize ok")?;
    preset_resize_ok.presetCfg.rcParams.averageBitRate = 1_000_000;

    let mut init_resize_ok = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, 320, 180);
    init_resize_ok
        .preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(tuning)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_resize_ok.presetCfg);

    session
        .reconfigure(ReconfigureParams::new(init_resize_ok).force_idr(true))
        .context("reconfigure resize + force idr")?;

    encode_one_black_frame(&session, 320, 180)?;
    session.end_of_stream().context("send eos")?;

    Ok(())
}

fn decode_with_options(
    cuda: Arc<CudaContext>,
    bitstream: &[u8],
    options: DecodeOptions,
) -> Result<Vec<nvidia_video_codec_sdk::DecodedRgbFrame>> {
    let mut decoder = Decoder::new(cuda, DecodeCodec::H264, options).context("create decoder")?;
    let mut frames = decoder
        .push_access_unit(bitstream, 0)
        .context("push bitstream")?;
    frames.extend(decoder.flush().context("flush decoder")?);
    Ok(frames)
}

#[test]
fn decoder_extended_options_smoke() -> Result<()> {
    let Some(cuda) = maybe_cuda() else {
        return Ok(());
    };

    let bitstream_path = PathBuf::from("output/playback_assets/sample_h264.bin");
    if !bitstream_path.exists() {
        eprintln!(
            "skip: sample bitstream not found at {}",
            bitstream_path.display()
        );
        return Ok(());
    }
    let bitstream = std::fs::read(&bitstream_path)
        .with_context(|| format!("read {}", bitstream_path.display()))?;

    let base_frames = decode_with_options(cuda.clone(), &bitstream, DecodeOptions::default())?;
    if base_frames.is_empty() {
        eprintln!("skip: decoded zero frames from sample_h264.bin");
        return Ok(());
    }

    let external_stream: CUstream = ptr::null_mut();
    let external_frames = decode_with_options(
        cuda.clone(),
        &bitstream,
        DecodeOptions {
            external_stream: Some(external_stream),
            av1_operating_point: Some(0),
            av1_all_layers: false,
            ..Default::default()
        },
    )?;
    assert_eq!(external_frames.len(), base_frames.len());

    let resized_frames = decode_with_options(
        cuda.clone(),
        &bitstream,
        DecodeOptions {
            crop_rect: Some(DecodeRect {
                left: 0,
                top: 0,
                right: 320,
                bottom: 180,
            }),
            resize_dim: Some((160, 90)),
            ..Default::default()
        },
    )?;
    assert!(!resized_frames.is_empty());
    assert!(resized_frames
        .iter()
        .all(|frame| frame.width == 160 && frame.height == 90));

    let mut sei_decoder = Decoder::new(
        cuda,
        DecodeCodec::H264,
        DecodeOptions {
            enable_sei_messages: true,
            ..Default::default()
        },
    )
    .context("create sei decoder")?;
    let _ = sei_decoder
        .push_access_unit(&bitstream, 0)
        .context("push bitstream for sei")?;
    let _ = sei_decoder.flush().context("flush sei decoder")?;
    let _sei_messages = sei_decoder.drain_sei_messages();

    Ok(())
}
