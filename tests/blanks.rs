//! Encoder smoke tests using generated white frames.
//!
//! This test intentionally uses a small deterministic workload so it can act
//! as a stability gate across a wide range of GPUs/drivers.

use std::{fs::OpenOptions, io::Write, path::Path, sync::Arc, thread, time::Duration};

use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_H264_GUID},
    EncodeError, Encoder, EncoderInitParams, ErrorKind,
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

fn write_white_frame(
    input_buffer: &mut nvidia_video_codec_sdk::Buffer<'_>,
    frame: &[u8],
    width: u32,
    height: u32,
) -> Result<(), EncodeError> {
    // Pitched writes avoid assuming that `pitch == width * bytes_per_pixel`.
    let mut lock = input_buffer.lock()?;
    unsafe {
        lock.write_pitched(frame, (width * 4) as usize, height as usize);
    }
    Ok(())
}

fn drain_bitstream(
    output_bitstream: &mut nvidia_video_codec_sdk::Bitstream<'_>,
    output_file: &mut Option<std::fs::File>,
) -> Result<(), EncodeError> {
    loop {
        // Use non-blocking lock + retry for stable handling of busy states.
        match output_bitstream.try_lock() {
            Ok(lock) => {
                if let Some(file) = output_file.as_mut() {
                    file.write_all(lock.data()).unwrap();
                }
                break;
            }
            Err(err)
                if err.kind() == ErrorKind::LockBusy || err.kind() == ErrorKind::EncoderBusy =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

fn encode_blanks<P: AsRef<Path>>(
    cuda_ctx: Arc<CudaContext>,
    file_path: Option<P>,
) -> Result<(), EncodeError> {
    const FRAMES: usize = 32;
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 360;
    const FRAMERATE: u32 = 30;
    const BUFFER_FORMAT: NV_ENC_BUFFER_FORMAT = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
    const ENCODE_GUID: GUID = NV_ENC_CODEC_H264_GUID;

    let frame = vec![255_u8; (WIDTH * HEIGHT * 4) as usize];
    let mut output_file = file_path.map(|path| {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .expect("Path should be valid.")
    });

    let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;
    let mut initialize_params = EncoderInitParams::new(ENCODE_GUID, WIDTH, HEIGHT);
    initialize_params
        .enable_picture_type_decision()
        .framerate(FRAMERATE, 1);
    let session = encoder.start_session(BUFFER_FORMAT, initialize_params)?;

    let mut input_buffer = session.create_input_buffer()?;
    let mut output_bitstream = session.create_output_bitstream()?;

    for _ in 0..FRAMES {
        write_white_frame(&mut input_buffer, &frame, WIDTH, HEIGHT)?;

        let produced_output = loop {
            match session.encode_picture(
                &mut input_buffer,
                &mut output_bitstream,
                Default::default(),
            ) {
                Ok(()) => break true,
                Err(err) if err.kind() == ErrorKind::EncoderBusy => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(err) if err.kind() == ErrorKind::NeedMoreInput => break false,
                Err(err) => return Err(err),
            }
        };

        if produced_output {
            drain_bitstream(&mut output_bitstream, &mut output_file)?;
        }
    }

    loop {
        match session.end_of_stream() {
            Ok(()) => break,
            Err(err)
                if err.kind() == ErrorKind::EncoderBusy
                    || err.kind() == ErrorKind::NeedMoreInput =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(err) => return Err(err),
        }
    }

    // Attempt to pull any final delayed output.
    for _ in 0..8 {
        match output_bitstream.try_lock() {
            Ok(lock) => {
                if let Some(file) = output_file.as_mut() {
                    file.write_all(lock.data()).unwrap();
                }
            }
            Err(err)
                if err.kind() == ErrorKind::LockBusy || err.kind() == ErrorKind::EncoderBusy =>
            {
                break;
            }
            Err(err) => return Err(err),
        }
    }

    Ok(())
}

#[test]
fn encoder_works() {
    let Some(cuda_ctx) = maybe_cuda() else {
        return;
    };
    encode_blanks::<&str>(cuda_ctx, None).unwrap();
}

#[test]
fn encode_in_parallel() {
    let Some(cuda_ctx) = maybe_cuda() else {
        return;
    };

    std::thread::scope(|scope| {
        for _ in 0..2 {
            let thread_cuda_ctx = cuda_ctx.clone();
            scope.spawn(move || encode_blanks::<&str>(thread_cuda_ctx, None).unwrap());
        }
    });
}
