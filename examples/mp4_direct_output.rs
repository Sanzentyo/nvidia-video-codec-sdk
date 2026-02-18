use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use cudarc::driver::CudaContext;
use image::GenericImageView;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
    },
    EncodePictureParams, Encoder, EncoderInitParams,
};

/// PNG画像ファイルを名前順に取得
fn get_png_files(dir_path: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("png") {
                    files.push(path);
                }
            }
        }
    }

    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    files
}

/// PNG画像をARGB形式に変換
fn load_png_as_argb(
    path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let resized_img = if width != target_width || height != target_height {
        image::imageops::resize(
            &rgb_img,
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        rgb_img
    };

    let mut argb_data = Vec::with_capacity((target_width * target_height * 4) as usize);

    for pixel in resized_img.pixels() {
        let [r, g, b] = pixel.0;
        argb_data.push(b); // Blue
        argb_data.push(g); // Green
        argb_data.push(r); // Red
        argb_data.push(255); // Alpha
    }

    Ok(argb_data)
}

/// フレーム情報とエンコード済みデータ
#[derive(Debug)]
struct EncodedFrame {
    data: Vec<u8>,
    pts: u64,
    dts: u64,
    frame_index: u64,
    is_keyframe: bool,
}

/// タイムスタンプ付きでエンコードし、フレーム情報を保持
fn encode_frames_with_metadata(
    input_dir: &Path,
    fps: u32,
) -> Result<Vec<EncodedFrame>, Box<dyn std::error::Error>> {
    let png_files = get_png_files(input_dir);
    if png_files.is_empty() {
        return Err("No PNG files found".into());
    }

    println!("Found {} PNG files to encode", png_files.len());

    // 最初の画像でサイズを取得
    let first_img = image::open(&png_files[0])?;
    let (width, height) = first_img.dimensions();

    println!("Image dimensions: {}x{}", width, height);

    // CUDA & エンコーダー初期化
    let cuda_ctx = CudaContext::new(0)?;
    let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;

    let encode_guid = NV_ENC_CODEC_H264_GUID;
    let preset_guid = NV_ENC_PRESET_P1_GUID;
    let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    let mut preset_config = encoder.get_preset_config(encode_guid, preset_guid, tuning_info)?;

    let mut initialize_params = EncoderInitParams::new(encode_guid, width, height);
    initialize_params
        .preset_guid(preset_guid)
        .tuning_info(tuning_info)
        .display_aspect_ratio(16, 9)
        .framerate(fps, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_config.presetCfg);

    let session = encoder.start_session(buffer_format, initialize_params)?;

    // バッファ作成
    let num_bufs = usize::try_from(preset_config.presetCfg.frameIntervalP).unwrap_or(1)
        + usize::try_from(preset_config.presetCfg.rcParams.lookaheadDepth).unwrap_or(0);

    let mut output_buffers: Vec<_> = (0..num_bufs)
        .map(|_| session.create_output_bitstream())
        .collect::<Result<Vec<_>, _>>()?;

    // フレーム読み込み
    let mut loaded_frames: Vec<(std::path::PathBuf, Vec<u8>)> = Vec::new();
    for png_path in &png_files {
        match load_png_as_argb(png_path, width, height) {
            Ok(data) => loaded_frames.push((png_path.clone(), data)),
            Err(e) => eprintln!("Failed to load PNG {}: {}", png_path.display(), e),
        }
    }

    if loaded_frames.is_empty() {
        return Err("No valid PNG frames were loaded".into());
    }

    println!("Starting encoding with metadata collection...");
    let mut input_buffer = session.create_input_buffer()?;
    let mut encoded_frames = Vec::new();

    // フレーム時間計算
    let frame_duration_us = 1_000_000 / fps as u64;
    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    // エンコードループ
    for (i, (_path, argb_data)) in loaded_frames.iter().enumerate() {
        println!("Processing frame {} / {}", i + 1, loaded_frames.len());

        let output_bitstream = &mut output_buffers[i % num_bufs];

        // フレームデータをバッファに書き込み
        {
            let mut buffer_lock = input_buffer.lock()?;
            unsafe {
                buffer_lock.write(argb_data);
            }
        }

        // PTS/DTS計算
        let pts = start_time + (i as u64 * frame_duration_us);
        let dts = pts; // シンプルなケースではPTS=DTS

        // エンコードパラメータ設定
        let encode_params = EncodePictureParams {
            input_timestamp: pts,
            ..Default::default()
        };

        // エンコード実行
        session.encode_picture(&mut input_buffer, output_bitstream, encode_params)?;

        // 結果取得
        let lock = output_bitstream.lock()?;
        let data = lock.data().to_vec();
        let is_keyframe = matches!(
            lock.picture_type(),
            nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR
                | nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I
        );

        println!(
            "Frame {}: PTS={}, Size={}B, Keyframe={}, Type={:?}",
            i + 1,
            pts,
            data.len(),
            is_keyframe,
            lock.picture_type()
        );

        encoded_frames.push(EncodedFrame {
            data,
            pts,
            dts,
            frame_index: i as u64,
            is_keyframe,
        });
    }

    Ok(encoded_frames)
}

/// 専用のMP4ライターを使用してMP4ファイルを作成
fn create_mp4_with_timestamps(
    encoded_frames: &[EncodedFrame],
    output_path: &Path,
    fps: u32,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // シンプルなアプローチ: AnnexB形式のH.264ストリームとして出力し、
    // 別途メタデータファイルを作成

    let h264_path = output_path.with_extension("h264");
    let mut h264_file = BufWriter::new(File::create(&h264_path)?);

    // H.264ストリーム書き込み
    for frame in encoded_frames {
        h264_file.write_all(&frame.data)?;
    }
    h264_file.flush()?;

    // メタデータファイル作成
    let metadata_path = output_path.with_extension("metadata.txt");
    let mut metadata_file = File::create(&metadata_path)?;

    writeln!(metadata_file, "# H.264 Stream Metadata")?;
    writeln!(metadata_file, "fps={}", fps)?;
    writeln!(metadata_file, "width={}", width)?;
    writeln!(metadata_file, "height={}", height)?;
    writeln!(metadata_file, "total_frames={}", encoded_frames.len())?;
    writeln!(metadata_file, "")?;
    writeln!(
        metadata_file,
        "# Frame Information (frame_index, pts, dts, size, keyframe)"
    )?;

    for frame in encoded_frames {
        writeln!(
            metadata_file,
            "{}, {}, {}, {}, {}",
            frame.frame_index,
            frame.pts,
            frame.dts,
            frame.data.len(),
            frame.is_keyframe
        )?;
    }

    println!("H.264 stream saved to: {}", h264_path.display());
    println!("Metadata saved to: {}", metadata_path.display());

    // ffmpegを使用してタイムスタンプ付きMP4を作成するためのスクリプトを生成
    let script_path = output_path.with_extension("convert.bat");
    let mut script_file = File::create(&script_path)?;

    writeln!(script_file, "@echo off")?;
    writeln!(
        script_file,
        "echo Converting H.264 stream to MP4 with proper timestamps..."
    )?;
    writeln!(
        script_file,
        "ffmpeg -f h264 -framerate {} -i \"{}\" -c copy -movflags +faststart \"{}\"",
        fps,
        h264_path.file_name().unwrap().to_string_lossy(),
        output_path.file_name().unwrap().to_string_lossy()
    )?;
    writeln!(script_file, "echo Conversion completed!")?;
    writeln!(script_file, "pause")?;

    println!("Conversion script saved to: {}", script_path.display());
    println!("Run the script to create MP4 with proper timestamps.");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = Path::new("input/save_frames/20250924_180745");
    let output_dir = Path::new("output/mp4_direct");
    let fps = 30;

    std::fs::create_dir_all(output_dir)?;

    // フレームをエンコードしてメタデータ収集
    let encoded_frames = encode_frames_with_metadata(input_dir, fps)?;

    // 最初のフレームからサイズを取得
    let first_img = image::open(&get_png_files(input_dir)[0])?;
    let (width, height) = first_img.dimensions();

    // MP4ファイル作成
    let output_path = output_dir.join("timestamped_output.mp4");
    create_mp4_with_timestamps(&encoded_frames, &output_path, fps, width, height)?;

    println!(
        "Encoding completed! {} frames processed.",
        encoded_frames.len()
    );
    println!(
        "Total encoded size: {} bytes",
        encoded_frames.iter().map(|f| f.data.len()).sum::<usize>()
    );

    Ok(())
}
