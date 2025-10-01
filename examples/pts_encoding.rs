use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use cudarc::driver::CudaContext;
use image::GenericImageView;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
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

/// PTS計算のための時間基準
struct PtsCalculator {
    start_time: SystemTime,
    frame_duration_us: u64, // マイクロ秒単位のフレーム間隔
    base_pts: u64,
}

impl PtsCalculator {
    fn new(fps: u32) -> Self {
        let frame_duration_us = 1_000_000 / fps as u64; // マイクロ秒単位
        let start_time = SystemTime::now();

        // 開始時刻をPTSベースとして使用
        let base_pts = start_time
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        Self {
            start_time,
            frame_duration_us,
            base_pts,
        }
    }

    /// フレーム番号からPTSを計算
    fn get_pts_for_frame(&self, frame_index: u64) -> u64 {
        self.base_pts + (frame_index * self.frame_duration_us)
    }

    /// 実際の経過時間からPTSを計算
    fn get_realtime_pts(&self) -> u64 {
        let elapsed = self.start_time.elapsed().unwrap_or(Duration::ZERO);
        self.base_pts + elapsed.as_micros() as u64
    }
}

/// PTS付きでエンコード
fn main() {
    let input_dir = Path::new("input/save_frames/20250924_180745");
    let output_dir = Path::new("output/pts_encoding");

    std::fs::create_dir_all(output_dir).expect("Creating output directory should succeed.");

    let png_files = get_png_files(input_dir);
    if png_files.is_empty() {
        eprintln!("No PNG files found in {}", input_dir.display());
        return;
    }

    println!("Found {} PNG files to encode with PTS", png_files.len());

    // 最初の画像でサイズを取得
    let first_img = image::open(&png_files[0]).expect("Failed to open first image");
    let (width, height) = first_img.dimensions();

    println!("Image dimensions: {}x{}", width, height);

    // CUDA & エンコーダー初期化
    let cuda_ctx = CudaContext::new(0).expect("CUDA should be available.");
    let encoder = Encoder::initialize_with_cuda(cuda_ctx)
        .expect("NVIDIA Video Codec SDK should be installed.");

    let encode_guid = NV_ENC_CODEC_H264_GUID;
    let preset_guid = NV_ENC_PRESET_P1_GUID;
    let profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
    let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    let mut preset_config = encoder
        .get_preset_config(encode_guid, preset_guid, tuning_info)
        .expect("Encoder should be able to create config based on presets.");

    // フレームレート設定
    let fps = 30u32;
    let mut initialize_params = EncoderInitParams::new(encode_guid, width, height);
    initialize_params
        .preset_guid(preset_guid)
        .tuning_info(tuning_info)
        .display_aspect_ratio(16, 9)
        .framerate(fps, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_config.presetCfg);

    let session = encoder
        .start_session(buffer_format, initialize_params)
        .expect("Encoder should be initialized correctly.");

    // バッファ作成
    let num_bufs = usize::try_from(preset_config.presetCfg.frameIntervalP)
        .expect("frame intervalP should be positive")
        + usize::try_from(preset_config.presetCfg.rcParams.lookaheadDepth)
            .expect("lookahead depth should be positive");

    let mut output_buffers: Vec<_> = (0..num_bufs)
        .map(|_| {
            session
                .create_output_bitstream()
                .expect("Output buffer should be created")
        })
        .collect();

    // 出力ファイル
    let mut out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_dir.join("pts_encoded_output.bin"))
        .expect("Output file should be created");

    // フレーム読み込み
    let mut loaded_frames: Vec<(std::path::PathBuf, Vec<u8>)> = Vec::new();
    for png_path in &png_files {
        match load_png_as_argb(png_path, width, height) {
            Ok(data) => loaded_frames.push((png_path.clone(), data)),
            Err(e) => eprintln!("Failed to load PNG {}: {}", png_path.display(), e),
        }
    }

    if loaded_frames.is_empty() {
        eprintln!("No valid PNG frames were loaded.");
        return;
    }

    // PTS計算器を初期化
    let pts_calculator = PtsCalculator::new(fps);

    println!("Starting encoding with PTS...");
    let encode_start = std::time::Instant::now();

    let mut input_buffer = session
        .create_input_buffer()
        .expect("Input buffer should be created");

    // エンコードループ（PTS付き）
    for (i, (path, argb_data)) in loaded_frames.iter().enumerate() {
        println!(
            "Processing frame {} / {}: {}",
            i + 1,
            loaded_frames.len(),
            path.display()
        );

        let output_bitstream = &mut output_buffers[i % num_bufs];

        // フレームデータをバッファに書き込み
        {
            let mut buffer_lock = input_buffer
                .lock()
                .expect("Input buffer should be lockable");
            unsafe {
                buffer_lock.write(argb_data);
            }
        }

        // PTSを計算
        let pts = pts_calculator.get_pts_for_frame(i as u64);

        // エンコードパラメータにPTSを設定
        let encode_params = EncodePictureParams {
            input_timestamp: pts,
            ..Default::default()
        };

        // エンコード実行
        session
            .encode_picture(&mut input_buffer, output_bitstream, encode_params)
            .expect("Encoder should be able to encode pictures");

        // 結果を取得
        let lock = output_bitstream
            .lock()
            .expect("Bitstream lock should be available");

        println!(
            "Frame {}: PTS={}, Output timestamp={}, duration={}, picture_type={:?}",
            i + 1,
            pts,
            lock.timestamp(),
            lock.duration(),
            lock.picture_type()
        );

        let data = lock.data();
        out_file.write_all(data).expect("Writing should succeed");
    }

    println!("Encoding completed! {} files processed.", png_files.len());
    println!("Total encoding time: {:.2?}", encode_start.elapsed());

    // PTS情報をログファイルに出力
    let mut pts_log = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_dir.join("pts_log.txt"))
        .expect("PTS log file should be created");

    writeln!(pts_log, "PTS Encoding Log").unwrap();
    writeln!(pts_log, "================").unwrap();
    writeln!(pts_log, "FPS: {}", fps).unwrap();
    writeln!(
        pts_log,
        "Frame duration (μs): {}",
        pts_calculator.frame_duration_us
    )
    .unwrap();
    writeln!(pts_log, "Base PTS: {}", pts_calculator.base_pts).unwrap();
    writeln!(pts_log, "Total frames: {}", loaded_frames.len()).unwrap();
    writeln!(pts_log, "").unwrap();

    for i in 0..loaded_frames.len() {
        let pts = pts_calculator.get_pts_for_frame(i as u64);
        writeln!(pts_log, "Frame {}: PTS = {} μs", i, pts).unwrap();
    }

    println!(
        "PTS log saved to: {}",
        output_dir.join("pts_log.txt").display()
    );
    println!(
        "Use: ffmpeg -i {} -vcodec copy output.mp4",
        output_dir.join("pts_encoded_output.bin").display()
    );
}
