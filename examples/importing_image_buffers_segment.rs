use std::{
    fs,
    path::Path,
};

use cudarc::driver::CudaContext;
use image::GenericImageView;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID,
        NV_ENC_PRESET_P1_GUID,
        NV_ENC_TUNING_INFO,
    },
    Encoder,
    EncoderInitParams,
};

/// 指定されたディレクトリ内のPNG画像ファイルを名前順に取得する
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
    
    // ファイル名で並び替え
    files.sort_by(|a, b| {
        a.file_name().cmp(&b.file_name())
    });
    
    files
}

/// PNG画像を読み込んでARGB形式のバイト配列に変換する
fn load_png_as_argb(path: &Path, target_width: u32, target_height: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    
    // リサイズが必要な場合はリサイズする
    let resized_img = if width != target_width || height != target_height {
        image::imageops::resize(&rgb_img, target_width, target_height, image::imageops::FilterType::Lanczos3)
    } else {
        rgb_img
    };
    
    // RGBからARGBに変換
    let mut argb_data = Vec::with_capacity((target_width * target_height * 4) as usize);
    
    for pixel in resized_img.pixels() {
        let [r, g, b] = pixel.0;
        // ARGB形式で格納 (Blue, Green, Red, Alpha)
        argb_data.push(b);  // Blue
        argb_data.push(g);  // Green
        argb_data.push(r);  // Red
        argb_data.push(255); // Alpha
    }
    
    Ok(argb_data)
}



/// PNG画像を読み込んでそれぞれを個別の.binファイルとしてエンコード
fn main() {
    // 入力フォルダと出力フォルダのパス
    let input_dir = Path::new("input/save_frames/20250924_180745");
    let output_dir = Path::new("output/image_buffers");
    
    // 出力ディレクトリを作成
    std::fs::create_dir_all(output_dir)
        .expect("Creating output directory should succeed.");

    // PNG画像ファイルを取得
    let png_files = get_png_files(input_dir);
    if png_files.is_empty() {
        eprintln!("No PNG files found in {}", input_dir.display());
        return;
    }

    println!("Found {} PNG files to encode", png_files.len());

    // 最初の画像を読み込んでサイズを取得
    let first_img_path = &png_files[0];
    let first_img = image::open(first_img_path)
        .expect("Failed to open first image to determine dimensions");
    let (width, height) = first_img.dimensions();
    
    println!("Image dimensions: {}x{}", width, height);

    // Create a new CudaContext to interact with cuda.
    let cuda_ctx = CudaContext::new(0).expect("Cuda should be installed correctly.");

    let encoder = Encoder::initialize_with_cuda(cuda_ctx.clone())
        .expect("NVIDIA Video Codec SDK should be installed correctly.");

    // Get all encode guids supported by the GPU.
    let encode_guids = encoder
        .get_encode_guids()
        .expect("The encoder should be able to get the supported guids.");
    let encode_guid = NV_ENC_CODEC_H264_GUID;
    assert!(encode_guids.contains(&encode_guid));

    // Get available preset guids based on encode guid.
    let preset_guids = encoder
        .get_preset_guids(encode_guid)
        .expect("The encoder should have a preset for H.264.");
    let preset_guid = NV_ENC_PRESET_P1_GUID;
    assert!(preset_guids.contains(&preset_guid));

    // Get available profiles based on encode guid.
    let profile_guids = encoder
        .get_profile_guids(encode_guid)
        .expect("The encoder should have a profile for H.264.");
    let profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
    assert!(profile_guids.contains(&profile_guid));

    // Get input formats based on the encode guid.
    let input_formats = encoder
        .get_supported_input_formats(encode_guid)
        .expect("The encoder should be able to get supported input buffer formats.");
    let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    assert!(input_formats.contains(&buffer_format));

    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    // Get the preset config based on the selected encode guid (H.264), selected
    // preset (`LOW_LATENCY`), and tuning info (`ULTRA_LOW_LATENCY`).
    let mut preset_config = encoder
        .get_preset_config(encode_guid, preset_guid, tuning_info)
        .expect("Encoder should be able to create config based on presets.");

    // Initialize a new encoder session based on the `preset_config`
    // we generated before.
    let mut initialize_params = EncoderInitParams::new(encode_guid, width, height);
    initialize_params
        .preset_guid(preset_guid)
        .tuning_info(tuning_info)
        .display_aspect_ratio(16, 9)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_config.presetCfg);
    let session = encoder
        .start_session(buffer_format, initialize_params)
        .expect("Encoder should be initialized correctly.");

    // まず全て読み込んでメモリに保持する (Path とデータをペアにする)
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

    let encode_start = std::time::Instant::now();

    // 入力バッファと出力ビットストリームを一回作成して使いまわす
    let mut input_buffer = session
        .create_input_buffer()
        .expect("The encoder should be able to create input buffers.");

    let mut output_buffer = session
        .create_output_bitstream()
        .expect("The encoder should be able to create bitstreams.");

    // 各フレームを順次エンコードして個別ファイルに保存
    for (i, (path, argb_data)) in loaded_frames.iter().enumerate() {
        println!("Processing frame {} / {}: {}", i + 1, loaded_frames.len(), path.display());

        // バッファをロックしてデータを書き込み
        {
            let mut buffer_lock = input_buffer
                .lock()
                .expect("Input buffer should be lockable.");
            unsafe {
                buffer_lock.write(&argb_data);
            }
        } // lock がここで落ちてアンロックされる

        // エンコード実行（同じ input_buffer / output_buffer を再利用）
        session
            .encode_picture(
                &mut input_buffer,
                &mut output_buffer,
                Default::default(),
            )
            .expect("Encoder should be able to encode valid pictures");

        // 出力をロックしてデータを取得し、個別ファイルに保存
        let lock = output_buffer
            .lock()
            .expect("Bitstream lock should be available.");

        let data = lock.data();
        
        // 画像ファイル名から出力ファイル名を生成
        let output_filename = path
            .file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("frame"))
            .to_string_lossy();
        let output_path = output_dir.join(format!("{}.bin", output_filename));

        // データを個別のファイルに保存
        std::fs::write(&output_path, data)
            .expect("Writing encoded data should succeed.");
            
        println!("Saved: {}", output_path.display());
    }

    println!("Encoding completed! {} files processed.", loaded_frames.len());
    println!("Total encoding time: {:.2?}", encode_start.elapsed());
}


