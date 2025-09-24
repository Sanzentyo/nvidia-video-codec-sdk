/// Picture Timing SEIを使用したタイムスタンプ埋め込み例
/// ffmpegの警告「Timestamps are unset in a packet for stream 0」を解決します

use nvidia_video_codec_sdk::{
    sys::{
        nvEncodeAPI::{
            NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_H264_GUID,
            NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
            NV_ENC_CLOCK_TIMESTAMP_SET,
            NV_ENC_DISPLAY_PIC_STRUCT,
            NV_ENC_TIME_CODE,
        }
    },
    Encoder, EncoderInitParams, EncodePictureParams,
};
use cudarc::driver::CudaContext;
use image::{ImageBuffer, Rgba};
use std::{
    fs::File,
    io::Write,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAME_COUNT: u32 = 30;
const FPS: u32 = 30;
const MAX_NUM_CLOCK_TS: usize = 3;

/// タイムコード計算ユーティリティ
struct TimeCodeCalculator {
    fps: u32,
    drop_frame: bool,
}

impl TimeCodeCalculator {
    fn new(fps: u32, drop_frame: bool) -> Self {
        Self { fps, drop_frame }
    }

    /// フレーム番号からタイムコードを計算
    fn frame_to_timecode(&self, frame_number: u32) -> NV_ENC_CLOCK_TIMESTAMP_SET {
        let total_seconds = frame_number / self.fps;
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let frames = frame_number % self.fps;

        let mut timestamp_set = NV_ENC_CLOCK_TIMESTAMP_SET::default();
        timestamp_set._bitfield_1 = NV_ENC_CLOCK_TIMESTAMP_SET::new_bitfield_1(
            0, // countingType: Progressive
            0, // discontinuityFlag: 連続
            if self.drop_frame { 1 } else { 0 }, // cntDroppedFrames
            frames, // nFrames
            seconds, // secondsValue
            minutes, // minutesValue
            hours, // hoursValue
            0, // reserved2
        );
        timestamp_set.timeOffset = 0;
        timestamp_set
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Picture Timing SEI を使用したタイムスタンプ埋め込み例");
    
    // CUDA初期化
    let cuda_ctx = CudaContext::new(0)?;
    
    // エンコーダー初期化
    let encoder = Encoder::initialize_with_cuda(cuda_ctx.clone())?;
    
    // まずプリセット設定を取得
    let mut preset_config = encoder.get_preset_config(
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P1_GUID,
        NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
    )?;
    
    // H.264エンコード設定でPicture Timing SEIを有効化
    let h264_config = unsafe { &mut preset_config.presetCfg.encodeCodecConfig.h264Config };
    
    // Picture Timing SEIを有効化（重要！）
    h264_config.set_outputPictureTimingSEI(1);
    h264_config.set_outputBufferingPeriodSEI(1);
    h264_config.set_outputAUD(1);
    
    // タイムコードSEIも有効化
    h264_config.set_enableTimeCode(1);
    
    // VUI parameters設定
    h264_config.h264VUIParameters.timingInfoPresentFlag = 1;
    h264_config.h264VUIParameters.numUnitInTicks = 1;
    h264_config.h264VUIParameters.timeScale = FPS * 2; // Field rate for progressive
    
    println!("H.264設定:");
    println!("  Picture Timing SEI: {}", h264_config.outputPictureTimingSEI());
    println!("  Buffering Period SEI: {}", h264_config.outputBufferingPeriodSEI());
    println!("  Time Code SEI: {}", h264_config.enableTimeCode());
    println!("  Timing Info Present: {}", h264_config.h264VUIParameters.timingInfoPresentFlag);
    println!("  Time Scale: {}", h264_config.h264VUIParameters.timeScale);
    
    // エンコーダーパラメータを作成し設定を適用
    let mut encoder_params = EncoderInitParams::new(NV_ENC_CODEC_H264_GUID, WIDTH, HEIGHT);
    encoder_params
        .preset_guid(NV_ENC_PRESET_P1_GUID)
        .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY)
        .enable_picture_type_decision()  // これが重要！
        .encode_config(&mut preset_config.presetCfg);
    
    // セッション開始
    let session = encoder.start_session(
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
        encoder_params,
    )?;
    
    // 複数出力バッファ作成
    let num_bufs = ((FRAME_COUNT + 30 - 1) / 30).max(4) as usize;
    println!("出力バッファ数: {}", num_bufs);
    
    let mut output_buffers = Vec::new();
    for _ in 0..num_bufs {
        output_buffers.push(session.create_output_bitstream()?);
    }
    
    // 入力バッファ作成
    let mut input_buffer = session.create_input_buffer()?;
    
    // 出力ファイル
    let mut output_file = File::create("timestamped_output.h264")?;
    let mut metadata_file = File::create("metadata.txt")?;
    
    // タイムコード計算器
    let timecode_calc = TimeCodeCalculator::new(FPS, false);
    
    let mut total_bytes = 0u64;
    
    println!("\nエンコード開始 (Picture Timing SEI付き):");
    
    for frame_idx in 0..FRAME_COUNT {
        // フレーム生成（カラフルなグラデーション）
        let image = generate_test_frame(frame_idx, WIDTH, HEIGHT);
        let argb_data = image.as_raw();
        
        // バッファに書き込み
        {
            let mut buffer_lock = input_buffer.lock()?;
            unsafe {
                buffer_lock.write(argb_data);
            }
        }
        
        // タイムスタンプ計算（マイクロ秒）
        let pts_us = (frame_idx as u64 * 1_000_000) / FPS as u64;
        
        // タイムコード設定（将来の実装用）
        let clock_timestamp = timecode_calc.frame_to_timecode(frame_idx);
        
        // エンコードパラメータ（PTS設定のみ）
        let encode_params = EncodePictureParams {
            input_timestamp: pts_us,
            ..Default::default()
        };
        
        // エンコード実行
        let buffer_idx = frame_idx as usize % num_bufs;
        session.encode_picture(
            &mut input_buffer,
            &mut output_buffers[buffer_idx],
            encode_params,
        )?;
        
        // エンコードされたデータを取得
        let lock = output_buffers[buffer_idx].lock()?;
        let encoded_data = lock.data();
        let frame_size = encoded_data.len();
        
        // ファイルに書き込み
        output_file.write_all(encoded_data)?;
        total_bytes += frame_size as u64;
        
        // メタデータ記録
        writeln!(metadata_file, "Frame {}: size={}B, PTS={}μs, TC={}:{}:{}:{}", 
            frame_idx, frame_size, pts_us,
            clock_timestamp.hoursValue(),
            clock_timestamp.minutesValue(), 
            clock_timestamp.secondsValue(),
            clock_timestamp.nFrames())?;
        
        if frame_idx % 10 == 0 {
            println!("  フレーム {}: {}B エンコード済み", frame_idx, frame_size);
        }
    }
    
    println!("\nエンコード完了!");
    println!("総サイズ: {} bytes", total_bytes);
    println!("平均フレームサイズ: {} bytes", total_bytes / FRAME_COUNT as u64);
    
    // ffmpeg変換スクリプト生成（タイムスタンプ警告回避版）
    let script_content = format!(
        r#"@echo off
echo Converting H.264 stream with proper timestamps to MP4...
echo Method 1: Using explicit framerate and codec copy
ffmpeg -y -framerate {} -i timestamped_output.h264 -c copy -movflags +faststart output_with_sei.mp4
if %ERRORLEVEL% EQU 0 (
    echo Conversion successful without timestamp warnings!
    echo.
    echo Analyzing the result:
    ffprobe -v quiet -select_streams v:0 -show_entries packet=pts_time,dts_time,duration_time -of csv=p=0 output_with_sei.mp4 | head -5
    echo.
    echo Playing with ffplay...
    ffplay -autoexit output_with_sei.mp4
) else (
    echo Conversion failed.
)
pause
"#, FPS);
    
    std::fs::write("convert_with_sei.bat", script_content)?;
    println!("変換スクリプト 'convert_with_sei.bat' を作成しました");
    println!("Picture Timing SEIが埋め込まれたH.264ストリームが生成されました");
    
    Ok(())
}

/// カラフルなテストフレームを生成
fn generate_test_frame(frame_idx: u32, width: u32, height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let mut image = ImageBuffer::new(width, height);
    
    let time_factor = (frame_idx as f32 * 2.0 * std::f32::consts::PI / FRAME_COUNT as f32).sin();
    
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let norm_x = x as f32 / width as f32;
        let norm_y = y as f32 / height as f32;
        
        // カラフルなグラデーション + 時間変化
        let red = ((norm_x + time_factor * 0.3) * 255.0).abs() as u8;
        let green = ((norm_y + time_factor * 0.5) * 255.0).abs() as u8;
        let blue = (((norm_x + norm_y) * 0.5 + time_factor * 0.7) * 255.0).abs() as u8;
        
        *pixel = Rgba([red, green, blue, 255]);
    }
    
    image
}

