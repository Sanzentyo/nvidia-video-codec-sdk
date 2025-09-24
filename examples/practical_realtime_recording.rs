use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use cudarc::driver::CudaContext;
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

/// フレームデータとメタデータ
#[derive(Clone)]
struct FrameData {
    data: Vec<u8>,
    timestamp: u64,
    frame_index: u64,
}

/// エンコード済みフレームデータ
struct EncodedFrame {
    data: Vec<u8>,
    frame_index: u64,
    picture_type: String,
    timestamp: u64,
}

/// 録画統計情報
struct RecordingStats {
    frames_generated: AtomicU64,
    frames_encoded: AtomicU64,
    frames_written: AtomicU64,
    frames_dropped: AtomicU64,
    start_time: Instant,
}

impl RecordingStats {
    fn new() -> Self {
        Self {
            frames_generated: AtomicU64::new(0),
            frames_encoded: AtomicU64::new(0),
            frames_written: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn print_stats(&self) {
        let generated = self.frames_generated.load(Ordering::Relaxed);
        let encoded = self.frames_encoded.load(Ordering::Relaxed);
        let written = self.frames_written.load(Ordering::Relaxed);
        let dropped = self.frames_dropped.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        if elapsed > 0.0 {
            println!(
                "Stats: Gen={}, Enc={}, Out={}, Drop={}, GenFPS={:.1}, EncFPS={:.1}",
                generated, encoded, written, dropped,
                generated as f64 / elapsed,
                encoded as f64 / elapsed
            );
        }
    }
}

/// 実用的なリアルタイム録画実装
struct PracticalRealtimeRecorder {
    width: u32,
    height: u32,
    fps: u32,
    output_file: String,
}

impl PracticalRealtimeRecorder {
    fn new(width: u32, height: u32, fps: u32, output_file: String) -> Self {
        Self {
            width,
            height,
            fps,
            output_file,
        }
    }

    /// 録画開始（チャンネルベースの実装）
    fn start_recording(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stats = Arc::new(RecordingStats::new());
        let running = Arc::new(AtomicBool::new(true));

        // チャンネル作成
        let (frame_tx, frame_rx) = mpsc::channel::<FrameData>();
        let (encoded_tx, encoded_rx) = mpsc::channel::<EncodedFrame>();

        // フレーム生成スレッド
        let generator_handle = self.start_frame_generator(
            Arc::clone(&stats),
            Arc::clone(&running),
            frame_tx,
        );

        // エンコードスレッド（メインスレッドで実行）
        let encoder_handle = self.start_encoder_thread(
            Arc::clone(&stats),
            Arc::clone(&running),
            frame_rx,
            encoded_tx,
        )?;

        // 出力スレッド
        let output_handle = self.start_output_thread(
            Arc::clone(&stats),
            Arc::clone(&running),
            encoded_rx,
        )?;

        // 統計表示スレッド
        let stats_handle = self.start_stats_thread(Arc::clone(&stats), Arc::clone(&running));

        println!("Recording started ({}x{} @ {}fps). Recording for 10 seconds...", 
                 self.width, self.height, self.fps);

        // 10秒間録画
        thread::sleep(Duration::from_secs(10));

        // 録画停止
        running.store(false, Ordering::Relaxed);

        // スレッド終了待ち
        generator_handle.join().map_err(|_| "Frame generator thread panicked")?;
        encoder_handle.join().map_err(|_| "Encoder thread panicked")?;
        output_handle.join().map_err(|_| "Output thread panicked")?;
        stats_handle.join().map_err(|_| "Stats thread panicked")?;

        stats.print_stats();
        println!("Recording completed!");

        Ok(())
    }

    /// フレーム生成スレッド
    fn start_frame_generator(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
        frame_tx: Sender<FrameData>,
    ) -> thread::JoinHandle<()> {
        let width = self.width;
        let height = self.height;
        let fps = self.fps;

        thread::spawn(move || {
            let frame_duration = Duration::from_nanos(1_000_000_000 / fps as u64);
            let mut next_frame_time = Instant::now();
            let mut frame_index = 0u64;

            while running.load(Ordering::Relaxed) {
                let now = Instant::now();
                
                if now >= next_frame_time {
                    // テスト用フレーム生成
                    let frame_data = generate_test_frame(width, height, frame_index, now);
                    
                    let frame = FrameData {
                        data: frame_data,
                        timestamp: now.duration_since(stats.start_time).as_micros() as u64,
                        frame_index,
                    };

                    // フレーム送信
                    match frame_tx.send(frame) {
                        Ok(_) => {
                            stats.frames_generated.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                            break; // チャンネルが閉じられた
                        }
                    }

                    frame_index += 1;
                    next_frame_time += frame_duration;
                } else {
                    // 次のフレーム時刻まで短時間待機
                    let sleep_time = next_frame_time.saturating_duration_since(now);
                    if sleep_time > Duration::from_millis(1) {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            }
            
            println!("Frame generator stopped");
        })
    }

    /// エンコードスレッド
    fn start_encoder_thread(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
        frame_rx: Receiver<FrameData>,
        encoded_tx: Sender<EncodedFrame>,
    ) -> Result<thread::JoinHandle<()>, Box<dyn std::error::Error>> {
        let width = self.width;
        let height = self.height;

        // エンコーダー初期化（メインスレッドで）
        let cuda_ctx = CudaContext::new(0)?;
        let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;

        let encode_guid = NV_ENC_CODEC_H264_GUID;
        let preset_guid = NV_ENC_PRESET_P1_GUID;
        let profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
        let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
        let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

        let mut preset_config = encoder.get_preset_config(encode_guid, preset_guid, tuning_info)?;

        let mut initialize_params = EncoderInitParams::new(encode_guid, width, height);
        initialize_params
            .preset_guid(preset_guid)
            .tuning_info(tuning_info)
            .display_aspect_ratio(16, 9)
            .framerate(self.fps, 1)
            .enable_picture_type_decision()
            .encode_config(&mut preset_config.presetCfg);

        let session = encoder.start_session(buffer_format, initialize_params)?;

        // バッファ作成（複数のバッファを循環使用）
        let buffer_count = 4;
        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();

        for _ in 0..buffer_count {
            input_buffers.push(session.create_input_buffer()?);
            output_buffers.push(session.create_output_bitstream()?);
        }

        let handle = thread::spawn(move || {
            let mut buffer_index = 0usize;
            
            while running.load(Ordering::Relaxed) {
                // フレーム受信（タイムアウト付き）
                let frame = match frame_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(frame) => frame,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };

                // バッファ選択
                let input_buffer = &mut input_buffers[buffer_index];
                let output_buffer = &mut output_buffers[buffer_index];
                buffer_index = (buffer_index + 1) % buffer_count;

                // フレームデータをバッファに書き込み
                {
                    let mut buffer_lock = input_buffer.lock().expect("Input buffer should be lockable");
                    unsafe {
                        buffer_lock.write(&frame.data);
                    }
                }

                // エンコード実行
                match session.encode_picture(input_buffer, output_buffer, Default::default()) {
                    Ok(_) => {
                        // エンコード結果を取得
                        let lock = output_buffer.lock().expect("Output buffer should be lockable");
                        let data = lock.data().to_vec();
                        let picture_type = format!("{:?}", lock.picture_type());

                        let encoded_frame = EncodedFrame {
                            data,
                            frame_index: frame.frame_index,
                            picture_type,
                            timestamp: frame.timestamp,
                        };

                        // エンコード済みフレーム送信
                        if encoded_tx.send(encoded_frame).is_ok() {
                            stats.frames_encoded.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        eprintln!("Encoding error for frame {}: {}", frame.frame_index, e);
                    }
                }
            }
            
            println!("Encoder stopped");
        });

        Ok(handle)
    }

    /// 出力スレッド
    fn start_output_thread(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
        encoded_rx: Receiver<EncodedFrame>,
    ) -> Result<thread::JoinHandle<()>, Box<dyn std::error::Error>> {
        let output_file = self.output_file.clone();
        
        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&output_file)?;

        let handle = thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                // エンコード済みフレーム受信
                let encoded_frame = match encoded_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(frame) => frame,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };

                // ファイルに書き込み
                match out_file.write_all(&encoded_frame.data) {
                    Ok(_) => {
                        stats.frames_written.fetch_add(1, Ordering::Relaxed);
                        if encoded_frame.frame_index % 30 == 0 {
                            println!("Written frame {} ({})", encoded_frame.frame_index, encoded_frame.picture_type);
                        }
                    }
                    Err(e) => {
                        eprintln!("Write error for frame {}: {}", encoded_frame.frame_index, e);
                    }
                }
            }
            
            println!("Output writer stopped");
        });

        Ok(handle)
    }

    /// 統計表示スレッド
    fn start_stats_thread(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(2));
                stats.print_stats();
            }
        })
    }
}

/// テスト用フレーム生成（時間による色変化）
fn generate_test_frame(width: u32, height: u32, frame_index: u64, now: Instant) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let time = now.elapsed().as_secs_f32();
    
    for y in 0..height {
        for x in 0..width {
            // 時間に応じて色が変化するパターン
            let red = (255.0 * (x as f32 / width as f32) * (0.5 + 0.5 * (time * 0.5).sin())) as u8;
            let green = (255.0 * (y as f32 / height as f32) * (0.5 + 0.5 * (time * 0.3).cos())) as u8;
            let blue = (255.0 * (0.5 + 0.5 * (time + frame_index as f32 * 0.1).sin())) as u8;
            
            // ARGB形式で格納
            data.push(blue);  // B
            data.push(green); // G  
            data.push(red);   // R
            data.push(255);   // A
        }
    }
    
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting practical realtime recording example...");
    
    let mut recorder = PracticalRealtimeRecorder::new(
        1920, 
        1080, 
        30, 
        "realtime_output.bin".to_string()
    );
    
    recorder.start_recording()?;
    
    println!("Use: ffmpeg -i realtime_output.bin -vcodec copy realtime_output.mp4");
    
    Ok(())
}