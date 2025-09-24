use std::{
    fs::OpenOptions,
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_PRESET_P1_GUID,
        NV_ENC_TUNING_INFO,
    },
    Encoder,
    EncoderInitParams,
};

/// フレームデータ
#[derive(Clone)]
struct FrameData {
    data: Vec<u8>,
    timestamp: u64,
    frame_index: u64,
}

/// 録画統計情報
struct RecordingStats {
    frames_generated: AtomicU64,
    frames_encoded: AtomicU64,
    frames_dropped: AtomicU64,
    start_time: Instant,
}

impl RecordingStats {
    fn new() -> Self {
        Self {
            frames_generated: AtomicU64::new(0),
            frames_encoded: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn print_stats(&self) {
        let generated = self.frames_generated.load(Ordering::Relaxed);
        let encoded = self.frames_encoded.load(Ordering::Relaxed);
        let dropped = self.frames_dropped.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        if elapsed > 0.0 {
            println!(
                "Stats: Generated={}, Encoded={}, Dropped={}, Gen FPS={:.1}, Enc FPS={:.1}",
                generated, encoded, dropped,
                generated as f64 / elapsed,
                encoded as f64 / elapsed
            );
        }
    }
}

/// シングルスレッドベースのリアルタイム録画
/// （実際のアプリケーションでは、フレーム取得とエンコードを同じスレッドで処理することが多い）
struct SimpleRealtimeRecorder {
    width: u32,
    height: u32,
    fps: u32,
    output_file: String,
}

impl SimpleRealtimeRecorder {
    fn new(width: u32, height: u32, fps: u32, output_file: String) -> Self {
        Self {
            width,
            height,
            fps,
            output_file,
        }
    }

    /// シングルスレッドでのリアルタイム録画
    fn start_recording(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stats = Arc::new(RecordingStats::new());
        let running = Arc::new(AtomicBool::new(true));

        // 統計表示スレッドのみ別スレッド
        let stats_handle = self.start_stats_thread(Arc::clone(&stats), Arc::clone(&running));

        // エンコーダー初期化
        let cuda_ctx = CudaContext::new(0)?;
        let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;

        let encode_guid = NV_ENC_CODEC_H264_GUID;
        let preset_guid = NV_ENC_PRESET_P1_GUID;
        let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
        let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

        let mut preset_config = encoder.get_preset_config(encode_guid, preset_guid, tuning_info)?;

        let mut initialize_params = EncoderInitParams::new(encode_guid, self.width, self.height);
        initialize_params
            .preset_guid(preset_guid)
            .tuning_info(tuning_info)
            .display_aspect_ratio(16, 9)
            .framerate(self.fps, 1)
            .enable_picture_type_decision()
            .encode_config(&mut preset_config.presetCfg);

        let session = encoder.start_session(buffer_format, initialize_params)?;

        // バッファ作成（効率的なローテーション用）
        let num_bufs = usize::try_from(preset_config.presetCfg.frameIntervalP)
            .expect("frame intervalP should always be positive.")
            + usize::try_from(preset_config.presetCfg.rcParams.lookaheadDepth)
                .expect("lookahead depth should always be positive.");

        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();

        for _ in 0..num_bufs {
            input_buffers.push(session.create_input_buffer()?);
            output_buffers.push(session.create_output_bitstream()?);
        }

        // 出力ファイル
        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.output_file)?;

        println!("Recording started ({}x{} @ {}fps, {} buffers)", 
                 self.width, self.height, self.fps, num_bufs);

        // メインレコーディングループ
        let frame_duration = Duration::from_nanos(1_000_000_000 / self.fps as u64);
        let mut next_frame_time = Instant::now();
        let mut frame_index = 0u64;
        let mut buffer_index = 0usize;
        let recording_duration = Duration::from_secs(10);
        let recording_start = Instant::now();

        while recording_start.elapsed() < recording_duration {
            let now = Instant::now();
            
            // フレーム生成タイミングチェック
            if now >= next_frame_time {
                // フレーム生成
                let frame_data = generate_test_frame(self.width, self.height, frame_index, now);
                stats.frames_generated.fetch_add(1, Ordering::Relaxed);

                // バッファ選択
                let input_buffer = &mut input_buffers[buffer_index];
                let output_buffer = &mut output_buffers[buffer_index];
                buffer_index = (buffer_index + 1) % num_bufs;

                // エンコード処理
                {
                    let mut buffer_lock = input_buffer.lock()?;
                    unsafe {
                        buffer_lock.write(&frame_data);
                    }
                }

                // エンコード実行
                match session.encode_picture(input_buffer, output_buffer, Default::default()) {
                    Ok(_) => {
                        let lock = output_buffer.lock()?;
                        let data = lock.data();
                        
                        // ファイルに書き込み
                        out_file.write_all(data)?;
                        stats.frames_encoded.fetch_add(1, Ordering::Relaxed);

                        if frame_index % 30 == 0 {
                            println!("Encoded frame {} ({:?})", frame_index, lock.picture_type());
                        }
                    }
                    Err(e) => {
                        eprintln!("Encoding error for frame {}: {}", frame_index, e);
                        stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                    }
                }

                frame_index += 1;
                next_frame_time += frame_duration;
            } else {
                // 次のフレーム時刻まで待機
                let sleep_time = next_frame_time.saturating_duration_since(now);
                if sleep_time > Duration::from_millis(1) {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }

        // 録画停止
        running.store(false, Ordering::Relaxed);
        stats_handle.join().map_err(|_| "Stats thread panicked")?;

        stats.print_stats();
        println!("Recording completed! Output saved to: {}", self.output_file);

        Ok(())
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

/// Producer-Consumer パターンのリアルタイム録画
/// （フレーム生成とエンコードを分離）
struct ProducerConsumerRecorder {
    width: u32,
    height: u32,
    fps: u32,
    output_file: String,
}

impl ProducerConsumerRecorder {
    fn new(width: u32, height: u32, fps: u32, output_file: String) -> Self {
        Self {
            width,
            height,
            fps,
            output_file,
        }
    }

    /// Producer-Consumer パターンでの録画
    fn start_recording(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let stats = Arc::new(RecordingStats::new());
        let running = Arc::new(AtomicBool::new(true));

        // チャンネル作成（バウンドチャンネルでバックプレッシャー制御）
        let (frame_tx, frame_rx) = mpsc::sync_channel::<FrameData>(10);

        // フレーム生成スレッド
        let generator_handle = self.start_frame_generator(
            Arc::clone(&stats),
            Arc::clone(&running),
            frame_tx,
        );

        // 統計表示スレッド
        let stats_handle = self.start_stats_thread(Arc::clone(&stats), Arc::clone(&running));

        println!("Recording started with Producer-Consumer pattern ({}x{} @ {}fps)", 
                 self.width, self.height, self.fps);

        // エンコード処理（メインスレッド）
        self.run_encoder(Arc::clone(&stats), Arc::clone(&running), frame_rx)?;

        // 録画停止
        running.store(false, Ordering::Relaxed);

        // スレッド終了待ち
        generator_handle.join().map_err(|_| "Frame generator thread panicked")?;
        stats_handle.join().map_err(|_| "Stats thread panicked")?;

        stats.print_stats();
        println!("Recording completed! Output saved to: {}", self.output_file);

        Ok(())
    }

    /// フレーム生成スレッド
    fn start_frame_generator(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
        frame_tx: mpsc::SyncSender<FrameData>,
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
                    let frame_data = generate_test_frame(width, height, frame_index, now);
                    
                    let frame = FrameData {
                        data: frame_data,
                        timestamp: now.duration_since(stats.start_time).as_micros() as u64,
                        frame_index,
                    };

                    // ノンブロッキング送信（バッファが満杯の場合はフレームドロップ）
                    match frame_tx.try_send(frame) {
                        Ok(_) => {
                            stats.frames_generated.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(mpsc::TrySendError::Full(_)) => {
                            stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(mpsc::TrySendError::Disconnected(_)) => {
                            break; // チャンネル切断
                        }
                    }

                    frame_index += 1;
                    next_frame_time += frame_duration;
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            
            println!("Frame generator stopped");
        })
    }

    /// エンコード処理（メインスレッド）
    fn run_encoder(
        &self,
        stats: Arc<RecordingStats>,
        running: Arc<AtomicBool>,
        frame_rx: mpsc::Receiver<FrameData>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // エンコーダー初期化
        let cuda_ctx = CudaContext::new(0)?;
        let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;

        let encode_guid = NV_ENC_CODEC_H264_GUID;
        let preset_guid = NV_ENC_PRESET_P1_GUID;
        let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
        let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

        let mut preset_config = encoder.get_preset_config(encode_guid, preset_guid, tuning_info)?;

        let mut initialize_params = EncoderInitParams::new(encode_guid, self.width, self.height);
        initialize_params
            .preset_guid(preset_guid)
            .tuning_info(tuning_info)
            .display_aspect_ratio(16, 9)
            .framerate(self.fps, 1)
            .enable_picture_type_decision()
            .encode_config(&mut preset_config.presetCfg);

        let session = encoder.start_session(buffer_format, initialize_params)?;

        // バッファ作成
        let num_bufs = 4; // シンプルに固定
        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();

        for _ in 0..num_bufs {
            input_buffers.push(session.create_input_buffer()?);
            output_buffers.push(session.create_output_bitstream()?);
        }

        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.output_file)?;

        let mut buffer_index = 0usize;
        let recording_duration = Duration::from_secs(10);

        // エンコードループ
        while stats.start_time.elapsed() < recording_duration {
            match frame_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(frame) => {
                    let input_buffer = &mut input_buffers[buffer_index];
                    let output_buffer = &mut output_buffers[buffer_index];
                    buffer_index = (buffer_index + 1) % num_bufs;

                    // エンコード処理
                    {
                        let mut buffer_lock = input_buffer.lock()?;
                        unsafe {
                            buffer_lock.write(&frame.data);
                        }
                    }

                    match session.encode_picture(input_buffer, output_buffer, Default::default()) {
                        Ok(_) => {
                            let lock = output_buffer.lock()?;
                            let data = lock.data();
                            out_file.write_all(data)?;
                            stats.frames_encoded.fetch_add(1, Ordering::Relaxed);

                            if frame.frame_index % 30 == 0 {
                                println!("Encoded frame {} ({:?})", frame.frame_index, lock.picture_type());
                            }
                        }
                        Err(e) => {
                            eprintln!("Encoding error: {}", e);
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        Ok(())
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

/// テスト用フレーム生成
fn generate_test_frame(width: u32, height: u32, frame_index: u64, now: Instant) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let time = now.elapsed().as_secs_f32();
    
    for y in 0..height {
        for x in 0..width {
            let red = (255.0 * (x as f32 / width as f32) * (0.5 + 0.5 * (time * 0.5).sin())) as u8;
            let green = (255.0 * (y as f32 / height as f32) * (0.5 + 0.5 * (time * 0.3).cos())) as u8;
            let blue = (255.0 * (0.5 + 0.5 * (time + frame_index as f32 * 0.1).sin())) as u8;
            
            data.push(blue);  // B
            data.push(green); // G
            data.push(red);   // R
            data.push(255);   // A
        }
    }
    
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Realtime recording examples");
    println!("1. Simple single-threaded approach");
    println!("2. Producer-Consumer pattern");
    
    // シンプルなシングルスレッドアプローチ
    println!("\n=== Simple Single-threaded Recording ===");
    let mut simple_recorder = SimpleRealtimeRecorder::new(
        1920, 
        1080, 
        30, 
        "simple_realtime.bin".to_string()
    );
    simple_recorder.start_recording()?;

    thread::sleep(Duration::from_secs(2));

    // Producer-Consumer パターン
    println!("\n=== Producer-Consumer Recording ===");
    let mut pc_recorder = ProducerConsumerRecorder::new(
        1920, 
        1080, 
        30, 
        "producer_consumer_realtime.bin".to_string()
    );
    pc_recorder.start_recording()?;

    println!("\nBoth recordings completed!");
    println!("Convert to video:");
    println!("ffmpeg -i simple_realtime.bin -vcodec copy simple_realtime.mp4");
    println!("ffmpeg -i producer_consumer_realtime.bin -vcodec copy producer_consumer_realtime.mp4");
    
    Ok(())
}