use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Condvar, Mutex,
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
        NV_ENC_PIC_TYPE,
    },
    Encoder,
    EncoderInitParams,
    Session,
    InputBuffer,
    OutputBitstream,
};

/// リアルタイム録画のための設定
#[derive(Clone)]
struct RecordingConfig {
    width: u32,
    height: u32,
    fps: u32,
    buffer_count: usize,
    output_file: String,
}

/// フレームデータとメタデータ
#[derive(Clone)]
struct FrameData {
    data: Vec<u8>,
    timestamp: u64,
    frame_index: u64,
}

/// バッファプールの管理
struct BufferPool {
    available_buffers: VecDeque<(InputBuffer, OutputBitstream)>,
    used_buffers: VecDeque<(InputBuffer, OutputBitstream, u64)>, // frame_indexも保持
    mutex: Mutex<()>,
    available_condvar: Condvar,
    used_condvar: Condvar,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            available_buffers: VecDeque::new(),
            used_buffers: VecDeque::new(),
            mutex: Mutex::new(()),
            available_condvar: Condvar::new(),
            used_condvar: Condvar::new(),
        }
    }

    /// バッファを初期化
    fn initialize(&mut self, session: &Session<InputBuffer>, buffer_count: usize) {
        let _lock = self.mutex.lock().unwrap();
        
        for _ in 0..buffer_count {
            let input_buffer = session
                .create_input_buffer()
                .expect("Input buffer should be created");
            let output_buffer = session
                .create_output_bitstream()
                .expect("Output buffer should be created");
            
            self.available_buffers.push_back((input_buffer, output_buffer));
        }
        
        println!("Initialized buffer pool with {} buffers", buffer_count);
    }

    /// 利用可能なバッファを取得
    fn get_available_buffer(&mut self) -> Option<(InputBuffer, OutputBitstream)> {
        let mut lock = self.mutex.lock().unwrap();
        
        // バッファが利用可能になるまで待機（タイムアウト付き）
        let timeout = Duration::from_millis(16); // ~60fps
        if self.available_buffers.is_empty() {
            lock = self.available_condvar.wait_timeout(lock, timeout).unwrap().0;
        }
        
        self.available_buffers.pop_front()
    }

    /// エンコード完了後にバッファを使用済みキューに追加
    fn mark_as_used(&mut self, buffers: (InputBuffer, OutputBitstream), frame_index: u64) {
        let _lock = self.mutex.lock().unwrap();
        self.used_buffers.push_back((buffers.0, buffers.1, frame_index));
        self.used_condvar.notify_one();
    }

    /// 使用済みバッファを取得（出力処理用）
    fn get_used_buffer(&mut self) -> Option<(InputBuffer, OutputBitstream, u64)> {
        let mut lock = self.mutex.lock().unwrap();
        
        if self.used_buffers.is_empty() {
            // 少し待機
            let timeout = Duration::from_millis(5);
            lock = self.used_condvar.wait_timeout(lock, timeout).unwrap().0;
        }
        
        self.used_buffers.pop_front()
    }

    /// バッファを利用可能キューに戻す
    fn return_buffer(&mut self, buffers: (InputBuffer, OutputBitstream)) {
        let _lock = self.mutex.lock().unwrap();
        self.available_buffers.push_back(buffers);
        self.available_condvar.notify_one();
    }
}

/// リアルタイム録画エンジン
struct RealtimeRecorder {
    config: RecordingConfig,
    session: Session<InputBuffer>,
    buffer_pool: Arc<Mutex<BufferPool>>,
    frame_queue: Arc<Mutex<VecDeque<FrameData>>>,
    frame_condvar: Arc<Condvar>,
    running: Arc<AtomicBool>,
    stats: Arc<RecordingStats>,
}

/// 録画統計情報
struct RecordingStats {
    frames_captured: AtomicU64,
    frames_encoded: AtomicU64,
    frames_dropped: AtomicU64,
    start_time: Instant,
}

impl RecordingStats {
    fn new() -> Self {
        Self {
            frames_captured: AtomicU64::new(0),
            frames_encoded: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn print_stats(&self) {
        let captured = self.frames_captured.load(Ordering::Relaxed);
        let encoded = self.frames_encoded.load(Ordering::Relaxed);
        let dropped = self.frames_dropped.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        
        println!(
            "Stats: Captured={}, Encoded={}, Dropped={}, FPS={:.1}, Encode Rate={:.1}",
            captured, encoded, dropped,
            captured as f64 / elapsed,
            encoded as f64 / elapsed
        );
    }
}

impl RealtimeRecorder {
    fn new(config: RecordingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // CUDA初期化
        let cuda_ctx = CudaContext::new(0)?;
        let encoder = Encoder::initialize_with_cuda(cuda_ctx)?;

        // エンコーダー設定
        let encode_guid = NV_ENC_CODEC_H264_GUID;
        let preset_guid = NV_ENC_PRESET_P1_GUID;
        let profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
        let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
        let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

        let mut preset_config = encoder.get_preset_config(encode_guid, preset_guid, tuning_info)?;

        // セッション初期化
        let mut initialize_params = EncoderInitParams::new(encode_guid, config.width, config.height);
        initialize_params
            .preset_guid(preset_guid)
            .tuning_info(tuning_info)
            .display_aspect_ratio(16, 9)
            .framerate(config.fps, 1)
            .enable_picture_type_decision()
            .encode_config(&mut preset_config.presetCfg);

        let session = encoder.start_session(buffer_format, initialize_params)?;

        // バッファプール初期化
        let mut buffer_pool = BufferPool::new();
        buffer_pool.initialize(&session, config.buffer_count);

        Ok(Self {
            config,
            session,
            buffer_pool: Arc::new(Mutex::new(buffer_pool)),
            frame_queue: Arc::new(Mutex::new(VecDeque::new())),
            frame_condvar: Arc::new(Condvar::new()),
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RecordingStats::new()),
        })
    }

    /// 録画開始
    fn start_recording(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.running.store(true, Ordering::Relaxed);
        
        // フレーム生成スレッド（実際のアプリケーションではスクリーンキャプチャなど）
        let frame_generator_handle = self.start_frame_generator();
        
        // エンコードスレッド
        let encoder_handle = self.start_encoder_thread();
        
        // 出力処理スレッド
        let output_handle = self.start_output_thread()?;
        
        // 統計表示スレッド
        let stats_handle = self.start_stats_thread();

        println!("Recording started. Press Ctrl+C to stop...");
        
        // 10秒間録画（実際のアプリケーションではユーザー入力待ち）
        thread::sleep(Duration::from_secs(10));
        
        // 録画停止
        self.running.store(false, Ordering::Relaxed);
        
        // スレッド終了待ち
        frame_generator_handle.join().unwrap();
        encoder_handle.join().unwrap();
        output_handle.join().unwrap();
        stats_handle.join().unwrap();
        
        self.stats.print_stats();
        println!("Recording stopped.");
        
        Ok(())
    }

    /// フレーム生成スレッド（実際にはスクリーンキャプチャなど）
    fn start_frame_generator(&self) -> thread::JoinHandle<()> {
        let frame_queue = Arc::clone(&self.frame_queue);
        let frame_condvar = Arc::clone(&self.frame_condvar);
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        thread::spawn(move || {
            let frame_duration = Duration::from_nanos(1_000_000_000 / config.fps as u64);
            let mut next_frame_time = Instant::now();
            let mut frame_index = 0u64;

            while running.load(Ordering::Relaxed) {
                let now = Instant::now();
                if now >= next_frame_time {
                    // テスト用のフレームデータ生成
                    let frame_data = generate_test_frame(
                        config.width, 
                        config.height, 
                        frame_index, 
                        now.elapsed().as_millis() as f32 / 1000.0
                    );
                    
                    let frame = FrameData {
                        data: frame_data,
                        timestamp: now.elapsed().as_micros() as u64,
                        frame_index,
                    };

                    // フレームキューに追加
                    {
                        let mut queue = frame_queue.lock().unwrap();
                        if queue.len() < 5 { // キューサイズ制限
                            queue.push_back(frame);
                            frame_condvar.notify_one();
                            stats.frames_captured.fetch_add(1, Ordering::Relaxed);
                        } else {
                            stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    frame_index += 1;
                    next_frame_time += frame_duration;
                } else {
                    // 次のフレーム時刻まで待機
                    thread::sleep(Duration::from_millis(1));
                }
            }
        })
    }

    /// エンコードスレッド
    fn start_encoder_thread(&self) -> thread::JoinHandle<()> {
        let frame_queue = Arc::clone(&self.frame_queue);
        let frame_condvar = Arc::clone(&self.frame_condvar);
        let buffer_pool = Arc::clone(&self.buffer_pool);
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);

        // セッションは Send + Sync ではないため、ここで作り直すか別の方法が必要
        // この例では簡略化のため、エンコード処理はメインスレッドで行う想定
        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                // フレームキューからフレーム取得
                let frame = {
                    let mut queue = frame_queue.lock().unwrap();
                    if queue.is_empty() {
                        let timeout = Duration::from_millis(5);
                        let (_queue, _timeout) = frame_condvar.wait_timeout(queue, timeout).unwrap();
                        continue;
                    }
                    queue.pop_front()
                };

                if let Some(frame_data) = frame {
                    // バッファプールから利用可能なバッファを取得
                    let buffers = {
                        let mut pool = buffer_pool.lock().unwrap();
                        pool.get_available_buffer()
                    };

                    if let Some((mut input_buffer, output_buffer)) = buffers {
                        // エンコード処理をここに実装
                        // 注意: Session は Send + Sync ではないため、
                        // 実際の実装では別のアプローチが必要
                        
                        // 一旦、使用済みキューに追加
                        {
                            let mut pool = buffer_pool.lock().unwrap();
                            pool.mark_as_used((input_buffer, output_buffer), frame_data.frame_index);
                        }
                        
                        stats.frames_encoded.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        })
    }

    /// 出力処理スレッド
    fn start_output_thread(&self) -> Result<thread::JoinHandle<()>, Box<dyn std::error::Error>> {
        let buffer_pool = Arc::clone(&self.buffer_pool);
        let running = Arc::clone(&self.running);
        let output_file = self.config.output_file.clone();

        let mut out_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&output_file)?;

        let handle = thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let used_buffer = {
                    let mut pool = buffer_pool.lock().unwrap();
                    pool.get_used_buffer()
                };

                if let Some((input_buffer, output_buffer, frame_index)) = used_buffer {
                    // 出力処理（簡略化）
                    println!("Processing output for frame {}", frame_index);
                    
                    // バッファを再利用キューに戻す
                    {
                        let mut pool = buffer_pool.lock().unwrap();
                        pool.return_buffer((input_buffer, output_buffer));
                    }
                }
                
                thread::sleep(Duration::from_millis(1));
            }
        });

        Ok(handle)
    }

    /// 統計表示スレッド
    fn start_stats_thread(&self) -> thread::JoinHandle<()> {
        let stats = Arc::clone(&self.stats);
        let running = Arc::clone(&self.running);

        thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
                stats.print_stats();
            }
        })
    }
}

/// テスト用フレーム生成
fn generate_test_frame(width: u32, height: u32, frame_index: u64, time: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    
    for y in 0..height {
        for x in 0..width {
            let red = ((255.0 * x as f32 / width as f32) as u8).wrapping_add((time * 50.0) as u8);
            let green = ((255.0 * y as f32 / height as f32) as u8).wrapping_add((frame_index % 255) as u8);
            let blue = ((time * 255.0) as u8).wrapping_add((frame_index % 128) as u8);
            
            // ARGB形式
            data.push(blue);  // B
            data.push(green); // G
            data.push(red);   // R
            data.push(255);   // A
        }
    }
    
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RecordingConfig {
        width: 1920,
        height: 1080,
        fps: 30,
        buffer_count: 8, // 8個のバッファを循環利用
        output_file: "realtime_output.bin".to_string(),
    };

    let mut recorder = RealtimeRecorder::new(config)?;
    recorder.start_recording()?;

    Ok(())
}