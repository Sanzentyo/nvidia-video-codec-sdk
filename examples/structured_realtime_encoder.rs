use std::{
    collections::VecDeque,
    env,
    fs::File,
    io::Write,
    mem,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT, NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
        NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG, NV_ENC_PRESET_P1_GUID,
        NV_ENC_TUNING_INFO,
    },
    Bitstream, Buffer, EncodePictureParams, Encoder, EncoderInitParams, Session,
};
use serde::Serialize;
use tracing::{debug, info, span, warn, Level, Span};
use tracing_subscriber::{fmt, EnvFilter};

/// フレームデータ: 所有しているピクセルデータとメタ情報をまとめた構造体。
#[derive(Debug, Clone)]
pub struct FrameData {
    pub frame_index: u64,
    pub pts: u64,
    pub captured_at: SystemTime,
    pub capture_instant: Instant,
    pub payload: Vec<u8>,
}

impl FrameData {
    pub fn new(frame_index: u64, pts: u64, payload: Vec<u8>) -> Self {
        Self {
            frame_index,
            pts,
            captured_at: SystemTime::now(),
            capture_instant: Instant::now(),
            payload,
        }
    }

    pub fn len(&self) -> usize {
        self.payload.len()
    }
}

/// ログファイルに書き出す JSON レコード。
#[derive(Debug, Serialize)]
struct FrameLogEntry {
    frame_index: u64,
    pts: u64,
    input_bytes: usize,
    output_bytes: usize,
    latency_ns: u64,
    copy_time_ns: u64,
    encode_time_ns: u64,
    captured_at_unix_micros: u128,
}

impl FrameLogEntry {
    fn new(
        frame: &FrameData,
        output_bytes: usize,
        latency: Duration,
        copy_time: Duration,
        encode_time: Duration,
    ) -> Self {
        Self {
            frame_index: frame.frame_index,
            pts: frame.pts,
            input_bytes: frame.len(),
            output_bytes,
            latency_ns: saturating_duration_ns(latency),
            copy_time_ns: saturating_duration_ns(copy_time),
            encode_time_ns: saturating_duration_ns(encode_time),
            captured_at_unix_micros: system_time_to_unix_micros(frame.captured_at),
        }
    }
}

/// エンコードセッションの統計情報。
#[derive(Debug, Default)]
struct SessionStats {
    frames_encoded: u64,
    encoded_bytes: u64,
    total_latency_ns: u128,
    max_latency_ns: u128,
}

impl SessionStats {
    fn record_success(&mut self, latency: Duration, encoded_bytes: usize) {
        self.frames_encoded += 1;
        self.encoded_bytes += encoded_bytes as u64;
        let latency_ns = latency.as_nanos();
        self.total_latency_ns += latency_ns;
        if latency_ns > self.max_latency_ns {
            self.max_latency_ns = latency_ns;
        }
    }

    fn into_summary(self, submitted: u64) -> SessionSummary {
        let avg_latency_ns = if self.frames_encoded > 0 {
            self.total_latency_ns / self.frames_encoded as u128
        } else {
            0
        };
        SessionSummary {
            frames_submitted: submitted,
            frames_encoded: self.frames_encoded,
            frames_dropped: submitted.saturating_sub(self.frames_encoded),
            encoded_bytes: self.encoded_bytes,
            average_latency_ms: (avg_latency_ns as f64) / 1_000_000.0,
            max_latency_ms: (self.max_latency_ns as f64) / 1_000_000.0,
        }
    }
}

/// セッション完了時に出力するサマリ。
#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub frames_submitted: u64,
    pub frames_encoded: u64,
    pub frames_dropped: u64,
    pub encoded_bytes: u64,
    pub average_latency_ms: f64,
    pub max_latency_ms: f64,
}

/// セッション中のログ出力を担当する簡易ライター。
struct FrameLogWriter<L: Write + Send + 'static> {
    inner: L,
}

impl<L: Write + Send + 'static> FrameLogWriter<L> {
    fn new(inner: L) -> Self {
        Self { inner }
    }

    fn write(&mut self, entry: &FrameLogEntry) -> Result<()> {
        serde_json::to_writer(&mut self.inner, entry).context("write frame log as json")?;
        self.inner
            .write_all(b"\n")
            .context("append newline to frame log")?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.inner.flush().context("flush frame log writer")
    }
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    width: u32,
    height: u32,
    fps: u32,
    codec_label: &'static str,
    encode_guid: nvidia_video_codec_sdk::sys::nvEncodeAPI::GUID,
    preset_guid: nvidia_video_codec_sdk::sys::nvEncodeAPI::GUID,
    buffer_format: NV_ENC_BUFFER_FORMAT,
    tuning_info: NV_ENC_TUNING_INFO,
}

impl EncoderConfig {
    fn h264(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            codec_label: "h264",
            encode_guid: NV_ENC_CODEC_H264_GUID,
            preset_guid: NV_ENC_PRESET_P1_GUID,
            buffer_format: NV_ENC_BUFFER_FORMAT_ARGB,
            tuning_info: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        }
    }

    fn hevc(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            codec_label: "hevc",
            encode_guid: NV_ENC_CODEC_HEVC_GUID,
            preset_guid: NV_ENC_PRESET_P1_GUID,
            buffer_format: NV_ENC_BUFFER_FORMAT_ARGB,
            tuning_info: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        }
    }

    fn codec_label(&self) -> &'static str {
        self.codec_label
    }
}

#[derive(Debug)]
enum EncoderLifecycle {
    Idle,
    Active,
}

impl EncoderLifecycle {
    fn try_acquire(&mut self) -> Result<StateGuard<'_>> {
        match self {
            EncoderLifecycle::Idle => {
                *self = EncoderLifecycle::Active;
                Ok(StateGuard { state: self })
            }
            EncoderLifecycle::Active => bail!("encoder session already active"),
        }
    }
}

struct StateGuard<'a> {
    state: &'a mut EncoderLifecycle,
}

impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        *self.state = EncoderLifecycle::Idle;
    }
}

struct SessionBuffers {
    inputs: VecDeque<Buffer<'static>>,
    outputs: VecDeque<Bitstream<'static>>,
}

impl SessionBuffers {
    fn new(session: &Session, count: usize) -> Result<Self> {
        let pool_size = count.max(3);
        let mut inputs = VecDeque::with_capacity(pool_size);
        let mut outputs = VecDeque::with_capacity(pool_size);
        for _ in 0..pool_size {
            let input = session
                .create_input_buffer()
                .context("create input buffer")?;
            let output = session
                .create_output_bitstream()
                .context("create output bitstream")?;
            // Safety: the session is stored in a pinned box alongside these buffers and
            // lives longer than any buffer. Drop order ensures buffers drop before session.
            inputs.push_back(unsafe { mem::transmute::<Buffer<'_>, Buffer<'static>>(input) });
            outputs
                .push_back(unsafe { mem::transmute::<Bitstream<'_>, Bitstream<'static>>(output) });
        }
        Ok(Self { inputs, outputs })
    }

    fn checkout(&mut self) -> Option<BufferPair> {
        let input = self.inputs.pop_front()?;
        let output = self.outputs.pop_front()?;
        Some(BufferPair { input, output })
    }

    fn checkin(&mut self, pair: BufferPair) {
        self.inputs.push_back(pair.input);
        self.outputs.push_back(pair.output);
    }
}

struct BufferPair {
    input: Buffer<'static>,
    output: Bitstream<'static>,
}

struct BufferPairGuard<'a> {
    pool: &'a mut SessionBuffers,
    pair: Option<BufferPair>,
}

impl<'a> BufferPairGuard<'a> {
    fn new(pool: &'a mut SessionBuffers, pair: BufferPair) -> Self {
        Self {
            pool,
            pair: Some(pair),
        }
    }

    fn pair_mut(&mut self) -> &mut BufferPair {
        self.pair.as_mut().expect("buffer pair already taken")
    }
}

impl Drop for BufferPairGuard<'_> {
    fn drop(&mut self) {
        if let Some(pair) = self.pair.take() {
            self.pool.checkin(pair);
        }
    }
}

/// エンコーダー本体。
pub struct StructuredRealtimeEncoder {
    cuda_ctx: Arc<CudaContext>,
    config: EncoderConfig,
    lifecycle: EncoderLifecycle,
    session_seq: u64,
}

impl StructuredRealtimeEncoder {
    pub fn new(device_index: u32, config: EncoderConfig) -> Result<Self> {
        let cuda_ctx = CudaContext::new(device_index as usize).context("create CUDA context")?;
        Ok(Self {
            cuda_ctx,
            config,
            lifecycle: EncoderLifecycle::Idle,
            session_seq: 0,
        })
    }

    pub fn default_h264(device_index: u32, width: u32, height: u32, fps: u32) -> Result<Self> {
        Self::new(device_index, EncoderConfig::h264(width, height, fps))
    }

    pub fn default_hevc(device_index: u32, width: u32, height: u32, fps: u32) -> Result<Self> {
        Self::new(device_index, EncoderConfig::hevc(width, height, fps))
    }

    pub fn video_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn fps(&self) -> u32 {
        self.config.fps
    }

    pub fn codec_label(&self) -> &'static str {
        self.config.codec_label()
    }

    pub fn start_session<'enc, W, L>(
        &'enc mut self,
        writer: W,
        log_writer: L,
    ) -> Result<SessionHandle<'enc, W, L>>
    where
        W: Write + Send + 'static,
        L: Write + Send + 'static,
    {
        let guard = self.lifecycle.try_acquire()?;
        self.session_seq = self.session_seq.wrapping_add(1);
        let session_span = span!(
            Level::INFO,
            "encoder_session",
            session_id = self.session_seq,
            width = self.config.width,
            height = self.config.height,
            fps = self.config.fps,
            codec = self.config.codec_label()
        );
        let span_for_session = session_span.clone();
        let _enter = session_span.enter();
        info!("starting encoder session");

        let encoder = Encoder::initialize_with_cuda(Arc::clone(&self.cuda_ctx))
            .context("initialize NVENC encoder")?;

        let mut preset_config = encoder
            .get_preset_config(
                self.config.encode_guid,
                self.config.preset_guid,
                self.config.tuning_info,
            )
            .context("fetch preset config")?;

        let mut init_params = EncoderInitParams::new(
            self.config.encode_guid,
            self.config.width,
            self.config.height,
        );
        init_params
            .preset_guid(self.config.preset_guid)
            .tuning_info(self.config.tuning_info)
            .display_aspect_ratio(16, 9)
            .framerate(self.config.fps, 1)
            .enable_picture_type_decision()
            .encode_config(&mut preset_config.presetCfg);

        let session = encoder
            .start_session(self.config.buffer_format, init_params)
            .context("start NVENC session")?;

        SessionHandle::new(
            span_for_session,
            guard,
            session,
            preset_config.presetCfg,
            writer,
            log_writer,
        )
    }
}

/// アクティブなセッション。StateGuardにより親エンコーダーの状態を管理する。
pub struct SessionHandle<'enc, W: Write + Send + 'static, L: Write + Send + 'static> {
    buffers: SessionBuffers,
    writer: W,
    log_writer: FrameLogWriter<L>,
    stats: SessionStats,
    frames_submitted: u64,
    session: Pin<Box<Session>>,
    span: Span,
    _state_guard: StateGuard<'enc>,
}

impl<'enc, W: Write + Send + 'static, L: Write + Send + 'static> SessionHandle<'enc, W, L> {
    fn new(
        span: Span,
        guard: StateGuard<'enc>,
        session: Session,
        preset_config: NV_ENC_CONFIG,
        writer: W,
        log_writer: L,
    ) -> Result<Self> {
        let session = Box::pin(session);
        let session_ref: &Session = Pin::as_ref(&session).get_ref();
        let buffer_count = usize::try_from(preset_config.frameIntervalP).unwrap_or(1)
            + usize::try_from(preset_config.rcParams.lookaheadDepth).unwrap_or(0)
            + 1;
        let buffers = SessionBuffers::new(session_ref, buffer_count)
            .with_context(|| format!("prepare buffer pool of size {buffer_count}"))?;
        Ok(Self {
            buffers,
            writer,
            log_writer: FrameLogWriter::new(log_writer),
            stats: SessionStats::default(),
            frames_submitted: 0,
            session,
            span,
            _state_guard: guard,
        })
    }

    pub fn encode_frame(&mut self, frame: FrameData) -> Result<()> {
        self.frames_submitted += 1;
        let frame_span = span!(
            Level::DEBUG,
            "encode_frame",
            frame_index = frame.frame_index,
            pts = frame.pts,
            payload_len = frame.len()
        );
        let _enter = frame_span.enter();

        let pair = self
            .buffers
            .checkout()
            .ok_or_else(|| anyhow!("buffer pool exhausted"))?;
        let mut guard = BufferPairGuard::new(&mut self.buffers, pair);
        let pair_mut = guard.pair_mut();

        let copy_start = Instant::now();
        {
            let mut lock = pair_mut
                .input
                .lock()
                .context("lock input buffer for writing")?;
            unsafe {
                lock.write(&frame.payload);
            }
        }
        let copy_time = copy_start.elapsed();

        let encode_start = Instant::now();
        self.session
            .as_ref()
            .encode_picture(
                &mut pair_mut.input,
                &mut pair_mut.output,
                EncodePictureParams {
                    input_timestamp: frame.pts,
                    ..Default::default()
                },
            )
            .context("encode frame")?;

        let encoded_size = {
            let lock = pair_mut
                .output
                .lock()
                .context("lock output bitstream for reading")?;
            let data = lock.data();
            self.writer
                .write_all(data)
                .context("write encoded bitstream")?;
            data.len()
        };
        let encode_time = encode_start.elapsed();

        let total_latency = frame.capture_instant.elapsed();
        let log_entry =
            FrameLogEntry::new(&frame, encoded_size, total_latency, copy_time, encode_time);
        self.log_writer.write(&log_entry)?;
        self.stats.record_success(total_latency, encoded_size);

        debug!(
            encoded_size,
            latency_ms = total_latency.as_secs_f64() * 1_000.0,
            "frame encoded"
        );
        Ok(())
    }

    pub fn finish(mut self) -> Result<SessionSummary> {
        let _enter = self.span.enter();
        info!("flushing encoder session");
        self.session
            .as_ref()
            .end_of_stream()
            .context("send end-of-stream to encoder")?;
        self.writer.flush().context("flush output writer")?;
        self.log_writer.finish()?;
        let summary = self.stats.into_summary(self.frames_submitted);
        info!(?summary, "session finished");
        Ok(summary)
    }
}

fn install_tracing() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?;
    let env_filter = if let Ok(directive) = "structured_realtime_encoder=debug".parse() {
        env_filter.add_directive(directive)
    } else {
        env_filter
    };
    let subscriber = fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_thread_ids(true);
    if subscriber.try_init().is_err() {
        warn!("tracing subscriber already installed");
    }
    Ok(())
}

fn generate_test_frame(width: u32, height: u32, frame_index: u64, fps: u32) -> FrameData {
    let mut payload = Vec::with_capacity((width * height * 4) as usize);
    let phase = frame_index as f32 * 0.1;
    let time = frame_index as f32 / fps as f32;
    for y in 0..height {
        for x in 0..width {
            let red = (255.0 * (x as f32 / width as f32) * (0.5 + 0.5 * (time * 0.5).sin())) as u8;
            let green =
                (255.0 * (y as f32 / height as f32) * (0.5 + 0.5 * (time * 0.3).cos())) as u8;
            let blue = (255.0 * (0.5 + 0.5 * (time + phase).sin())) as u8;
            payload.push(blue);
            payload.push(green);
            payload.push(red);
            payload.push(255);
        }
    }
    let pts = frame_index * 1_000_000 / fps as u64;
    FrameData::new(frame_index, pts, payload)
}

fn ensure_output_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory for {}", parent.display()))?;
        }
    }
    Ok(())
}

/// Example usage:
///
/// ```sh
/// cargo run --example structured_realtime_encoder -- --codec hevc
/// cargo run --example structured_realtime_encoder -- hevc
/// ```
///
/// 引数を省略した場合は H.264 を使用します。
fn main() -> Result<()> {
    install_tracing()?;

    let mut codec_arg = "h264".to_string();
    let mut args_iter = env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        if let Some(value) = arg.strip_prefix("--codec=") {
            codec_arg = value.to_ascii_lowercase();
        } else if arg == "--codec" {
            let value = args_iter
                .next()
                .ok_or_else(|| anyhow!("--codec requires a value"))?;
            codec_arg = value.to_ascii_lowercase();
        } else if !arg.starts_with("--") {
            codec_arg = arg.to_ascii_lowercase();
        }
    }

    let video_width = 1920;
    let video_height = 1080;
    let fps_target = 30;

    let config = match codec_arg.as_str() {
        "h264" => EncoderConfig::h264(video_width, video_height, fps_target),
        "h265" | "hevc" => EncoderConfig::hevc(video_width, video_height, fps_target),
        other => return Err(anyhow!("unsupported codec '{other}'. use 'h264' or 'hevc'")),
    };

    let mut encoder = StructuredRealtimeEncoder::new(0, config)?;
    let codec_label = encoder.codec_label();

    let output_path = PathBuf::from(format!("output/structured_realtime_{codec_label}.bin"));
    let log_path = PathBuf::from(format!(
        "output/structured_realtime_{codec_label}_log.jsonl"
    ));
    ensure_output_dir(output_path.as_path())?;
    ensure_output_dir(log_path.as_path())?;

    info!(codec = codec_label, "selected codec");

    let fps = encoder.fps();
    let (width, height) = encoder.video_size();
    let data_writer = File::create(&output_path).context("open output bitstream file")?;
    let log_writer = File::create(&log_path).context("open structured log file")?;

    let mut session = encoder.start_session(data_writer, log_writer)?;
    let total_frames = fps as u64 * 60; // 60秒分
    for frame_index in 0..total_frames {
        let frame = generate_test_frame(width, height, frame_index, fps);
        session.encode_frame(frame)?;
    }
    let summary = session.finish()?;
    info!(?summary, "encoding session summary");

    Ok(())
}

fn saturating_duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn system_time_to_unix_micros(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros())
        .unwrap_or(0)
}
