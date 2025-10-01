use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use anyhow::{bail, Context, Result};
use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB,
        NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
    },
    Encoder, EncoderInitParams, Session,
};
use serde::Serialize;
use tracing::{info, info_span, instrument, warn, Level};

/// 在りもののシンプルな色グラデーションからフレームを生成する。
#[derive(Debug, Clone)]
struct FrameData {
    data: Vec<u8>,
    frame_index: u64,
    captured_at: Instant,
}

/// JSONL に書き出すフレーム単位のログ。
#[derive(Debug, Serialize)]
struct EncodedFrameLog {
    frame_index: u64,
    latency_ms: f64,
    encoded_at_micros: u128,
}

impl EncodedFrameLog {
    fn from_latency(frame_index: u64, latency: Duration) -> Self {
        Self {
            frame_index,
            latency_ms: latency.as_secs_f64() * 1_000.0,
            encoded_at_micros: system_time_micros(),
        }
    }
}

/// セッションの概要。
#[derive(Debug, Serialize)]
struct SessionSummaryLog {
    total_frames: u64,
    total_latency_ms: f64,
    max_latency_ms: f64,
    duration_ms: f64,
}

impl SessionSummaryLog {
    fn from_stats(stats: &SessionStats, duration: Duration) -> Self {
        let total_latency_ms = stats.total_latency.as_secs_f64() * 1_000.0;
        let max_latency_ms = stats
            .max_latency
            .map_or(0.0, |lat| lat.as_secs_f64() * 1_000.0);
        Self {
            total_frames: stats.frames_encoded,
            total_latency_ms,
            max_latency_ms,
            duration_ms: duration.as_secs_f64() * 1_000.0,
        }
    }
}

#[derive(Debug)]
struct SessionStats {
    frames_encoded: u64,
    total_latency: Duration,
    max_latency: Option<Duration>,
}

impl SessionStats {
    fn new() -> Self {
        Self {
            frames_encoded: 0,
            total_latency: Duration::default(),
            max_latency: None,
        }
    }

    fn record_frame(&mut self, latency: Duration) {
        self.frames_encoded += 1;
        self.total_latency += latency;
        self.max_latency = Some(
            self.max_latency
                .map_or(latency, |current| current.max(latency)),
        );
    }
}

/// JSONL 形式で serde::Serialize を書き込むヘルパー。
struct JsonlWriter<W: Write> {
    inner: W,
}

impl<W: Write> JsonlWriter<W> {
    fn new(inner: W) -> Self {
        Self { inner }
    }

    fn write_entry<T: Serialize>(&mut self, value: &T) -> Result<()> {
        serde_json::to_writer(&mut self.inner, value).context("failed to serialize log entry")?;
        self.inner
            .write_all(b"\n")
            .context("failed to write log newline")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush().context("failed to flush log writer")
    }
}

#[derive(Clone, Debug)]
struct EncoderConfig {
    width: u32,
    height: u32,
    fps: u32,
    encode_guid: GUID,
    preset_guid: GUID,
    tuning_info: NV_ENC_TUNING_INFO,
    buffer_format: NV_ENC_BUFFER_FORMAT,
}

impl EncoderConfig {
    fn default_h264(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            encode_guid: NV_ENC_CODEC_H264_GUID,
            preset_guid: NV_ENC_PRESET_P1_GUID,
            tuning_info: NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            buffer_format: NV_ENC_BUFFER_FORMAT_ARGB,
        }
    }
}

enum EncoderLifecycle<W: Write, L: Write> {
    Idle,
    Recording(RecordingSession<W, L>),
    Closed,
}

struct RecordingSession<W: Write, L: Write> {
    session: Session,
    bitstream_writer: W,
    log_writer: JsonlWriter<L>,
    stats: SessionStats,
    started_at: Instant,
    span: tracing::Span,
}

impl<W: Write, L: Write> RecordingSession<W, L> {
    fn new(session: Session, bitstream_writer: W, log_writer: L, span: tracing::Span) -> Self {
        Self {
            session,
            bitstream_writer,
            log_writer: JsonlWriter::new(log_writer),
            stats: SessionStats::new(),
            started_at: Instant::now(),
            span,
        }
    }

    #[instrument(skip(self, frame), fields(frame_index = frame.frame_index))]
    fn encode_frame(&mut self, frame: FrameData) -> Result<()> {
        let frame_span = info_span!("frame", frame_index = frame.frame_index);
        let _enter = frame_span.enter();

        let mut input_buffer = self
            .session
            .create_input_buffer()
            .context("failed to create input buffer")?;
        let mut output_buffer = self
            .session
            .create_output_bitstream()
            .context("failed to create output bitstream")?;

        {
            let mut lock = input_buffer.lock().context("failed to lock input buffer")?;
            unsafe {
                lock.write(&frame.data);
            }
        }

        self.session
            .encode_picture(&mut input_buffer, &mut output_buffer, Default::default())
            .context("failed to encode picture")?;

        let bitstream_lock = output_buffer
            .lock()
            .context("failed to lock output bitstream")?;
        self.bitstream_writer
            .write_all(bitstream_lock.data())
            .context("failed to write encoded bitstream")?;

        let latency = frame.captured_at.elapsed();
        self.stats.record_frame(latency);
        self.log_writer
            .write_entry(&EncodedFrameLog::from_latency(frame.frame_index, latency))?;

        Ok(())
    }

    fn finalize(mut self) -> Result<SessionSummaryLog> {
        let duration = self.started_at.elapsed();
        let summary = SessionSummaryLog::from_stats(&self.stats, duration);
        self.log_writer.write_entry(&summary)?;
        self.log_writer.flush()?;
        self.bitstream_writer
            .flush()
            .context("failed to flush bitstream writer")?;
        drop(self.span);
        Ok(summary)
    }
}

struct RealtimeEncoder<W: Write, L: Write> {
    config: EncoderConfig,
    cuda_context: Arc<CudaContext>,
    state: EncoderLifecycle<W, L>,
}

impl<W: Write, L: Write> RealtimeEncoder<W, L> {
    fn new(config: EncoderConfig) -> Result<Self> {
        let cuda_context = CudaContext::new(0).context("failed to create CUDA context")?;
        Ok(Self {
            config,
            cuda_context,
            state: EncoderLifecycle::Idle,
        })
    }

    #[instrument(skip(self, bitstream_writer, log_writer))]
    fn start_session(&mut self, bitstream_writer: W, log_writer: L) -> Result<()> {
        match self.state {
            EncoderLifecycle::Idle => {}
            EncoderLifecycle::Recording(_) => bail!("session already active"),
            EncoderLifecycle::Closed => bail!("encoder has been closed"),
        }

        info!(
            width = self.config.width,
            height = self.config.height,
            fps = self.config.fps,
            "initializing encoder"
        );

        let encoder = Encoder::initialize_with_cuda(Arc::clone(&self.cuda_context))
            .context("failed to initialize encoder")?;

        let mut preset_config = encoder
            .get_preset_config(
                self.config.encode_guid,
                self.config.preset_guid,
                self.config.tuning_info,
            )
            .context("failed to fetch preset config")?;

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
            .context("failed to start encoder session")?;

        let session_span = info_span!(
            "recording_session",
            width = self.config.width,
            height = self.config.height,
            fps = self.config.fps
        );

        self.state = EncoderLifecycle::Recording(RecordingSession::new(
            session,
            bitstream_writer,
            log_writer,
            session_span,
        ));

        info!("session started");
        Ok(())
    }

    #[instrument(skip(self, frame))]
    fn encode_frame(&mut self, frame: FrameData) -> Result<()> {
        match &mut self.state {
            EncoderLifecycle::Recording(session) => session.encode_frame(frame),
            EncoderLifecycle::Idle => bail!("session has not been started"),
            EncoderLifecycle::Closed => bail!("encoder has been closed"),
        }
    }

    fn finish_session(&mut self) -> Result<SessionSummaryLog> {
        let state = std::mem::replace(&mut self.state, EncoderLifecycle::Closed);
        match state {
            EncoderLifecycle::Recording(session) => {
                let summary = session.finalize()?;
                self.state = EncoderLifecycle::Idle;
                Ok(summary)
            }
            EncoderLifecycle::Idle => {
                self.state = EncoderLifecycle::Idle;
                bail!("no active session")
            }
            EncoderLifecycle::Closed => {
                self.state = EncoderLifecycle::Closed;
                bail!("encoder already closed")
            }
        }
    }

    fn close(&mut self) {
        self.state = EncoderLifecycle::Closed;
    }
}

struct FrameGenerator {
    width: u32,
    height: u32,
    frame_index: u64,
    started_at: Instant,
}

impl FrameGenerator {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_index: 0,
            started_at: Instant::now(),
        }
    }

    fn next_frame(&mut self) -> FrameData {
        let now = Instant::now();
        let mut data = Vec::with_capacity((self.width * self.height * 4) as usize);
        let t = self.started_at.elapsed().as_secs_f32();

        for y in 0..self.height {
            for x in 0..self.width {
                let red = (255.0
                    * (x as f32 / self.width as f32)
                    * (0.5 + 0.5 * (t * 0.5 + self.frame_index as f32 * 0.02).sin()))
                    as u8;
                let green = (255.0
                    * (y as f32 / self.height as f32)
                    * (0.5 + 0.5 * (t * 0.3 + self.frame_index as f32 * 0.01).cos()))
                    as u8;
                let blue = (255.0 * (0.5 + 0.5 * (t + self.frame_index as f32 * 0.1).sin())) as u8;

                data.push(blue);
                data.push(green);
                data.push(red);
                data.push(255);
            }
        }

        let frame = FrameData {
            data,
            frame_index: self.frame_index,
            captured_at: now,
        };
        self.frame_index += 1;
        frame
    }
}

fn system_time_micros() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or_default()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_max_level(Level::INFO)
        .compact()
        .init();

    let config = EncoderConfig::default_h264(1280, 720, 30);
    let mut encoder: RealtimeEncoder<BufWriter<File>, BufWriter<File>> =
        RealtimeEncoder::new(config.clone())?;
    let mut generator = FrameGenerator::new(config.width, config.height);

    info!("starting first recording run");
    let bitstream_writer_1 = BufWriter::new(File::create("structured_run_01.h264")?);
    let log_writer_1 = BufWriter::new(File::create("structured_run_01.log.jsonl")?);
    encoder.start_session(bitstream_writer_1, log_writer_1)?;
    for _ in 0..(config.fps * 2) {
        let frame = generator.next_frame();
        encoder.encode_frame(frame)?;
    }
    let summary_one = encoder.finish_session()?;
    info!(?summary_one, "first session finished");

    info!("starting second recording run");
    let bitstream_writer_2 = BufWriter::new(File::create("structured_run_02.h264")?);
    let log_writer_2 = BufWriter::new(File::create("structured_run_02.log.jsonl")?);
    encoder.start_session(bitstream_writer_2, log_writer_2)?;
    for _ in 0..(config.fps * 2) {
        let frame = generator.next_frame();
        encoder.encode_frame(frame)?;
    }
    let summary_two = encoder.finish_session()?;
    info!(?summary_two, "second session finished");

    encoder.close();
    warn!("encoder closed; further calls will error");

    Ok(())
}
