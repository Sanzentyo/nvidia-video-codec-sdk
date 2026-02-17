use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use eframe::egui::{self, ColorImage};
use nvidia_video_codec_sdk::{DecodeCodec, DecodeOptions, DecodedRgbFrame, Decoder};
use serde::Deserialize;

#[derive(Debug, Clone)]
struct Cli {
    input: PathBuf,
    index: PathBuf,
    loop_playback: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct PlaybackIndex {
    codec: String,
    access_units: Vec<AccessUnitEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct AccessUnitEntry {
    offset: u64,
    len: u64,
    timestamp_90k: i64,
}

#[derive(Debug, Clone)]
struct UiFrame {
    width: u32,
    height: u32,
    timestamp_90k: i64,
    rgba: Vec<u8>,
}

struct DecoderApp {
    frames: Vec<UiFrame>,
    texture: Option<egui::TextureHandle>,
    current_index: usize,
    displayed_index: Option<usize>,
    loop_playback: bool,
    started_at: Instant,
    base_timestamp_90k: i64,
    finished_once: bool,
}

impl DecoderApp {
    fn new(frames: Vec<UiFrame>, loop_playback: bool) -> Self {
        let base_timestamp_90k = frames.first().map_or(0, |f| f.timestamp_90k);
        Self {
            frames,
            texture: None,
            current_index: 0,
            displayed_index: None,
            loop_playback,
            started_at: Instant::now(),
            base_timestamp_90k,
            finished_once: false,
        }
    }

    fn elapsed_90k(&self) -> i64 {
        (self.started_at.elapsed().as_secs_f64() * 90_000.0) as i64
    }

    fn advance_playback(&mut self) {
        if self.frames.is_empty() {
            return;
        }
        if self.current_index + 1 >= self.frames.len() {
            self.finished_once = true;
            if self.loop_playback {
                self.started_at = Instant::now();
                self.current_index = 0;
                self.base_timestamp_90k = self.frames[0].timestamp_90k;
                self.finished_once = false;
            }
            return;
        }
        let now_90k = self.base_timestamp_90k + self.elapsed_90k();
        while self.current_index + 1 < self.frames.len()
            && self.frames[self.current_index + 1].timestamp_90k <= now_90k
        {
            self.current_index += 1;
        }
    }

    fn update_texture_if_needed(&mut self, ui: &mut egui::Ui) {
        if self.displayed_index == Some(self.current_index) {
            return;
        }
        let frame = &self.frames[self.current_index];
        let image = ColorImage::from_rgba_unmultiplied(
            [frame.width as usize, frame.height as usize],
            &frame.rgba,
        );
        match &mut self.texture {
            Some(texture) if texture.size() == [frame.width as usize, frame.height as usize] => {
                texture.set(image, egui::TextureOptions::LINEAR);
            }
            _ => {
                self.texture = Some(ui.ctx().load_texture(
                    "decoded-video-frame-indexed",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
        self.displayed_index = Some(self.current_index);
    }
}

impl eframe::App for DecoderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(16));
        self.advance_playback();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("NVDEC + egui indexed playback");
            ui.label(format!(
                "frame: {}/{}",
                self.current_index.saturating_add(1),
                self.frames.len()
            ));
            ui.label(format!(
                "timestamp_90k: {}",
                self.frames[self.current_index].timestamp_90k
            ));
            if self.finished_once && !self.loop_playback {
                ui.label("end of stream");
            }

            self.update_texture_if_needed(ui);
            if let Some(texture) = &self.texture {
                ui.add(
                    egui::Image::from_texture(egui::load::SizedTexture::from_handle(texture))
                        .shrink_to_fit(),
                );
            }
        });
    }
}

fn main() -> Result<()> {
    let cli = parse_cli(std::env::args().skip(1).collect())?;
    let frames = decode_indexed_file(&cli)?;
    if frames.is_empty() {
        bail!("decoder produced no frames");
    }

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "NVDEC Indexed Playback",
        options,
        Box::new(move |_cc| Ok(Box::new(DecoderApp::new(frames, cli.loop_playback)))),
    )
    .map_err(|err| anyhow!("eframe failed: {err}"))?;
    Ok(())
}

fn decode_indexed_file(cli: &Cli) -> Result<Vec<UiFrame>> {
    let bitstream =
        fs::read(&cli.input).with_context(|| format!("failed to read {}", cli.input.display()))?;
    let index_data =
        fs::read(&cli.index).with_context(|| format!("failed to read {}", cli.index.display()))?;
    let index: PlaybackIndex =
        serde_json::from_slice(&index_data).context("failed to parse index json")?;
    let codec = parse_codec(&index.codec)?;
    let decode_options = DecodeOptions {
        av1_annexb: false,
        ..DecodeOptions::default()
    };

    let cuda_ctx = CudaContext::new(0).context("failed to create CUDA context")?;
    let mut decoder =
        Decoder::new(cuda_ctx, codec, decode_options).context("failed to create decoder")?;

    let mut decoded = Vec::new();
    for (i, entry) in index.access_units.iter().enumerate() {
        let start = usize::try_from(entry.offset).context("offset does not fit usize")?;
        let len = usize::try_from(entry.len).context("len does not fit usize")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("offset+len overflow at index #{i}"))?;
        if end > bitstream.len() {
            bail!(
                "index entry out of range at #{i}: end={} > file_size={}",
                end,
                bitstream.len()
            );
        }
        let mut frames = decoder
            .push_access_unit(&bitstream[start..end], entry.timestamp_90k)
            .with_context(|| format!("failed to decode access unit #{i}"))?;
        decoded.append(&mut frames);
    }
    decoded.extend(decoder.flush().context("failed to flush decoder")?);

    Ok(decoded
        .into_iter()
        .map(|f| UiFrame {
            width: f.width,
            height: f.height,
            timestamp_90k: f.timestamp_90k,
            rgba: rgb_to_rgba(f),
        })
        .collect())
}

fn rgb_to_rgba(frame: DecodedRgbFrame) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((frame.width * frame.height * 4) as usize);
    for px in frame.data.chunks_exact(3) {
        rgba.push(px[0]);
        rgba.push(px[1]);
        rgba.push(px[2]);
        rgba.push(255);
    }
    rgba
}

fn parse_codec(value: &str) -> Result<DecodeCodec> {
    match value {
        "h264" | "H264" => Ok(DecodeCodec::H264),
        "h265" | "H265" | "hevc" | "HEVC" => Ok(DecodeCodec::H265),
        "av1" | "AV1" => Ok(DecodeCodec::Av1),
        _ => bail!("invalid codec in index file: {value}"),
    }
}

fn parse_cli(args: Vec<String>) -> Result<Cli> {
    let mut input = None;
    let mut index = None;
    let mut loop_playback = false;
    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--input requires a value"))?;
                input = Some(PathBuf::from(value));
            }
            "--index" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--index requires a value"))?;
                index = Some(PathBuf::from(value));
            }
            "--loop" => {
                loop_playback = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    let input = input.ok_or_else(|| anyhow!("--input is required"))?;
    let index = index.unwrap_or_else(|| input.with_extension("json"));
    Ok(Cli {
        input,
        index,
        loop_playback,
    })
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --example playback_egui_indexed -- --input <FILE.bin> [--index <FILE.json>] [--loop]");
    println!();
    println!("Generate sample assets first:");
    println!("  cargo run --example generate_playback_assets");
}
