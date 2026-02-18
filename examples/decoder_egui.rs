use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::CudaContext;
use eframe::egui::{self, ColorImage};
use nvidia_video_codec_sdk::{DecodeCodec, DecodeOptions, DecodedRgbFrame, Decoder};
use rtc::media::io::h26x_reader::H26xReader;
use rtc::shared::error::Error as RtcError;
use std::io::Cursor;

const CLOCK_RATE_90K: i64 = 90_000;

#[derive(Debug, Clone)]
struct Cli {
    input: PathBuf,
    codec: DecodeCodec,
    fps: u32,
    loop_playback: bool,
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
        (self.started_at.elapsed().as_secs_f64() * CLOCK_RATE_90K as f64) as i64
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
                    "decoded-video-frame",
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
            ui.heading("NVDEC + egui video viewer");
            ui.label(format!(
                "frame: {}/{}  loop: {}",
                self.current_index.saturating_add(1),
                self.frames.len(),
                self.loop_playback
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
    let frames = decode_annexb_file(&cli)?;
    if frames.is_empty() {
        bail!("decoder produced no frames");
    }

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "NVDEC Video Viewer",
        options,
        Box::new(move |_cc| Ok(Box::new(DecoderApp::new(frames, cli.loop_playback)))),
    )
    .map_err(|err| anyhow!("eframe failed: {err}"))?;
    Ok(())
}

fn decode_annexb_file(cli: &Cli) -> Result<Vec<UiFrame>> {
    let bitstream =
        fs::read(&cli.input).with_context(|| format!("failed to read {}", cli.input.display()))?;
    let access_units = build_access_units(&bitstream, cli.codec)?;
    if access_units.is_empty() {
        bail!(
            "no access units found in {} (input must be Annex-B stream)",
            cli.input.display()
        );
    }

    let timestamp_step = i64::from(CLOCK_RATE_90K as u32 / cli.fps);
    let cuda_ctx = CudaContext::new(0).context("failed to create CUDA context")?;
    let mut decoder = Decoder::new(cuda_ctx, cli.codec, DecodeOptions::default())
        .context("failed to create decoder")?;

    let mut decoded = Vec::new();
    for (i, access_unit) in access_units.iter().enumerate() {
        let timestamp_90k = i as i64 * timestamp_step;
        let mut frames = decoder
            .push_access_unit(access_unit, timestamp_90k)
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

fn build_access_units(bitstream: &[u8], codec: DecodeCodec) -> Result<Vec<Vec<u8>>> {
    let nals = match codec {
        DecodeCodec::H264 => read_h26x_nals(bitstream, false)?,
        DecodeCodec::H265 => read_h26x_nals(bitstream, true)?,
        DecodeCodec::Av1 => split_annexb_nals(bitstream),
    };
    if nals.is_empty() {
        return Ok(Vec::new());
    }
    let access_units = match codec {
        DecodeCodec::H264 => group_h264_access_units(&nals),
        DecodeCodec::H265 => group_h265_access_units(&nals),
        DecodeCodec::Av1 => nals.into_iter().map(|nal| with_start_code(&nal)).collect(),
    };
    Ok(access_units)
}

fn read_h26x_nals(bitstream: &[u8], is_hevc: bool) -> Result<Vec<Vec<u8>>> {
    let mut reader = H26xReader::new(Cursor::new(bitstream), 4096, is_hevc);
    let mut nals = Vec::new();

    loop {
        match reader.next_nal() {
            Ok(nal) => nals.push(nal.data().to_vec()),
            Err(RtcError::ErrIoEOF) => break,
            Err(err) => return Err(anyhow!("failed to parse h26x Annex-B stream: {err}")),
        }
    }

    Ok(nals)
}

fn split_annexb_nals(data: &[u8]) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();
    let mut cursor = 0usize;
    while let Some((start, start_len)) = find_start_code(data, cursor) {
        let nal_start = start + start_len;
        let next_start = find_start_code(data, nal_start)
            .map(|(idx, _)| idx)
            .unwrap_or(data.len());
        if next_start > nal_start {
            nals.push(data[nal_start..next_start].to_vec());
        }
        cursor = next_start;
    }
    nals
}

fn group_h264_access_units(nals: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut has_vcl = false;

    for nal in nals {
        let Some(nal_type) = nal.first().map(|b| b & 0x1f) else {
            continue;
        };
        let is_vcl = (1..=5).contains(&nal_type);
        let starts_new_picture = is_vcl && h264_first_mb_in_slice_is_zero(nal);
        if starts_new_picture && has_vcl && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            has_vcl = false;
        }
        append_annexb_nal(&mut current, nal);
        if is_vcl {
            has_vcl = true;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn group_h265_access_units(nals: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut has_vcl = false;

    for nal in nals {
        if nal.len() < 3 {
            continue;
        }
        let nal_type = (nal[0] >> 1) & 0x3f;
        let is_vcl = nal_type <= 31;
        let starts_new_picture = is_vcl && (nal[2] & 0x80) != 0;

        if starts_new_picture && has_vcl && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            has_vcl = false;
        }
        append_annexb_nal(&mut current, nal);
        if is_vcl {
            has_vcl = true;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn append_annexb_nal(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
}

fn with_start_code(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len() + 4);
    append_annexb_nal(&mut out, nal);
    out
}

fn h264_first_mb_in_slice_is_zero(nal: &[u8]) -> bool {
    if nal.len() < 2 {
        return false;
    }
    let rbsp = remove_emulation_prevention(&nal[1..]);
    let mut reader = BitReader::new(&rbsp);
    matches!(reader.read_ue(), Some(0))
}

fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

#[derive(Debug)]
struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bit(&mut self) -> Option<u8> {
        if self.bit_pos >= self.data.len() * 8 {
            return None;
        }
        let byte = self.data[self.bit_pos / 8];
        let shift = 7 - (self.bit_pos % 8);
        self.bit_pos += 1;
        Some((byte >> shift) & 1)
    }

    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zero_bits = 0usize;
        while let Some(bit) = self.read_bit() {
            if bit == 0 {
                leading_zero_bits += 1;
            } else {
                break;
            }
        }
        if leading_zero_bits > 31 {
            return None;
        }
        let mut suffix = 0u32;
        for _ in 0..leading_zero_bits {
            suffix = (suffix << 1) | u32::from(self.read_bit()?);
        }
        Some(((1u32 << leading_zero_bits) - 1) + suffix)
    }
}

fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    if from >= data.len() {
        return None;
    }
    let mut i = from;
    while i + 3 <= data.len() {
        if i + 4 <= data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            return Some((i, 4));
        }
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some((i, 3));
        }
        i += 1;
    }
    None
}

fn parse_cli(args: Vec<String>) -> Result<Cli> {
    let mut input = None;
    let mut codec = DecodeCodec::H264;
    let mut fps = 30u32;
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
            "--codec" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--codec requires a value"))?;
                codec = parse_codec(value)?;
            }
            "--fps" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--fps requires a value"))?;
                fps = value
                    .parse::<u32>()
                    .with_context(|| format!("invalid --fps value: {value}"))?;
                if fps == 0 {
                    bail!("--fps must be >= 1");
                }
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
    Ok(Cli {
        input,
        codec,
        fps,
        loop_playback,
    })
}

fn parse_codec(value: &str) -> Result<DecodeCodec> {
    match value {
        "h264" | "H264" => Ok(DecodeCodec::H264),
        "h265" | "H265" | "hevc" | "HEVC" => Ok(DecodeCodec::H265),
        "av1" | "AV1" => Ok(DecodeCodec::Av1),
        _ => bail!("invalid --codec value: {value} (use h264|h265|av1)"),
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --example decoder_egui -- --input <FILE> [--codec h264|h265|av1] [--fps 30] [--loop]");
    println!();
    println!("Notes:");
    println!("  - Input must be an Annex-B elementary stream (.h264/.h265).");
    println!("  - --fps is used only to synthesize timestamps for playback timing.");
}
