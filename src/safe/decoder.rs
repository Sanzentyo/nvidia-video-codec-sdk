//! Defines [`Decoder`] and related types.

use std::{
    collections::VecDeque,
    ffi::{c_int, c_longlong, c_ulong, c_void},
    fmt, ptr,
    sync::{Arc, Mutex},
};

use cudarc::driver::{
    result::{self, DriverError},
    sys::CUresult,
    CudaContext,
};

use crate::sys::{
    cuviddec::{
        cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
        cudaVideoSurfaceFormat, cuvidCreateDecoder, cuvidDecodePicture, cuvidDestroyDecoder,
        cuvidGetDecoderCaps, cuvidMapVideoFrame64, cuvidReconfigureDecoder, cuvidUnmapVideoFrame64,
        CUvideodecoder, CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDPICPARAMS, CUVIDPROCPARAMS,
        CUVIDRECONFIGUREDECODERINFO,
    },
    nvcuvid::{
        cuvidCreateVideoParser, cuvidDestroyVideoParser, cuvidParseVideoData, CUvideopacketflags,
        CUvideoparser, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS,
        CUVIDSOURCEDATAPACKET,
    },
};

/// Video codec to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// AV1.
    Av1,
}

impl DecodeCodec {
    fn as_cuda_codec(self) -> cudaVideoCodec {
        match self {
            Self::H264 => cudaVideoCodec::cudaVideoCodec_H264,
            Self::H265 => cudaVideoCodec::cudaVideoCodec_HEVC,
            Self::Av1 => cudaVideoCodec::cudaVideoCodec_AV1,
        }
    }
}

/// Decoder options.
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// Parser reordering depth in frames.
    pub max_display_delay: u32,
    /// Parser clock rate used for packet timestamps.
    pub timestamp_clock_rate: u32,
    /// Initial parser decode surfaces before sequence callback refines the value.
    pub initial_decode_surfaces: u32,
    /// For AV1, `true` if the input stream is Annex-B, `false` for low-overhead OBU.
    pub av1_annexb: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_display_delay: 0,
            timestamp_clock_rate: 90_000,
            initial_decode_surfaces: 1,
            av1_annexb: false,
        }
    }
}

/// One decoded RGB frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRgbFrame {
    /// Visible frame width in pixels.
    pub width: u32,
    /// Visible frame height in pixels.
    pub height: u32,
    /// Frame timestamp in the configured parser clock rate (default 90 kHz).
    pub timestamp_90k: i64,
    /// RGB24 bytes (`width * height * 3`).
    pub data: Vec<u8>,
}

impl DecodedRgbFrame {
    /// Converts to `(data, width, height)` for direct interoperability with
    /// image-u8 style constructors.
    #[must_use]
    pub fn into_tuple(self) -> (Vec<u8>, u32, u32) {
        (self.data, self.width, self.height)
    }
}

/// Decoder error.
#[derive(Debug, Clone)]
pub enum DecodeError {
    /// CUDA Driver API error.
    Cuda(DriverError),
    /// NVDEC API error.
    Nvdec {
        /// Operation name.
        operation: &'static str,
        /// CUDA error code.
        code: CUresult,
    },
    /// Unsupported input or platform capability.
    Unsupported(String),
    /// Invalid caller input.
    InvalidInput(String),
    /// Internal decoder state error.
    Internal(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(err) => write!(f, "cuda error: {err:?}"),
            Self::Nvdec { operation, code } => {
                write!(f, "{operation} failed with {code:?}")
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<DriverError> for DecodeError {
    fn from(value: DriverError) -> Self {
        Self::Cuda(value)
    }
}

/// NVDEC parser + decoder wrapper.
#[derive(Debug)]
pub struct Decoder {
    ctx: Arc<CudaContext>,
    parser: CUvideoparser,
    bridge: Box<CallbackBridge>,
}

impl Decoder {
    /// Creates a decoder bound to an existing CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error when the codec/output format is unsupported, parser
    /// creation fails, or CUDA context binding fails.
    pub fn new(
        cuda_ctx: Arc<CudaContext>,
        codec: DecodeCodec,
        options: DecodeOptions,
    ) -> Result<Self, DecodeError> {
        cuda_ctx.bind_to_thread()?;
        check_decoder_caps(codec)?;

        let mut bridge = Box::new(CallbackBridge {
            codec,
            options,
            state: Mutex::new(DecoderState::default()),
        });

        let bridge_ptr = ptr::from_mut(bridge.as_mut()).cast::<c_void>();
        let mut parser_params = CUVIDPARSERPARAMS {
            CodecType: codec.as_cuda_codec(),
            ulMaxNumDecodeSurfaces: options.initial_decode_surfaces.max(1),
            ulClockRate: options.timestamp_clock_rate,
            ulErrorThreshold: 0,
            ulMaxDisplayDelay: options.max_display_delay,
            pUserData: bridge_ptr,
            pfnSequenceCallback: Some(sequence_callback),
            pfnDecodePicture: Some(decode_callback),
            pfnDisplayPicture: Some(display_callback),
            ..Default::default()
        };
        parser_params.set_bAnnexb(u32::from(codec == DecodeCodec::Av1 && options.av1_annexb));

        let mut parser = ptr::null_mut();
        check_nvdec(
            // SAFETY: parser pointer and params are valid for the call.
            unsafe { cuvidCreateVideoParser(&mut parser, &mut parser_params) },
            "cuvidCreateVideoParser",
        )?;

        Ok(Self {
            ctx: cuda_ctx,
            parser,
            bridge,
        })
    }

    /// Pushes one complete access unit into the parser.
    ///
    /// Returns all RGB frames that became available after this call.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or mapping fails.
    pub fn push_access_unit(
        &mut self,
        access_unit: &[u8],
        timestamp_90k: i64,
    ) -> Result<Vec<DecodedRgbFrame>, DecodeError> {
        if access_unit.is_empty() {
            return Err(DecodeError::InvalidInput(
                "access unit must not be empty".to_string(),
            ));
        }

        self.ctx.bind_to_thread()?;
        self.ensure_no_callback_error()?;

        let payload_size = c_ulong::try_from(access_unit.len()).map_err(|_| {
            DecodeError::InvalidInput("access unit size does not fit into c_ulong".to_string())
        })?;
        let flags = (CUvideopacketflags::CUVID_PKT_TIMESTAMP as c_ulong)
            | (CUvideopacketflags::CUVID_PKT_ENDOFPICTURE as c_ulong);
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags,
            payload_size,
            payload: access_unit.as_ptr(),
            timestamp: timestamp_90k as c_longlong,
        };

        check_nvdec(
            // SAFETY: parser and packet pointers are valid for the call.
            unsafe { cuvidParseVideoData(self.parser, &mut packet) },
            "cuvidParseVideoData",
        )?;
        self.ensure_no_callback_error()?;

        self.drain_display_queue()
    }

    /// Flushes the decoder by sending an end-of-stream packet.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or frame draining fails.
    pub fn flush(&mut self) -> Result<Vec<DecodedRgbFrame>, DecodeError> {
        self.ctx.bind_to_thread()?;
        self.ensure_no_callback_error()?;

        let flags = (CUvideopacketflags::CUVID_PKT_ENDOFSTREAM as c_ulong)
            | (CUvideopacketflags::CUVID_PKT_NOTIFY_EOS as c_ulong);
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        check_nvdec(
            // SAFETY: parser and packet pointers are valid for the call.
            unsafe { cuvidParseVideoData(self.parser, &mut packet) },
            "cuvidParseVideoData",
        )?;
        self.ensure_no_callback_error()?;

        self.drain_display_queue()
    }

    fn ensure_no_callback_error(&self) -> Result<(), DecodeError> {
        let state = lock_state(&self.bridge.state);
        match &state.sticky_error {
            Some(err) => Err(err.clone()),
            None => Ok(()),
        }
    }

    fn drain_display_queue(&mut self) -> Result<Vec<DecodedRgbFrame>, DecodeError> {
        self.ctx.bind_to_thread()?;

        let mut frames = Vec::new();
        loop {
            let (entry, decoder, layout) = {
                let mut state = lock_state(&self.bridge.state);
                if let Some(err) = &state.sticky_error {
                    return Err(err.clone());
                }

                let Some(entry) = state.display_queue.pop_front() else {
                    break;
                };
                let Some(decoder) = state.decoder else {
                    return Err(DecodeError::Internal(
                        "display callback fired before decoder initialization".to_string(),
                    ));
                };
                if !state.layout.is_valid() {
                    return Err(DecodeError::Internal(
                        "display callback fired before layout initialization".to_string(),
                    ));
                }
                (entry, decoder, state.layout)
            };

            frames.push(self.map_frame(decoder, entry, layout)?);
        }

        self.ensure_no_callback_error()?;
        Ok(frames)
    }

    fn map_frame(
        &self,
        decoder: CUvideodecoder,
        entry: DisplayQueueEntry,
        layout: OutputLayout,
    ) -> Result<DecodedRgbFrame, DecodeError> {
        let mut proc_params = CUVIDPROCPARAMS {
            progressive_frame: entry.progressive_frame,
            top_field_first: entry.top_field_first,
            second_field: 0,
            unpaired_field: 0,
            ..Default::default()
        };
        let mut dev_ptr: u64 = 0;
        let mut pitch: u32 = 0;

        check_nvdec(
            // SAFETY: decoder handle and output pointers are valid.
            unsafe {
                cuvidMapVideoFrame64(
                    decoder,
                    entry.picture_index,
                    &mut dev_ptr,
                    &mut pitch,
                    &mut proc_params,
                )
            },
            "cuvidMapVideoFrame64",
        )?;

        let mapped = self.copy_mapped_nv12_to_rgb(dev_ptr, pitch as usize, layout, entry.timestamp);
        let unmap = check_nvdec(
            // SAFETY: mapped device pointer belongs to this decoder and must be unmapped once.
            unsafe { cuvidUnmapVideoFrame64(decoder, dev_ptr) },
            "cuvidUnmapVideoFrame64",
        );

        match (mapped, unmap) {
            (Ok(frame), Ok(())) => Ok(frame),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }

    fn copy_mapped_nv12_to_rgb(
        &self,
        dev_ptr: u64,
        pitch: usize,
        layout: OutputLayout,
        timestamp: i64,
    ) -> Result<DecodedRgbFrame, DecodeError> {
        let coded_height = usize::try_from(layout.coded_height).map_err(|_| {
            DecodeError::Internal("coded_height does not fit into usize".to_string())
        })?;
        let nv12_bytes = pitch
            .checked_mul(coded_height + (coded_height / 2))
            .ok_or_else(|| DecodeError::Internal("NV12 byte size overflow".to_string()))?;
        let mut nv12 = vec![0_u8; nv12_bytes];

        // SAFETY: `dev_ptr` points to `nv12_bytes` mapped bytes for this frame.
        unsafe { result::memcpy_dtoh_sync(nv12.as_mut_slice(), dev_ptr)? };
        let rgb = nv12_to_rgb24(&nv12, pitch, layout)?;

        Ok(DecodedRgbFrame {
            width: layout.visible_width,
            height: layout.visible_height,
            timestamp_90k: timestamp,
            data: rgb,
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        if !self.parser.is_null() {
            // SAFETY: parser was created by `cuvidCreateVideoParser` and is owned by self.
            let _ = unsafe { cuvidDestroyVideoParser(self.parser) }.result();
            self.parser = ptr::null_mut();
        }

        let decoder = {
            let mut state = lock_state(&self.bridge.state);
            state.decoder.take()
        };
        if let Some(decoder) = decoder {
            // SAFETY: decoder was created by `cuvidCreateDecoder` and is owned by self.
            let _ = unsafe { cuvidDestroyDecoder(decoder) }.result();
        }
    }
}

#[derive(Debug)]
struct CallbackBridge {
    codec: DecodeCodec,
    options: DecodeOptions,
    state: Mutex<DecoderState>,
}

#[derive(Debug, Clone, Copy, Default)]
struct DisplayQueueEntry {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    timestamp: i64,
}

#[derive(Debug, Clone, Copy, Default)]
struct OutputLayout {
    coded_width: u32,
    coded_height: u32,
    visible_left: u32,
    visible_top: u32,
    visible_width: u32,
    visible_height: u32,
}

impl OutputLayout {
    fn is_valid(self) -> bool {
        self.coded_width > 0
            && self.coded_height > 0
            && self.visible_width > 0
            && self.visible_height > 0
    }
}

#[derive(Debug, Default)]
struct DecoderState {
    decoder: Option<CUvideodecoder>,
    sticky_error: Option<DecodeError>,
    display_queue: VecDeque<DisplayQueueEntry>,
    layout: OutputLayout,
}

impl DecoderState {
    fn set_error_once(&mut self, err: DecodeError) {
        if self.sticky_error.is_none() {
            self.sticky_error = Some(err);
        }
    }

    fn configure_decoder(
        &mut self,
        codec: DecodeCodec,
        options: DecodeOptions,
        format: &CUVIDEOFORMAT,
    ) -> Result<c_int, DecodeError> {
        if format.bit_depth_luma_minus8 != 0 || format.bit_depth_chroma_minus8 != 0 {
            return Err(DecodeError::Unsupported(
                "only 8-bit input is currently supported".to_string(),
            ));
        }
        if format.chroma_format != cudaVideoChromaFormat::cudaVideoChromaFormat_420 {
            return Err(DecodeError::Unsupported(
                "only 4:2:0 chroma format is currently supported".to_string(),
            ));
        }
        if format.coded_width == 0 || format.coded_height == 0 {
            return Err(DecodeError::InvalidInput(
                "sequence callback reported zero dimensions".to_string(),
            ));
        }

        let layout = derive_output_layout(format);
        let num_surfaces = u32::from(format.min_num_decode_surfaces.max(1));
        if let Some(decoder) = self.decoder {
            let mut reconfigure_info = build_reconfigure_info(
                format,
                layout.visible_width,
                layout.visible_height,
                num_surfaces,
            );
            check_nvdec(
                // SAFETY: decoder is valid and reconfigure info is initialized.
                unsafe { cuvidReconfigureDecoder(decoder, &mut reconfigure_info) },
                "cuvidReconfigureDecoder",
            )?;
        } else {
            let mut create_info = build_decode_create_info(
                codec,
                options,
                format,
                layout.visible_width,
                layout.visible_height,
                num_surfaces,
            );
            let mut decoder = ptr::null_mut();
            check_nvdec(
                // SAFETY: output pointer and create info are valid.
                unsafe { cuvidCreateDecoder(&mut decoder, &mut create_info) },
                "cuvidCreateDecoder",
            )?;
            self.decoder = Some(decoder);
        }

        self.layout = layout;
        Ok(num_surfaces as c_int)
    }
}

unsafe extern "C" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> c_int {
    let Some(bridge) = bridge_from_user_data(user_data) else {
        return 0;
    };
    if format.is_null() {
        let mut state = lock_state(&bridge.state);
        state.set_error_once(DecodeError::InvalidInput(
            "null CUVIDEOFORMAT pointer in sequence callback".to_string(),
        ));
        return 0;
    }

    let mut state = lock_state(&bridge.state);
    let result = state.configure_decoder(
        bridge.codec,
        bridge.options,
        // SAFETY: pointer checked for null above.
        unsafe { &*format },
    );
    match result {
        Ok(num_surfaces) => num_surfaces,
        Err(err) => {
            state.set_error_once(err);
            0
        }
    }
}

unsafe extern "C" fn decode_callback(
    user_data: *mut c_void,
    pic_params: *mut CUVIDPICPARAMS,
) -> c_int {
    let Some(bridge) = bridge_from_user_data(user_data) else {
        return 0;
    };
    if pic_params.is_null() {
        let mut state = lock_state(&bridge.state);
        state.set_error_once(DecodeError::InvalidInput(
            "null CUVIDPICPARAMS pointer in decode callback".to_string(),
        ));
        return 0;
    }

    let mut state = lock_state(&bridge.state);
    let Some(decoder) = state.decoder else {
        state.set_error_once(DecodeError::Internal(
            "decode callback fired before decoder creation".to_string(),
        ));
        return 0;
    };

    let result = check_nvdec(
        // SAFETY: decoder and picture params are valid by callback contract.
        unsafe { cuvidDecodePicture(decoder, pic_params) },
        "cuvidDecodePicture",
    );
    match result {
        Ok(()) => 1,
        Err(err) => {
            state.set_error_once(err);
            0
        }
    }
}

unsafe extern "C" fn display_callback(
    user_data: *mut c_void,
    display_info: *mut CUVIDPARSERDISPINFO,
) -> c_int {
    let Some(bridge) = bridge_from_user_data(user_data) else {
        return 0;
    };
    if display_info.is_null() {
        return 1;
    }

    // SAFETY: pointer checked for null above.
    let display_info = unsafe { &*display_info };
    let mut state = lock_state(&bridge.state);
    state.display_queue.push_back(DisplayQueueEntry {
        picture_index: display_info.picture_index,
        progressive_frame: display_info.progressive_frame,
        top_field_first: display_info.top_field_first,
        timestamp: display_info.timestamp,
    });
    1
}

fn check_decoder_caps(codec: DecodeCodec) -> Result<(), DecodeError> {
    let mut caps = CUVIDDECODECAPS {
        eCodecType: codec.as_cuda_codec(),
        eChromaFormat: cudaVideoChromaFormat::cudaVideoChromaFormat_420,
        nBitDepthMinus8: 0,
        ..Default::default()
    };
    check_nvdec(
        // SAFETY: pointer is valid for the duration of the call.
        unsafe { cuvidGetDecoderCaps(&mut caps) },
        "cuvidGetDecoderCaps",
    )?;

    if caps.bIsSupported == 0 {
        return Err(DecodeError::Unsupported(format!(
            "{codec:?} decoder is not supported by this GPU",
        )));
    }
    let nv12_mask = 1_u16 << (cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12 as u32);
    if (caps.nOutputFormatMask & nv12_mask) == 0 {
        return Err(DecodeError::Unsupported(
            "NV12 output is not supported by the hardware decoder".to_string(),
        ));
    }
    Ok(())
}

fn build_decode_create_info(
    codec: DecodeCodec,
    _options: DecodeOptions,
    format: &CUVIDEOFORMAT,
    _visible_width: u32,
    _visible_height: u32,
    num_surfaces: u32,
) -> CUVIDDECODECREATEINFO {
    CUVIDDECODECREATEINFO {
        ulWidth: format.coded_width as c_ulong,
        ulHeight: format.coded_height as c_ulong,
        ulNumDecodeSurfaces: num_surfaces as c_ulong,
        CodecType: codec.as_cuda_codec(),
        ChromaFormat: format.chroma_format,
        ulCreationFlags: cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as c_ulong,
        bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
        ulIntraDecodeOnly: 0,
        ulMaxWidth: format.coded_width as c_ulong,
        ulMaxHeight: format.coded_height as c_ulong,
        display_area: to_create_rect(
            format.display_area.left,
            format.display_area.top,
            format.display_area.right,
            format.display_area.bottom,
        ),
        OutputFormat: cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
        DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,
        ulTargetWidth: format.coded_width as c_ulong,
        ulTargetHeight: format.coded_height as c_ulong,
        ulNumOutputSurfaces: 2,
        vidLock: ptr::null_mut(),
        target_rect: to_create_target_rect(
            0,
            0,
            format.coded_width as i32,
            format.coded_height as i32,
        ),
        enableHistogram: 0,
        ..Default::default()
    }
}

fn build_reconfigure_info(
    format: &CUVIDEOFORMAT,
    _visible_width: u32,
    _visible_height: u32,
    num_surfaces: u32,
) -> CUVIDRECONFIGUREDECODERINFO {
    CUVIDRECONFIGUREDECODERINFO {
        ulWidth: format.coded_width,
        ulHeight: format.coded_height,
        ulTargetWidth: format.coded_width,
        ulTargetHeight: format.coded_height,
        ulNumDecodeSurfaces: num_surfaces,
        display_area: to_reconfigure_rect(
            format.display_area.left,
            format.display_area.top,
            format.display_area.right,
            format.display_area.bottom,
        ),
        target_rect: to_reconfigure_target_rect(
            0,
            0,
            format.coded_width as i32,
            format.coded_height as i32,
        ),
        ..Default::default()
    }
}

fn derive_output_layout(format: &CUVIDEOFORMAT) -> OutputLayout {
    let coded_width = format.coded_width;
    let coded_height = format.coded_height;

    let left = format.display_area.left.max(0) as u32;
    let top = format.display_area.top.max(0) as u32;
    let mut right = format.display_area.right.max(0) as u32;
    let mut bottom = format.display_area.bottom.max(0) as u32;

    if right == 0 || right > coded_width {
        right = coded_width;
    }
    if bottom == 0 || bottom > coded_height {
        bottom = coded_height;
    }

    let (visible_left, visible_top, visible_width, visible_height) = if right > left && bottom > top
    {
        (left, top, right - left, bottom - top)
    } else {
        (0, 0, coded_width, coded_height)
    };

    OutputLayout {
        coded_width,
        coded_height,
        visible_left,
        visible_top,
        visible_width,
        visible_height,
    }
}

fn nv12_to_rgb24(nv12: &[u8], pitch: usize, layout: OutputLayout) -> Result<Vec<u8>, DecodeError> {
    let coded_height = usize::try_from(layout.coded_height)
        .map_err(|_| DecodeError::Internal("coded_height does not fit into usize".to_string()))?;
    let visible_left = usize::try_from(layout.visible_left)
        .map_err(|_| DecodeError::Internal("visible_left does not fit into usize".to_string()))?;
    let visible_top = usize::try_from(layout.visible_top)
        .map_err(|_| DecodeError::Internal("visible_top does not fit into usize".to_string()))?;
    let visible_width = usize::try_from(layout.visible_width)
        .map_err(|_| DecodeError::Internal("visible_width does not fit into usize".to_string()))?;
    let visible_height = usize::try_from(layout.visible_height)
        .map_err(|_| DecodeError::Internal("visible_height does not fit into usize".to_string()))?;

    if visible_left + visible_width > pitch {
        return Err(DecodeError::Internal(
            "visible rect exceeds mapped pitch".to_string(),
        ));
    }
    if visible_top + visible_height > coded_height {
        return Err(DecodeError::Internal(
            "visible rect exceeds mapped height".to_string(),
        ));
    }

    let luma_size = pitch
        .checked_mul(coded_height)
        .ok_or_else(|| DecodeError::Internal("luma plane size overflow".to_string()))?;
    if nv12.len() < luma_size + (luma_size / 2) {
        return Err(DecodeError::Internal(
            "NV12 buffer is smaller than expected".to_string(),
        ));
    }
    let uv_base = luma_size;

    let mut rgb = vec![0_u8; visible_width * visible_height * 3];
    for y in 0..visible_height {
        let src_y = visible_top + y;
        let y_row = src_y * pitch;
        let uv_row = uv_base + (src_y / 2) * pitch;
        let dst_row = y * visible_width * 3;

        for x in 0..visible_width {
            let src_x = visible_left + x;
            let y_value = i32::from(nv12[y_row + src_x]);
            let uv_index = uv_row + (src_x & !1);
            let u_value = i32::from(nv12[uv_index]);
            let v_value = i32::from(nv12[uv_index + 1]);

            let c = (y_value - 16).max(0);
            let d = u_value - 128;
            let e = v_value - 128;

            let r = clip_to_u8((298 * c + 409 * e + 128) >> 8);
            let g = clip_to_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
            let b = clip_to_u8((298 * c + 516 * d + 128) >> 8);

            let dst = dst_row + x * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }

    Ok(rgb)
}

fn clip_to_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn check_nvdec(status: CUresult, operation: &'static str) -> Result<(), DecodeError> {
    status.result().map_err(|err| DecodeError::Nvdec {
        operation,
        code: err.0,
    })
}

fn lock_state<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn to_create_rect(
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> crate::sys::cuviddec::_CUVIDDECODECREATEINFO__bindgen_ty_1 {
    crate::sys::cuviddec::_CUVIDDECODECREATEINFO__bindgen_ty_1 {
        left: to_c_short(left),
        top: to_c_short(top),
        right: to_c_short(right),
        bottom: to_c_short(bottom),
    }
}

fn to_reconfigure_rect(
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> crate::sys::cuviddec::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_1 {
    crate::sys::cuviddec::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_1 {
        left: to_c_short(left),
        top: to_c_short(top),
        right: to_c_short(right),
        bottom: to_c_short(bottom),
    }
}

fn to_reconfigure_target_rect(
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> crate::sys::cuviddec::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_2 {
    crate::sys::cuviddec::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_2 {
        left: to_c_short(left),
        top: to_c_short(top),
        right: to_c_short(right),
        bottom: to_c_short(bottom),
    }
}

fn to_create_target_rect(
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> crate::sys::cuviddec::_CUVIDDECODECREATEINFO__bindgen_ty_2 {
    crate::sys::cuviddec::_CUVIDDECODECREATEINFO__bindgen_ty_2 {
        left: to_c_short(left),
        top: to_c_short(top),
        right: to_c_short(right),
        bottom: to_c_short(bottom),
    }
}

fn to_c_short(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn bridge_from_user_data(user_data: *mut c_void) -> Option<&'static CallbackBridge> {
    if user_data.is_null() {
        None
    } else {
        // SAFETY: user_data was created from Box<CallbackBridge> in Decoder::new and
        // lives until Decoder drop.
        Some(unsafe { &*user_data.cast::<CallbackBridge>() })
    }
}
