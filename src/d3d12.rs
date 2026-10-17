//! Thread-affine external D3D12 NVENC session and explicit retirement.
//!
//! The input image stays on the producer's device. Only the compressed access
//! unit is copied into a Rust-owned vector. Failed or uncertain driver operations
//! retain all referenced resources; explicit close reports physical retirement.

use crate::{
    runtime::{LoadError, RuntimeNvenc},
    sys::nvEncodeAPI::{
        NVENCAPI_VERSION, NVENCSTATUS, NV_ENCODE_API_FUNCTION_LIST, NV_ENC_BUFFER_FORMAT,
        NV_ENC_BUFFER_USAGE, NV_ENC_CONFIG_VER, NV_ENC_DEVICE_TYPE, NV_ENC_FENCE_POINT_D3D12_VER,
        NV_ENC_INITIALIZE_PARAMS, NV_ENC_INITIALIZE_PARAMS_VER, NV_ENC_INPUT_PTR,
        NV_ENC_INPUT_RESOURCE_D3D12, NV_ENC_INPUT_RESOURCE_D3D12_VER, NV_ENC_INPUT_RESOURCE_TYPE,
        NV_ENC_LOCK_BITSTREAM, NV_ENC_LOCK_BITSTREAM_VER, NV_ENC_MAP_INPUT_RESOURCE,
        NV_ENC_MAP_INPUT_RESOURCE_VER, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS,
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_OUTPUT_RESOURCE_D3D12,
        NV_ENC_OUTPUT_RESOURCE_D3D12_VER, NV_ENC_PIC_FLAGS, NV_ENC_PIC_PARAMS,
        NV_ENC_PIC_PARAMS_VER, NV_ENC_PIC_STRUCT, NV_ENC_PIC_TYPE, NV_ENC_PRESET_CONFIG,
        NV_ENC_PRESET_CONFIG_VER, NV_ENC_PRESET_P3_GUID, NV_ENC_REGISTERED_PTR,
        NV_ENC_REGISTER_RESOURCE, NV_ENC_REGISTER_RESOURCE_VER,
    },
};
use std::{
    cell::RefCell,
    ffi::{c_void, CStr},
    mem::ManuallyDrop,
    num::NonZeroU64,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};
use windows::{
    core::{IUnknown, Interface},
    Win32::Graphics::{
        Direct3D12::{
            ID3D12Device, ID3D12DeviceChild, ID3D12Fence, ID3D12Resource, D3D12_FENCE_FLAG_NONE,
            D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        },
        Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC},
    },
};

/// Encoding failure, preserving the driver operation and exact status.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    /// Driver discovery failed.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// D3D12 operation failed.
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
    /// Configuration, resource, timestamp, or output violated the contract.
    #[error("D3D12 NVENC contract: {0}")]
    Contract(&'static str),
    /// Driver rejected an operation.
    #[error("NVENC {operation}: {status:?} {detail}")]
    Driver {
        /// Failing operation.
        operation: &'static str,
        /// Raw driver status.
        status: NVENCSTATUS,
        /// Driver diagnostic, capped at 384 characters.
        detail: String,
    },
    /// Completion was not observed; the encoder retains the submission.
    #[error("NVENC completion deadline; resources remain retained")]
    CompletionDeadline,
    /// Pre-configuration admission failed and its one physical close also failed.
    #[error("NVENC initialization failed: {cause}; session close failed: {cleanup}")]
    InitializationCleanup {
        /// Original admission or read-only query failure.
        #[source]
        cause: Box<EncodeError>,
        /// Close failure; the complete session remains retained.
        cleanup: Box<EncodeError>,
    },
}

/// Positive external producer fence point. `UINT64_MAX` is device removal.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FenceValue(NonZeroU64);
impl FenceValue {
    /// Validates a positive, non-removal fence value.
    /// # Errors
    /// Rejects zero and `UINT64_MAX`.
    pub fn new(value: u64) -> Result<Self, EncodeError> {
        if value == u64::MAX {
            return Err(EncodeError::Contract("invalid fence value"));
        }
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(EncodeError::Contract("zero fence value"))
    }
    /// Returns the native fence value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Positive exact microseconds, representable in Android's signed nanoseconds.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimestampMicros(NonZeroU64);
impl TimestampMicros {
    /// Validates an exact timestamp without rescaling or saturation.
    /// # Errors
    /// Rejects zero and values exceeding signed nanosecond representation.
    pub fn new(value: u64) -> Result<Self, EncodeError> {
        if value > (i64::MAX as u64) / 1000 {
            return Err(EncodeError::Contract(
                "timestamp exceeds signed nanoseconds",
            ));
        }
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(EncodeError::Contract("zero timestamp"))
    }
    /// Returns exact microseconds.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

mod capabilities;
mod config;
#[cfg(test)]
mod hardware_tests;

pub use config::{Codec, EncoderCapabilities, EncoderConfig, H264Config, HevcLosslessConfig};

/// Owned compressed packet, independent of the encoder's mapped output.
#[derive(Debug)]
pub struct EncodedAccessUnit {
    bytes: Vec<u8>,
    timestamp: TimestampMicros,
    idr: bool,
    codec: Codec,
}
impl EncodedAccessUnit {
    /// Borrows Annex-B data in the packet's declared codec.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Exact input timestamp acknowledged by NVENC.
    #[must_use]
    pub const fn timestamp(&self) -> TimestampMicros {
        self.timestamp
    }
    /// Whether NVENC identified this packet as IDR.
    #[must_use]
    pub const fn is_idr(&self) -> bool {
        self.idr
    }
    /// Coding syntax of the owned access unit.
    #[must_use]
    pub const fn codec(&self) -> Codec {
        self.codec
    }
}

/// Owns the codec session, COM references, registrations, mappings, and driver.
///
/// This value cannot move between threads. Create and close it on the codec
/// worker. An unsuccessful close retains ownership and can be inspected/retried
/// only when the state permits it; dropping uncertain state quarantines it for
/// process lifetime rather than freeing resources still referenced by a driver.
pub struct D3d12Encoder {
    inner: Option<Box<Inner>>,
}
impl std::fmt::Debug for D3d12Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3d12Encoder")
            .field("open", &self.inner.is_some())
            .field(
                "quarantined",
                &self
                    .inner
                    .as_ref()
                    .is_some_and(|i| i.retirement_phase == RetirementPhase::Uncertain),
            )
            .finish()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionPhase {
    NotOpened,
    Opened,
    Initialized,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmissionPhase {
    Idle,
    Submitted,
    Locked,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetirementPhase {
    Open,
    EosAttempted,
    Uncertain,
}
struct Inner {
    _permit: AdmissionPermit,
    driver: RuntimeNvenc,
    device: ID3D12Device,
    input: ID3D12Resource,
    producer: ID3D12Fence,
    output: Option<ID3D12Resource>,
    done: Option<ID3D12Fence>,
    session: *mut c_void,
    session_phase: SessionPhase,
    input_registered: NV_ENC_REGISTERED_PTR,
    output_registered: NV_ENC_REGISTERED_PTR,
    input_mapped: NV_ENC_INPUT_PTR,
    output_mapped: NV_ENC_INPUT_PTR,
    input_wrapper: NV_ENC_INPUT_RESOURCE_D3D12,
    output_wrapper: NV_ENC_OUTPUT_RESOURCE_D3D12,
    config: EncoderConfig,
    capabilities: Option<EncoderCapabilities>,
    submission_phase: SubmissionPhase,
    retirement_phase: RetirementPhase,
    output_value: u64,
    last_producer: u64,
    last_timestamp: u64,
}
// If driver completion/cleanup is uncertain, releasing resources is unsafe.
// ManuallyDrop preserves their ownership even when the codec worker exits.
// Application admission must bound live/quarantined sessions; this is a terminal
// safety retention, never a normal reusable pool or a retry-on-Drop mechanism.
thread_local! { static QUARANTINED: RefCell<Vec<ManuallyDrop<Box<Inner>>>> = const { RefCell::new(Vec::new()) }; }

static ADMITTED: AtomicUsize = AtomicUsize::new(0);
struct AdmissionPermit;
impl AdmissionPermit {
    fn acquire() -> Result<Self, EncodeError> {
        ADMITTED
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < 2).then_some(count + 1)
            })
            .map(|_| Self)
            .map_err(|_| {
                EncodeError::Contract(
                    "D3D12 encoder capacity full; retained sessions require process restart",
                )
            })
    }
}
impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        ADMITTED.fetch_sub(1, Ordering::AcqRel);
    }
}

impl D3d12Encoder {
    /// Registers the producer's NV12 texture on its actual D3D12 device.
    ///
    /// # Safety
    /// The caller must provide a valid single-plane-mip NV12 texture in COMMON
    /// state, coordinate every producer write with `producer`, and ensure no
    /// concurrent codec consumes or mutates these resources. COM references are
    /// retained; the producer queue must not write while `encode` is running or
    /// after an uncertain submission until physical retirement is proved.
    ///
    /// # Errors
    /// Rejects different devices, incompatible texture descriptors, unavailable
    /// driver/API, and failed resource/session initialization.
    pub unsafe fn new(
        device: ID3D12Device,
        input: ID3D12Resource,
        producer: ID3D12Fence,
        config: EncoderConfig,
    ) -> Result<Self, EncodeError> {
        // SAFETY: GetDesc/GetDevice operate on live owned COM references only.
        let desc = unsafe { input.GetDesc() };
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.Format != DXGI_FORMAT_NV12
            || desc.Width != u64::from(config.width())
            || desc.Height != config.height()
            || desc.DepthOrArraySize != 1
            || desc.MipLevels != 1
            || desc.SampleDesc.Count != 1
        {
            return Err(EncodeError::Contract(
                "input descriptor differs from NV12 configuration",
            ));
        }
        verify_device(&device, &input)?;
        verify_device(&device, &producer)?;
        let permit = AdmissionPermit::acquire()?;
        let driver = RuntimeNvenc::load()?;
        let mut result = Self {
            inner: Some(Box::new(Inner {
                _permit: permit,
                driver,
                device,
                input,
                producer,
                output: None,
                done: None,
                session: ptr::null_mut(),
                session_phase: SessionPhase::NotOpened,
                input_registered: ptr::null_mut(),
                output_registered: ptr::null_mut(),
                input_mapped: ptr::null_mut(),
                output_mapped: ptr::null_mut(),
                input_wrapper: NV_ENC_INPUT_RESOURCE_D3D12::default(),
                output_wrapper: NV_ENC_OUTPUT_RESOURCE_D3D12::default(),
                config,
                capabilities: None,
                submission_phase: SubmissionPhase::Idle,
                retirement_phase: RetirementPhase::Open,
                output_value: 0,
                last_producer: 0,
                last_timestamp: 0,
            })),
        };
        let owner = result
            .inner
            .as_mut()
            .ok_or(EncodeError::Contract("missing owner"))?;
        if let Err(error) = owner.initialize() {
            // A failed driver operation after a session handle was returned may
            // still have consumed registrations. Constructor failure is not
            // proof that releasing this partially initialized owner is safe.
            if !owner.session.is_null() {
                owner.retirement_phase = RetirementPhase::Uncertain;
            }
            return Err(error);
        }
        Ok(result)
    }

    /// Returns actual driver capability facts from this session's construction.
    /// # Errors
    /// Rejects a closed or incompletely initialized owner.
    pub fn capabilities(&self) -> Result<EncoderCapabilities, EncodeError> {
        self.inner
            .as_ref()
            .and_then(|inner| inner.capabilities)
            .ok_or(EncodeError::Contract(
                "encoder capabilities are not available",
            ))
    }

    /// Encodes after the producer's increasing fence point, without CPU image readback.
    ///
    /// # Safety
    /// `ready` must identify this input image's completed producer work, including
    /// its transition to COMMON. No writer may reuse the image during the call.
    /// On failure, check/close the encoder before any image reuse; the driver may
    /// have consumed a submission even when it returned an error or timed out.
    ///
    /// # Errors
    /// Rejects non-increasing fence/PTS, retired/uncertain state, driver failures,
    /// completion deadline, invalid packet sizes, and timestamp mismatch.
    pub unsafe fn encode(
        &mut self,
        ready: FenceValue,
        timestamp: TimestampMicros,
        force_idr: bool,
    ) -> Result<EncodedAccessUnit, EncodeError> {
        let owner = self
            .inner
            .as_mut()
            .ok_or(EncodeError::Contract("encoder is closed"))?;
        if owner.retirement_phase != RetirementPhase::Open
            || owner.submission_phase != SubmissionPhase::Idle
        {
            return Err(EncodeError::Contract("encoder is not reusable"));
        }
        if ready.get() <= owner.last_producer || timestamp.get() <= owner.last_timestamp {
            return Err(EncodeError::Contract("fence and timestamp must increase"));
        }
        let outcome = owner.encode(ready, timestamp, force_idr);
        if outcome.is_err() {
            owner.retirement_phase = RetirementPhase::Uncertain;
        }
        outcome
    }

    /// Physically destroys the session and registrations only from a proven idle state.
    ///
    /// # Errors
    /// Returns an error while resources remain retained. No failed EOS/unlock or
    /// cleanup driver call is blindly repeated.
    pub fn try_close(&mut self) -> Result<(), EncodeError> {
        if let Some(owner) = &mut self.inner {
            owner.close()?;
        }
        self.inner = None;
        Ok(())
    }
}
impl Drop for D3d12Encoder {
    fn drop(&mut self) {
        if self.try_close().is_err() {
            if let Some(owner) = self.inner.take() {
                // Disable destruction before touching TLS: worker TLS teardown
                // or a reentrant borrow must never drop the unsafe-to-free owner.
                let owner = ManuallyDrop::new(owner);
                let _ = QUARANTINED.try_with(|retained| {
                    if let Ok(mut retained) = retained.try_borrow_mut() {
                        retained.push(owner);
                    }
                });
            }
        }
    }
}

fn verify_device<T: Interface>(device: &ID3D12Device, resource: &T) -> Result<(), EncodeError> {
    let child: ID3D12DeviceChild = resource.cast()?;
    let mut resource_device: Option<ID3D12Device> = None;
    // SAFETY: a live device-child interface writes an AddRef'd device to the
    // initialized output. IUnknown casts compare canonical COM identities.
    unsafe {
        child.GetDevice(&raw mut resource_device)?;
    }
    let resource_device = resource_device.ok_or(EncodeError::Contract("resource has no device"))?;
    if device.cast::<IUnknown>()? != resource_device.cast::<IUnknown>()? {
        return Err(EncodeError::Contract(
            "external resources belong to another D3D12 device",
        ));
    }
    Ok(())
}

impl Inner {
    fn api(&self) -> &NV_ENCODE_API_FUNCTION_LIST {
        self.driver.functions()
    }
    fn check(&self, operation: &'static str, status: NVENCSTATUS) -> Result<(), EncodeError> {
        if status == NVENCSTATUS::NV_ENC_SUCCESS {
            return Ok(());
        }
        let detail = if self.session_phase == SessionPhase::NotOpened || self.session.is_null() {
            String::new()
        } else if let Some(last_error) = self.api().nvEncGetLastErrorString {
            // SAFETY: the driver owns a NUL-terminated diagnostic for this live
            // session. It is copied before the next driver call or session close.
            let diagnostic = unsafe { last_error(self.session) };
            if diagnostic.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(diagnostic) }
                    .to_string_lossy()
                    .chars()
                    .take(384)
                    .collect()
            }
        } else {
            String::new()
        };
        Err(EncodeError::Driver {
            operation,
            status,
            detail,
        })
    }
    fn initialize(&mut self) -> Result<(), EncodeError> {
        let identity: IUnknown = self.device.cast()?;
        let mut open = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS {
            version: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            apiVersion: NVENCAPI_VERSION,
            deviceType: NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_DIRECTX,
            device: identity.as_raw(),
            ..Default::default()
        };
        // SAFETY: all callbacks were checked by RuntimeNvenc; structs use pinned
        // ABI versions, initialized reserved fields, and owned live COM pointers.
        let status = unsafe {
            self.api()
                .nvEncOpenEncodeSessionEx
                .ok_or(EncodeError::Contract("missing open"))?(
                &raw mut open, &raw mut self.session
            )
        };
        self.check("open D3D12 session", status)?;
        self.session_phase = SessionPhase::Opened;
        if self.session.is_null() {
            return Err(EncodeError::Contract("driver returned null session"));
        }
        // Queries/admission cannot register or submit GPU work. A definite
        // rejection closes this merely opened session exactly once. Configure
        // and registration failures below keep their existing uncertain owner.
        let query = self.query_capabilities();
        let capabilities = config::admit_or_close(query, self.config, || self.close())?;
        self.capabilities = Some(capabilities);
        self.configure()?;
        self.create_output()?;
        self.register_resources()
    }

    fn configure(&mut self) -> Result<(), EncodeError> {
        let mut preset = NV_ENC_PRESET_CONFIG {
            version: NV_ENC_PRESET_CONFIG_VER,
            ..Default::default()
        };
        preset.presetCfg.version = NV_ENC_CONFIG_VER;
        // SAFETY: live session and writable versioned configuration.
        self.check("get preset", unsafe {
            self.api()
                .nvEncGetEncodePresetConfigEx
                .ok_or(EncodeError::Contract("missing preset"))?(
                self.session,
                self.config.guid(),
                NV_ENC_PRESET_P3_GUID,
                self.config.tuning(),
                &raw mut preset,
            )
        })?;
        let mut config = self.config.apply(preset.presetCfg);
        let mut init = NV_ENC_INITIALIZE_PARAMS {
            version: NV_ENC_INITIALIZE_PARAMS_VER,
            encodeGUID: self.config.guid(),
            presetGUID: NV_ENC_PRESET_P3_GUID,
            tuningInfo: self.config.tuning(),
            maxEncodeWidth: self.config.width(),
            encodeWidth: self.config.width(),
            darWidth: self.config.width(),
            maxEncodeHeight: self.config.height(),
            encodeHeight: self.config.height(),
            darHeight: self.config.height(),
            frameRateNum: self.config.fps(),
            frameRateDen: 1,
            enablePTD: 1,
            bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12,
            encodeConfig: &raw mut config,
            ..Default::default()
        };
        // SAFETY: versioned init/config live through the synchronous call; NVENC
        // copies configuration and uses the already owned D3D12 device.
        self.check("initialize encoder", unsafe {
            self.api()
                .nvEncInitializeEncoder
                .ok_or(EncodeError::Contract("missing initialize"))?(
                self.session, &raw mut init
            )
        })?;
        self.session_phase = SessionPhase::Initialized;
        Ok(())
    }

    fn create_output(&mut self) -> Result<(), EncodeError> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(self.config.capacity()),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: valid readback buffer descriptor, initialized outputs receiving
        // owned COM references; initial COPY_DEST state is required by NVENC.
        unsafe {
            self.device.CreateCommittedResource(
                &raw const heap,
                D3D12_HEAP_FLAG_NONE,
                &raw const desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &raw mut self.output,
            )?;
        }
        // SAFETY: live device creates an independent owned codec-completion fence.
        self.done = Some(unsafe { self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? });
        Ok(())
    }

    fn register_resources(&mut self) -> Result<(), EncodeError> {
        let mut register = NV_ENC_REGISTER_RESOURCE {
            version: NV_ENC_REGISTER_RESOURCE_VER,
            resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX,
            resourceToRegister: self.input.as_raw(),
            width: self.config.width(),
            height: self.config.height(),
            bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12,
            bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE,
            ..Default::default()
        };
        // SAFETY: owned input satisfies constructor contract and remains retained
        // through unregistration; driver writes registration into this struct.
        let status = unsafe {
            self.api()
                .nvEncRegisterResource
                .ok_or(EncodeError::Contract("missing register"))?(
                self.session, &raw mut register
            )
        };
        self.input_registered = register.registeredResource;
        self.check("register input", status)?;
        register = NV_ENC_REGISTER_RESOURCE {
            version: NV_ENC_REGISTER_RESOURCE_VER,
            resourceType: NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX,
            resourceToRegister: self
                .output
                .as_ref()
                .ok_or(EncodeError::Contract("missing output"))?
                .as_raw(),
            width: self.config.capacity(),
            height: 1,
            bufferFormat: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_U8,
            bufferUsage: NV_ENC_BUFFER_USAGE::NV_ENC_OUTPUT_BITSTREAM,
            ..Default::default()
        };
        // SAFETY: owned readback output stays live until explicit unregistration.
        let status = unsafe {
            self.api()
                .nvEncRegisterResource
                .ok_or(EncodeError::Contract("missing register"))?(
                self.session, &raw mut register
            )
        };
        self.output_registered = register.registeredResource;
        self.check("register output", status)?;
        if self.input_registered.is_null() || self.output_registered.is_null() {
            return Err(EncodeError::Contract("driver returned null registration"));
        }
        Ok(())
    }

    fn map(&self, registered: NV_ENC_REGISTERED_PTR) -> Result<NV_ENC_INPUT_PTR, EncodeError> {
        let mut map = NV_ENC_MAP_INPUT_RESOURCE {
            version: NV_ENC_MAP_INPUT_RESOURCE_VER,
            registeredResource: registered,
            ..Default::default()
        };
        // SAFETY: registration belongs to this live session; map struct is writable
        // and remains live for the call. Owner retains the mapped handle afterwards.
        self.check("map resource", unsafe {
            self.api()
                .nvEncMapInputResource
                .ok_or(EncodeError::Contract("missing map"))?(self.session, &raw mut map)
        })?;
        if map.mappedResource.is_null() {
            return Err(EncodeError::Contract("driver returned null mapping"));
        }
        Ok(map.mappedResource)
    }
    fn unmap(&mut self) -> Result<(), EncodeError> {
        for output in [true, false] {
            let handle = if output {
                self.output_mapped
            } else {
                self.input_mapped
            };
            if handle.is_null() {
                continue;
            }
            // SAFETY: known mapped handle from this session; no submitted/locked
            // codec operation remains when this cleanup is invoked.
            let status = unsafe {
                self.api()
                    .nvEncUnmapInputResource
                    .ok_or(EncodeError::Contract("missing unmap"))?(
                    self.session, handle
                )
            };
            if status != NVENCSTATUS::NV_ENC_SUCCESS {
                self.retirement_phase = RetirementPhase::Uncertain;
            }
            self.check("unmap resource", status)?;
            if output {
                self.output_mapped = ptr::null_mut();
            } else {
                self.input_mapped = ptr::null_mut();
            }
        }
        Ok(())
    }
    fn encode(
        &mut self,
        ready: FenceValue,
        timestamp: TimestampMicros,
        force_idr: bool,
    ) -> Result<EncodedAccessUnit, EncodeError> {
        self.input_mapped = self.map(self.input_registered)?;
        self.output_mapped = self.map(self.output_registered)?;
        self.output_value = self
            .output_value
            .checked_add(1)
            .filter(|v| *v != u64::MAX)
            .ok_or(EncodeError::Contract("output fence exhausted"))?;
        self.input_wrapper = NV_ENC_INPUT_RESOURCE_D3D12 {
            version: NV_ENC_INPUT_RESOURCE_D3D12_VER,
            pInputBuffer: self.input_mapped,
            ..Default::default()
        };
        self.input_wrapper.inputFencePoint.version = NV_ENC_FENCE_POINT_D3D12_VER;
        self.input_wrapper.inputFencePoint.pFence = self.producer.as_raw();
        self.input_wrapper.inputFencePoint.waitValue = ready.get();
        self.input_wrapper.inputFencePoint.set_bWait(1);
        self.output_wrapper = NV_ENC_OUTPUT_RESOURCE_D3D12 {
            version: NV_ENC_OUTPUT_RESOURCE_D3D12_VER,
            pOutputBuffer: self.output_mapped,
            ..Default::default()
        };
        self.output_wrapper.outputFencePoint.version = NV_ENC_FENCE_POINT_D3D12_VER;
        self.output_wrapper.outputFencePoint.pFence = self
            .done
            .as_ref()
            .ok_or(EncodeError::Contract("missing completion fence"))?
            .as_raw();
        self.output_wrapper.outputFencePoint.signalValue = self.output_value;
        self.output_wrapper.outputFencePoint.set_bSignal(1);
        let first = self.last_timestamp == 0;
        let mut picture = NV_ENC_PIC_PARAMS {
            version: NV_ENC_PIC_PARAMS_VER,
            inputBuffer: ptr::from_mut(&mut self.input_wrapper).cast(),
            outputBitstream: ptr::from_mut(&mut self.output_wrapper).cast(),
            bufferFmt: NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12,
            inputWidth: self.config.width(),
            inputHeight: self.config.height(),
            inputTimeStamp: timestamp.get(),
            inputDuration: u64::from(1_000_000 / self.config.fps()),
            pictureStruct: NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME,
            ..Default::default()
        };
        if force_idr || first {
            picture.encodePicFlags = NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
                | NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS as u32;
        }
        self.submission_phase = SubmissionPhase::Submitted;
        // SAFETY: mapped handles wrapped with the required D3D12 fence structures,
        // kept at stable Box-owned addresses until observed codec completion.
        self.check("encode picture", unsafe {
            self.api()
                .nvEncEncodePicture
                .ok_or(EncodeError::Contract("missing encode"))?(
                self.session, &raw mut picture
            )
        })?;
        self.wait_output()?;
        self.collect_packet(ready, timestamp, first)
    }

    fn collect_packet(
        &mut self,
        ready: FenceValue,
        timestamp: TimestampMicros,
        first: bool,
    ) -> Result<EncodedAccessUnit, EncodeError> {
        let mut locked = NV_ENC_LOCK_BITSTREAM {
            version: NV_ENC_LOCK_BITSTREAM_VER,
            outputBitstream: ptr::from_mut(&mut self.output_wrapper).cast(),
            ..Default::default()
        };
        // SAFETY: completed output wrapper belongs to the session and remains live
        // through lock, copying, and exactly one unlock attempt.
        self.check("lock output", unsafe {
            self.api()
                .nvEncLockBitstream
                .ok_or(EncodeError::Contract("missing lock"))?(
                self.session, &raw mut locked
            )
        })?;
        self.submission_phase = SubmissionPhase::Locked;
        if locked.bitstreamBufferPtr.is_null()
            || locked.bitstreamSizeInBytes == 0
            || locked.bitstreamSizeInBytes > self.config.capacity()
            || locked.bitstreamSizeInBytes > 8 * 1024 * 1024
            || locked.outputTimeStamp != timestamp.get()
        {
            return Err(EncodeError::Contract(
                "bounded output packet or exact PTS differs",
            ));
        }
        let idr = locked.pictureType == NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR;
        if first && !idr {
            return Err(EncodeError::Contract("first packet must be IDR"));
        }
        // SAFETY: successful LockBitstream returns this readable allocation of
        // the checked size; it remains locked and live throughout the copy.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                locked.bitstreamBufferPtr.cast::<u8>(),
                locked.bitstreamSizeInBytes as usize,
            )
        }
        .to_vec();
        // SAFETY: exactly one unlock for this successful lock; any failure makes
        // the owner uncertain and prevents an unlock retry or early release.
        let unlock = self
            .api()
            .nvEncUnlockBitstream
            .ok_or(EncodeError::Contract("missing unlock"))?;
        let status =
            unsafe { unlock(self.session, ptr::from_mut(&mut self.output_wrapper).cast()) };
        self.check("unlock output", status)?;
        self.submission_phase = SubmissionPhase::Idle;
        self.unmap()?;
        self.last_producer = ready.get();
        self.last_timestamp = timestamp.get();
        Ok(EncodedAccessUnit {
            bytes,
            timestamp,
            idr,
            codec: self.config.codec(),
        })
    }
    fn wait_output(&self) -> Result<(), EncodeError> {
        let started = Instant::now();
        let done = self
            .done
            .as_ref()
            .ok_or(EncodeError::Contract("missing completion fence"))?;
        loop {
            // SAFETY: live completion fence owned by this encoder.
            let value = unsafe { done.GetCompletedValue() };
            if value == u64::MAX {
                return Err(EncodeError::Contract("D3D12 device removed"));
            }
            if value >= self.output_value {
                return Ok(());
            }
            if started.elapsed() >= Duration::from_secs(5) {
                return Err(EncodeError::CompletionDeadline);
            }
            thread::sleep(Duration::from_millis(1));
        }
    }
    fn close(&mut self) -> Result<(), EncodeError> {
        if self.retirement_phase == RetirementPhase::Uncertain
            || self.submission_phase != SubmissionPhase::Idle
        {
            return Err(EncodeError::Contract(
                "uncertain or in-flight session remains retained",
            ));
        }
        self.unmap()?;
        if self.session.is_null() {
            return Ok(());
        }
        if self.session_phase == SessionPhase::Initialized
            && self.retirement_phase == RetirementPhase::Open
        {
            self.retirement_phase = RetirementPhase::EosAttempted;
            let mut eos = NV_ENC_PIC_PARAMS {
                version: NV_ENC_PIC_PARAMS_VER,
                encodePicFlags: NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS as u32,
                ..Default::default()
            };
            // SAFETY: session is idle, all packets unlocked/unmapped; synchronous
            // IP-only mode has no pending buffered input. EOS is attempted once.
            let status = unsafe {
                self.api()
                    .nvEncEncodePicture
                    .ok_or(EncodeError::Contract("missing encode"))?(
                    self.session, &raw mut eos
                )
            };
            if status != NVENCSTATUS::NV_ENC_SUCCESS {
                self.retirement_phase = RetirementPhase::Uncertain;
            }
            self.check("end of stream", status)?;
        }
        for output in [true, false] {
            let registered = if output {
                self.output_registered
            } else {
                self.input_registered
            };
            if registered.is_null() {
                continue;
            }
            // SAFETY: known registration with no mappings or submitted work;
            // successful unregister is recorded before moving to the next handle.
            let status = unsafe {
                self.api()
                    .nvEncUnregisterResource
                    .ok_or(EncodeError::Contract("missing unregister"))?(
                    self.session, registered
                )
            };
            if status != NVENCSTATUS::NV_ENC_SUCCESS {
                self.retirement_phase = RetirementPhase::Uncertain;
            }
            self.check("unregister resource", status)?;
            if output {
                self.output_registered = ptr::null_mut();
            } else {
                self.input_registered = ptr::null_mut();
            }
        }
        // SAFETY: idle session with no live registrations/mappings; on failure it
        // remains retained and is never destroyed again through Drop.
        let status = unsafe {
            self.api()
                .nvEncDestroyEncoder
                .ok_or(EncodeError::Contract("missing destroy"))?(self.session)
        };
        if status != NVENCSTATUS::NV_ENC_SUCCESS {
            self.retirement_phase = RetirementPhase::Uncertain;
        }
        self.check("destroy encoder", status)?;
        self.session = ptr::null_mut();
        Ok(())
    }
}
