//! Fallible NVENC driver loading without a CUDA context or a static link dependency.
//!
//! Keep this owner alive until every encoder session created from its function
//! table has been destroyed. Calling the raw callbacks remains unsafe; their
//! resource and synchronization contracts are those of the pinned NVENC header.

use std::{fmt, marker::PhantomData, rc::Rc};

use crate::sys::nvEncodeAPI::{
    NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION, NVENCSTATUS, NV_ENCODE_API_FUNCTION_LIST,
    NV_ENCODE_API_FUNCTION_LIST_VER,
};

/// Driver discovery and function-table negotiation failed.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The platform driver library or a required exported symbol was unavailable.
    #[error("NVENC driver loading failed: {0}")]
    Library(#[from] libloading::Error),
    /// A driver entry point rejected the call.
    #[error("NVENC {operation} failed with {status:?}")]
    Driver {
        /// Entry point that failed.
        operation: &'static str,
        /// Exact driver status.
        status: NVENCSTATUS,
    },
    /// The installed driver cannot serve the pinned API.
    #[error("NVENC driver API {supported:#x} is older than required API {required:#x}")]
    OlderDriver {
        /// Driver API in the major << 4 | minor representation.
        supported: u32,
        /// Required API in the same representation.
        required: u32,
    },
    /// The negotiated function table omitted a required encoder callback.
    #[error("NVENC function table is missing {0}")]
    MissingCallback(&'static str),
}

/// Owns the dynamically loaded driver and a checked encoder function table.
///
/// This owner is deliberately thread affine. Create it on the codec worker and
/// retain it there through explicit session destruction; no global lazy loader
/// or CUDA device-zero initialization is performed.
pub struct RuntimeNvenc {
    _library: libloading::Library,
    functions: NV_ENCODE_API_FUNCTION_LIST,
    supported_version: u32,
    thread_affinity: PhantomData<Rc<()>>,
}

impl fmt::Debug for RuntimeNvenc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeNvenc")
            .field("supported_version", &self.supported_version)
            .finish_non_exhaustive()
    }
}

impl RuntimeNvenc {
    /// Loads the platform driver, verifies its API, and negotiates the table.
    ///
    /// Windows loads `nvEncodeAPI64.dll` only from System32, preventing a DLL
    /// next to the application or in its current directory from being selected.
    /// Linux uses the installed `libnvidia-encode.so.1` driver soname.
    ///
    /// # Errors
    /// Returns discovery, API-version, driver-status, or missing-callback errors.
    pub fn load() -> Result<Self, LoadError> {
        type Maximum = unsafe extern "C" fn(*mut u32) -> NVENCSTATUS;
        type Create = unsafe extern "C" fn(*mut NV_ENCODE_API_FUNCTION_LIST) -> NVENCSTATUS;
        let library = load_driver()?;
        // SAFETY: these names and signatures are exported by the pinned NVENC
        // C ABI. The library stays owned here; neither symbol escapes this scope.
        let maximum = unsafe { library.get::<Maximum>(b"NvEncodeAPIGetMaxSupportedVersion\0")? };
        // SAFETY: same ABI and library-lifetime conditions as above.
        let create = unsafe { library.get::<Create>(b"NvEncodeAPICreateInstance\0")? };
        let mut supported_version = 0;
        // SAFETY: the output points to an initialized, writable u32 for this call.
        check("get maximum API version", unsafe {
            maximum(&raw mut supported_version)
        })?;
        let required = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
        if supported_version < required {
            return Err(LoadError::OlderDriver {
                supported: supported_version,
                required,
            });
        }
        let mut functions = NV_ENCODE_API_FUNCTION_LIST {
            version: NV_ENCODE_API_FUNCTION_LIST_VER,
            ..Default::default()
        };
        // SAFETY: the struct is zero-initialized except its pinned ABI version;
        // the driver writes a table of exactly that version into its live storage.
        check("create API instance", unsafe { create(&raw mut functions) })?;
        validate_encoder_callbacks(&functions)?;
        Ok(Self {
            _library: library,
            functions,
            supported_version,
            thread_affinity: PhantomData,
        })
    }

    /// Borrows the table while the driver remains loaded.
    #[must_use]
    pub fn functions(&self) -> &NV_ENCODE_API_FUNCTION_LIST {
        // The explicit owner field must remain alive even though callbacks are
        // already copied into the table. It is never loaded and then discarded.
        &self.functions
    }

    /// Returns the driver's maximum API version (major << 4 | minor).
    #[must_use]
    pub const fn supported_version(&self) -> u32 {
        self.supported_version
    }
}

fn check(operation: &'static str, status: NVENCSTATUS) -> Result<(), LoadError> {
    if status == NVENCSTATUS::NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(LoadError::Driver { operation, status })
    }
}

fn validate_encoder_callbacks(table: &NV_ENCODE_API_FUNCTION_LIST) -> Result<(), LoadError> {
    let callbacks = [
        (
            "nvEncOpenEncodeSessionEx",
            table.nvEncOpenEncodeSessionEx.is_some(),
        ),
        (
            "nvEncGetEncodePresetConfigEx",
            table.nvEncGetEncodePresetConfigEx.is_some(),
        ),
        (
            "nvEncInitializeEncoder",
            table.nvEncInitializeEncoder.is_some(),
        ),
        (
            "nvEncRegisterResource",
            table.nvEncRegisterResource.is_some(),
        ),
        (
            "nvEncUnregisterResource",
            table.nvEncUnregisterResource.is_some(),
        ),
        (
            "nvEncMapInputResource",
            table.nvEncMapInputResource.is_some(),
        ),
        (
            "nvEncUnmapInputResource",
            table.nvEncUnmapInputResource.is_some(),
        ),
        ("nvEncEncodePicture", table.nvEncEncodePicture.is_some()),
        ("nvEncLockBitstream", table.nvEncLockBitstream.is_some()),
        ("nvEncUnlockBitstream", table.nvEncUnlockBitstream.is_some()),
        ("nvEncDestroyEncoder", table.nvEncDestroyEncoder.is_some()),
    ];
    for (name, present) in callbacks {
        if !present {
            return Err(LoadError::MissingCallback(name));
        }
    }
    Ok(())
}

#[cfg(windows)]
fn load_driver() -> Result<libloading::Library, libloading::Error> {
    // SAFETY: this loads only NVIDIA's installed driver from the system library
    // directory. Its exported C ABI is negotiated before any encoder operation.
    unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvEncodeAPI64.dll",
            libloading::os::windows::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Into::into)
    }
}

#[cfg(unix)]
fn load_driver() -> Result<libloading::Library, libloading::Error> {
    // SAFETY: this opens the installed NVIDIA driver soname; exported signatures
    // are checked above and the library owner outlives all borrowed callbacks.
    unsafe { libloading::Library::new("libnvidia-encode.so.1") }
}
