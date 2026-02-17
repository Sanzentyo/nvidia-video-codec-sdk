//! Auto-generated bindings to NVIDIA Video Codec SDK.
//!
//! The bindings were generated using [bindgen](https://github.com/rust-lang/rust-bindgen)
//! using the scripts `sys/linux_sys/bindgen.sh` and
//! `sys/windows_sys/bindgen.ps1` for the respective operating system.

mod guid;
mod version;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "linux")]
#[path = "linux_sys/mod.rs"]
mod linux_bindings;
#[cfg(target_os = "linux")]
pub use self::linux_bindings::*;

#[allow(warnings)]
#[rustfmt::skip]
#[cfg(target_os = "windows")]
#[path = "windows_sys/mod.rs"]
mod windows_bindings;
#[cfg(target_os = "windows")]
pub use self::windows_bindings::*;
