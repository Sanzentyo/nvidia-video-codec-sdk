//! Safe wrapper around the raw bindings.
//!
//! Largely unfinished, so you might still have to dip into
//! [`sys`](crate::sys) for the missing functionality.

mod api;
mod buffer;
mod builders;
mod decoder;
mod encoder;
mod result;
mod session;

pub use api::{EncodeAPI, ENCODE_API};
pub use buffer::{
    Bitstream, BitstreamLock, Buffer, BufferLock, EncoderInput, EncoderOutput, RegisteredResource,
};
pub use decoder::{DecodeCodec, DecodeOptions, DecodeRect, DecodedRgbFrame, Decoder, SeiMessage};
pub use encoder::{Encoder, EncoderInitParams};
pub use result::{DecodeError, EncodeError, ErrorKind};
#[cfg(target_os = "windows")]
pub use session::WindowsAsyncEvent;
pub use session::{CodecPictureParams, EncodePictureParams, ReconfigureParams, Session};
