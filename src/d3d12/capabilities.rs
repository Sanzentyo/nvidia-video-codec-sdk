use super::{EncodeError, EncoderCapabilities, EncoderConfig, Inner};
use crate::sys::nvEncodeAPI::{
    GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CAPS, NV_ENC_CAPS_PARAM, NV_ENC_CAPS_PARAM_VER,
};
impl Inner {
    pub(super) fn query_capabilities(&self) -> Result<EncoderCapabilities, EncodeError> {
        let mut guids = [GUID::default(); 64];
        let mut count = 0;
        // SAFETY: this live session owns the checked function table. Fixed arrays
        // provide exactly the declared writable capacity for synchronous queries.
        self.check("enumerate codecs", unsafe {
            self.api()
                .nvEncGetEncodeGUIDs
                .ok_or(EncodeError::Contract("missing codec enumeration"))?(
                self.session,
                guids.as_mut_ptr(),
                64,
                &raw mut count,
            )
        })?;
        if count == 0 || count > 64 || !guids[..count as usize].contains(&self.config.guid()) {
            return Err(EncodeError::Contract("selected codec was not enumerated"));
        }
        count = 0;
        // SAFETY: same live session and initialized bounded GUID output storage.
        self.check("enumerate profiles", unsafe {
            self.api()
                .nvEncGetEncodeProfileGUIDs
                .ok_or(EncodeError::Contract("missing profile enumeration"))?(
                self.session,
                self.config.guid(),
                guids.as_mut_ptr(),
                64,
                &raw mut count,
            )
        })?;
        if count == 0 || count > 64 || !guids[..count as usize].contains(&self.config.profile()) {
            return Err(EncodeError::Contract(
                "selected codec profile was not enumerated",
            ));
        }
        let mut formats = [NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; 64];
        count = 0;
        // SAFETY: initialized format array and exact writable capacity; no output escapes.
        self.check("enumerate input formats", unsafe {
            self.api()
                .nvEncGetInputFormats
                .ok_or(EncodeError::Contract("missing input-format enumeration"))?(
                self.session,
                self.config.guid(),
                formats.as_mut_ptr(),
                64,
                &raw mut count,
            )
        })?;
        if count == 0
            || count > 64
            || !formats[..count as usize].contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12)
        {
            return Err(EncodeError::Contract("NV12 input was not enumerated"));
        }
        let lossless_supported = if matches!(self.config, EncoderConfig::HevcLossless(_)) {
            let mut params = NV_ENC_CAPS_PARAM {
                version: NV_ENC_CAPS_PARAM_VER,
                capsToQuery: NV_ENC_CAPS::NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE,
                ..Default::default()
            };
            let mut value = -1;
            // SAFETY: versioned initialized input and one exclusive writable scalar.
            self.check("query lossless capability", unsafe {
                self.api()
                    .nvEncGetEncodeCaps
                    .ok_or(EncodeError::Contract("missing capability query"))?(
                    self.session,
                    self.config.guid(),
                    &raw mut params,
                    &raw mut value,
                )
            })?;
            if !(0..=1).contains(&value) {
                return Err(EncodeError::Contract("invalid lossless capability value"));
            }
            Some(value == 1)
        } else {
            None
        };
        Ok(EncoderCapabilities {
            codec_supported: true,
            profile_supported: true,
            nv12_supported: true,
            lossless_supported,
        })
    }
}
