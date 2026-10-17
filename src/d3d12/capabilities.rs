use super::{EncodeError, EncoderCapabilities, EncoderConfig, Inner};
use crate::sys::nvEncodeAPI::{
    GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CAPS, NV_ENC_CAPS_PARAM, NV_ENC_CAPS_PARAM_VER,
};
impl Inner {
    pub(super) fn query_capabilities(&self) -> Result<EncoderCapabilities, EncodeError> {
        let mut guids = [GUID::default(); 64];
        let mut count = 0;
        // SAFETY: the live session writes one exclusive initialized count.
        self.check("count codecs", unsafe {
            self.api()
                .nvEncGetEncodeGUIDCount
                .ok_or(EncodeError::Contract("missing codec count"))?(
                self.session, &raw mut count
            )
        })?;
        let capacity = bounded_count(count)?;
        // SAFETY: this live session owns the checked function table. Fixed arrays
        // provide exactly the declared writable capacity for synchronous queries.
        self.check("enumerate codecs", unsafe {
            self.api()
                .nvEncGetEncodeGUIDs
                .ok_or(EncodeError::Contract("missing codec enumeration"))?(
                self.session,
                guids.as_mut_ptr(),
                capacity,
                &raw mut count,
            )
        })?;
        if count == 0 || count > capacity || !guids[..count as usize].contains(&self.config.guid())
        {
            return Err(EncodeError::Contract("selected codec was not enumerated"));
        }
        count = 0;
        // SAFETY: the live session and selected codec write one exclusive count.
        self.check("count profiles", unsafe {
            self.api()
                .nvEncGetEncodeProfileGUIDCount
                .ok_or(EncodeError::Contract("missing profile count"))?(
                self.session,
                self.config.guid(),
                &raw mut count,
            )
        })?;
        let capacity = bounded_count(count)?;
        // SAFETY: same live session and initialized bounded GUID output storage.
        self.check("enumerate profiles", unsafe {
            self.api()
                .nvEncGetEncodeProfileGUIDs
                .ok_or(EncodeError::Contract("missing profile enumeration"))?(
                self.session,
                self.config.guid(),
                guids.as_mut_ptr(),
                capacity,
                &raw mut count,
            )
        })?;
        if count == 0
            || count > capacity
            || !guids[..count as usize].contains(&self.config.profile())
        {
            return Err(EncodeError::Contract(
                "selected codec profile was not enumerated",
            ));
        }
        let mut formats = [NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_UNDEFINED; 64];
        count = 0;
        // SAFETY: the live session and selected codec write one exclusive count.
        self.check("count input formats", unsafe {
            self.api()
                .nvEncGetInputFormatCount
                .ok_or(EncodeError::Contract("missing input-format count"))?(
                self.session,
                self.config.guid(),
                &raw mut count,
            )
        })?;
        let capacity = bounded_count(count)?;
        // SAFETY: initialized format array and exact writable capacity; no output escapes.
        self.check("enumerate input formats", unsafe {
            self.api()
                .nvEncGetInputFormats
                .ok_or(EncodeError::Contract("missing input-format enumeration"))?(
                self.session,
                self.config.guid(),
                formats.as_mut_ptr(),
                capacity,
                &raw mut count,
            )
        })?;
        if count == 0
            || count > capacity
            || !formats[..count as usize].contains(&NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12)
        {
            return Err(EncodeError::Contract("NV12 input was not enumerated"));
        }
        let lossless_supported = self.query_lossless()?;
        Ok(EncoderCapabilities {
            codec_supported: true,
            profile_supported: true,
            nv12_supported: true,
            lossless_supported,
        })
    }
    fn query_lossless(&self) -> Result<Option<bool>, EncodeError> {
        if matches!(self.config, EncoderConfig::HevcLossless(_)) {
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
            Ok(Some(value == 1))
        } else {
            Ok(None)
        }
    }
}
fn bounded_count(count: u32) -> Result<u32, EncodeError> {
    if (1..=64).contains(&count) {
        Ok(count)
    } else {
        Err(EncodeError::Contract(
            "driver enumeration count exceeds fixed storage",
        ))
    }
}
