use super::EncodeError;
use crate::sys::nvEncodeAPI::{
    GUID, NV_ENC_BIT_DEPTH, NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG,
    NV_ENC_CONFIG_H264_VUI_PARAMETERS, NV_ENC_H264_PROFILE_HIGH_GUID,
    NV_ENC_HEVC_PROFILE_MAIN_GUID, NV_ENC_MULTI_PASS, NV_ENC_PARAMS_RC_MODE, NV_ENC_QP,
    NV_ENC_QP_MAP_MODE, NV_ENC_TUNING_INFO, NV_ENC_VUI_COLOR_PRIMARIES, NV_ENC_VUI_MATRIX_COEFFS,
    NV_ENC_VUI_TRANSFER_CHARACTERISTIC, NV_ENC_VUI_VIDEO_FORMAT,
};

/// Video coding syntax, independent of rate control and resource ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// H.264/AVC Annex-B.
    H264,
    /// H.265/HEVC Annex-B.
    Hevc,
}
#[derive(Debug, Clone, Copy)]
struct ImageConfig {
    width: u32,
    height: u32,
    fps: u32,
    capacity: u32,
}
impl ImageConfig {
    fn new(width: u32, height: u32, fps: u32) -> Result<Self, EncodeError> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || !(1..=120).contains(&fps)
        {
            return Err(EncodeError::Contract(
                "NV12 dimensions or frame rate out of bounds",
            ));
        }
        let capacity = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|v| v.checked_mul(3))
            .ok_or(EncodeError::Contract("capacity overflow"))?;
        if capacity > 32 * 1024 * 1024 {
            return Err(EncodeError::Contract("compressed output allocation bound"));
        }
        Ok(Self {
            width,
            height,
            fps,
            capacity: u32::try_from(capacity)
                .map_err(|_| EncodeError::Contract("capacity overflow"))?,
        })
    }
}
/// Immutable H.264 High/P3/ULL/CBR, IP-only, BT.709 limited-range policy.
#[derive(Debug, Clone, Copy)]
pub struct H264Config {
    image: ImageConfig,
    bitrate: u32,
}
impl H264Config {
    /// Validates even NV12 dimensions, fps1..=120, bitrate1..=60000000 and
    /// compressed output allocation no greater than32MiB.
    /// # Errors
    /// Rejects an invalid dimension, frame rate, bitrate or allocation size.
    pub fn new(width: u32, height: u32, fps: u32, bitrate: u32) -> Result<Self, EncodeError> {
        if !(1..=60_000_000).contains(&bitrate) {
            return Err(EncodeError::Contract("bitrate out of bounds"));
        }
        Ok(Self {
            image: ImageConfig::new(width, height, fps)?,
            bitrate,
        })
    }
    /// Coded NV12 width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.image.width
    }
    /// Coded NV12 height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.image.height
    }
}
/// Immutable HEVC Main8/NV12/P3/lossless, IP-only, BT.709 limited-range policy.
/// No bitrate or VBV constraint is imposed. Driver capability is required.
#[derive(Debug, Clone, Copy)]
pub struct HevcLosslessConfig {
    image: ImageConfig,
}
impl HevcLosslessConfig {
    /// Validates even NV12 dimensions, fps1..=120 and output allocation<=32MiB.
    /// # Errors
    /// Rejects an invalid dimension, frame rate or allocation size.
    pub fn new(width: u32, height: u32, fps: u32) -> Result<Self, EncodeError> {
        Ok(Self {
            image: ImageConfig::new(width, height, fps)?,
        })
    }
}
/// Explicit immutable policy consumed by the external D3D12 encoder.
#[derive(Debug, Clone, Copy)]
pub enum EncoderConfig {
    /// H.264 High, ultra-low-latency constant bitrate.
    H264(H264Config),
    /// HEVC Main8, genuine lossless tuning and QP0 without a bitrate limit.
    HevcLossless(HevcLosslessConfig),
}
impl EncoderConfig {
    fn image(self) -> ImageConfig {
        match self {
            Self::H264(c) => c.image,
            Self::HevcLossless(c) => c.image,
        }
    }
    /// Selected video coding syntax.
    #[must_use]
    pub const fn codec(self) -> Codec {
        match self {
            Self::H264(_) => Codec::H264,
            Self::HevcLossless(_) => Codec::Hevc,
        }
    }
    pub(super) fn width(self) -> u32 {
        self.image().width
    }
    pub(super) fn height(self) -> u32 {
        self.image().height
    }
    pub(super) fn fps(self) -> u32 {
        self.image().fps
    }
    pub(super) fn capacity(self) -> u32 {
        self.image().capacity
    }
    pub(super) const fn guid(self) -> GUID {
        match self {
            Self::H264(_) => NV_ENC_CODEC_H264_GUID,
            Self::HevcLossless(_) => NV_ENC_CODEC_HEVC_GUID,
        }
    }
    pub(super) const fn profile(self) -> GUID {
        match self {
            Self::H264(_) => NV_ENC_H264_PROFILE_HIGH_GUID,
            Self::HevcLossless(_) => NV_ENC_HEVC_PROFILE_MAIN_GUID,
        }
    }
    pub(super) const fn tuning(self) -> NV_ENC_TUNING_INFO {
        match self {
            Self::H264(_) => NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            Self::HevcLossless(_) => NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOSSLESS,
        }
    }
    pub(super) fn apply(self, mut config: NV_ENC_CONFIG) -> NV_ENC_CONFIG {
        config.profileGUID = self.profile();
        config.gopLength = 60;
        config.frameIntervalP = 1;
        config.rcParams.set_enableLookahead(0);
        match self {
            Self::H264(c) => {
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
                config.rcParams.averageBitRate = c.bitrate;
                config.rcParams.maxBitRate = c.bitrate;
                config.rcParams.vbvBufferSize = c.bitrate / c.image.fps;
                config.rcParams.vbvInitialDelay = config.rcParams.vbvBufferSize;
                // SAFETY: this preset and policy both select the H.264 union member.
                let h264 = unsafe { &mut config.encodeCodecConfig.h264Config };
                h264.idrPeriod = 60;
                h264.set_repeatSPSPPS(1);
                h264.chromaFormatIDC = 1;
                set_vui(&mut h264.h264VUIParameters);
            }
            Self::HevcLossless(_) => {
                config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
                config.rcParams.constQP = NV_ENC_QP {
                    qpInterP: 0,
                    qpInterB: 0,
                    qpIntra: 0,
                };
                config.rcParams.averageBitRate = 0;
                config.rcParams.maxBitRate = 0;
                config.rcParams.vbvBufferSize = 0;
                config.rcParams.vbvInitialDelay = 0;
                config.rcParams.set_enableAQ(0);
                config.rcParams.set_enableTemporalAQ(0);
                config.rcParams.set_enableMinQP(0);
                config.rcParams.set_enableMaxQP(0);
                config.rcParams.set_enableInitialRCQP(0);
                config.rcParams.lookaheadDepth = 0;
                config.rcParams.multiPass = NV_ENC_MULTI_PASS::NV_ENC_MULTI_PASS_DISABLED;
                config.rcParams.qpMapMode = NV_ENC_QP_MAP_MODE::NV_ENC_QP_MAP_DISABLED;
                // SAFETY: this preset and policy both select the HEVC union member.
                let hevc = unsafe { &mut config.encodeCodecConfig.hevcConfig };
                hevc.idrPeriod = 60;
                hevc.set_repeatSPSPPS(1);
                hevc.set_chromaFormatIDC(1);
                hevc.inputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                hevc.outputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                set_vui(&mut hevc.hevcVUIParameters);
            }
        }
        config
    }
}
fn set_vui(vui: &mut NV_ENC_CONFIG_H264_VUI_PARAMETERS) {
    vui.videoSignalTypePresentFlag = 1;
    vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
    vui.videoFullRangeFlag = 0;
    vui.colourDescriptionPresentFlag = 1;
    vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
    vui.transferCharacteristics =
        NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
    vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
}

// The caller is between session-open and configure, with no registered work.
pub(super) fn admit_or_close(
    queried: Result<EncoderCapabilities, EncodeError>,
    config: EncoderConfig,
    close: impl FnOnce() -> Result<(), EncodeError>,
) -> Result<EncoderCapabilities, EncodeError> {
    match queried.and_then(|facts| {
        facts.admit(config)?;
        Ok(facts)
    }) {
        Ok(facts) => Ok(facts),
        Err(cause) => match close() {
            Ok(()) => Err(cause),
            Err(cleanup) => Err(EncodeError::InitializationCleanup {
                cause: Box::new(cause),
                cleanup: Box::new(cleanup),
            }),
        },
    }
}

/// Actual bounded capability enumeration for this encoder's selected codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderCapabilities {
    /// Selected codec appeared in the driver's GUID list.
    pub codec_supported: bool,
    /// Required High or Main profile appeared in the codec's profile list.
    pub profile_supported: bool,
    /// NV12 appeared in the codec's input-format list.
    pub nv12_supported: bool,
    /// Raw codec-wide lossless capability, queried for the lossless policy only.
    pub lossless_supported: Option<bool>,
}
impl EncoderCapabilities {
    pub(super) fn admit(self, config: EncoderConfig) -> Result<(), EncodeError> {
        if !self.codec_supported
            || !self.profile_supported
            || !self.nv12_supported
            || (matches!(config, EncoderConfig::HevcLossless(_))
                && self.lossless_supported != Some(true))
        {
            return Err(EncodeError::Contract(
                "selected codec/profile/NV12/lossless capability unavailable",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preset_vui_fields_outside_the_color_policy_are_preserved() {
        let mut vui = NV_ENC_CONFIG_H264_VUI_PARAMETERS {
            chromaSampleLocationFlag: 1,
            chromaSampleLocationTop: 2,
            chromaSampleLocationBot: 3,
            timingInfoPresentFlag: 1,
            numUnitInTicks: 1001,
            timeScale: 60000,
            ..Default::default()
        };
        set_vui(&mut vui);
        assert_eq!(
            [
                vui.chromaSampleLocationFlag,
                vui.chromaSampleLocationTop,
                vui.chromaSampleLocationBot
            ],
            [1, 2, 3]
        );
        assert_eq!(
            [vui.timingInfoPresentFlag, vui.numUnitInTicks, vui.timeScale],
            [1, 1001, 60000]
        );
        assert_eq!(vui.videoFullRangeFlag, 0);
        assert_eq!(
            vui.colourPrimaries,
            NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709
        );
    }
    #[test]
    fn definite_admission_failures_close_once_and_preserve_both_errors() {
        let policy =
            EncoderConfig::HevcLossless(HevcLosslessConfig::new(1152, 576, 15).expect("config"));
        let good = EncoderCapabilities {
            codec_supported: true,
            profile_supported: true,
            nv12_supported: true,
            lossless_supported: Some(true),
        };
        assert!(admit_or_close(Ok(good), policy, || panic!(
            "admitted session must remain open"
        ))
        .is_ok());
        let closes = std::cell::Cell::new(0);
        let unsupported = EncoderCapabilities {
            lossless_supported: Some(false),
            ..good
        };
        assert!(admit_or_close(Ok(unsupported), policy, || {
            closes.set(closes.get() + 1);
            Ok(())
        })
        .is_err());
        assert_eq!(closes.get(), 1);
        let failure = admit_or_close(Err(EncodeError::Contract("query failed")), policy, || {
            Err(EncodeError::Contract("close failed"))
        });
        let Err(EncodeError::InitializationCleanup { cause, cleanup }) = failure else {
            panic!("both errors required")
        };
        assert!(cause.to_string().contains("query failed"));
        assert!(cleanup.to_string().contains("close failed"));
    }
    #[test]
    fn explicit_policies_select_distinct_guid_tuning_profile_and_active_config() {
        let h = EncoderConfig::H264(H264Config::new(1088, 1088, 15, 32_000_000).expect("H264"));
        let v = EncoderConfig::HevcLossless(HevcLosslessConfig::new(1152, 576, 15).expect("HEVC"));
        assert_eq!(h.guid(), NV_ENC_CODEC_H264_GUID);
        assert_eq!(h.profile(), NV_ENC_H264_PROFILE_HIGH_GUID);
        assert_eq!(
            h.tuning(),
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
        );
        assert_eq!(v.guid(), NV_ENC_CODEC_HEVC_GUID);
        assert_eq!(v.profile(), NV_ENC_HEVC_PROFILE_MAIN_GUID);
        assert_eq!(v.tuning(), NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_LOSSLESS);
        let hc = h.apply(NV_ENC_CONFIG::default());
        let vc = v.apply(NV_ENC_CONFIG::default());
        assert_eq!(hc.rcParams.averageBitRate, 32_000_000);
        assert_eq!(
            hc.rcParams.rateControlMode,
            NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR
        );
        assert_eq!(
            vc.rcParams.rateControlMode,
            NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP
        );
        assert_eq!(vc.rcParams.constQP.qpIntra, 0);
        assert_eq!(vc.rcParams.constQP.qpInterP, 0);
        assert_eq!(vc.rcParams.constQP.qpInterB, 0);
        assert_eq!(
            [
                vc.rcParams.averageBitRate,
                vc.rcParams.maxBitRate,
                vc.rcParams.vbvBufferSize,
                vc.rcParams.vbvInitialDelay
            ],
            [0; 4]
        );
        assert_eq!(vc.frameIntervalP, 1);
        assert_eq!(vc.gopLength, 60);
        // SAFETY: vc was constructed from the HEVC policy, selecting this union.
        let hevc = unsafe { vc.encodeCodecConfig.hevcConfig };
        assert_eq!(hevc.chromaFormatIDC(), 1);
        assert_eq!(hevc.inputBitDepth, NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8);
        assert_eq!(hevc.outputBitDepth, NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8);
        assert_eq!(hevc.hevcVUIParameters.videoFullRangeFlag, 0);
    }
    #[test]
    fn lossless_configuration_rejects_resource_and_rate_boundaries() {
        for (w, h, f) in [
            (0, 2, 15),
            (2, 3, 15),
            (2, 2, 0),
            (2, 2, 121),
            (u32::MAX - 1, u32::MAX - 1, 15),
            (4096, 4096, 15),
        ] {
            assert!(HevcLosslessConfig::new(w, h, f).is_err());
        }
        assert!(H264Config::new(2, 2, 15, 0).is_err());
        assert!(HevcLosslessConfig::new(1152, 1664, 15).is_ok());
    }
    #[test]
    fn every_required_capability_is_enforced_without_another_policy() {
        let c =
            EncoderConfig::HevcLossless(HevcLosslessConfig::new(1152, 576, 15).expect("config"));
        let good = EncoderCapabilities {
            codec_supported: true,
            profile_supported: true,
            nv12_supported: true,
            lossless_supported: Some(true),
        };
        assert!(good.admit(c).is_ok());
        for bad in [
            EncoderCapabilities {
                codec_supported: false,
                ..good
            },
            EncoderCapabilities {
                profile_supported: false,
                ..good
            },
            EncoderCapabilities {
                nv12_supported: false,
                ..good
            },
            EncoderCapabilities {
                lossless_supported: Some(false),
                ..good
            },
            EncoderCapabilities {
                lossless_supported: None,
                ..good
            },
        ] {
            assert!(bad.admit(c).is_err());
        }
    }
}
