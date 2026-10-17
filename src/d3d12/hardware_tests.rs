use super::*;
use windows::Win32::Graphics::{Direct3D::D3D_FEATURE_LEVEL_11_0, Direct3D12::*};

#[test]
#[ignore = "requires an NVIDIA D3D12 device and NVENC driver"]
fn hardware_lossless_session_initializes_and_closes() -> Result<(), EncodeError> {
    let mut device = None;
    // SAFETY: initialized owned output; the default adapter must support NVENC.
    unsafe {
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &raw mut device)?;
    }
    let device: ID3D12Device = device.ok_or(EncodeError::Contract("no D3D12 device"))?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1152,
        Height: 576,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_NV12,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut input = None;
    // SAFETY: initialized descriptor and exclusive owned output.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &raw mut input,
        )?;
    }
    let input: ID3D12Resource = input.ok_or(EncodeError::Contract("no NV12 input"))?;
    let producer = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
    eprintln!("NVENC: constructing lossless HEVC session");
    // SAFETY: no producer writes occur; all resources have the same device and COMMON state.
    let mut encoder = unsafe {
        D3d12Encoder::new(
            device,
            input,
            producer,
            EncoderConfig::HevcLossless(HevcLosslessConfig::new(1152, 576, 15)?),
        )?
    };
    assert_eq!(encoder.capabilities()?.lossless_supported, Some(true));
    encoder.try_close()
}
