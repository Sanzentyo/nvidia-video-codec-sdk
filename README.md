# NVIDIA Video Codec SDK

[![crates.io](https://img.shields.io/crates/v/nvidia-video-codec-sdk?style=for-the-badge)](https://crates.io/crates/nvidia-video-codec-sdk)
[![docs.rs](https://img.shields.io/docsrs/nvidia-video-codec-sdk?label=docs.rs%20latest&style=for-the-badge)](https://docs.rs/nvidia-video-codec-sdk)

Rust bindings for [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk).

The documentation is also hosted on GitHub Pages
[here](https://viliamvadocz.github.io/nvidia-video-codec-sdk/nvidia_video_codec_sdk/).

Versions:
- NVIDIA Video Codec SDK 12.1.14
- CUDA 12.2 (older CUDA versions should also work)

## Installation

The build script will try to automatically locate your NVIDIA Video Codec SDK installation.
You can help it by setting the environment variable `NVIDIA_VIDEO_CODEC_SDK_PATH` to the directory containing the library files. 
- `nvEncodeAPI.lib` and `nvcuvid.lib` on Windows,
- `libnvidia-encode.so` and `libnvcuvid.so` on Linux.
## External D3D12 encoding policies

The `d3d12` feature exposes explicit `EncoderConfig::H264` and
`EncoderConfig::HevcLossless` policies. HEVC uses Main8/NV12, lossless tuning and
QP0 without bitrate/VBV constraints. Construction checks the selected codec,
profile, input format and required lossless capability on the actual device.
Both policies use the same thread-affine registration, producer-fence and
physical-retirement owner. Packets report their codec and own Annex-B bytes.

`HevcLosslessConfig::new(width, height, fps, ColorRange::Full)` explicitly
selects full-range BT.709 VUI; use `ColorRange::Limited` for limited-range VUI.
H.264 retains its explicit limited-range policy. The range is immutable and
available through `HevcLosslessConfig::color_range()`. This signals the input's
range; it does not rescale pixels or establish end-to-end decoder sample fidelity.
