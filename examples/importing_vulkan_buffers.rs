use std::{
    fs::{File, OpenOptions},
    io::Write,
    ptr::NonNull,
    sync::Arc,
};

use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_CODEC_H264_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID, NV_ENC_PRESET_P1_GUID, NV_ENC_TUNING_INFO,
    },
    Encoder, EncoderInitParams,
};
use vulkano::VulkanObject;
use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
        QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::{
        DeviceMemory, ExternalMemoryHandleTypes, MappedMemoryRange, MemoryAllocateInfo,
        MemoryMapFlags, MemoryMapInfo, MemoryPropertyFlags,
    },
    VulkanLibrary,
};

#[cfg(windows)]
use ash::vk;
#[cfg(windows)]
use std::os::windows::io::FromRawHandle;

/// Returns the color `(r, g, b, alpha)` of a pixel on the screen relative to
/// its position on a screen:
///
/// Top right will be red,
/// bottom left will be green,
/// all colors will shift towards having more blue as `time` increases.
///
/// # Arguments
///
/// * `width`, `height` - Width and height of the screen.
/// * `x`, `y` - Coordinates of the pixel on the screen.
/// * time - Fraction indicating what part of the animation we are in [0,1]
fn get_color(width: u32, height: u32, x: u32, y: u32, time: f32) -> (u8, u8, u8, u8) {
    let alpha = 255;
    let red = (255 * x / width) as u8;
    let green = (255 * y / height) as u8;
    let blue = (255. * time) as u8;
    (blue, green, red, alpha)
}

/// Generates test frame inputs and sets `buf` to that input.
///
/// # Arguments
///
/// * `buf` - The buffer in which to put the generated input.
/// * `width`, `height` - The size of the frames to generate input for.
/// * `i`, `i_max` - The current frame and total amount of frames.
fn generate_test_input(buf: &mut [u8], width: u32, height: u32, i: u32, i_max: u32) {
    assert_eq!(buf.len(), (width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = width * y + x;
            let index = (pixel * 4) as usize;
            let color = get_color(width, height, x, y, i as f32 / i_max as f32);
            buf[index] = color.0;
            buf[index + 1] = color.1;
            buf[index + 2] = color.2;
            buf[index + 3] = color.3;
        }
    }
}

/// Initialize Vulkan and find the desired memory type index.
///
/// The `memory_type_index` corresponds to the memory type which is
/// `HOST_VISIBLE`  which is needed so that we can map device memory later in
/// the example.
fn initialize_vulkan() -> (Arc<Device>, u32) {
    // Initialize Vulkan library.
    let vulkan_library = VulkanLibrary::new().expect("Vulkan should be installed correctly");
    let instance = Instance::new(
        vulkan_library,
        InstanceCreateInfo::application_from_cargo_toml(),
    )
    .expect("Vulkan should be installed correctly");

    let (memory_type_index, physical_device) = instance
        .enumerate_physical_devices()
        .expect("There should be some device capable of encoding")
        .filter_map(|pd| {
            matches!(pd.properties().device_type, PhysicalDeviceType::DiscreteGpu)
                .then_some(())
                .and_then(|()| {
                    pd.memory_properties()
                        .memory_types
                        .iter()
                        .position(|mt| {
                            mt.property_flags
                                .contains(MemoryPropertyFlags::HOST_VISIBLE)
                        })
                        .map(|index| (index as u32, pd))
                })
        })
        .next()
        .expect(
            "There should be at least one GPU which supports a memory type that is `HOST_VISIBLE`",
        );

    // Create a Vulkan device.
    // Pick an external-memory extension based on what the physical device actually supports.
    let physical_device_name = physical_device.properties().device_name.clone();

    let mut enabled_exts = DeviceExtensions::default();
    #[cfg(unix)]
    let (supports_fd, chosen_ext_name) = {
        let supports_fd = physical_device
            .supported_extensions()
            .khr_external_memory_fd;
        assert!(
            supports_fd,
            "The physical device should support khr_external_memory_fd on unix"
        );
        enabled_exts.khr_external_memory = true;
        enabled_exts.khr_external_memory_fd = true;
        let chosen_ext_name = "khr_external_memory_fd";
        (supports_fd, chosen_ext_name)
    };
    #[cfg(windows)]
    let (supports_win32, chosen_ext_name) = {
        let supports_win32 = physical_device
            .supported_extensions()
            .khr_external_memory_win32;
        assert!(
            supports_win32,
            "The physical device should support khr_external_memory_win32 on windows"
        );
        enabled_exts.khr_external_memory = true;
        enabled_exts.khr_external_memory_win32 = true;
        let chosen_ext_name = "khr_external_memory_win32";
        (supports_win32, chosen_ext_name)
    };

    // Try to create the device and provide richer diagnostics on failure.
    let supported_features = physical_device.supported_features();
    let enabled_features = DeviceFeatures {
        memory_map_placed: supported_features.memory_map_placed,
        ..Default::default()
    };

    let device_create_info = DeviceCreateInfo {
        queue_create_infos: vec![QueueCreateInfo::default()],
        enabled_extensions: enabled_exts,
        enabled_features,
        ..Default::default()
    };

    let (vulkan_device, _queues) = match Device::new(physical_device, device_create_info) {
        Ok((dev, queues)) => (dev, queues),
        Err(err) => {
            eprintln!("Failed to create device: {:?}", err);
            #[cfg(unix)]
            eprintln!(
                "Physical device '{}' supported extensions: khr_external_memory_fd={}",
                physical_device_name, supports_fd
            );
            #[cfg(windows)]
            eprintln!(
                "Physical device '{}' supported extensions: khr_external_memory_win32={}",
                physical_device_name, supports_win32
            );
            eprintln!(
                "DeviceCreateInfo requested extensions: khr_external_memory={} khr_external_memory_fd={} khr_external_memory_win32={}",
                enabled_exts.khr_external_memory,
                enabled_exts.khr_external_memory_fd,
                enabled_exts.khr_external_memory_win32
            );
            panic!(
                "Vulkan should be installed correctly and device should support selected extension: {}",
                chosen_ext_name
            )
        }
    };

    (vulkan_device, memory_type_index)
}

/// Creates an encoded bitstream for a 128 frame, 1920x1080 video.
/// This bitstream will be written to ./test.bin
/// To view this bitstream use a decoder like ffmpeg.
///
/// For ffmpeg use `ffmpeg -i test.bin -vcodec copy test.mp4` to
/// decode the video.
fn main() {
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;
    const FRAMES: u32 = 128;

    let (vulkan_device, memory_type_index) = initialize_vulkan();

    // Create a new CudaContext to interact with cuda.
    let cuda_ctx = CudaContext::new(0).expect("Cuda should be installed correctly.");

    let encoder = Encoder::initialize_with_cuda(cuda_ctx.clone())
        .expect("NVIDIA Video Codec SDK should be installed correctly.");

    // Get all encode guids supported by the GPU.
    let encode_guids = encoder
        .get_encode_guids()
        .expect("The encoder should be able to get the supported guids.");
    let encode_guid = NV_ENC_CODEC_H264_GUID;
    assert!(encode_guids.contains(&encode_guid));

    // Get available preset guids based on encode guid.
    let preset_guids = encoder
        .get_preset_guids(encode_guid)
        .expect("The encoder should have a preset for H.264.");
    let preset_guid = NV_ENC_PRESET_P1_GUID;
    assert!(preset_guids.contains(&preset_guid));

    // Get available profiles based on encode guid.
    let profile_guids = encoder
        .get_profile_guids(encode_guid)
        .expect("The encoder should have a profile for H.264.");
    let profile_guid = NV_ENC_H264_PROFILE_HIGH_GUID;
    assert!(profile_guids.contains(&profile_guid));

    // Get input formats based on the encode guid.
    let input_formats = encoder
        .get_supported_input_formats(encode_guid)
        .expect("The encoder should be able to get supported input buffer formats.");
    let buffer_format = NV_ENC_BUFFER_FORMAT_ARGB;
    assert!(input_formats.contains(&buffer_format));

    let tuning_info = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    // Get the preset config based on the selected encode guid (H.264), selected
    // preset (`LOW_LATENCY`), and tuning info (`ULTRA_LOW_LATENCY`).
    let mut preset_config = encoder
        .get_preset_config(encode_guid, preset_guid, tuning_info)
        .expect("Encoder should be able to create config based on presets.");

    // Initialize a new encoder session based on the `preset_config`
    // we generated before.
    let mut initialize_params = EncoderInitParams::new(encode_guid, WIDTH, HEIGHT);
    initialize_params
        .preset_guid(preset_guid)
        .tuning_info(tuning_info)
        .display_aspect_ratio(16, 9)
        .framerate(30, 1)
        .enable_picture_type_decision()
        .encode_config(&mut preset_config.presetCfg);
    let session = encoder
        .start_session(buffer_format, initialize_params)
        .expect("Encoder should be initialized correctly.");

    // Calculate the number of buffers we need based on the interval of P frames and
    // the look ahead depth.
    let num_bufs = usize::try_from(preset_config.presetCfg.frameIntervalP)
        .expect("frame intervalP should always be positive.")
        + usize::try_from(preset_config.presetCfg.rcParams.lookaheadDepth)
            .expect("lookahead depth should always be positive.");

    let mut output_buffers: Vec<_> = (0..num_bufs)
        .map(|_| {
            session
                .create_output_bitstream()
                .expect("The encoder should be able to create bitstreams.")
        })
        .collect();

    // Write result to output file "example_output.bin".
    let mut out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("example_output.bin")
        .expect("Permissions and available space should allow creating a new file.");

    // Generate each of the frames with Vulkan.
    let file_descriptors = (0..FRAMES)
        .map(|f| {
            create_buffer(
                vulkan_device.clone(),
                memory_type_index,
                WIDTH,
                HEIGHT,
                f,
                FRAMES,
            )
        })
        .collect::<Vec<_>>();

    // Encode each of the frames.
    for (i, file_descriptor) in file_descriptors.into_iter().enumerate() {
        println!("Encoding frame {:>3} / {FRAMES}", i + 1);
        let output_bitstream = &mut output_buffers[i % num_bufs];

        // Import file descriptor using CUDA.
        let external_memory = unsafe {
            cuda_ctx.import_external_memory(file_descriptor.into(), (WIDTH * HEIGHT * 4) as u64)
        }
        .expect("File descriptor should be valid for importing.");
        let mapped_buffer = external_memory
            .map_all()
            .expect("External memory should be mappable.");

        // Register and map with NVENC.
        let mut registered_resource = session
            .register_cuda_resource(WIDTH * 4, mapped_buffer)
            .expect("Buffer should be mapped and available for registration with NVENC.");

        session
            .encode_picture(
                &mut registered_resource,
                output_bitstream,
                Default::default(),
            )
            .expect("Encoder should be able to encode valid pictures");

        // Immediately locking is probably inefficient
        // (you should encode multiple before locking),
        // but for simplicity we just lock immediately.
        let lock = output_bitstream
            .lock()
            .expect("Bitstream lock should be available.");
        dbg!(lock.frame_index());
        dbg!(lock.timestamp());
        dbg!(lock.duration());
        dbg!(lock.picture_type());

        let data = lock.data();
        out_file
            .write_all(data)
            .expect("Writing should succeed because `out_file` was opened with write permissions.");
    }
}

/// Allocates memory on a Vulkan [`Device`] and returns a [`File`] (file
/// descriptor) to that data.
///
/// Will be used to create file descriptors for the invidual frames.
///
/// # Arguments
///
/// * `vulkan_device` - The device where the data should be allocated.
/// * `memory_type_index` - The index of the memory type that should be
///   allocated.
/// * `width`, `height` - The size of data to store.
/// * `i`, `i_max`: - The current frame and maximum frame index.
fn create_buffer(
    vulkan_device: Arc<Device>,
    memory_type_index: u32,
    width: u32,
    height: u32,
    i: u32,
    i_max: u32,
) -> File {
    let size = (width * height * 4) as u64;

    // Allocate memory with Vulkan.
    let mut memory = DeviceMemory::allocate(
        vulkan_device.clone(),
        MemoryAllocateInfo {
            allocation_size: size,
            memory_type_index,
            #[cfg(unix)]
            export_handle_types: ExternalMemoryHandleTypes::OPAQUE_FD,
            #[cfg(windows)]
            export_handle_types: ExternalMemoryHandleTypes::OPAQUE_WIN32,
            ..Default::default()
        },
    )
    .expect("There should be space to allocate vulkan memory on the device");

    // Map and write to the memory.
    // If memory_map_placed is supported, reserve an address and map there.
    // Otherwise, use the pointer returned by Vulkan's mapping state.
    let placed_supported = vulkan_device
        .physical_device()
        .supported_features()
        .memory_map_placed;

    let write_ptr: *mut std::ffi::c_void = if placed_supported {
        let mmap_mut = memmap2::MmapOptions::new()
            .len(memory.allocation_size() as usize)
            .map_anon()
            .expect("Mapping anonymous memory should work");
        let address = mmap_mut.as_ptr() as *mut std::ffi::c_void;

        // Keep the mmap alive until after unmap by binding it in the same scope.
        unsafe {
            memory.map_placed(
                MemoryMapInfo {
                    flags: MemoryMapFlags::PLACED,
                    size: memory.allocation_size(),
                    ..Default::default()
                },
                NonNull::new(address).expect("The mapped address should not be null"),
            )
        }
        .unwrap();

        // Write using the placed address
        // Note: `mmap_mut` must live until after unmap; ensure scope encloses writes and unmap.
        let content = unsafe {
            std::slice::from_raw_parts_mut(address as *mut u8, memory.allocation_size() as usize)
        };
        generate_test_input(content, width, height, i, i_max);

        // Flush host writes
        unsafe {
            memory
                .flush_range(MappedMemoryRange {
                    offset: 0,
                    size,
                    ..Default::default()
                })
                .expect("Flush should succeed for host-visible memory");
        }

        // Unmap before `mmap_mut` drops
        memory
            .unmap(Default::default())
            .expect("unmap should be sucessful on host-mapped device");

        // Return value unused in placed branch after unmap; set to null.
        std::ptr::null_mut()
    } else {
        memory
            .map(MemoryMapInfo {
                flags: MemoryMapFlags::empty(),
                size: memory.allocation_size(),
                ..Default::default()
            })
            .unwrap();

        let ptr = memory
            .mapping_state()
            .expect("memory should be mapped")
            .ptr()
            .as_ptr();

        let content = unsafe {
            std::slice::from_raw_parts_mut(ptr as *mut u8, memory.allocation_size() as usize)
        };
        generate_test_input(content, width, height, i, i_max);

        unsafe {
            memory
                .flush_range(MappedMemoryRange {
                    offset: 0,
                    size,
                    ..Default::default()
                })
                .expect("Flush should succeed for host-visible memory");
        }

        // Unmap after writes
        memory
            .unmap(Default::default())
            .expect("unmap should be sucessful on host-mapped device");

        ptr
    };

    let _ = write_ptr; // silence unused in placed branch

    // Export the memory. On Windows export a Win32 HANDLE and wrap into File; on Unix export an FD.
    #[cfg(unix)]
    {
        memory
            .export_fd(vulkano::memory::ExternalMemoryHandleType::OpaqueFd)
            .expect("The memory should be exportable as an OpaqueFd on UNIX")
    }

    #[cfg(windows)]
    {
        use std::mem::MaybeUninit;
        let info_vk = vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory.handle())
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);

        let fns = vulkan_device.fns();
        let mut output = MaybeUninit::<vk::HANDLE>::uninit();
        let result = unsafe {
            (fns.khr_external_memory_win32.get_memory_win32_handle_khr)(
                vulkan_device.handle(),
                &info_vk,
                output.as_mut_ptr(),
            )
        };
        if result != vk::Result::SUCCESS {
            panic!(
                "The memory should be exportable as an OpaqueWin32 handle on Windows: {:?}",
                result
            );
        }
        let raw_handle = unsafe { output.assume_init() };
        // SAFETY: OPAQUE_WIN32 requires the caller to close the handle. Wrapping in File transfers
        // ownership and ensures CloseHandle is called on drop.
        unsafe { File::from_raw_handle(raw_handle as _) }
    }
}
