use std::{
    fs::{DirEntry, File},
    io::Read,
};

use h264_parser::AnnexBParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::read_dir("./output")?
        .filter_map(Result::ok)
        .filter(|e: &DirEntry| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter(|e: &DirEntry| {
            e.path()
                .extension()
                .map(|ext| ext == "bin")
                .unwrap_or(false)
        })
        .map(|e| File::open(e.path()))
        .map(Result::unwrap)
        .map(|mut f| {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            buf
        })
        .collect::<Vec<_>>();

    let mut parser = AnnexBParser::new();
    parser.push(&file[0]);
    for _ in 0..2 {
        if let Ok(Some(au)) = parser.next_access_unit() {
            println!("Frame: keyframe={}", au.is_keyframe());

            if let Some(ref sps) = au.sps {
                println!("  Resolution: {}x{}", sps.width, sps.height);
            }

            for nal in au.nals() {
                println!("  NAL: {:?}", nal.nal_type);
            }
        }
    }

    for (i, data) in file.iter_mut().skip(1).enumerate() {
        parser.push(data);
        if let Ok(Some(au)) = parser.next_access_unit() {
            print!("{i}: ");
            println!("Frame: keyframe={}", au.is_keyframe());

            if let Some(ref sps) = au.sps {
                println!("  Resolution: {}x{}", sps.width, sps.height);
            }

            for nal in au.nals() {
                println!("  NAL: {:?}", nal.nal_type);
            }
        }
    }

    println!("Draining remaining AUs...");

    while let Ok(Some(au)) = parser.next_access_unit() {
        println!("Frame: keyframe={}", au.is_keyframe());

        if let Some(ref sps) = au.sps {
            println!("  Resolution: {}x{}", sps.width, sps.height);
        }

        for nal in au.nals() {
            println!("  NAL: {:?}", nal.nal_type);
        }
    }

    Ok(())
}
