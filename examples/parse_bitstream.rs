use std::{fs::File, io::Read};

use h264_parser::AnnexBParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("example_output_hevc.bin")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut parser = AnnexBParser::new();
    parser.push(&buffer);

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
