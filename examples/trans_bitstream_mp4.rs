use h264_parser::AnnexBParser;
use shiguredo_mp4::BoxHeader;
use std::fs::File;
use std::io::Read;

const MAX_FRAG_COUNT: usize = 16;

fn main() {
    let mut file = File::open("example_output.mp4").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let mut offset = 0;
    let mut au_count = 0;
    let mut parser = AnnexBParser::new();

    while offset < buffer.len() && au_count < MAX_FRAG_COUNT {
        let (box_header, box_size) = BoxHeader::read(&buffer[offset..]).unwrap();
        if &box_header.box_type == b"mdat" {
            let mdat_data = &buffer[offset + box_header.size as usize..offset + box_size as usize];
            parser.push(mdat_data);

            while let Ok(Some(au)) = parser.next_access_unit() {
                println!("Frame {}: keyframe={}", au_count, au.is_keyframe());

                if let Some(ref sps) = au.sps {
                    println!("  Resolution: {}x{}", sps.width, sps.height);
                }

                for nal in au.nals() {
                    println!("  NAL: {:?}", nal.nal_type);
                }

                au_count += 1;
            }
        }
        offset += box_size as usize;
    }
}