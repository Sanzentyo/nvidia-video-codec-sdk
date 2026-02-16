use anyhow::{anyhow, bail, Context, Result};
use bytes::{Bytes, BytesMut};
use rtc_rtp::{
    codec::{
        av1::{Av1Depacketizer, Av1Payloader},
        h264::H264Payloader,
        h264::{H264Packet, ANNEXB_NALUSTART_CODE},
        h265::{H265Packet, H265Payload, HevcPayloader},
    },
    header::Header,
    packet::Packet,
    packetizer::{Depacketizer, Payloader},
};

use crate::support::rtp_reorder::{RtpOrderingMode, RtpReorderBuffer};

/// RTP codecs supported by the test pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtpCodec {
    H264,
    H265,
    Av1,
}

/// Access unit produced by RTP depacketization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub timestamp_90k: i64,
    pub data: Vec<u8>,
}

/// Incremental RTP packet to access-unit assembler.
#[derive(Debug)]
pub struct RtpToAccessUnit {
    codec: RtpCodec,
    reorder: RtpReorderBuffer,
    current_timestamp: Option<u32>,
    current_au: BytesMut,
    h264: H264Packet,
    h265: H265Packet,
    h265_fu_buffer: BytesMut,
    av1: Av1Depacketizer,
}

impl RtpToAccessUnit {
    /// Creates a new assembler.
    #[must_use]
    pub fn new(codec: RtpCodec, ordering_mode: RtpOrderingMode) -> Self {
        Self {
            codec,
            reorder: RtpReorderBuffer::new(ordering_mode),
            current_timestamp: None,
            current_au: BytesMut::new(),
            h264: H264Packet::default(),
            h265: H265Packet::default(),
            h265_fu_buffer: BytesMut::new(),
            av1: Av1Depacketizer::new(),
        }
    }

    /// Pushes one RTP packet and returns newly completed access units.
    pub fn push_packet(&mut self, packet: Packet) -> Result<Vec<AccessUnit>> {
        let mut out = Vec::new();
        for ordered in self.reorder.push(packet) {
            self.handle_ordered_packet(ordered, &mut out)?;
        }
        Ok(out)
    }

    /// Flushes reorder and frame assembly state.
    pub fn flush(&mut self) -> Vec<AccessUnit> {
        let mut out = Vec::new();
        for ordered in self.reorder.flush() {
            if let Err(err) = self.handle_ordered_packet(ordered, &mut out) {
                eprintln!("rtp flush warning: {err}");
            }
        }
        self.flush_current(&mut out);
        out
    }

    fn handle_ordered_packet(&mut self, packet: Packet, out: &mut Vec<AccessUnit>) -> Result<()> {
        if let Some(current_timestamp) = self.current_timestamp {
            if packet.header.timestamp != current_timestamp {
                self.flush_current(out);
            }
        }
        self.current_timestamp = Some(packet.header.timestamp);

        let payload = self
            .depacketize_payload(&packet)
            .with_context(|| format!("depacketize {:?}", self.codec))?;
        if !payload.is_empty() {
            self.current_au.extend_from_slice(&payload);
        }

        if packet.header.marker {
            self.flush_current(out);
        }
        Ok(())
    }

    fn flush_current(&mut self, out: &mut Vec<AccessUnit>) {
        let Some(timestamp) = self.current_timestamp else {
            return;
        };
        if self.current_au.is_empty() {
            return;
        }
        let data = std::mem::take(&mut self.current_au).to_vec();
        out.push(AccessUnit {
            timestamp_90k: i64::from(timestamp),
            data,
        });
    }

    fn depacketize_payload(&mut self, packet: &Packet) -> Result<Bytes> {
        match self.codec {
            RtpCodec::H264 => self
                .h264
                .depacketize(&packet.payload)
                .context("h264 depacketize"),
            RtpCodec::H265 => self.depacketize_h265(packet),
            RtpCodec::Av1 => self
                .av1
                .depacketize(&packet.payload)
                .context("av1 depacketize"),
        }
    }

    fn depacketize_h265(&mut self, packet: &Packet) -> Result<Bytes> {
        self.h265
            .depacketize(&packet.payload)
            .context("h265 depacketize")?;

        match self.h265.payload() {
            H265Payload::H265SingleNALUnitPacket(single) => {
                let mut out = BytesMut::new();
                append_annexb_nal_with_header(
                    &mut out,
                    single.payload_header().0,
                    single.payload().as_ref(),
                );
                Ok(out.freeze())
            }
            H265Payload::H265AggregationPacket(aggregation) => {
                let mut out = BytesMut::new();
                if let Some(first) = aggregation.first_unit() {
                    append_annexb_nal(&mut out, first.nal_unit().as_ref());
                }
                for unit in aggregation.other_units() {
                    append_annexb_nal(&mut out, unit.nal_unit().as_ref());
                }
                Ok(out.freeze())
            }
            H265Payload::H265FragmentationUnitPacket(fu) => {
                if fu.fu_header().s() {
                    self.h265_fu_buffer.clear();
                    self.h265_fu_buffer
                        .extend_from_slice(&reconstruct_fu_header(
                            fu.payload_header().0,
                            fu.fu_header().fu_type(),
                        ));
                } else if self.h265_fu_buffer.is_empty() {
                    return Ok(Bytes::new());
                }

                self.h265_fu_buffer.extend_from_slice(fu.payload().as_ref());
                if fu.fu_header().e() {
                    let nal = std::mem::take(&mut self.h265_fu_buffer).freeze();
                    let mut out = BytesMut::new();
                    append_annexb_nal(&mut out, nal.as_ref());
                    Ok(out.freeze())
                } else {
                    Ok(Bytes::new())
                }
            }
            H265Payload::H265PACIPacket(_) => {
                bail!("H265 PACI payload is currently unsupported in this pipeline")
            }
        }
    }
}

/// Packetizes one codec access unit into RTP packets.
pub fn packetize_access_unit(
    codec: RtpCodec,
    access_unit: &[u8],
    timestamp_90k: u32,
    sequence_start: u16,
    mtu: usize,
) -> Result<Vec<Packet>> {
    if access_unit.is_empty() {
        bail!("access unit must not be empty");
    }
    if mtu <= 12 {
        bail!("mtu must be larger than RTP header size");
    }

    let payload = Bytes::copy_from_slice(access_unit);
    let payload_mtu = mtu - 12;
    let payloads = match codec {
        RtpCodec::H264 => {
            let mut payloader = H264Payloader::default();
            payloader
                .payload(payload_mtu, &payload)
                .context("h264 packetize")?
        }
        RtpCodec::H265 => {
            let mut payloader = HevcPayloader::default();
            payloader
                .payload(payload_mtu, &payload)
                .context("h265 packetize")?
        }
        RtpCodec::Av1 => {
            let mut payloader = Av1Payloader::default();
            payloader
                .payload(payload_mtu, &payload)
                .context("av1 packetize")?
        }
    };
    if payloads.is_empty() {
        return Err(anyhow!("codec payloader produced no RTP payloads"));
    }

    let payload_count = payloads.len();
    let mut packets = Vec::with_capacity(payload_count);
    for (index, payload) in payloads.into_iter().enumerate() {
        packets.push(Packet {
            header: Header {
                version: 2,
                marker: index + 1 == payload_count,
                payload_type: 96,
                sequence_number: sequence_start.wrapping_add(index as u16),
                timestamp: timestamp_90k,
                ssrc: 1,
                ..Default::default()
            },
            payload,
        });
    }
    Ok(packets)
}

fn append_annexb_nal(out: &mut BytesMut, nal_data: &[u8]) {
    if nal_data.is_empty() {
        return;
    }
    out.extend_from_slice(ANNEXB_NALUSTART_CODE.as_ref());
    out.extend_from_slice(nal_data);
}

fn append_annexb_nal_with_header(out: &mut BytesMut, header: u16, payload: &[u8]) {
    let header_bytes = [(header >> 8) as u8, header as u8];
    out.extend_from_slice(ANNEXB_NALUSTART_CODE.as_ref());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(payload);
}

fn reconstruct_fu_header(original_header: u16, fu_type: u8) -> [u8; 2] {
    let high = (((original_header >> 8) as u8) & 0x81) | ((fu_type & 0x3f) << 1);
    let low = original_header as u8;
    [high, low]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h265_packet(sequence: u16, marker: bool, payload: &[u8]) -> Packet {
        Packet {
            header: Header {
                version: 2,
                payload_type: 96,
                sequence_number: sequence,
                timestamp: 42,
                marker,
                ssrc: 1,
                ..Default::default()
            },
            payload: Bytes::copy_from_slice(payload),
        }
    }

    #[test]
    fn h265_fu_fragments_are_reassembled_into_one_access_unit() {
        let mut assembler = RtpToAccessUnit::new(RtpCodec::H265, RtpOrderingMode::InOrder);

        let start = h265_packet(10, false, &[0x62, 0x01, 0x93, 0xaa, 0xbb]);
        let middle = h265_packet(11, false, &[0x62, 0x01, 0x13, 0xcc]);
        let end = h265_packet(12, true, &[0x62, 0x01, 0x53, 0xdd]);

        assert!(assembler.push_packet(start).unwrap().is_empty());
        assert!(assembler.push_packet(middle).unwrap().is_empty());

        let out = assembler.push_packet(end).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].timestamp_90k, 42);
        assert_eq!(
            out[0].data,
            vec![0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaa, 0xbb, 0xcc, 0xdd]
        );
    }
}
