use std::collections::BTreeMap;

use bytes::Bytes;
use rtc_rtp::{header::Header, packet::Packet};

/// RTP packet ordering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtpOrderingMode {
    /// Emit only contiguous packets from the current sequence.
    InOrder,
    /// Keep a small reordering window and recover from gaps when exceeded.
    LightReorder { window: u16 },
}

impl Default for RtpOrderingMode {
    fn default() -> Self {
        Self::LightReorder { window: 64 }
    }
}

/// Reorders incoming RTP packets by sequence number.
#[derive(Debug, Default)]
pub struct RtpReorderBuffer {
    mode: RtpOrderingMode,
    next_sequence: Option<u16>,
    buffer: BTreeMap<u16, Packet>,
}

impl RtpReorderBuffer {
    /// Creates a new reorder buffer.
    #[must_use]
    pub fn new(mode: RtpOrderingMode) -> Self {
        Self {
            mode,
            next_sequence: None,
            buffer: BTreeMap::new(),
        }
    }

    /// Pushes one packet and returns packets that are now ready in order.
    pub fn push(&mut self, packet: Packet) -> Vec<Packet> {
        let sequence = packet.header.sequence_number;
        self.buffer.entry(sequence).or_insert(packet);

        if self.next_sequence.is_none() {
            self.next_sequence = self.buffer.first_key_value().map(|(&seq, _)| seq);
        }

        let mut ready = Vec::new();
        self.drain_contiguous(&mut ready);
        if ready.is_empty() {
            self.recover_gap_if_needed(&mut ready);
        }
        ready
    }

    /// Drains all buffered packets in current order.
    pub fn flush(&mut self) -> Vec<Packet> {
        let mut ready = Vec::new();
        self.drain_contiguous(&mut ready);

        while let Some((&seq, _)) = self.buffer.first_key_value() {
            if let Some(packet) = self.buffer.remove(&seq) {
                ready.push(packet);
            }
        }
        self.next_sequence = None;
        ready
    }

    fn drain_contiguous(&mut self, ready: &mut Vec<Packet>) {
        while let Some(expected) = self.next_sequence {
            let Some(packet) = self.buffer.remove(&expected) else {
                break;
            };
            ready.push(packet);
            self.next_sequence = Some(expected.wrapping_add(1));
        }
    }

    fn recover_gap_if_needed(&mut self, ready: &mut Vec<Packet>) {
        let RtpOrderingMode::LightReorder { window } = self.mode else {
            return;
        };
        let (Some(expected), Some((&min_seq, _))) =
            (self.next_sequence, self.buffer.first_key_value())
        else {
            return;
        };
        let gap = min_seq.wrapping_sub(expected);
        if gap > window {
            self.next_sequence = Some(min_seq);
            self.drain_contiguous(ready);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(sequence: u16) -> Packet {
        Packet {
            header: Header {
                sequence_number: sequence,
                ..Default::default()
            },
            payload: Bytes::new(),
        }
    }

    #[test]
    fn in_order_mode_keeps_contiguous_packets() {
        let mut reorder = RtpReorderBuffer::new(RtpOrderingMode::InOrder);

        let out1 = reorder.push(packet(10));
        assert_eq!(out1.len(), 1);
        assert_eq!(out1[0].header.sequence_number, 10);

        let out2 = reorder.push(packet(12));
        assert!(out2.is_empty());

        let out3 = reorder.push(packet(11));
        assert_eq!(out3.len(), 2);
        assert_eq!(out3[0].header.sequence_number, 11);
        assert_eq!(out3[1].header.sequence_number, 12);
    }

    #[test]
    fn light_reorder_recovers_when_gap_exceeds_window() {
        let mut reorder = RtpReorderBuffer::new(RtpOrderingMode::LightReorder { window: 2 });

        let out1 = reorder.push(packet(100));
        assert_eq!(out1.len(), 1);
        assert_eq!(out1[0].header.sequence_number, 100);

        let out2 = reorder.push(packet(104));
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].header.sequence_number, 104);

        let out3 = reorder.push(packet(105));
        assert_eq!(out3.len(), 1);
        assert_eq!(out3[0].header.sequence_number, 105);
    }

    #[test]
    fn flush_returns_remaining_packets() {
        let mut reorder = RtpReorderBuffer::new(RtpOrderingMode::InOrder);
        let _ = reorder.push(packet(1));
        let _ = reorder.push(packet(3));

        let flushed = reorder.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].header.sequence_number, 3);
    }
}
