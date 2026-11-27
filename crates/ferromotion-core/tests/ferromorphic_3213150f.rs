//! Full-state v1 bytes from ferromorphic
//! `3213150fd863f67d7027d1e3bebc8ea052886a75`.
//!
//! The walk follows the layout in that commit's
//! `docs/contracts/ferromotion-clock.md` and the field widths in
//! `Sim::full_state_bytes` and `Sim::state_bytes`. It reads the clock header
//! inside `state_bytes`. It does not decode neuron records and it does not
//! restore a simulation.

use ferromotion_core::digest_sha256;

const MAGIC: &[u8] = b"ferromorphic-sim-state\0";

#[derive(Debug, PartialEq, Eq)]
enum Refuse {
    Truncated,
    BadMagic,
    UnknownVersion,
    ChecksumMismatch,
    TrailingBytes,
    BadValue,
}

#[derive(Debug, PartialEq, Eq)]
struct ClockHeader {
    state_bytes: Vec<u8>,
    tick: u64,
    origin_ns: u64,
    period_ns: u64,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Refuse> {
        let end = self.at.checked_add(n).ok_or(Refuse::Truncated)?;
        if end > self.bytes.len() {
            return Err(Refuse::Truncated);
        }
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, Refuse> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Refuse> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, Refuse> {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }

    fn presence(&mut self) -> Result<Option<u64>, Refuse> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u64()?)),
            _ => Err(Refuse::BadValue),
        }
    }

    fn count(&mut self, item: usize) -> Result<usize, Refuse> {
        let n = usize::try_from(self.u64()?).map_err(|_| Refuse::Truncated)?;
        let need = n.checked_mul(item).ok_or(Refuse::Truncated)?;
        if need > self.bytes.len() - self.at {
            return Err(Refuse::Truncated);
        }
        Ok(n)
    }

    fn finish(&self) -> Result<(), Refuse> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(Refuse::TrailingBytes)
        }
    }
}

fn parse_state_bytes(bytes: &[u8]) -> Result<(u64, Option<u64>, Option<u64>), Refuse> {
    let mut r = Reader { bytes, at: 0 };
    let _dt_bits = r.u64()?;
    let _mode = r.u8()?;
    let tick = r.u64()?;
    let _epoch = r.u64()?;
    let origin_ns = r.presence()?;
    let period_ns = r.presence()?;
    let _exact = r.u8()?;
    for _ in 0..6 {
        let _ledger = r.u64()?;
    }
    let slots = r.count(8)?;
    for _ in 0..slots {
        let deliveries = r.count(12)?;
        for _ in 0..deliveries {
            let _post = r.u32()?;
            let _weight_bits = r.u64()?;
        }
    }
    let inject = r.count(8)?;
    for _ in 0..inject {
        let _bits = r.u64()?;
    }
    let touched = r.count(4)?;
    for _ in 0..touched {
        let _index = r.u32()?;
    }
    let as_of = r.count(8)?;
    for _ in 0..as_of {
        let _tick = r.u64()?;
    }
    r.finish()?;
    Ok((tick, origin_ns, period_ns))
}

/// Structural read of full-state v1. The checksum is checked before the fields,
/// which is why a short suffix is [`Refuse::ChecksumMismatch`].
fn parse_full_state_v1(bytes: &[u8]) -> Result<ClockHeader, Refuse> {
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err(if bytes.len() < MAGIC.len() && MAGIC.starts_with(bytes) {
            Refuse::Truncated
        } else {
            Refuse::BadMagic
        });
    }
    let mut r = Reader { bytes, at: 0 };
    r.take(MAGIC.len())?;
    let version = r.u32()?;
    if version != 1 {
        return Err(Refuse::UnknownVersion);
    }
    if bytes.len() < r.at + 32 {
        return Err(Refuse::Truncated);
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if digest_sha256(body)
        != digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    {
        return Err(Refuse::ChecksumMismatch);
    }
    let mut r = Reader {
        bytes: body,
        at: r.at,
    };
    let tag_len = usize::try_from(r.u32()?).map_err(|_| Refuse::Truncated)?;
    let tag = r.take(tag_len)?;
    if tag != b"lif" {
        return Err(Refuse::BadValue);
    }
    let core_len = usize::try_from(r.u64()?).map_err(|_| Refuse::Truncated)?;
    let state_bytes = r.take(core_len)?.to_vec();
    let (tick, origin_ns, period_ns) = parse_state_bytes(&state_bytes)?;
    let n = usize::try_from(r.u64()?).map_err(|_| Refuse::Truncated)?;
    let n_syn = usize::try_from(r.u64()?).map_err(|_| Refuse::Truncated)?;
    let _max_delay = r.u32()?;
    let offsets = n.checked_add(1).ok_or(Refuse::Truncated)?;
    for _ in 0..offsets {
        let _offset = r.u64()?;
    }
    for _ in 0..n_syn {
        let _post = r.u32()?;
        let _weight_bits = r.u64()?;
        let _delay = r.u32()?;
    }
    let neurons = usize::try_from(r.u64()?).map_err(|_| Refuse::Truncated)?;
    let record = usize::try_from(r.u32()?).map_err(|_| Refuse::Truncated)?;
    let total = neurons.checked_mul(record).ok_or(Refuse::Truncated)?;
    r.take(total)?;
    r.finish()?;
    let (Some(origin_ns), Some(period_ns)) = (origin_ns, period_ns) else {
        return Err(Refuse::BadValue);
    };
    Ok(ClockHeader {
        state_bytes,
        tick,
        origin_ns,
        period_ns,
    })
}

fn file(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/acceptance/contracts/fixtures/ferromorphic-3213150f")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn ferromorphic_3213150f_full_state_matches_its_sha256_lengths_and_clock_header() {
    let full = file("full-state-v1-lif50.bin");
    let state = file("full-state-v1-lif50-state-bytes.bin");
    let listed = file("full-state-v1-lif50.sha256");
    assert_eq!(full.len(), 970);
    assert_eq!(state.len(), 280);
    assert_eq!(
        digest_sha256(&full),
        "7d9bc9c1400bd2b14d4d5336bb713b91733353b9e0a17fa6b1852ecbb05617f1"
    );
    assert_eq!(
        digest_sha256(&state),
        "01e52c73cf8376050d5b95581a6a8e15389668ad2f067d6e517b7d94fc753f13"
    );
    assert_eq!(
        listed,
        b"7d9bc9c1400bd2b14d4d5336bb713b91733353b9e0a17fa6b1852ecbb05617f1  full-state-v1-lif50.bin\n01e52c73cf8376050d5b95581a6a8e15389668ad2f067d6e517b7d94fc753f13  full-state-v1-lif50-state-bytes.bin\n"
    );

    let header = parse_full_state_v1(&full).unwrap();
    let offset = MAGIC.len() + 4 + 4 + b"lif".len() + 8;
    assert_eq!(&full[offset..offset + state.len()], state.as_slice());
    assert_eq!(header.state_bytes, state);
    assert_eq!(header.tick, 50);
    assert_eq!(header.origin_ns, 7000);
    assert_eq!(header.period_ns, 100_000);
}

#[test]
fn full_state_v1_refuses_truncation_and_trailing_bytes() {
    let full = file("full-state-v1-lif50.bin");
    assert_eq!(parse_full_state_v1(&[]), Err(Refuse::Truncated));
    assert_eq!(parse_full_state_v1(&full[..10]), Err(Refuse::Truncated));
    assert_eq!(
        parse_full_state_v1(&full[..full.len() - 1]),
        Err(Refuse::ChecksumMismatch)
    );
    let mut extra = full.clone();
    extra.push(0);
    assert_eq!(parse_full_state_v1(&extra), Err(Refuse::ChecksumMismatch));

    let mut body = full[..full.len() - 32].to_vec();
    body.push(0);
    let digest = digest_sha256(&body);
    let mut padded = body;
    for i in (0..digest.len()).step_by(2) {
        padded.push(u8::from_str_radix(&digest[i..i + 2], 16).unwrap());
    }
    assert_eq!(parse_full_state_v1(&padded), Err(Refuse::TrailingBytes));
}
