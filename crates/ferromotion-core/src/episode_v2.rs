//! Envelope v2 and chunk-binding 1, the text grammar of ferromorphic
//! `30812e26027cafa50e17c848e5afe57b1f246838`.
//!
//! Durations are reduced integer rationals in nanoseconds. The grid test is a
//! checked `u128` cross-multiplication. `dt_bits` is one round-to-nearest-even
//! of that rational, by binary long division, not an `f64` division.
//!
//! A chunk body with `n_action_steps == 0` is still [`crate::smolvla_receiver::CodecError::Horizon`].
//! On a binding, that same refusal is [`EpisodeError::NoActionSteps`].
//! `n > 50` is `Horizon` on the chunk body and [`EpisodeError::TooManyActionSteps`] here.

use crate::smolvla_receiver::{CodecError, sha256};

/// Header line of envelope v2 text, without its newline.
pub const EPISODE_ENVELOPE_V2_HEADER: &str = "ferromorphic-episode 2";
/// The `version` value an envelope v2 record must carry.
pub const EPISODE_ENVELOPE_V2_VERSION: u32 = 2;
/// Header line of a chunk binding record, without its newline.
pub const CHUNK_BINDING_HEADER: &str = "ferromorphic-chunk-binding 1";
/// Largest `n_action_steps` [`verify_chunk_binding`] accepts.
pub const MAX_ACTION_STEPS: u32 = 50;

const NS_PER_S: u128 = 1_000_000_000;
const EPISODE_PREFIX: &str = "ferromorphic-episode ";
const BINDING_PREFIX: &str = "ferromorphic-chunk-binding ";
const ENVELOPE_V2_KEYS: [&str; 9] = [
    "version",
    "epoch",
    "dt_num_ns",
    "dt_den",
    "dt_bits",
    "period_num_ns",
    "period_den",
    "origin_ns",
    "episode_nonce",
];
const CHUNK_BINDING_KEYS: [&str; 9] = [
    "envelope_sha256",
    "epoch",
    "chunk_index",
    "t0_num_ns",
    "t0_den",
    "period_num_ns",
    "period_den",
    "n_action_steps",
    "chunk_sha256",
];

/// Why an envelope v2 record, a chunk binding record, or a binding check was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeError {
    /// The first line names this record kind at a version this crate does not parse.
    UnknownVersion,
    /// The first line is not a header of this record kind.
    UnknownHeader,
    /// The `version` line disagrees with the header.
    VersionMismatch,
    /// The text is not terminated by `\n`.
    MissingNewline,
    /// Bytes follow the last canonical line.
    TrailingBytes,
    /// A line is not `key value`, or its key is not one of the record's keys.
    UnknownLine,
    /// A canonical line is absent.
    MissingLine,
    /// A canonical line appears twice.
    DuplicateLine,
    /// Canonical lines are present but not in header order.
    ReorderedLine,
    /// A decimal is empty, signed, or has a non-canonical leading zero.
    BadDecimal,
    /// A hex value has the wrong length or a byte outside `0-9a-f`.
    BadHex,
    /// A value or a product does not fit. Nothing wraps.
    Overflow,
    /// A rational has denominator zero.
    ZeroDenominator,
    /// A duration has numerator zero.
    ZeroDuration,
    /// A rational is not in lowest terms.
    NotReduced { num_ns: u64, den: u64 },
    /// `dt_bits` is not the round-to-nearest-even value of the rational `dt`.
    DtBitsMismatch { expected: u64, got: u64 },
    /// `period / dt` is not a positive integer.
    PeriodNotMultipleOfDt,
    /// `envelope_sha256` is not the SHA-256 of the envelope text.
    EnvelopeHashMismatch,
    /// The binding epoch is not the envelope epoch.
    EpochMismatch { binding: u64, envelope: u64 },
    /// The binding period is not the envelope period.
    PeriodNotEnvelope,
    /// The binding period is not the chunk period.
    PeriodNotChunk,
    /// The binding `t0` is not the chunk `t0`.
    T0NotChunk,
    /// The binding `n` is not the chunk `n`.
    StepsNotChunk { binding: u32, chunk: u32 },
    /// `n_action_steps` is zero. The chunk-body name for this refusal is
    /// [`CodecError::Horizon`].
    NoActionSteps,
    /// `n_action_steps` is above [`MAX_ACTION_STEPS`].
    TooManyActionSteps { n: u32 },
    /// `t0` is not `origin + k * period` for an integer `k >= 0`.
    OffGrid,
    /// `t0` is before `origin_ns`.
    BeforeOrigin,
    /// `chunk_index` is not strictly greater than the previous one.
    ChunkIndexNotIncreasing { previous: u64, got: u64 },
    /// `t0` is earlier than the previous accepted `t0`.
    Late,
    /// `t0` equals the previous accepted `t0`.
    RepeatedT0,
    /// The previous state names another envelope or epoch.
    ForeignPrevious,
}

/// A time in nanoseconds as a reduced fraction `num_ns / den`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NsRational {
    num_ns: u64,
    den: u64,
}

impl NsRational {
    /// Accept `num_ns / den` only when it is already in lowest terms.
    pub fn try_new(num_ns: u64, den: u64) -> Result<Self, EpisodeError> {
        if den == 0 {
            return Err(EpisodeError::ZeroDenominator);
        }
        if gcd(num_ns, den) != 1 {
            return Err(EpisodeError::NotReduced { num_ns, den });
        }
        Ok(Self { num_ns, den })
    }

    /// A whole number of nanoseconds, `ns / 1`.
    #[must_use]
    pub fn from_ns(ns: u64) -> Self {
        Self { num_ns: ns, den: 1 }
    }

    #[must_use]
    pub fn num_ns(self) -> u64 {
        self.num_ns
    }

    #[must_use]
    pub fn den(self) -> u64 {
        self.den
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn mul_u128(a: u128, b: u128) -> Result<u128, EpisodeError> {
    a.checked_mul(b).ok_or(EpisodeError::Overflow)
}

fn rational_lt(left: NsRational, right: NsRational) -> Result<bool, EpisodeError> {
    let l = mul_u128(u128::from(left.num_ns), u128::from(right.den))?;
    let r = mul_u128(u128::from(right.num_ns), u128::from(left.den))?;
    Ok(l < r)
}

/// Bits of `num_ns / (den * 10^9)` seconds, rounded once to nearest, ties to even.
pub fn rne_f64_seconds(num_ns: u64, den: u64) -> Result<u64, EpisodeError> {
    if den == 0 {
        return Err(EpisodeError::ZeroDenominator);
    }
    if num_ns == 0 {
        return Err(EpisodeError::ZeroDuration);
    }
    let d = mul_u128(u128::from(den), NS_PER_S)?;
    rne_f64_ratio(u128::from(num_ns), d)
}

fn rne_f64_ratio(n: u128, d: u128) -> Result<u64, EpisodeError> {
    if n == 0 || d == 0 || d > 1u128 << 127 {
        return Err(EpisodeError::Overflow);
    }
    let int = n / d;
    let mut rem = n % d;
    let mut sig = Significand::default();
    let mut exp: i32;
    if int > 0 {
        let len = 128 - int.leading_zeros();
        exp = i32::try_from(len).map_err(|_| EpisodeError::Overflow)? - 1;
        for i in (0..len).rev() {
            sig.push((int >> i) & 1 == 1);
        }
    } else {
        exp = 0;
        loop {
            exp -= 1;
            rem = mul_u128(rem, 2)?;
            if rem >= d {
                rem -= d;
                sig.push(true);
                break;
            }
        }
    }
    while sig.guard.is_none() {
        rem = mul_u128(rem, 2)?;
        let bit = rem >= d;
        if bit {
            rem -= d;
        }
        sig.push(bit);
    }
    let sticky = sig.sticky || rem != 0;
    let mut mant = sig.mant;
    if sig.guard == Some(true) && (sticky || mant & 1 == 1) {
        mant += 1;
        if mant == 1u64 << 53 {
            mant >>= 1;
            exp += 1;
        }
    }
    let biased = u64::try_from(exp + 1023).map_err(|_| EpisodeError::Overflow)?;
    Ok((biased << 52) | (mant & ((1u64 << 52) - 1)))
}

#[derive(Default)]
struct Significand {
    mant: u64,
    taken: u32,
    guard: Option<bool>,
    sticky: bool,
}

impl Significand {
    fn push(&mut self, bit: bool) {
        if self.taken < 53 {
            self.mant = (self.mant << 1) | u64::from(bit);
            self.taken += 1;
        } else if self.guard.is_none() {
            self.guard = Some(bit);
        } else {
            self.sticky |= bit;
        }
    }
}

/// `period / dt` as a whole number of neural steps.
pub fn steps_per_action(period: NsRational, dt: NsRational) -> Result<u64, EpisodeError> {
    if period.num_ns == 0 || dt.num_ns == 0 {
        return Err(EpisodeError::ZeroDuration);
    }
    let num = mul_u128(u128::from(period.num_ns), u128::from(dt.den))?;
    let den = mul_u128(u128::from(period.den), u128::from(dt.num_ns))?;
    if num % den != 0 {
        return Err(EpisodeError::PeriodNotMultipleOfDt);
    }
    u64::try_from(num / den).map_err(|_| EpisodeError::Overflow)
}

/// Envelope v2. `dt_bits` and `steps_per_action` are derived, not stored from the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpisodeEnvelopeV2 {
    epoch: u64,
    dt: NsRational,
    dt_bits: u64,
    period: NsRational,
    origin_ns: u64,
    episode_nonce: [u8; 32],
    steps_per_action: u64,
}

impl EpisodeEnvelopeV2 {
    pub fn try_new(
        epoch: u64,
        dt: NsRational,
        period: NsRational,
        origin_ns: u64,
        episode_nonce: [u8; 32],
    ) -> Result<Self, EpisodeError> {
        let dt_bits = rne_f64_seconds(dt.num_ns, dt.den)?;
        let steps_per_action = steps_per_action(period, dt)?;
        Ok(Self {
            epoch,
            dt,
            dt_bits,
            period,
            origin_ns,
            episode_nonce,
            steps_per_action,
        })
    }

    #[must_use]
    pub fn version(&self) -> u32 {
        EPISODE_ENVELOPE_V2_VERSION
    }
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    #[must_use]
    pub fn dt(&self) -> NsRational {
        self.dt
    }
    #[must_use]
    pub fn dt_bits(&self) -> u64 {
        self.dt_bits
    }
    #[must_use]
    pub fn period(&self) -> NsRational {
        self.period
    }
    #[must_use]
    pub fn origin_ns(&self) -> u64 {
        self.origin_ns
    }
    #[must_use]
    pub fn episode_nonce(&self) -> [u8; 32] {
        self.episode_nonce
    }
    #[must_use]
    pub fn steps_per_action(&self) -> u64 {
        self.steps_per_action
    }

    #[must_use]
    pub fn to_text(&self) -> String {
        format!(
            "{EPISODE_ENVELOPE_V2_HEADER}\nversion {}\nepoch {}\ndt_num_ns {}\ndt_den {}\ndt_bits {:016x}\nperiod_num_ns {}\nperiod_den {}\norigin_ns {}\nepisode_nonce {}\n",
            EPISODE_ENVELOPE_V2_VERSION,
            self.epoch,
            self.dt.num_ns,
            self.dt.den,
            self.dt_bits,
            self.period.num_ns,
            self.period.den,
            self.origin_ns,
            hex_lower(&self.episode_nonce)
        )
    }

    #[must_use]
    pub fn sha256(&self) -> [u8; 32] {
        sha256(self.to_text().as_bytes())
    }

    pub fn from_text(text: &str) -> Result<Self, EpisodeError> {
        let v = parse_record(
            text,
            EPISODE_ENVELOPE_V2_HEADER,
            EPISODE_PREFIX,
            &ENVELOPE_V2_KEYS,
        )?;
        let version = u32::try_from(parse_decimal(v[0])?).map_err(|_| EpisodeError::Overflow)?;
        if version != EPISODE_ENVELOPE_V2_VERSION {
            return Err(EpisodeError::VersionMismatch);
        }
        let epoch = parse_decimal(v[1])?;
        let dt = NsRational::try_new(parse_decimal(v[2])?, parse_decimal(v[3])?)?;
        let dt_bits = parse_hex_u64(v[4])?;
        let period = NsRational::try_new(parse_decimal(v[5])?, parse_decimal(v[6])?)?;
        let origin_ns = parse_decimal(v[7])?;
        let episode_nonce = parse_hex_32(v[8])?;
        let envelope = Self::try_new(epoch, dt, period, origin_ns, episode_nonce)?;
        if envelope.dt_bits != dt_bits {
            return Err(EpisodeError::DtBitsMismatch {
                expected: envelope.dt_bits,
                got: dt_bits,
            });
        }
        Ok(envelope)
    }
}

/// `ferromorphic-chunk-binding 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkBinding {
    envelope_sha256: [u8; 32],
    epoch: u64,
    chunk_index: u64,
    t0: NsRational,
    period: NsRational,
    n_action_steps: u32,
    chunk_sha256: [u8; 32],
}

impl ChunkBinding {
    pub fn try_new(
        envelope_sha256: [u8; 32],
        epoch: u64,
        chunk_index: u64,
        t0: NsRational,
        period: NsRational,
        n_action_steps: u32,
        chunk_sha256: [u8; 32],
    ) -> Result<Self, EpisodeError> {
        if period.num_ns == 0 {
            return Err(EpisodeError::ZeroDuration);
        }
        Ok(Self {
            envelope_sha256,
            epoch,
            chunk_index,
            t0,
            period,
            n_action_steps,
            chunk_sha256,
        })
    }

    #[must_use]
    pub fn for_envelope(
        envelope: &EpisodeEnvelopeV2,
        chunk_index: u64,
        t0: NsRational,
        n_action_steps: u32,
        chunk_sha256: [u8; 32],
    ) -> Self {
        Self {
            envelope_sha256: envelope.sha256(),
            epoch: envelope.epoch(),
            chunk_index,
            t0,
            period: envelope.period(),
            n_action_steps,
            chunk_sha256,
        }
    }

    #[must_use]
    pub fn envelope_sha256(&self) -> [u8; 32] {
        self.envelope_sha256
    }
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    #[must_use]
    pub fn chunk_index(&self) -> u64 {
        self.chunk_index
    }
    #[must_use]
    pub fn t0(&self) -> NsRational {
        self.t0
    }
    #[must_use]
    pub fn period(&self) -> NsRational {
        self.period
    }
    #[must_use]
    pub fn n_action_steps(&self) -> u32 {
        self.n_action_steps
    }
    #[must_use]
    pub fn chunk_sha256(&self) -> [u8; 32] {
        self.chunk_sha256
    }

    #[must_use]
    pub fn to_text(&self) -> String {
        format!(
            "{CHUNK_BINDING_HEADER}\nenvelope_sha256 {}\nepoch {}\nchunk_index {}\nt0_num_ns {}\nt0_den {}\nperiod_num_ns {}\nperiod_den {}\nn_action_steps {}\nchunk_sha256 {}\n",
            hex_lower(&self.envelope_sha256),
            self.epoch,
            self.chunk_index,
            self.t0.num_ns,
            self.t0.den,
            self.period.num_ns,
            self.period.den,
            self.n_action_steps,
            hex_lower(&self.chunk_sha256)
        )
    }

    pub fn from_text(text: &str) -> Result<Self, EpisodeError> {
        let v = parse_record(
            text,
            CHUNK_BINDING_HEADER,
            BINDING_PREFIX,
            &CHUNK_BINDING_KEYS,
        )?;
        let envelope_sha256 = parse_hex_32(v[0])?;
        let epoch = parse_decimal(v[1])?;
        let chunk_index = parse_decimal(v[2])?;
        let t0 = NsRational::try_new(parse_decimal(v[3])?, parse_decimal(v[4])?)?;
        let period = NsRational::try_new(parse_decimal(v[5])?, parse_decimal(v[6])?)?;
        let n_action_steps =
            u32::try_from(parse_decimal(v[7])?).map_err(|_| EpisodeError::Overflow)?;
        let chunk_sha256 = parse_hex_32(v[8])?;
        Self::try_new(
            envelope_sha256,
            epoch,
            chunk_index,
            t0,
            period,
            n_action_steps,
            chunk_sha256,
        )
    }
}

/// What the receiver decoded from the chunk, for [`verify_chunk_binding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkFacts {
    pub t0: NsRational,
    pub period: NsRational,
    pub n_action_steps: u32,
}

/// The last accepted binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingState {
    envelope_sha256: [u8; 32],
    epoch: u64,
    chunk_index: u64,
    t0: NsRational,
}

impl BindingState {
    #[must_use]
    pub fn chunk_index(&self) -> u64 {
        self.chunk_index
    }
    #[must_use]
    pub fn t0(&self) -> NsRational {
        self.t0
    }
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Check one binding. `n == 0` is [`CodecError::Horizon`] on a chunk body and
/// [`EpisodeError::NoActionSteps`] here. `n > 50` is the same `Horizon` on a
/// chunk body and [`EpisodeError::TooManyActionSteps`] here.
pub fn verify_chunk_binding(
    envelope: &EpisodeEnvelopeV2,
    binding: &ChunkBinding,
    chunk: &ChunkFacts,
    previous: Option<&BindingState>,
) -> Result<BindingState, EpisodeError> {
    let envelope_sha256 = envelope.sha256();
    if binding.envelope_sha256 != envelope_sha256 {
        return Err(EpisodeError::EnvelopeHashMismatch);
    }
    if binding.epoch != envelope.epoch {
        return Err(EpisodeError::EpochMismatch {
            binding: binding.epoch,
            envelope: envelope.epoch,
        });
    }
    if binding.period != envelope.period {
        return Err(EpisodeError::PeriodNotEnvelope);
    }
    if binding.period != chunk.period {
        return Err(EpisodeError::PeriodNotChunk);
    }
    if binding.t0 != chunk.t0 {
        return Err(EpisodeError::T0NotChunk);
    }
    if binding.n_action_steps != chunk.n_action_steps {
        return Err(EpisodeError::StepsNotChunk {
            binding: binding.n_action_steps,
            chunk: chunk.n_action_steps,
        });
    }
    match crate::smolvla_receiver::check_horizon(binding.n_action_steps, MAX_ACTION_STEPS) {
        Err(CodecError::Horizon) if binding.n_action_steps == 0 => {
            return Err(EpisodeError::NoActionSteps);
        }
        Err(CodecError::Horizon) => {
            return Err(EpisodeError::TooManyActionSteps {
                n: binding.n_action_steps,
            });
        }
        Ok(()) => {}
        Err(_) => {
            return Err(EpisodeError::TooManyActionSteps {
                n: binding.n_action_steps,
            });
        }
    }
    require_on_grid(envelope.origin_ns, envelope.period, binding.t0)?;
    if let Some(prev) = previous {
        if prev.envelope_sha256 != envelope_sha256 || prev.epoch != binding.epoch {
            return Err(EpisodeError::ForeignPrevious);
        }
        if binding.chunk_index <= prev.chunk_index {
            return Err(EpisodeError::ChunkIndexNotIncreasing {
                previous: prev.chunk_index,
                got: binding.chunk_index,
            });
        }
        if binding.t0 == prev.t0 {
            return Err(EpisodeError::RepeatedT0);
        }
        if rational_lt(binding.t0, prev.t0)? {
            return Err(EpisodeError::Late);
        }
    }
    Ok(BindingState {
        envelope_sha256,
        epoch: binding.epoch,
        chunk_index: binding.chunk_index,
        t0: binding.t0,
    })
}

fn require_on_grid(origin_ns: u64, period: NsRational, t0: NsRational) -> Result<(), EpisodeError> {
    let lhs = mul_u128(u128::from(t0.num_ns), u128::from(period.den))?;
    let base = mul_u128(
        mul_u128(u128::from(origin_ns), u128::from(period.den))?,
        u128::from(t0.den),
    )?;
    if lhs < base {
        return Err(EpisodeError::BeforeOrigin);
    }
    let step = mul_u128(u128::from(period.num_ns), u128::from(t0.den))?;
    if step == 0 || (lhs - base) % step != 0 {
        return Err(EpisodeError::OffGrid);
    }
    Ok(())
}

fn parse_record<'a, const N: usize>(
    text: &'a str,
    header: &str,
    prefix: &str,
    keys: &[&str; N],
) -> Result<[&'a str; N], EpisodeError> {
    if text.is_empty() {
        return Err(EpisodeError::MissingLine);
    }
    let Some(body) = text.strip_suffix('\n') else {
        return Err(EpisodeError::MissingNewline);
    };
    let lines: Vec<&str> = body.split('\n').collect();
    let first = lines.first().copied().unwrap_or("");
    if first != header {
        return Err(if first.starts_with(prefix) {
            EpisodeError::UnknownVersion
        } else {
            EpisodeError::UnknownHeader
        });
    }
    let mut values = [""; N];
    let mut at = 0usize;
    for line in &lines[1..] {
        if at == N {
            return Err(match line_key(line) {
                Some(key) if keys.contains(&key) => EpisodeError::DuplicateLine,
                _ => EpisodeError::TrailingBytes,
            });
        }
        let Some((key, value)) = line.split_once(' ') else {
            return Err(EpisodeError::UnknownLine);
        };
        if key != keys[at] {
            return Err(if keys[..at].contains(&key) {
                EpisodeError::DuplicateLine
            } else if keys[at + 1..].contains(&key) {
                if lines[1..].iter().any(|l| line_key(l) == Some(keys[at])) {
                    EpisodeError::ReorderedLine
                } else {
                    EpisodeError::MissingLine
                }
            } else {
                EpisodeError::UnknownLine
            });
        }
        values[at] = value;
        at += 1;
    }
    if at != N {
        return Err(EpisodeError::MissingLine);
    }
    Ok(values)
}

fn line_key(line: &str) -> Option<&str> {
    line.split_once(' ').map(|(key, _)| key)
}

fn parse_decimal(s: &str) -> Result<u64, EpisodeError> {
    let bytes = s.as_bytes();
    if bytes.is_empty()
        || !bytes.iter().all(u8::is_ascii_digit)
        || (bytes.len() > 1 && bytes[0] == b'0')
    {
        return Err(EpisodeError::BadDecimal);
    }
    let mut value = 0u64;
    for &b in bytes {
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .ok_or(EpisodeError::Overflow)?;
    }
    Ok(value)
}

fn hex_digit(b: u8) -> Result<u8, EpisodeError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(EpisodeError::BadHex),
    }
}

fn parse_hex_u64(s: &str) -> Result<u64, EpisodeError> {
    let bytes = s.as_bytes();
    if bytes.len() != 16 {
        return Err(EpisodeError::BadHex);
    }
    let mut value = 0u64;
    for &b in bytes {
        value = (value << 4) | u64::from(hex_digit(b)?);
    }
    Ok(value)
}

fn parse_hex_32(s: &str) -> Result<[u8; 32], EpisodeError> {
    let bytes = s.as_bytes();
    if bytes.len() != 64 {
        return Err(EpisodeError::BadHex);
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = (hex_digit(bytes[i * 2])? << 4) | hex_digit(bytes[i * 2 + 1])?;
    }
    Ok(out)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(char::from(HEX[usize::from(b >> 4)]));
        s.push(char::from(HEX[usize::from(b & 15)]));
    }
    s
}
