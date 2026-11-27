//! SmolVLA action-chunk codec version 3.
//!
//! The byte layout is ferric `docs/acceptance/contracts/smolvla-ferromotion-v3.md`,
//! sha256 [`crate::smolvla_receiver::FERRIC_V3_CONTRACT_SHA256`], which is unchanged.
//! The decoder follows `crates/ferric-accept/src/smolvla_v3.rs` at
//! `b9fd7597c24c6f05b162c572a126749998f5c5a2`. A decoded chunk is not model output.
//! [`predict_action_chunk`] refuses with [`V3Error::ProposedEntryInvoked`].

use crate::smolvla_receiver::{LEROBOT_UPSTREAM, SMOLVLA_HUB_ID, sha256};

const CHUNK_MAGIC: &[u8] = b"FERRIC-SMOLVLA-CHUNK\0";
const CHUNK_VERSION_V3: u32 = 3;

const WEIGHT_SHA256: &str = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb";
const STATS_FILE_SHA256: &str = "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8";
const SO100_ACTION_MEAN: &str = "e22c3af117ece699b816d50f60df8a1e7e4728e1babf5297634b74b8483ad924";
const SO100_ACTION_STD: &str = "a09262820c5532e2fffb6da68c7ee6d9670d0da7080f70aadd0b34b7af8755b9";
const SO100_BLUE_ACTION_MEAN: &str =
    "8f178ffad9bca313223b2ebf56102779a017dbb0409c557259fa8888e02155fb";
const SO100_BLUE_ACTION_STD: &str =
    "a4762d32295adf3bf7d652ea86cb3e7f2cc575b496e7e30a220a572e927a6820";
const SO100_RED_ACTION_MEAN: &str =
    "f85d5119f589c058249515c89ed780b4bba4fdaa30b81277bb738e513b7e6cc8";
const SO100_RED_ACTION_STD: &str =
    "4deb2fcd2af0aa4667f6b09748b5e527e543ca36c13e204b75273e7cc62c8bf6";
const STATE_MEAN_SHA256: &str = "c0087077ccbf5362ee8dab2ec5cae2ee915123b176ade6c0be1fad3feb7a7eca";
const STATE_STD_SHA256: &str = "09d52284cad4193fdaf1fe35c89dab6aa2711514b3043296b7811c13cb8f13ef";

/// Joint limit bits. These are the f32 values in the ferric SO-100 table.
const JOINTS: &[(&str, &str, u32, u32)] = &[
    ("main_shoulder_pan", "deg", 0xc214_b600, 0x4291_8c00),
    ("main_shoulder_lift", "deg", 0xc2b2_f200, 0x4224_8800),
    ("main_elbow_flex", "deg", 0xc244_2c00, 0x4295_3d00),
    ("main_wrist_flex", "deg", 0x411a_b000, 0x42c1_2f00),
    ("main_wrist_roll", "deg", 0x4261_0000, 0x42f7_2600),
    ("main_gripper", "percent_range", 0x0000_0000, 0x42c8_0000),
];

/// Why a version-3 chunk was refused. `ProposedEntryInvoked` is not a decode error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3Error {
    EvenSeed,
    MissingHash,
    BadHash,
    BadStatsKey,
    StatsKeyMismatch,
    Embodiment,
    Unreduced,
    ZeroDenominator,
    Period,
    BeforeOrigin,
    OffGrid,
    Overflow,
    ActionSteps,
    ActionDim,
    JointCount,
    JointName,
    JointUnit,
    JointFrame,
    JointLimit,
    ValueOutsideLimit,
    NonFiniteValue,
    BadMagic,
    BadVersion,
    Truncated,
    TrailingBytes,
    BadUtf8,
    BadDtype,
    BadOrder,
    /// `predict_action_chunk` was called. A fixture decode is not that call.
    ProposedEntryInvoked,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointV3 {
    pub name: String,
    pub unit: String,
    pub frame: String,
    pub min: f32,
    pub max: f32,
}

/// One version-3 chunk. The noise payload is not stored here.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionChunkV3 {
    pub lerobot_commit: String,
    pub model_id: String,
    pub weight_sha256: String,
    pub stats_sha256: String,
    pub stats_key: String,
    pub action_mean_sha256: String,
    pub action_std_sha256: String,
    pub state_mean_sha256: String,
    pub state_std_sha256: String,
    pub noise_sha256: String,
    pub seed: u64,
    pub embodiment: String,
    pub t0_num_ns: u64,
    pub t0_den: u64,
    pub period_num_ns: u64,
    pub period_den: u64,
    pub n_action_steps: u32,
    pub action_dim: u32,
    pub joints: Vec<JointV3>,
    pub values: Vec<f32>,
}

/// The model call. It does not decode a fixture and it does not return actions.
pub fn predict_action_chunk() -> Result<ActionChunkV3, V3Error> {
    Err(V3Error::ProposedEntryInvoked)
}

/// Decode a version-3 canonical chunk and return SHA-256 of those bytes.
pub fn decode_action_chunk_v3(bytes: &[u8]) -> Result<(ActionChunkV3, [u8; 32]), V3Error> {
    if bytes.len() < CHUNK_MAGIC.len() || &bytes[..CHUNK_MAGIC.len()] != CHUNK_MAGIC {
        return Err(V3Error::BadMagic);
    }
    let mut i = CHUNK_MAGIC.len();
    let version = read_u32(bytes, &mut i)?;
    if version != CHUNK_VERSION_V3 {
        return Err(V3Error::BadVersion);
    }
    let lerobot_commit = read_str(bytes, &mut i)?;
    let model_id = read_str(bytes, &mut i)?;
    let weight_sha256 = read_hash(bytes, &mut i)?;
    let stats_sha256 = read_hash(bytes, &mut i)?;
    let stats_key = read_str(bytes, &mut i)?;
    let action_mean_sha256 = read_hash(bytes, &mut i)?;
    let action_std_sha256 = read_hash(bytes, &mut i)?;
    let state_mean_sha256 = read_hash(bytes, &mut i)?;
    let state_std_sha256 = read_hash(bytes, &mut i)?;
    let noise_sha256 = read_hash(bytes, &mut i)?;
    let seed = read_u64(bytes, &mut i)?;
    let embodiment = read_str(bytes, &mut i)?;
    let t0_num_ns = read_u64(bytes, &mut i)?;
    let t0_den = read_u64(bytes, &mut i)?;
    let period_num_ns = read_u64(bytes, &mut i)?;
    let period_den = read_u64(bytes, &mut i)?;
    let n_action_steps = read_u32(bytes, &mut i)?;
    let action_dim = read_u32(bytes, &mut i)?;
    if !(1..=50).contains(&n_action_steps) {
        return Err(V3Error::ActionSteps);
    }
    if action_dim != 6 {
        return Err(V3Error::ActionDim);
    }
    let joint_count = read_u32(bytes, &mut i)?;
    if joint_count != 6 || joint_count != action_dim {
        return Err(V3Error::JointCount);
    }
    let value_count = n_action_steps as usize * action_dim as usize;
    let min_tail = 6 * 20 + 2 + value_count * 4;
    if bytes.len().saturating_sub(i) < min_tail {
        return Err(V3Error::Truncated);
    }
    let mut joints = Vec::with_capacity(6);
    for _ in 0..6 {
        joints.push(JointV3 {
            name: read_str(bytes, &mut i)?,
            unit: read_str(bytes, &mut i)?,
            frame: read_str(bytes, &mut i)?,
            min: read_f32(bytes, &mut i)?,
            max: read_f32(bytes, &mut i)?,
        });
    }
    if read_u8(bytes, &mut i)? != 1 {
        return Err(V3Error::BadDtype);
    }
    if read_u8(bytes, &mut i)? != 1 {
        return Err(V3Error::BadOrder);
    }
    if bytes.len().saturating_sub(i) < value_count * 4 {
        return Err(V3Error::Truncated);
    }
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        values.push(read_f32(bytes, &mut i)?);
    }
    if i != bytes.len() {
        return Err(V3Error::TrailingBytes);
    }
    let chunk = ActionChunkV3 {
        lerobot_commit,
        model_id,
        weight_sha256,
        stats_sha256,
        stats_key,
        action_mean_sha256,
        action_std_sha256,
        state_mean_sha256,
        state_std_sha256,
        noise_sha256,
        seed,
        embodiment,
        t0_num_ns,
        t0_den,
        period_num_ns,
        period_den,
        n_action_steps,
        action_dim,
        joints,
        values,
    };
    validate_decoded(&chunk)?;
    Ok((chunk, sha256(bytes)))
}

fn validate_decoded(chunk: &ActionChunkV3) -> Result<(), V3Error> {
    if chunk.seed & 1 == 0 {
        return Err(V3Error::EvenSeed);
    }
    if chunk.lerobot_commit != LEROBOT_UPSTREAM || chunk.model_id != SMOLVLA_HUB_ID {
        return Err(V3Error::Embodiment);
    }
    if chunk.weight_sha256 != WEIGHT_SHA256 || chunk.stats_sha256 != STATS_FILE_SHA256 {
        return Err(V3Error::BadHash);
    }
    check_stats(chunk)?;
    if chunk.embodiment != chunk.stats_key {
        return Err(V3Error::Embodiment);
    }
    check_reduced(chunk.t0_num_ns, chunk.t0_den)?;
    check_reduced(chunk.period_num_ns, chunk.period_den)?;
    if chunk.period_num_ns != 100_000_000 || chunk.period_den != 3 {
        return Err(V3Error::Period);
    }
    if !(1..=50).contains(&chunk.n_action_steps) {
        return Err(V3Error::ActionSteps);
    }
    if chunk.action_dim != 6 {
        return Err(V3Error::ActionDim);
    }
    if chunk.joints.len() != chunk.action_dim as usize {
        return Err(V3Error::JointCount);
    }
    if chunk.values.len() != chunk.n_action_steps as usize * chunk.action_dim as usize {
        return Err(V3Error::ActionDim);
    }
    for (i, joint) in chunk.joints.iter().enumerate() {
        let (name, unit, min, max) = JOINTS[i];
        if joint.name != name {
            return Err(V3Error::JointName);
        }
        if joint.unit != unit {
            return Err(V3Error::JointUnit);
        }
        if joint.frame != "so100_joint" {
            return Err(V3Error::JointFrame);
        }
        if joint.min.to_bits() != min || joint.max.to_bits() != max {
            return Err(V3Error::JointLimit);
        }
    }
    for t in 0..chunk.n_action_steps {
        for d in 0..chunk.action_dim {
            let v = chunk.values[(t * chunk.action_dim + d) as usize];
            if !v.is_finite() {
                return Err(V3Error::NonFiniteValue);
            }
            let joint = &chunk.joints[d as usize];
            if v < joint.min || v > joint.max {
                return Err(V3Error::ValueOutsideLimit);
            }
        }
        step_numerator(chunk, t)?;
    }
    Ok(())
}

fn step_numerator(chunk: &ActionChunkV3, step: u32) -> Result<(), V3Error> {
    let k = u128::from(step);
    let _num = u128::from(chunk.t0_num_ns)
        .checked_mul(u128::from(chunk.period_den))
        .ok_or(V3Error::Overflow)?
        .checked_add(
            k.checked_mul(u128::from(chunk.period_num_ns))
                .ok_or(V3Error::Overflow)?
                .checked_mul(u128::from(chunk.t0_den))
                .ok_or(V3Error::Overflow)?,
        )
        .ok_or(V3Error::Overflow)?;
    let _den = u128::from(chunk.t0_den)
        .checked_mul(u128::from(chunk.period_den))
        .ok_or(V3Error::Overflow)?;
    Ok(())
}

fn check_stats(chunk: &ActionChunkV3) -> Result<(), V3Error> {
    let (mean, std) = match chunk.stats_key.as_str() {
        "so100" => (SO100_ACTION_MEAN, SO100_ACTION_STD),
        "so100-blue" => (SO100_BLUE_ACTION_MEAN, SO100_BLUE_ACTION_STD),
        "so100-red" => (SO100_RED_ACTION_MEAN, SO100_RED_ACTION_STD),
        _ => return Err(V3Error::BadStatsKey),
    };
    if chunk.action_mean_sha256 != mean || chunk.action_std_sha256 != std {
        return Err(V3Error::StatsKeyMismatch);
    }
    if chunk.state_mean_sha256 != STATE_MEAN_SHA256 || chunk.state_std_sha256 != STATE_STD_SHA256 {
        return Err(V3Error::StatsKeyMismatch);
    }
    Ok(())
}

fn check_reduced(num: u64, den: u64) -> Result<(), V3Error> {
    if den == 0 {
        return Err(V3Error::ZeroDenominator);
    }
    if gcd(num, den) != 1 {
        return Err(V3Error::Unreduced);
    }
    Ok(())
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn is_placeholder_hash(s: &str) -> bool {
    s.len() == 64 && (s.bytes().all(|c| c == b'1') || s.bytes().all(|c| c == b'2'))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn read_exact<'a>(bytes: &'a [u8], i: &mut usize, n: usize) -> Result<&'a [u8], V3Error> {
    let end = i.checked_add(n).ok_or(V3Error::Truncated)?;
    if end > bytes.len() {
        return Err(V3Error::Truncated);
    }
    let s = &bytes[*i..end];
    *i = end;
    Ok(s)
}

fn read_u8(bytes: &[u8], i: &mut usize) -> Result<u8, V3Error> {
    Ok(read_exact(bytes, i, 1)?[0])
}

fn read_u32(bytes: &[u8], i: &mut usize) -> Result<u32, V3Error> {
    let s = read_exact(bytes, i, 4)?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], i: &mut usize) -> Result<u64, V3Error> {
    let s = read_exact(bytes, i, 8)?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], i: &mut usize) -> Result<f32, V3Error> {
    let s = read_exact(bytes, i, 4)?;
    Ok(f32::from_le_bytes(s.try_into().unwrap()))
}

fn read_hash(bytes: &[u8], i: &mut usize) -> Result<String, V3Error> {
    let s = read_exact(bytes, i, 64)?;
    let text = std::str::from_utf8(s).map_err(|_| V3Error::BadUtf8)?;
    if is_placeholder_hash(text) {
        return Err(V3Error::MissingHash);
    }
    if !is_sha256(text) {
        return Err(V3Error::BadHash);
    }
    Ok(text.to_string())
}

fn read_str(bytes: &[u8], i: &mut usize) -> Result<String, V3Error> {
    let n = usize::try_from(read_u32(bytes, i)?).map_err(|_| V3Error::Truncated)?;
    let rest = bytes.len().checked_sub(*i).ok_or(V3Error::Truncated)?;
    if n > rest {
        return Err(V3Error::Truncated);
    }
    let s = read_exact(bytes, i, n)?;
    std::str::from_utf8(s)
        .map(|t| t.to_string())
        .map_err(|_| V3Error::BadUtf8)
}
