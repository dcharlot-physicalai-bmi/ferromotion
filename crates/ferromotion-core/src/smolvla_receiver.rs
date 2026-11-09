//! Independent receiver for Ferric SmolVLA codec version 2.
//!
//! The contract document and the protocol fixture were read from
//! `dcharlot-physicalai-bmi/ferric` at `3ee11c015e3478c7c4aa0409499346f79ab1c8ce`
//! (`docs/acceptance/contracts/smolvla-ferromotion.md` and
//! `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json`). The empty-horizon
//! fixture is Ferric `c05228929efc3bcb4210b475dd959370fa69893a`. Codec version 1
//! is withdrawn. This file does not copy Ferric's encoder.
//!
//! A digest that matches the fixture is a protocol check. It does not authorize
//! a rollout, load weights, or record a measurement. The entry
//! `ferric.smolvla.predict_action_chunk.v1` remains proposed.

/// Contract document pin. Not a statement that the entry has been released.
pub const FERRIC_CONTRACT_REVISION: &str = "3ee11c015e3478c7c4aa0409499346f79ab1c8ce";
pub const PROPOSED_FERRIC_ENTRY: &str = "ferric.smolvla.predict_action_chunk.v1";
pub const PREDICT_ACTION_CHUNK: &str = "predict_action_chunk";
pub const SELECT_ACTION_IS_NOT_THE_ENTRY: &str = "select_action";
pub const LEROBOT_UPSTREAM: &str = "a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a";
pub const SMOLVLA_HUB_ID: &str = "lerobot/smolvla_base";
pub const DIGEST_DOMAIN: &str = "ferric.smolvla.action-chunk";
pub const CODEC_VERSION: u32 = 2;
pub const CODEC_V1_WITHDRAWN: bool = true;
pub const CHUNK_SIZE: u32 = 50;
pub const STATE_WIDTH: usize = 32;
pub const TOKEN_MAX: usize = 48;
pub const PRE_UNPAD_STEPS: usize = 50;
pub const PRE_UNPAD_WIDTH: usize = 32;

pub const PREPROCESS_STEPS: &[&str] = &[
    "rename",
    "batch",
    "trailing newline",
    "tokenize right/longest max 48",
    "device",
    "normalize",
];

pub const IMAGE_SIDE: usize = 512;
pub const IMAGE_ALIGN_CORNERS: bool = false;
pub const IMAGE_PAD_LEFT: usize = 0;
pub const IMAGE_PAD_TOP: usize = 0;
pub const IMAGE_INTERPOLATION: &str = "bilinear";

/// Golden protocol digest from the pinned nonempty fixture. Not a model output.
pub const GOLDEN_DIGEST: &str = "6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86";

/// Ferric commit that published the empty-horizon fixture. Not an interoperability claim.
pub const FERRIC_EMPTY_FIXTURE_REVISION: &str = "c05228929efc3bcb4210b475dd959370fa69893a";

/// Digest of `fixtures/smolvla-empty-chunk-v2.json` canonical bytes. Not a model output.
pub const EMPTY_GOLDEN_DIGEST: &str = "6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e";

const MAGIC: &[u8] = b"FERRIC-SMOLVLA-CHUNK\0";

#[derive(Clone, Debug, PartialEq)]
pub struct JointSpec {
    pub name: String,
    pub unit: String,
    pub frame: String,
    pub min: f32,
    pub max: f32,
}

/// One action chunk in dataset-native units. No fps conversion is applied.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkV2 {
    pub lerobot_commit: String,
    pub model_id: String,
    pub weight_sha256: String,
    pub stats_sha256: String,
    pub noise_sha256: String,
    pub embodiment: String,
    pub t0_ns: u64,
    pub step_duration_ns: u64,
    pub n_action_steps: u32,
    pub action_dim: u32,
    pub joints: Vec<JointSpec>,
    pub values_time_major: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecError {
    MissingNoiseProvenance,
    MissingWeightProvenance,
    MissingStatsProvenance,
    BadProvenance,
    EmptyEmbodiment,
    EmptyJointField,
    JointCount,
    JointLimits,
    Horizon,
    /// Action index is not in `0..steps`. An empty chunk has no index 0.
    TimestampIndex { index: u32, steps: u32 },
    ActionDim,
    ChunkShape,
    NonFinite,
    ValueOutOfRange,
    StepDuration,
    TimestampOverflow,
    /// Encode succeeded. The entry is still proposed, so rollout stays closed.
    EntryProposed,
    AlohaDelta,
    TokenLength,
    PixelRange,
    StateTooWide,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlReject {
    Codec(CodecError),
    PhysicalActuation,
    SelectAction,
}

/// Protocol conformance is not inference fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceFidelity {
    NotExecuted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolReceipt {
    pub digest_sha256: String,
    pub matches_golden_fixture: bool,
    pub rollout_authorized: bool,
    pub model_executed: bool,
    pub inference: InferenceFidelity,
    pub measured_receipts: u32,
}

pub fn accepts_entry(name: &str) -> bool {
    name == PROPOSED_FERRIC_ENTRY || name == PREDICT_ACTION_CHUNK
}

pub fn normalize_forward(x: f32, mean: f32, std: f32) -> f32 {
    (x - mean) / (std + 1e-8)
}

/// `y * std + mean`. The forward epsilon is not applied again.
pub fn normalize_inverse(y: f32, mean: f32, std: f32) -> f32 {
    y * std + mean
}

pub fn pad_state(state: &[f32]) -> Result<[f32; STATE_WIDTH], CodecError> {
    if state.len() > STATE_WIDTH {
        return Err(CodecError::StateTooWide);
    }
    let mut out = [0.0f32; STATE_WIDTH];
    out[..state.len()].copy_from_slice(state);
    Ok(out)
}

pub fn refuse_aloha_delta(enabled: bool) -> Result<(), CodecError> {
    if enabled { Err(CodecError::AlohaDelta) } else { Ok(()) }
}

pub fn refuse_token_len(len: usize) -> Result<(), CodecError> {
    if len > TOKEN_MAX { Err(CodecError::TokenLength) } else { Ok(()) }
}

pub fn refuse_pixel(pixel: f32) -> Result<(), CodecError> {
    if pixel.is_finite() && (0.0..=1.0).contains(&pixel) { Ok(()) } else { Err(CodecError::PixelRange) }
}

/// `n_action_steps > chunk_size` only. Zero is inside the helper.
/// This does not decide which version-2 payloads encode.
pub fn check_horizon(n_action_steps: u32, chunk_size: u32) -> Result<(), CodecError> {
    if n_action_steps > chunk_size {
        Err(CodecError::Horizon)
    } else {
        Ok(())
    }
}

/// Shape of the value matrix. Provenance and joints are not this layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionChunk {
    pub n_action_steps: u32,
    pub action_dim: u32,
    pub values: Vec<f32>,
}

impl ActionChunk {
    /// `try_new(0, 1, [])` is a valid empty matrix. `action_dim == 0` is not.
    pub fn try_new(n_action_steps: u32, action_dim: u32, values: Vec<f32>) -> Result<Self, CodecError> {
        check_horizon(n_action_steps, CHUNK_SIZE)?;
        if action_dim == 0 || action_dim > PRE_UNPAD_WIDTH as u32 {
            return Err(CodecError::ActionDim);
        }
        let expect = (n_action_steps as usize).checked_mul(action_dim as usize).ok_or(CodecError::ChunkShape)?;
        if values.len() != expect {
            return Err(CodecError::ChunkShape);
        }
        Ok(ActionChunk { n_action_steps, action_dim, values })
    }
}

/// Timestamp of action index `index`. Zero duration is refused before the index check.
/// `index >= steps` is [`CodecError::TimestampIndex`], including index 0 of an empty chunk.
/// No fps is read and no model runs.
pub fn action_timestamp_ns(t0_ns: u64, step_duration_ns: u64, index: u32, steps: u32) -> Result<u64, CodecError> {
    if step_duration_ns == 0 {
        return Err(CodecError::StepDuration);
    }
    if index >= steps {
        return Err(CodecError::TimestampIndex { index, steps });
    }
    let delta = u64::from(index).checked_mul(step_duration_ns).ok_or(CodecError::TimestampOverflow)?;
    t0_ns.checked_add(delta).ok_or(CodecError::TimestampOverflow)
}

fn provenance(field: &str, empty: CodecError) -> Result<(), CodecError> {
    if field.is_empty() {
        return Err(empty);
    }
    if field.len() == 64 && field.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        Ok(())
    } else {
        Err(CodecError::BadProvenance)
    }
}

fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_str(buf: &mut Vec<u8>, value: &str) -> Result<(), CodecError> {
    let len = u32::try_from(value.len()).map_err(|_| CodecError::ChunkShape)?;
    push_u32(buf, len);
    buf.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_f32(buf: &mut Vec<u8>, value: f32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Little-endian canonical bytes. The SHA-256 is not appended.
pub fn encode(chunk: &ChunkV2) -> Result<Vec<u8>, CodecError> {
    provenance(&chunk.weight_sha256, CodecError::MissingWeightProvenance)?;
    provenance(&chunk.stats_sha256, CodecError::MissingStatsProvenance)?;
    provenance(&chunk.noise_sha256, CodecError::MissingNoiseProvenance)?;
    if chunk.embodiment.is_empty() {
        return Err(CodecError::EmptyEmbodiment);
    }
    check_horizon(chunk.n_action_steps, CHUNK_SIZE)?;
    if chunk.action_dim == 0 || chunk.action_dim > PRE_UNPAD_WIDTH as u32 {
        return Err(CodecError::ActionDim);
    }
    if chunk.joints.len() != chunk.action_dim as usize {
        return Err(CodecError::JointCount);
    }
    if chunk.step_duration_ns == 0 {
        return Err(CodecError::StepDuration);
    }
    let width = chunk.action_dim as usize;
    let rows = chunk.n_action_steps as usize;
    let expect = rows.checked_mul(width).ok_or(CodecError::ChunkShape)?;
    if chunk.values_time_major.len() != expect {
        return Err(CodecError::ChunkShape);
    }
    for k in 0..chunk.n_action_steps {
        action_timestamp_ns(chunk.t0_ns, chunk.step_duration_ns, k, chunk.n_action_steps)?;
    }
    for joint in &chunk.joints {
        if joint.name.is_empty() || joint.unit.is_empty() || joint.frame.is_empty() {
            return Err(CodecError::EmptyJointField);
        }
        if !joint.min.is_finite() || !joint.max.is_finite() || joint.min >= joint.max {
            return Err(CodecError::JointLimits);
        }
    }
    for (index, value) in chunk.values_time_major.iter().enumerate() {
        if !value.is_finite() {
            return Err(CodecError::NonFinite);
        }
        let joint = &chunk.joints[index % width];
        if *value < joint.min || *value > joint.max {
            return Err(CodecError::ValueOutOfRange);
        }
    }

    let mut buf = Vec::with_capacity(64 + chunk.values_time_major.len() * 4);
    buf.extend_from_slice(MAGIC);
    push_u32(&mut buf, CODEC_VERSION);
    push_str(&mut buf, &chunk.lerobot_commit)?;
    push_str(&mut buf, &chunk.model_id)?;
    buf.extend_from_slice(chunk.weight_sha256.as_bytes());
    buf.extend_from_slice(chunk.stats_sha256.as_bytes());
    buf.extend_from_slice(chunk.noise_sha256.as_bytes());
    push_str(&mut buf, &chunk.embodiment)?;
    push_u64(&mut buf, chunk.t0_ns);
    push_u64(&mut buf, chunk.step_duration_ns);
    push_u32(&mut buf, chunk.n_action_steps);
    push_u32(&mut buf, chunk.action_dim);
    push_u32(&mut buf, chunk.joints.len() as u32);
    for joint in &chunk.joints {
        push_str(&mut buf, &joint.name)?;
        push_str(&mut buf, &joint.unit)?;
        push_str(&mut buf, &joint.frame)?;
        push_f32(&mut buf, joint.min);
        push_f32(&mut buf, joint.max);
    }
    buf.push(1);
    buf.push(1);
    for value in &chunk.values_time_major {
        push_f32(&mut buf, *value);
    }
    Ok(buf)
}

pub fn digest_sha256(bytes: &[u8]) -> String {
    sha256(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Same checks as [`encode`], then [`CodecError::EntryProposed`].
/// An empty horizon is not a second refusal while the entry is proposed.
/// An empty-horizon execution gate, if one is added, runs only after this gate clears.
/// A golden digest does not authorize a rollout.
pub fn authorize_rollout(chunk: &ChunkV2) -> Result<(), CodecError> {
    encode(chunk)?;
    Err(CodecError::EntryProposed)
}

/// Read a version-2 body. This is the byte layout, not the authorization gate.
pub fn parse_chunk(bytes: &[u8]) -> Result<ChunkV2, CodecError> {
    let mut cur = Cursor { bytes, at: 0 };
    let magic = cur.take(MAGIC.len())?;
    if magic != MAGIC {
        return Err(CodecError::ChunkShape);
    }
    if cur.u32()? != CODEC_VERSION {
        return Err(CodecError::ChunkShape);
    }
    let chunk = ChunkV2 {
        lerobot_commit: cur.string()?,
        model_id: cur.string()?,
        weight_sha256: cur.hex64()?,
        stats_sha256: cur.hex64()?,
        noise_sha256: cur.hex64()?,
        embodiment: cur.string()?,
        t0_ns: cur.u64()?,
        step_duration_ns: cur.u64()?,
        n_action_steps: cur.u32()?,
        action_dim: cur.u32()?,
        joints: {
            let n = cur.u32()? as usize;
            let mut joints = Vec::with_capacity(n);
            for _ in 0..n {
                joints.push(JointSpec {
                    name: cur.string()?,
                    unit: cur.string()?,
                    frame: cur.string()?,
                    min: cur.f32()?,
                    max: cur.f32()?,
                });
            }
            joints
        },
        values_time_major: {
            let flags = cur.take(2)?;
            if flags != [1, 1] {
                return Err(CodecError::ChunkShape);
            }
            let rest = cur.bytes.len() - cur.at;
            if !rest.is_multiple_of(4) {
                return Err(CodecError::ChunkShape);
            }
            let n = rest / 4;
            let mut values = Vec::with_capacity(n);
            for _ in 0..n {
                values.push(cur.f32()?);
            }
            values
        },
    };
    if cur.at != cur.bytes.len() {
        return Err(CodecError::ChunkShape);
    }
    Ok(chunk)
}

/// Parse, then require [`encode`] to reproduce the same bytes.
pub fn verify_chunk(bytes: &[u8]) -> Result<ChunkV2, CodecError> {
    let chunk = parse_chunk(bytes)?;
    if encode(&chunk)? != bytes {
        return Err(CodecError::ChunkShape);
    }
    Ok(chunk)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        let end = self.at.checked_add(n).ok_or(CodecError::ChunkShape)?;
        let out = self.bytes.get(self.at..end).ok_or(CodecError::ChunkShape)?;
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn string(&mut self) -> Result<String, CodecError> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| CodecError::ChunkShape)
    }

    fn hex64(&mut self) -> Result<String, CodecError> {
        let bytes = self.take(64)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| CodecError::BadProvenance)
    }
}

/// The entry stays proposed, so rollout stays unauthorized even after a digest match.
pub fn protocol_receipt(chunk: &ChunkV2, reference_digest: &str) -> Result<ProtocolReceipt, CodecError> {
    let bytes = encode(chunk)?;
    let digest = digest_sha256(&bytes);
    let _ = authorize_rollout(chunk);
    Ok(ProtocolReceipt {
        matches_golden_fixture: digest == reference_digest,
        digest_sha256: digest,
        rollout_authorized: false,
        model_executed: false,
        inference: InferenceFidelity::NotExecuted,
        measured_receipts: 0,
    })
}

pub fn accept_for_control(chunk: &ChunkV2, entry: &str, physical_actuation: bool, reference_digest: &str) -> Result<ProtocolReceipt, ControlReject> {
    if physical_actuation {
        return Err(ControlReject::PhysicalActuation);
    }
    if !accepts_entry(entry) {
        return Err(ControlReject::SelectAction);
    }
    protocol_receipt(chunk, reference_digest).map_err(ControlReject::Codec)
}

fn sha256(message: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state = [
        0x6a09e667_u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (message.len() as u64).saturating_mul(8);
    let mut data = Vec::with_capacity(message.len() + 72);
    data.extend_from_slice(message);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let (blocks, _) = data.as_chunks::<64>();
    for block in blocks {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) =
            (state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(add);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
fn protocol_fixture() -> ChunkV2 {
    ChunkV2 {
        lerobot_commit: LEROBOT_UPSTREAM.into(),
        model_id: SMOLVLA_HUB_ID.into(),
        weight_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        stats_sha256: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into(),
        noise_sha256: "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff".into(),
        embodiment: "protocol-test-embodiment".into(),
        t0_ns: 1_000_000_000,
        step_duration_ns: 20_000_000,
        n_action_steps: 2,
        action_dim: 2,
        joints: vec![
            JointSpec { name: "j0".into(), unit: "dataset_label_a".into(), frame: "embodiment_joint".into(), min: -1.0, max: 1.0 },
            JointSpec { name: "j1".into(), unit: "dataset_label_b".into(), frame: "embodiment_joint".into(), min: -1.0, max: 1.0 },
        ],
        values_time_major: vec![0.25, 0.125, -0.5, -0.125],
    }
}

#[cfg(test)]
fn json_string(file: &str, key: &str) -> String {
    let pat = format!("\"{key}\": \"");
    let start = file.find(&pat).unwrap_or_else(|| panic!("missing {key}")) + pat.len();
    let end = file[start..].find('"').unwrap();
    file[start..start + end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json");

    #[test]
    fn sha256_of_abc_matches_the_known_vector() {
        assert_eq!(
            digest_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn pinned_fixture_bytes_match_the_golden_digest_and_do_not_authorize_a_rollout() {
        assert!(FIXTURE.contains("\"kind\": \"protocol_test\""));
        assert!(FIXTURE.contains("\"not_model_evidence\": true"));
        assert!(FIXTURE.contains("\"not_rollout_evidence\": true"));
        assert!(FIXTURE.contains("\"not_energy_evidence\": true"));
        let canonical = json_string(FIXTURE, "canonical_hex");
        let digest = json_string(FIXTURE, "digest_sha256");
        assert_eq!(digest, GOLDEN_DIGEST);
        let bytes = encode(&protocol_fixture()).unwrap();
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, canonical);
        assert_eq!(digest_sha256(&bytes), GOLDEN_DIGEST);
        let dt = 20_000_000_f64 * 1e-9;
        let mut clock = crate::ferromorphic_clock::BoundaryClock::new(4, dt).unwrap();
        clock.episode_start(1_000_000_000, 20_000_000).unwrap();
        assert_eq!(clock.epoch, 1);
        assert_eq!(clock.dt_bits, dt.to_bits());
        let envelope = crate::ferromorphic_clock::EpisodeEnvelope::try_new(
            crate::ferromorphic_clock::EPISODE_ENVELOPE_VERSION,
            clock.epoch,
            dt.to_bits(),
            20_000_000,
            1_000_000_000,
        )
        .unwrap();
        assert_eq!(
            crate::ferromorphic_clock::bind_v2_timestamps(1_000_000_000, 20_000_000, 2, Some(envelope), &clock),
            Ok(())
        );
        assert_eq!(encode(&protocol_fixture()).unwrap(), bytes);
        let receipt = protocol_receipt(&protocol_fixture(), GOLDEN_DIGEST).unwrap();
        assert!(receipt.matches_golden_fixture);
        assert!(!receipt.rollout_authorized);
        assert!(!receipt.model_executed);
        assert_eq!(receipt.inference, InferenceFidelity::NotExecuted);
        assert_eq!(receipt.measured_receipts, 0);
        assert_eq!(authorize_rollout(&protocol_fixture()), Err(CodecError::EntryProposed));
        const { assert!(CODEC_V1_WITHDRAWN); }
        assert_eq!(DIGEST_DOMAIN, "ferric.smolvla.action-chunk");
        assert_eq!(FERRIC_CONTRACT_REVISION.len(), 40);
    }

    #[test]
    fn empty_noise_is_missing_noise_provenance_and_incomplete_hashes_refuse() {
        let mut chunk = protocol_fixture();
        chunk.noise_sha256.clear();
        assert_eq!(encode(&chunk), Err(CodecError::MissingNoiseProvenance));
        chunk.noise_sha256 = "zz".into();
        assert_eq!(encode(&chunk), Err(CodecError::BadProvenance));
        chunk.noise_sha256 = protocol_fixture().noise_sha256;
        chunk.weight_sha256.clear();
        assert_eq!(encode(&chunk), Err(CodecError::MissingWeightProvenance));
        chunk.weight_sha256 = "ABCDEF".into();
        assert_eq!(encode(&chunk), Err(CodecError::BadProvenance));
        chunk.weight_sha256 = protocol_fixture().weight_sha256;
        chunk.stats_sha256.clear();
        assert_eq!(encode(&chunk), Err(CodecError::MissingStatsProvenance));
    }

    #[test]
    fn mutated_shape_limits_and_time_are_refused() {
        let mut chunk = protocol_fixture();
        chunk.n_action_steps = 51;
        assert_eq!(encode(&chunk), Err(CodecError::Horizon));
        chunk.n_action_steps = 0;
        assert_eq!(encode(&chunk), Err(CodecError::ChunkShape));
        chunk.n_action_steps = 2;
        chunk.action_dim = 0;
        assert_eq!(encode(&chunk), Err(CodecError::ActionDim));
        chunk.action_dim = 33;
        assert_eq!(encode(&chunk), Err(CodecError::ActionDim));
        chunk.action_dim = 2;
        chunk.joints.pop();
        assert_eq!(encode(&chunk), Err(CodecError::JointCount));
        chunk.joints = protocol_fixture().joints;
        chunk.joints[0].name.clear();
        assert_eq!(encode(&chunk), Err(CodecError::EmptyJointField));
        chunk.joints[0].name = "j0".into();
        chunk.joints[0].min = 1.0;
        assert_eq!(encode(&chunk), Err(CodecError::JointLimits));
        chunk.joints[0].min = -1.0;
        chunk.values_time_major[0] = 2.0;
        assert_eq!(encode(&chunk), Err(CodecError::ValueOutOfRange));
        chunk.values_time_major[0] = f32::NAN;
        assert_eq!(encode(&chunk), Err(CodecError::NonFinite));
        chunk.values_time_major[0] = 0.25;
        chunk.step_duration_ns = 0;
        assert_eq!(encode(&chunk), Err(CodecError::StepDuration));
        chunk.step_duration_ns = u64::MAX;
        chunk.t0_ns = 1;
        assert_eq!(encode(&chunk), Err(CodecError::TimestampOverflow));
        chunk.embodiment.clear();
        chunk.step_duration_ns = 20_000_000;
        chunk.t0_ns = 1_000_000_000;
        assert_eq!(encode(&chunk), Err(CodecError::EmptyEmbodiment));
        let mut flipped = encode(&protocol_fixture()).unwrap();
        flipped[0] ^= 0x01;
        assert_ne!(digest_sha256(&flipped), GOLDEN_DIGEST);
    }

    #[test]
    fn timestamps_are_checked_and_a_digest_match_still_refuses_actuation_and_select_action() {
        assert_eq!(action_timestamp_ns(1_000_000_000, 20_000_000, 0, 2), Ok(1_000_000_000));
        assert_eq!(action_timestamp_ns(1_000_000_000, 20_000_000, 1, 2), Ok(1_020_000_000));
        assert_eq!(action_timestamp_ns(1, 0, 0, 1), Err(CodecError::StepDuration));
        assert!(FIXTURE.contains("1000000000") && FIXTURE.contains("1020000000"));
        let got = accept_for_control(&protocol_fixture(), PROPOSED_FERRIC_ENTRY, false, GOLDEN_DIGEST).unwrap();
        assert!(got.matches_golden_fixture && !got.rollout_authorized && !got.model_executed);
        assert_eq!(
            accept_for_control(&protocol_fixture(), PROPOSED_FERRIC_ENTRY, true, GOLDEN_DIGEST),
            Err(ControlReject::PhysicalActuation)
        );
        assert_eq!(
            accept_for_control(&protocol_fixture(), SELECT_ACTION_IS_NOT_THE_ENTRY, false, GOLDEN_DIGEST),
            Err(ControlReject::SelectAction)
        );
        assert!(accepts_entry(PREDICT_ACTION_CHUNK));
    }

    #[test]
    fn chunk_index_does_not_become_a_ferromorphic_tick_without_an_epoch() {
        assert_eq!(
            crate::ferromorphic_clock::map_chunk_index_to_tick(1, false, 0),
            Ok(crate::ferromorphic_clock::ChunkTickMap::EpochUnresolved)
        );
        assert_eq!(
            crate::ferromorphic_clock::map_chunk_index_to_tick(1, true, 0),
            Ok(crate::ferromorphic_clock::ChunkTickMap::Mapped { tick: 1, spikes_visible_at: 2 })
        );
        assert_eq!(
            crate::ferromorphic_clock::map_chunk_index_to_tick(u64::MAX, true, 0),
            Err(crate::ferromorphic_clock::TimebaseError::Overflow)
        );
        const { assert!(!crate::ferromorphic_clock::JOINT_LOOP_EXECUTED); }
        assert!(crate::ferromorphic_clock::UNRESOLVED.contains(&crate::ferromorphic_clock::UnresolvedPolicy::AccumulatorRemainsInsideDynamicsLab));
        assert_eq!(crate::ferromorphic_clock::TICK_ARITHMETIC, "checked_u64_fail_closed");
        const { assert!(!crate::ferromorphic_clock::EPISODE_ENVELOPE_INTEROPERABLE); }
        const { assert!(!crate::ferromorphic_clock::FERROMORPHIC_BIND_VERIFIED); }
    }

    #[test]
    fn normalization_pad_and_bounds_follow_the_contract_and_are_not_inference() {
        let y = normalize_forward(3.0, 1.0, 2.0);
        assert_eq!(y, (3.0 - 1.0) / (2.0 + 1e-8));
        assert_eq!(normalize_inverse(y, 1.0, 2.0), y * 2.0 + 1.0);
        let through_zero_std = normalize_inverse(normalize_forward(3.0, 1.0, 0.0), 1.0, 0.0);
        assert_eq!(through_zero_std, 1.0);
        assert_ne!(through_zero_std, 3.0);
        let padded = pad_state(&[0.5]).unwrap();
        assert_eq!(padded[0], 0.5);
        assert!(padded[1..].iter().all(|v| *v == 0.0));
        assert_eq!(pad_state(&[0.0; 33]), Err(CodecError::StateTooWide));
        assert_eq!(refuse_aloha_delta(true), Err(CodecError::AlohaDelta));
        assert_eq!(refuse_token_len(49), Err(CodecError::TokenLength));
        assert_eq!(refuse_pixel(1.1), Err(CodecError::PixelRange));
        assert_eq!(refuse_pixel(0.0), Ok(()));
        const {
            assert!(!IMAGE_ALIGN_CORNERS);
            assert!(IMAGE_PAD_LEFT == 0 && IMAGE_PAD_TOP == 0 && IMAGE_SIDE == 512);
            assert!(IMAGE_INTERPOLATION.len() == 8);
            assert!(PRE_UNPAD_STEPS == 50 && PRE_UNPAD_WIDTH == 32);
        }
        assert_eq!(PREPROCESS_STEPS[5], "normalize");
    }

    fn empty_protocol_chunk() -> ChunkV2 {
        let mut chunk = protocol_fixture();
        chunk.n_action_steps = 0;
        chunk.values_time_major.clear();
        chunk
    }

    fn decode_hex(hex: &str) -> Vec<u8> {
        (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
    }

    /// Nonempty golden stays the published v2 fixture. The empty fixture shares its identity.
    #[test]
    fn nonempty_golden_is_unchanged_and_matches_the_empty_identity() {
        let full = protocol_fixture();
        let empty = empty_protocol_chunk();
        assert_eq!(full.n_action_steps, 2);
        assert_eq!(full.values_time_major.len(), 4);
        assert_eq!(digest_sha256(&encode(&full).unwrap()), GOLDEN_DIGEST);
        assert_eq!(empty.action_dim, full.action_dim);
        assert_eq!(empty.joints, full.joints);
        assert_eq!(empty.model_id, full.model_id);
        assert_eq!(empty.weight_sha256, full.weight_sha256);
        assert_eq!(empty.stats_sha256, full.stats_sha256);
        assert_eq!(empty.noise_sha256, full.noise_sha256);
        assert_eq!(empty.embodiment, full.embodiment);
        assert_eq!(empty.t0_ns, full.t0_ns);
        assert_eq!(empty.step_duration_ns, full.step_duration_ns);
        assert_eq!(empty.n_action_steps, 0);
        assert!(empty.values_time_major.is_empty());
        assert_ne!(encode(&empty).unwrap(), encode(&full).unwrap());
    }

    /// Published empty fixture at Ferric `c052289`. This encoder recomputes the bytes.
    /// Ferric names the authorization error `ProposedEntryInvoked`; this receiver's
    /// corresponding refusal is `EntryProposed`. Neither test runs a model.
    #[test]
    fn published_empty_horizon_matches_canonical_bytes_and_refuses_index_zero() {
        const EMPTY: &str = include_str!("../../../docs/acceptance/contracts/fixtures/smolvla-empty-chunk-v2.json");
        assert!(EMPTY.contains("\"kind\": \"protocol_test\""));
        assert!(EMPTY.contains("\"not_model_evidence\": true"));
        assert!(EMPTY.contains("\"not_rollout_evidence\": true"));
        assert!(EMPTY.contains("\"not_energy_evidence\": true"));
        assert!(EMPTY.contains("\"model_status\": \"proposed\""));
        assert!(EMPTY.contains("\"n_action_steps\": 0"));
        assert!(EMPTY.contains("\"action_dim\": 2"));
        assert!(EMPTY.contains("\"values_time_major\": []"));
        assert!(EMPTY.contains("\"execution_error\": \"ProposedEntryInvoked\""));
        assert!(EMPTY.contains("\"timestamp_index_zero\": \"TimestampIndex\""));
        assert_eq!(json_string(EMPTY, "digest_sha256"), EMPTY_GOLDEN_DIGEST);
        assert_eq!(check_horizon(0, 50), Ok(()));
        assert_eq!(check_horizon(0, 0), Ok(()));
        let shape = ActionChunk::try_new(0, 2, Vec::new()).unwrap();
        let mut chunk = empty_protocol_chunk();
        assert_eq!(shape.n_action_steps, chunk.n_action_steps);
        assert_eq!(shape.action_dim, chunk.action_dim);
        chunk.values_time_major = shape.values;
        let bytes = encode(&chunk).unwrap();
        let published = decode_hex(&json_string(EMPTY, "canonical_hex"));
        assert_eq!(bytes, published);
        assert_eq!(bytes.len(), 449);
        assert_eq!(digest_sha256(&bytes), EMPTY_GOLDEN_DIGEST);
        assert_eq!(verify_chunk(&bytes).unwrap(), chunk);
        assert_eq!(
            action_timestamp_ns(chunk.t0_ns, chunk.step_duration_ns, 0, 0),
            Err(CodecError::TimestampIndex { index: 0, steps: 0 })
        );
        let receipt = protocol_receipt(&chunk, EMPTY_GOLDEN_DIGEST).unwrap();
        assert!(receipt.matches_golden_fixture);
        assert!(!receipt.rollout_authorized);
        assert!(!receipt.model_executed);
        assert_eq!(receipt.inference, InferenceFidelity::NotExecuted);
        assert_eq!(receipt.measured_receipts, 0);
        assert_eq!(authorize_rollout(&chunk), Err(CodecError::EntryProposed));
        assert_eq!(FERRIC_EMPTY_FIXTURE_REVISION.len(), 40);
    }

    #[test]
    fn empty_horizon_negative_cases_stay_on_their_own_layers() {
        let chunk = empty_protocol_chunk();
        let mut bad = chunk.clone();
        bad.weight_sha256.clear();
        assert_eq!(encode(&bad), Err(CodecError::MissingWeightProvenance));
        assert_eq!(authorize_rollout(&bad), Err(CodecError::MissingWeightProvenance));
        bad = chunk.clone();
        bad.stats_sha256.clear();
        assert_eq!(encode(&bad), Err(CodecError::MissingStatsProvenance));
        bad = chunk.clone();
        bad.noise_sha256.clear();
        assert_eq!(encode(&bad), Err(CodecError::MissingNoiseProvenance));
        assert_eq!(authorize_rollout(&bad), Err(CodecError::MissingNoiseProvenance));
        assert_eq!(ActionChunk::try_new(0, chunk.action_dim, vec![0.0]), Err(CodecError::ChunkShape));
        bad = chunk.clone();
        bad.step_duration_ns = 0;
        assert_eq!(encode(&bad), Err(CodecError::StepDuration));
        assert_eq!(ActionChunk::try_new(0, 0, Vec::new()), Err(CodecError::ActionDim));
        assert_eq!(check_horizon(51, 50), Err(CodecError::Horizon));
    }
}
