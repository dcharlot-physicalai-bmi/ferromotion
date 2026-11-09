//! Per-call-site contraction. The published default for every site is [`Contraction::Contracted`].
//!
//! A site may name another contraction only in `docs/acceptance/oracles/manifest.json` after a
//! matched oracle reproduces that choice. No such selection is in force. Arm64 golden flags are
//! unproven provenance.

/// How one call site rounds a product-sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contraction {
    /// Published `mul_add` chain. This is the build default.
    Contracted,
    /// Each product rounds, then the sums follow the SSE schedule in
    /// `mju_mulQuat` of libmujoco `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
    /// (`mulpd`/`addpd`/`subpd`, no FMA). Not the published default.
    Unfused,
}

/// `mul_quat` build selection. Contracted until a manifest entry changes it.
pub const MUL_QUAT: Contraction = Contraction::Contracted;

/// Named source oracle. Selecting it is explicit. The host architecture does not select it.
pub const ORACLE_X86_CONTRACT_OFF_ID: &str = "mujoco-3.13.0-x86_64-contract-off";
pub const ORACLE_X86_CONTRACT_OFF_SHA256: &str = "64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc";

/// Active contraction for the three opt-in sites. The published default leaves every site contracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildPolicy {
    pub id: &'static str,
    pub artifact_sha256: &'static str,
    pub mul_quat: Contraction,
    pub rot_vec_quat: Contraction,
    pub mul_mat_vec3: Contraction,
}

pub const DEFAULT_POLICY: BuildPolicy = BuildPolicy {
    id: "published-default",
    artifact_sha256: "published-default",
    mul_quat: Contraction::Contracted,
    rot_vec_quat: Contraction::Contracted,
    mul_mat_vec3: Contraction::Contracted,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError {
    UnknownIdentity,
    InconsistentIdentity,
    UnknownSite,
}

use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<BuildPolicy> = const { Cell::new(DEFAULT_POLICY) };
}

pub fn active_policy() -> BuildPolicy {
    ACTIVE.with(|slot| slot.get())
}

pub fn reset_policy() {
    ACTIVE.with(|slot| slot.set(DEFAULT_POLICY));
}

/// Install a named policy. A sha that does not match the id is refused. An unknown site is refused.
/// Neither case falls back to the default.
pub fn install_policy(id: &str, artifact_sha256: &str, sites: &str) -> Result<BuildPolicy, PolicyError> {
    let mut policy = match id {
        "published-default" => {
            if artifact_sha256 != "published-default" || !sites.is_empty() {
                return Err(PolicyError::InconsistentIdentity);
            }
            DEFAULT_POLICY
        }
        ORACLE_X86_CONTRACT_OFF_ID => {
            if artifact_sha256 != ORACLE_X86_CONTRACT_OFF_SHA256 {
                return Err(PolicyError::InconsistentIdentity);
            }
            BuildPolicy {
                id: ORACLE_X86_CONTRACT_OFF_ID,
                artifact_sha256: ORACLE_X86_CONTRACT_OFF_SHA256,
                // `mju_mulQuat` in that artifact is the unfused SSE schedule.
                // `rot_vec_quat` stays contracted until the site is named.
                mul_quat: Contraction::Unfused,
                rot_vec_quat: Contraction::Contracted,
                mul_mat_vec3: Contraction::Contracted,
            }
        }
        _ => return Err(PolicyError::UnknownIdentity),
    };
    if id == ORACLE_X86_CONTRACT_OFF_ID {
        for site in sites.split(',').filter(|site| !site.is_empty()) {
            match site {
                "mul_quat" => policy.mul_quat = Contraction::Unfused,
                "rot_vec_quat" => policy.rot_vec_quat = Contraction::Unfused,
                // Reverted after the post-mesh remeasure: unfusing this site moved
                // `qacc` from 24/636 back to 23/636. The formula stays in
                // `mul_mat_vec3_with` for the vector file. It is not selectable.
                "mul_mat_vec3" => return Err(PolicyError::UnknownSite),
                _ => return Err(PolicyError::UnknownSite),
            }
        }
    }
    ACTIVE.with(|slot| slot.set(policy));
    Ok(policy)
}

pub fn install_policy_from_env() -> Result<BuildPolicy, PolicyError> {
    let id = std::env::var("FERROMOTION_BUILD_POLICY").unwrap_or_else(|_| "published-default".to_string());
    let sha = std::env::var("FERROMOTION_ORACLE_SHA256").unwrap_or_else(|_| {
        if id == "published-default" { "published-default".to_string() } else { String::new() }
    });
    let sites = std::env::var("FERROMOTION_CONTRACT_SITES").unwrap_or_default();
    install_policy(&id, &sha, &sites)
}

pub fn policy_receipt(policy: BuildPolicy) -> String {
    format!(
        "policy id={} sha256={} mul_quat={:?} rot_vec_quat={:?} mul_mat_vec3={:?}",
        policy.id, policy.artifact_sha256, policy.mul_quat, policy.rot_vec_quat, policy.mul_mat_vec3
    )
}

pub fn mul_quat(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let out = mul_quat_with(active_policy().mul_quat, a, b);
    let mut inputs = [0.0; 8];
    inputs[..4].copy_from_slice(a);
    inputs[4..].copy_from_slice(b);
    record_op("mul_quat", &inputs, &out);
    out
}

/// One executed call. Present only while [`capture_ops`] is running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSample {
    pub op: &'static str,
    pub inputs: Vec<u64>,
    pub outputs: Vec<u64>,
}

use std::cell::RefCell;

thread_local! {
    static OP_TRACE: RefCell<Option<Vec<OpSample>>> = const { RefCell::new(None) };
}

/// Record `op` when a capture is active. A capture that is off does not store the call.
pub fn record_op(op: &'static str, inputs: &[f64], outputs: &[f64]) {
    OP_TRACE.with(|slot| {
        let mut guard = slot.borrow_mut();
        if let Some(buf) = guard.as_mut() {
            buf.push(OpSample {
                op,
                inputs: inputs.iter().copied().map(f64::to_bits).collect(),
                outputs: outputs.iter().copied().map(f64::to_bits).collect(),
            });
        }
    });
}

/// Run `f` and return the instrumented calls it made, in order.
pub fn capture_ops<R>(f: impl FnOnce() -> R) -> (R, Vec<OpSample>) {
    OP_TRACE.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
    let value = f();
    let samples = OP_TRACE.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (value, samples)
}

pub fn rot_vec_quat_with(policy: Contraction, v: &[f64; 3], q: &[f64; 4]) -> [f64; 3] {
    match policy {
        Contraction::Contracted => {
            let t = [
                (-q[3]).mul_add(v[1], q[0].mul_add(v[0], q[2] * v[2])),
                (-q[1]).mul_add(v[2], q[0].mul_add(v[1], q[3] * v[0])),
                (-q[2]).mul_add(v[0], q[0].mul_add(v[2], q[1] * v[1])),
            ];
            [
                2.0f64.mul_add(q[2].mul_add(t[2], -(q[3] * t[1])), v[0]),
                2.0f64.mul_add(q[3].mul_add(t[0], -(q[1] * t[2])), v[1]),
                2.0f64.mul_add(q[1].mul_add(t[1], -(q[2] * t[0])), v[2]),
            ]
        }
        Contraction::Unfused => {
            let t0 = (q[0] * v[0] + q[2] * v[2]) - q[3] * v[1];
            let t1 = (q[0] * v[1] + q[3] * v[0]) - q[1] * v[2];
            let t2 = (q[0] * v[2] + q[1] * v[1]) - q[2] * v[0];
            [
                v[0] + 2.0 * (q[2] * t2 - q[3] * t1),
                v[1] + 2.0 * (q[3] * t0 - q[1] * t2),
                v[2] + 2.0 * (q[1] * t1 - q[2] * t0),
            ]
        }
    }
}

pub fn mul_mat_vec3_with(policy: Contraction, m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    match policy {
        Contraction::Contracted => [
            m[2].mul_add(v[2], m[0].mul_add(v[0], m[1] * v[1])),
            m[5].mul_add(v[2], m[3].mul_add(v[0], m[4] * v[1])),
            m[8].mul_add(v[2], m[6].mul_add(v[0], m[7] * v[1])),
        ],
        Contraction::Unfused => [
            (m[0] * v[0] + m[1] * v[1]) + m[2] * v[2],
            (m[3] * v[0] + m[4] * v[1]) + m[5] * v[2],
            (m[6] * v[0] + m[7] * v[1]) + m[8] * v[2],
        ],
    }
}

pub fn mul_quat_with(policy: Contraction, a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    match policy {
        Contraction::Contracted => [
            (-a[3]).mul_add(b[3], (-a[2]).mul_add(b[2], a[0].mul_add(b[0], -(a[1] * b[1])))),
            (-a[3]).mul_add(b[2], a[2].mul_add(b[3], a[0].mul_add(b[1], a[1] * b[0]))),
            a[3].mul_add(b[1], a[2].mul_add(b[0], a[0].mul_add(b[2], -(a[1] * b[3])))),
            a[3].mul_add(b[0], (-a[2]).mul_add(b[1], a[0].mul_add(b[3], a[1] * b[2]))),
        ],
        Contraction::Unfused => {
            // Products first, then the lanes of the `64156abe` `mju_mulQuat` schedule:
            // r0 = ((p00 - p11) - p22) - p33
            // r1 = ((p10 + p01) + p23) - p32
            // r2 = ((p02 - p13) + p20) + p31
            // r3 = ((p12 + p03) - p21) + p30
            let (p00, p11, p22, p33) = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
            let (p01, p10, p23, p32) = (a[0] * b[1], a[1] * b[0], a[2] * b[3], a[3] * b[2]);
            let (p02, p13, p20, p31) = (a[0] * b[2], a[1] * b[3], a[2] * b[0], a[3] * b[1]);
            let (p03, p12, p21, p30) = (a[0] * b[3], a[1] * b[2], a[2] * b[1], a[3] * b[0]);
            [
                ((p00 - p11) - p22) - p33,
                ((p10 + p01) + p23) - p32,
                ((p02 - p13) + p20) + p31,
                ((p12 + p03) - p21) + p30,
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mul_quat_stays_contracted_and_unfused_matches_the_pinned_wheel_vector() {
        reset_policy();
        assert_eq!(MUL_QUAT, Contraction::Contracted);
        assert_eq!(active_policy(), DEFAULT_POLICY);
        let manifest = include_str!("../../../docs/acceptance/oracles/manifest.json");
        assert!(manifest.contains("\"mul_quat\": \"contracted\""));
        let a = [0.9, 0.1, -0.2, 0.3];
        let b = [0.8, -0.1, 0.4, 0.2];
        let contracted = mul_quat(&a, &b);
        assert_eq!(contracted.map(f64::to_bits), mul_quat_with(Contraction::Contracted, &a, &b).map(f64::to_bits));
        let wheel = [
            0x3fe8000000000000,
            0xbfc5c28f5c28f5c2,
            0x3fc3333333333333,
            0x3fdc28f5c28f5c29,
        ];
        let unfused = mul_quat_with(Contraction::Unfused, &a, &b);
        assert_eq!(unfused.map(f64::to_bits), wheel);
        assert_ne!(contracted.map(f64::to_bits), wheel);
        let (again, samples) = capture_ops(|| mul_quat(&a, &b));
        assert_eq!(again.map(f64::to_bits), contracted.map(f64::to_bits));
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].op, "mul_quat");
        assert_eq!(samples[0].outputs, contracted.map(f64::to_bits).to_vec());
        assert_ne!(samples[0].outputs, wheel.to_vec());
    }

    #[test]
    fn contract_off_is_opt_in_and_a_wrong_sha_is_refused() {
        reset_policy();
        assert_eq!(
            install_policy(ORACLE_X86_CONTRACT_OFF_ID, "0".repeat(64).as_str(), "mul_quat"),
            Err(PolicyError::InconsistentIdentity)
        );
        assert_eq!(active_policy(), DEFAULT_POLICY);
        assert_eq!(install_policy("linux", ORACLE_X86_CONTRACT_OFF_SHA256, "mul_quat"), Err(PolicyError::UnknownIdentity));
        assert_eq!(
            install_policy(ORACLE_X86_CONTRACT_OFF_ID, ORACLE_X86_CONTRACT_OFF_SHA256, "solver"),
            Err(PolicyError::UnknownSite)
        );
        assert_eq!(
            install_policy(ORACLE_X86_CONTRACT_OFF_ID, ORACLE_X86_CONTRACT_OFF_SHA256, "mul_mat_vec3"),
            Err(PolicyError::UnknownSite),
            "mul_mat_vec3 stays unselected after the qacc regression"
        );
        assert_eq!(active_policy(), DEFAULT_POLICY);
        let policy = install_policy(ORACLE_X86_CONTRACT_OFF_ID, ORACLE_X86_CONTRACT_OFF_SHA256, "mul_quat").unwrap();
        assert_eq!(policy.mul_quat, Contraction::Unfused);
        assert_eq!(policy.rot_vec_quat, Contraction::Contracted);
        assert_eq!(policy.mul_mat_vec3, Contraction::Contracted);
        let a = [0.9, 0.1, -0.2, 0.3];
        let b = [0.8, -0.1, 0.4, 0.2];
        assert_eq!(mul_quat(&a, &b).map(f64::to_bits), mul_quat_with(Contraction::Unfused, &a, &b).map(f64::to_bits));
        let v = [0.2, -0.4, 0.6];
        let q = [0.5, 0.5, 0.5, 0.5];
        assert_eq!(
            rot_vec_quat_with(active_policy().rot_vec_quat, &v, &q).map(f64::to_bits),
            rot_vec_quat_with(Contraction::Contracted, &v, &q).map(f64::to_bits)
        );
        reset_policy();
        assert_eq!(mul_quat(&a, &b).map(f64::to_bits), mul_quat_with(Contraction::Contracted, &a, &b).map(f64::to_bits));
    }

    #[test]
    fn unfused_sites_match_the_named_oracle_on_random_and_adversarial_vectors() {
        let text = include_str!("../../../docs/acceptance/oracles/x86_64-contract-off-op-vectors.txt");
        let mut n = 0usize;
        for line in text.lines().filter(|line| !line.is_empty()) {
            let mut parts = line.split('\t');
            let op = parts.next().unwrap();
            let inputs: Vec<f64> = parts.next().unwrap().split(',').map(|s| f64::from_bits(s.parse().unwrap())).collect();
            let want: Vec<u64> = parts.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect();
            let got = match op {
                "mul_quat" => mul_quat_with(Contraction::Unfused, inputs[..4].try_into().unwrap(), inputs[4..8].try_into().unwrap()).map(f64::to_bits).to_vec(),
                "rot_vec_quat" => rot_vec_quat_with(Contraction::Unfused, inputs[..3].try_into().unwrap(), inputs[3..7].try_into().unwrap()).map(f64::to_bits).to_vec(),
                "mul_mat_vec3" => mul_mat_vec3_with(Contraction::Unfused, inputs[..9].try_into().unwrap(), inputs[9..12].try_into().unwrap()).map(f64::to_bits).to_vec(),
                other => panic!("{other}"),
            };
            assert_eq!(got, want, "{op}");
            n += 1;
        }
        assert!(n >= 100);
    }
}
