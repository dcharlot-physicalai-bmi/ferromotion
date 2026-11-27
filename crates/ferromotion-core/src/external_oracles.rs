//! External oracles and the SmolVLA closed-loop contract.
//!
//! MuJoCo Warp and Newton are upstream references (Google DeepMind, NVIDIA, and the
//! Newton project). SmolVLA is a Hugging Face / LeRobot model. None of those names
//! is an Institute product, and citing them is not an endorsement. Ferromotion's
//! dynamics stay an independent Rust implementation. Upstream code is an oracle,
//! not a dependency of the simulator.
//!
//! Pins below were read from the public GitHub and Hugging Face APIs on 2026-11-07.
//! They are revisions, not proof that a benchmark was run.

use crate::joint_acceptance::EvidenceKind;

/// MuJoCo Warp `main` at the pin. Apache-2.0. README at this sha says Warp
/// differentiability is not yet available.
pub const MUJOCO_WARP_REVISION: &str = "4d1c64bc077d725247e9bf5e2ac3f357749ff804";
/// Newton `main` at the pin. Code Apache-2.0. Documentation CC-BY-4.0.
pub const NEWTON_REVISION: &str = "1a497c00ed255db343e39eec4d614864c9aa9b95";
/// `lerobot/smolvla_base` model-card revision returned by the Hub API. Apache-2.0.
/// Weights were not downloaded.
pub const SMOLVLA_REVISION: &str = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6";
/// SmolVLA paper. Citation only; the PDF is not mirrored here.
pub const SMOLVLA_ARXIV: &str = "2506.01844";

/// One cell of a model × feature × solver matrix. `Blocked` is not a pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleCell {
    /// Bit-exact or within a stated error, from an execution on this host.
    Executed { matched: bool, detail: String },
    /// The backend, device, or gradient tape was not available.
    Blocked { reason: String },
    /// Not part of this run.
    NotRun,
}

impl OracleCell {
    pub fn is_pass(&self) -> bool {
        matches!(self, OracleCell::Executed { matched: true, .. })
    }
}

/// What a matched-fidelity comparison is allowed to claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamOracle {
    pub name: &'static str,
    pub owner: &'static str,
    pub url: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
}

pub fn warp_oracle() -> UpstreamOracle {
    UpstreamOracle {
        name: "MuJoCo Warp",
        owner: "Google DeepMind and NVIDIA",
        url: "https://github.com/google-deepmind/mujoco_warp",
        revision: MUJOCO_WARP_REVISION,
        license: "Apache-2.0",
    }
}

pub fn newton_oracle() -> UpstreamOracle {
    UpstreamOracle {
        name: "Newton",
        owner: "Newton project (Linux Foundation; initiated by Disney Research, Google DeepMind, and NVIDIA)",
        url: "https://github.com/newton-physics/newton",
        revision: NEWTON_REVISION,
        license: "Apache-2.0 code; CC-BY-4.0 documentation",
    }
}

/// Hardware and backend actually observed. Absent GPU is blocked, not a zero-error pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarpNewtonProbe {
    pub gpu_device: Option<String>,
    pub warp_importable: bool,
    pub newton_importable: bool,
    pub batch: Option<u32>,
    pub steps_executed: u32,
    /// Wall time of a real rollout, if one ran.
    pub elapsed_ms: Option<u64>,
    /// True only when a gradient was evaluated on this backend and contact regime.
    pub gradient_evaluated: bool,
}

impl WarpNewtonProbe {
    /// This host, 2026-11-07: no NVIDIA device, `warp` and `newton` not installed.
    pub fn this_host() -> Self {
        WarpNewtonProbe {
            gpu_device: None,
            warp_importable: false,
            newton_importable: false,
            batch: None,
            steps_executed: 0,
            elapsed_ms: None,
            gradient_evaluated: false,
        }
    }
}

/// Classify one oracle. GPU execution would still not imply differentiability.
pub fn warp_newton_probe(probe: &WarpNewtonProbe, want_gradient: bool) -> OracleCell {
    if probe.gpu_device.is_none() {
        return OracleCell::Blocked {
            reason: "no GPU device was visible to this process".into(),
        };
    }
    if !probe.warp_importable && !probe.newton_importable {
        return OracleCell::Blocked {
            reason: "neither mujoco_warp nor newton imported".into(),
        };
    }
    if probe.steps_executed == 0 {
        return OracleCell::Blocked {
            reason: "packages imported but no rollout was executed".into(),
        };
    }
    if want_gradient && !probe.gradient_evaluated {
        return OracleCell::Blocked {
            reason: "forward GPU execution is not a gradient test; differentiability was not evaluated for this backend and contact regime".into(),
        };
    }
    OracleCell::Executed {
        matched: false,
        detail:
            "a rollout ran; fidelity error against MuJoCo was not attached, so this is not a match"
                .into(),
    }
}

/// Observation → action-chunk → simulated rollout → safe recovery.
/// No physical actuation. Integrated inference requires action agreement and a rollout record.
#[derive(Clone, Debug, PartialEq)]
pub struct SmolVlaContract {
    pub paper: &'static str,
    pub checkpoint: String,
    pub preprocessing: Option<String>,
    pub normalization: Option<String>,
    /// Unit and frame of each action component, for example "radian, joint".
    pub action_unit: Option<String>,
    pub action: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub action_time_ns: u64,
    pub now_ns: u64,
    pub max_age_ns: u64,
    pub reference_sha256: Option<String>,
    pub produced_sha256: Option<String>,
    pub rollout: EvidenceKind,
    pub held_out_success: Option<f64>,
    pub browser_or_device_executed: bool,
    pub physical_actuation: bool,
}

impl SmolVlaContract {
    /// Scope only. The checkpoint pin is the Hub revision. Weights were not loaded,
    /// so there is no action vector and no rollout.
    pub fn scoped() -> Self {
        SmolVlaContract {
            paper: SMOLVLA_ARXIV,
            checkpoint: SMOLVLA_REVISION.into(),
            preprocessing: None,
            normalization: None,
            action_unit: None,
            action: Vec::new(),
            low: -1.0,
            high: 1.0,
            action_time_ns: 0,
            now_ns: 0,
            max_age_ns: 0,
            reference_sha256: None,
            produced_sha256: None,
            rollout: EvidenceKind::NotMeasured,
            held_out_success: None,
            browser_or_device_executed: false,
            physical_actuation: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionReject {
    Empty,
    NonFinite,
    OutOfRange,
    Stale,
    PhysicalActuation,
}

/// Reject an action chunk before it can be applied, including in simulation.
pub fn reject_action(contract: &SmolVlaContract) -> Option<ActionReject> {
    if contract.physical_actuation {
        return Some(ActionReject::PhysicalActuation);
    }
    if contract.action.is_empty() {
        return Some(ActionReject::Empty);
    }
    if contract.action.iter().any(|x| !x.is_finite()) {
        return Some(ActionReject::NonFinite);
    }
    if contract
        .action
        .iter()
        .any(|x| *x < contract.low || *x > contract.high)
    {
        return Some(ActionReject::OutOfRange);
    }
    if contract.now_ns.saturating_sub(contract.action_time_ns) > contract.max_age_ns {
        return Some(ActionReject::Stale);
    }
    None
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceClaim {
    NotIntegrated { missing: Vec<&'static str> },
    Integrated,
}

/// Action-output agreement and rollout evidence are required before calling inference integrated.
/// A browser or device fit is not implied by a checkpoint pin.
pub fn inference_claim(contract: &SmolVlaContract) -> InferenceClaim {
    let mut missing = Vec::new();
    if contract.physical_actuation {
        missing.push("physical actuation is out of scope");
    }
    if reject_action(contract).is_some() {
        missing.push("action chunk was rejected");
    }
    if contract.preprocessing.is_none()
        || contract.normalization.is_none()
        || contract.action_unit.is_none()
    {
        missing.push("preprocessing, normalization, or action units are unspecified");
    }
    match (&contract.reference_sha256, &contract.produced_sha256) {
        (Some(r), Some(p)) if r == p && r.len() == 64 => {}
        _ => missing.push("action output does not match a pinned reference digest"),
    }
    if !matches!(contract.rollout, EvidenceKind::IndependentlyExecuted { .. }) {
        missing.push("no independently executed simulated rollout");
    }
    if contract.held_out_success.is_none() {
        missing.push("held-out success was not measured");
    }
    if missing.is_empty() {
        InferenceClaim::Integrated
    } else {
        InferenceClaim::NotIntegrated { missing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_host_warp_and_newton_probe_is_blocked_not_a_pass() {
        let cell = warp_newton_probe(&WarpNewtonProbe::this_host(), true);
        assert!(!cell.is_pass());
        assert!(matches!(cell, OracleCell::Blocked { .. }));
    }

    #[test]
    fn a_gpu_forward_without_a_gradient_is_still_blocked() {
        let probe = WarpNewtonProbe {
            gpu_device: Some("nvidia-placeholder".into()),
            warp_importable: true,
            newton_importable: false,
            batch: Some(64),
            steps_executed: 20,
            elapsed_ms: Some(12),
            gradient_evaluated: false,
        };
        let cell = warp_newton_probe(&probe, true);
        assert!(!cell.is_pass());
        match cell {
            OracleCell::Blocked { reason } => assert!(reason.contains("gradient"), "{reason}"),
            other => panic!("expected blocked, got {other:?}"),
        }
    }

    #[test]
    fn scoped_smolvla_is_not_integrated_and_does_not_actuate() {
        let contract = SmolVlaContract::scoped();
        assert!(matches!(
            reject_action(&contract),
            Some(ActionReject::Empty)
        ));
        assert!(!contract.browser_or_device_executed);
        assert!(!contract.physical_actuation);
        assert!(matches!(
            inference_claim(&contract),
            InferenceClaim::NotIntegrated { .. }
        ));
        assert_eq!(contract.checkpoint, SMOLVLA_REVISION);
    }

    #[test]
    fn nonfinite_stale_and_out_of_range_actions_are_rejected() {
        let mut c = SmolVlaContract::scoped();
        c.action = vec![0.0, f64::NAN];
        c.max_age_ns = 10;
        c.now_ns = 5;
        c.action_time_ns = 5;
        assert_eq!(reject_action(&c), Some(ActionReject::NonFinite));
        c.action = vec![2.0];
        assert_eq!(reject_action(&c), Some(ActionReject::OutOfRange));
        c.action = vec![0.25];
        c.action_time_ns = 0;
        c.now_ns = 11;
        assert_eq!(reject_action(&c), Some(ActionReject::Stale));
        c.physical_actuation = true;
        c.now_ns = 0;
        assert_eq!(reject_action(&c), Some(ActionReject::PhysicalActuation));
    }

    #[test]
    fn pins_are_hex_revisions_and_named_owners() {
        for sha in [MUJOCO_WARP_REVISION, NEWTON_REVISION, SMOLVLA_REVISION] {
            assert!(
                sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()),
                "{sha}"
            );
        }
        assert!(warp_oracle().license.contains("Apache-2.0"));
        assert!(newton_oracle().license.contains("CC-BY-4.0"));
        assert_eq!(SMOLVLA_ARXIV, "2506.01844");
    }
}
