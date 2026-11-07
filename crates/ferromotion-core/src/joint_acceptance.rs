//! One closed-loop task connecting ferromotion's dynamics with the pinned revisions of Ferric,
//! ferrotherm, and ferromorphic.
//!
//! Joint acceptance requires three things at once: the trajectory agrees with its reference, the
//! deterministic receipts replay to the same bytes, and the whole-system energy traces are
//! synchronized measurements. Simulator agreement is not that. This module does not record a
//! measurement, a hardware-in-the-loop run, or a calibrated joule figure; those slots stay
//! [`EvidenceKind::NotMeasured`] until a provenance record exists.

use std::collections::BTreeMap;

/// Ferric `main` at the pin used by [`PINNED_CLOSED_LOOP_TASK`].
pub const FERRIC_REVISION: &str = "67f78760c0824f70ff1a4fc1877788a57d926c31";
/// ferrotherm `main` at the same pin. Normative sampler program: `spec/ftp-v1.md`.
pub const FERROTHERM_REVISION: &str = "3c474719e193d4f441201975b16419223e83d160";
/// ferromorphic `main` at the same pin. Event, NIR, and mapping sources: `eventprop.rs`, `nir.rs`,
/// `mapping.rs`.
pub const FERROMORPHIC_REVISION: &str = "8dabe3a45e589908f524b595da09bbadbc8aa103";

const TASK_ID: &str = "ferromotion.joint.v1";

/// How a number or digest entered the record. A prior claim and a synthetic fixture are not an
/// execution of this revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvidenceKind {
    /// Produced by running the pinned revision in this record, and named by `instrument` when the
    /// quantity is a physical measurement.
    IndependentlyExecuted { instrument: Option<String> },
    /// An author's earlier report. Not re-run for this record.
    PriorClaim,
    /// The run was attempted and did not produce the quantity.
    Blocked { reason: String },
    /// Constructed in a unit test to exercise the predicate. Not a physical result.
    SyntheticFixture,
    /// No measurement was made.
    NotMeasured,
}

impl EvidenceKind {
    fn is_independent_measurement(&self) -> bool {
        matches!(self, EvidenceKind::IndependentlyExecuted { instrument: Some(name) } if !name.is_empty())
    }

    #[cfg(test)]
    fn is_independent_execution(&self) -> bool {
        matches!(self, EvidenceKind::IndependentlyExecuted { .. })
    }
}

/// A scalar trace aligned to the task clock. `samples` is empty when nothing was measured.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyTrace {
    pub kind: EvidenceKind,
    pub unit: &'static str,
    pub samples: Vec<f64>,
}

impl EnergyTrace {
    pub fn not_measured() -> Self {
        EnergyTrace {
            kind: EvidenceKind::NotMeasured,
            unit: "J",
            samples: Vec::new(),
        }
    }
}

/// A Ferric tier bound (`ferric-tier`): memory, time, and fabric. Not an energy measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBound {
    pub bytes: u64,
    pub nanoseconds: u64,
    pub fabric: String,
}

/// Receipts for the single task, each tagged with the revision that would have produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct JointReceipt {
    pub task_id: String,
    pub steps: usize,
    /// ferromotion `qpos`/`qvel` digest at each step, hex SHA-256 of the little-endian bit patterns.
    pub trajectory_sha256: String,
    pub reference_sha256: String,
    /// `ferric-certify` digest of the deterministic kernel outputs.
    pub ferric_certify_sha256: Option<String>,
    pub ferric_revision: String,
    pub ferric_tier: Option<ResourceBound>,
    /// `ferric-joule`. Empty and [`EvidenceKind::NotMeasured`] until a run records it.
    pub ferric_joule: EnergyTrace,
    /// ferrotherm `.ftp` program digest (`spec/ftp-v1.md` at [`FERROTHERM_REVISION`]).
    pub ferrotherm_ftp_sha256: Option<String>,
    pub ferrotherm_revision: String,
    /// Calibrated sampler/fabric energy. Not inferred from `.ftp` operation costs.
    pub ferrotherm_energy: EnergyTrace,
    /// ferromorphic `eventprop` / `nir` / `mapping` / policy digests.
    pub ferromorphic_revision: String,
    pub event_sha256: Option<String>,
    pub nir_sha256: Option<String>,
    pub mapping_sha256: Option<String>,
    pub policy_sha256: Option<String>,
}

/// The one closed-loop task. Fields that are not measurements say so.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoopTask {
    pub id: &'static str,
    pub ferromotion_revision: String,
    pub receipt: JointReceipt,
    /// Second execution of the deterministic trajectory digest. Agreement is reproducibility.
    pub replay_sha256: String,
}

/// Why joint acceptance did not hold, or that it did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Accepted,
    Rejected {
        reasons: BTreeMap<&'static str, String>,
    },
}

impl Decision {
    pub fn accepted(&self) -> bool {
        matches!(self, Decision::Accepted)
    }
}

fn sha256_hex_ok(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn git_sha_ok(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn trace_synchronized(
    trace: &EnergyTrace,
    steps: usize,
    label: &str,
    reasons: &mut BTreeMap<&'static str, String>,
    key: &'static str,
) {
    if !trace.kind.is_independent_measurement() {
        reasons.insert(
            key,
            format!(
                "{label} is {:?}, not an instrumented measurement",
                evidence_name(&trace.kind)
            ),
        );
        return;
    }
    if trace.unit != "J" {
        reasons.insert(
            key,
            format!(
                "{label} unit is {}, joint acceptance counts joules",
                trace.unit
            ),
        );
    }
    if trace.samples.len() != steps || trace.samples.iter().any(|x| !x.is_finite()) {
        reasons.insert(
            key,
            format!(
                "{label} has {} samples for {steps} steps",
                trace.samples.len()
            ),
        );
    }
}

fn evidence_name(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::IndependentlyExecuted {
            instrument: Some(_),
        } => "an instrumented execution",
        EvidenceKind::IndependentlyExecuted { instrument: None } => {
            "an execution with no instrument"
        }
        EvidenceKind::PriorClaim => "a prior claim",
        EvidenceKind::Blocked { .. } => "blocked",
        EvidenceKind::SyntheticFixture => "a synthetic fixture",
        EvidenceKind::NotMeasured => "not measured",
    }
}

/// Decide the pinned task. Acceptance is reference agreement, a matching replay, and synchronized
/// measured energy from both Ferric and ferrotherm. A dynamics digest alone does not pass.
pub fn decide(task: &ClosedLoopTask) -> Decision {
    let mut reasons = BTreeMap::new();
    let r = &task.receipt;
    if r.task_id != task.id {
        reasons.insert(
            "task",
            format!("receipt task {} is not {}", r.task_id, task.id),
        );
    }
    if !git_sha_ok(&task.ferromotion_revision)
        || !git_sha_ok(&r.ferric_revision)
        || !git_sha_ok(&r.ferrotherm_revision)
        || !git_sha_ok(&r.ferromorphic_revision)
    {
        reasons.insert(
            "revision",
            "a pinned revision is not a 40-digit git sha".into(),
        );
    }
    if r.ferric_revision != FERRIC_REVISION
        || r.ferrotherm_revision != FERROTHERM_REVISION
        || r.ferromorphic_revision != FERROMORPHIC_REVISION
    {
        reasons.insert(
            "pin",
            "a sibling revision does not match the pin in this source".into(),
        );
    }
    if !sha256_hex_ok(&r.trajectory_sha256)
        || !sha256_hex_ok(&r.reference_sha256)
        || !sha256_hex_ok(&task.replay_sha256)
    {
        reasons.insert(
            "digest",
            "a trajectory digest is not a sha256 hex string".into(),
        );
    } else {
        if r.trajectory_sha256 != r.reference_sha256 {
            reasons.insert(
                "reference",
                "trajectory digest disagrees with the reference".into(),
            );
        }
        if r.trajectory_sha256 != task.replay_sha256 {
            reasons.insert(
                "replay",
                "re-executing the trajectory did not reproduce the digest".into(),
            );
        }
    }
    if !r
        .ferric_certify_sha256
        .as_deref()
        .is_some_and(sha256_hex_ok)
    {
        reasons.insert("ferric_certify", "ferric-certify digest is absent".into());
    }
    if r.ferric_tier.is_none() {
        reasons.insert("ferric_tier", "ferric-tier resource bound is absent".into());
    }
    for (name, digest) in [
        ("event", &r.event_sha256),
        ("nir", &r.nir_sha256),
        ("mapping", &r.mapping_sha256),
        ("policy", &r.policy_sha256),
    ] {
        match digest {
            Some(d) if sha256_hex_ok(d) => {}
            _ => {
                reasons.insert(
                    "ferromorphic",
                    format!("ferromorphic {name} receipt is absent"),
                );
                break;
            }
        }
    }
    if r.ferrotherm_ftp_sha256
        .as_ref()
        .is_none_or(|d| !sha256_hex_ok(d))
    {
        reasons.insert("ftp", "ferrotherm .ftp digest is absent".into());
    }
    trace_synchronized(
        &r.ferric_joule,
        r.steps,
        "ferric-joule",
        &mut reasons,
        "ferric_joule",
    );
    trace_synchronized(
        &r.ferrotherm_energy,
        r.steps,
        "ferrotherm energy",
        &mut reasons,
        "ferrotherm_energy",
    );
    if reasons.is_empty() {
        Decision::Accepted
    } else {
        Decision::Rejected { reasons }
    }
}

/// The task as this revision can state it. Dynamics pins are real git shas. Energy is not measured.
pub fn pinned_task(ferromotion_revision: &str) -> ClosedLoopTask {
    let receipt = JointReceipt {
        task_id: TASK_ID.into(),
        steps: 0,
        trajectory_sha256: "0".repeat(64),
        reference_sha256: "0".repeat(64),
        ferric_certify_sha256: None,
        ferric_revision: FERRIC_REVISION.into(),
        ferric_tier: None,
        ferric_joule: EnergyTrace::not_measured(),
        ferrotherm_ftp_sha256: None,
        ferrotherm_revision: FERROTHERM_REVISION.into(),
        ferrotherm_energy: EnergyTrace::not_measured(),
        ferromorphic_revision: FERROMORPHIC_REVISION.into(),
        event_sha256: None,
        nir_sha256: None,
        mapping_sha256: None,
        policy_sha256: None,
    };
    ClosedLoopTask {
        id: TASK_ID,
        ferromotion_revision: ferromotion_revision.into(),
        replay_sha256: "0".repeat(64),
        receipt,
    }
}

/// Placeholder revision until a commit sha is supplied by the caller. The exported task uses the
/// parent of this change, recorded in the conformance pin, not a guessed measurement.
pub const PINNED_CLOSED_LOOP_TASK_PARENT: &str = "5daca8aac1b513a18c54792383d021cf1d27af56";

/// The closed-loop task at the ferromotion revision this work started from. Rejected: energy and the
/// cross-repo receipts were not executed here.
pub const PINNED_CLOSED_LOOP_TASK: &str = PINNED_CLOSED_LOOP_TASK_PARENT;

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    fn complete_synthetic() -> ClosedLoopTask {
        let mut task = pinned_task(PINNED_CLOSED_LOOP_TASK_PARENT);
        let d = digest(0x11);
        task.receipt.steps = 4;
        task.receipt.trajectory_sha256 = d.clone();
        task.receipt.reference_sha256 = d.clone();
        task.replay_sha256 = d.clone();
        task.receipt.ferric_certify_sha256 = Some(digest(0x22));
        task.receipt.ferric_tier = Some(ResourceBound {
            bytes: 1024,
            nanoseconds: 1_000,
            fabric: "cpu".into(),
        });
        task.receipt.ferrotherm_ftp_sha256 = Some(digest(0x33));
        task.receipt.event_sha256 = Some(digest(0x41));
        task.receipt.nir_sha256 = Some(digest(0x42));
        task.receipt.mapping_sha256 = Some(digest(0x43));
        task.receipt.policy_sha256 = Some(digest(0x44));
        let energy = |kind| EnergyTrace {
            kind,
            unit: "J",
            samples: vec![0.1, 0.1, 0.1, 0.1],
        };
        task.receipt.ferric_joule = energy(EvidenceKind::SyntheticFixture);
        task.receipt.ferrotherm_energy = energy(EvidenceKind::SyntheticFixture);
        task
    }

    #[test]
    fn the_pinned_task_is_not_a_measured_acceptance() {
        let task = pinned_task(PINNED_CLOSED_LOOP_TASK_PARENT);
        let decision = decide(&task);
        assert!(!decision.accepted());
        let Decision::Rejected { reasons } = decision else {
            panic!("rejected")
        };
        assert!(reasons.contains_key("ferric_joule"), "{reasons:?}");
        assert!(reasons.contains_key("ferrotherm_energy"), "{reasons:?}");
        assert!(matches!(
            task.receipt.ferric_joule.kind,
            EvidenceKind::NotMeasured
        ));
        assert!(task.receipt.ferric_joule.samples.is_empty());
        assert!(task.receipt.ferrotherm_energy.samples.is_empty());
    }

    #[test]
    fn a_synthetic_energy_trace_does_not_count_as_joint_acceptance() {
        let decision = decide(&complete_synthetic());
        assert!(!decision.accepted());
        let Decision::Rejected { reasons } = decision else {
            panic!("rejected")
        };
        assert!(
            reasons["ferric_joule"].contains("synthetic"),
            "{}",
            reasons["ferric_joule"]
        );
    }

    #[test]
    fn an_execution_without_an_instrument_is_not_a_measurement() {
        let mut task = complete_synthetic();
        let kind = EvidenceKind::IndependentlyExecuted { instrument: None };
        task.receipt.ferric_joule.kind = kind.clone();
        task.receipt.ferrotherm_energy.kind = EvidenceKind::IndependentlyExecuted {
            instrument: Some(String::new()),
        };
        let Decision::Rejected { reasons } = decide(&task) else {
            panic!("rejected")
        };
        assert!(
            reasons.contains_key("ferric_joule") && reasons.contains_key("ferrotherm_energy"),
            "{reasons:?}"
        );
        assert!(!kind.is_independent_measurement());
        assert!(kind.is_independent_execution());
    }

    #[test]
    fn reference_disagreement_and_a_divergent_replay_are_rejected() {
        let mut task = complete_synthetic();
        task.receipt.ferric_joule.kind = EvidenceKind::IndependentlyExecuted {
            instrument: Some("bench-a".into()),
        };
        task.receipt.ferrotherm_energy.kind = EvidenceKind::IndependentlyExecuted {
            instrument: Some("bench-a".into()),
        };
        task.receipt.reference_sha256 = digest(0x99);
        task.replay_sha256 = digest(0x98);
        let Decision::Rejected { reasons } = decide(&task) else {
            panic!("rejected")
        };
        assert!(
            reasons.contains_key("reference") && reasons.contains_key("replay"),
            "{reasons:?}"
        );
    }

    #[test]
    fn a_complete_instrumented_record_is_the_only_acceptance() {
        let mut task = complete_synthetic();
        task.receipt.ferric_joule.kind = EvidenceKind::IndependentlyExecuted {
            instrument: Some("ferric-joule-bench".into()),
        };
        task.receipt.ferrotherm_energy.kind = EvidenceKind::IndependentlyExecuted {
            instrument: Some("ferrotherm-meter".into()),
        };
        assert!(decide(&task).accepted());
        task.receipt.ferrotherm_energy.samples.pop();
        assert!(
            !decide(&task).accepted(),
            "a short energy trace is not synchronized"
        );
    }

    #[test]
    fn sibling_pins_are_forty_hex_digits() {
        for sha in [
            FERRIC_REVISION,
            FERROTHERM_REVISION,
            FERROMORPHIC_REVISION,
            PINNED_CLOSED_LOOP_TASK_PARENT,
        ] {
            assert!(git_sha_ok(sha), "{sha}");
        }
    }
}
