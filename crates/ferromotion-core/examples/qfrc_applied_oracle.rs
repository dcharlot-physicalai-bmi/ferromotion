//! Bit-exact `qfrc_applied` against the dump from `scripts/qfrc_applied_oracle.py`.
//!
//! The dump is MuJoCo 3.13.0 contract-off (`64156abe`). This process must be on that
//! same policy (`FERROMOTION_BUILD_POLICY` / `FERROMOTION_ORACLE_SHA256`). Comparison
//! is `f64::to_bits`. The published default is not installed.

use ferromotion_core::{MjNativeState, tree_from_mjcf};

fn parse_bits(field: &str) -> Vec<f64> {
    field
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| u64::from_str_radix(s, 16).map(f64::from_bits))
        .collect::<Result<Vec<_>, _>>()
        .expect("hex bits")
}

fn same(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn main() {
    let policy = ferromotion_core::install_policy_from_env().unwrap_or_else(|err| {
        eprintln!("policy refused: {err:?}");
        std::process::exit(2);
    });
    eprintln!("{}", ferromotion_core::policy_receipt(policy));
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: qfrc_applied_oracle <model.xml> <dump.txt>");
        std::process::exit(2);
    }
    let xml = std::fs::read_to_string(&args[1]).unwrap_or_else(|err| panic!("read xml: {err}"));
    let model = tree_from_mjcf(&xml, &|_| None).unwrap_or_else(|err| panic!("load: {err}"));
    let text = std::fs::read_to_string(&args[2]).unwrap_or_else(|err| panic!("read dump: {err}"));

    let mut qfrc = Vec::new();
    let mut start = MjNativeState::default();
    let mut steps: Vec<(MjNativeState, Vec<f64>)> = Vec::new();
    for line in text.lines() {
        let (key, field) = line.split_once('\t').unwrap_or((line, ""));
        let vals = parse_bits(field);
        match key {
            "qfrc" => qfrc = vals,
            "qpos" if steps.is_empty() => start.qpos = vals,
            "qvel" if steps.is_empty() => start.qvel = vals,
            "act" if steps.is_empty() => start.act = vals,
            "s" => steps.push((MjNativeState::default(), Vec::new())),
            "qpos" => steps.last_mut().unwrap().0.qpos = vals,
            "qvel" => steps.last_mut().unwrap().0.qvel = vals,
            "act" => steps.last_mut().unwrap().0.act = vals,
            "warm" => steps.last_mut().unwrap().0.qacc_warmstart = vals,
            "qfrc_constraint" => steps.last_mut().unwrap().1 = vals,
            other => panic!("unknown dump key {other}"),
        }
    }
    assert!(
        qfrc.iter().any(|v| v.to_bits() != 0),
        "dump qfrc_applied is zero"
    );
    let mut st = start;
    st.qacc_warmstart = vec![0.0; st.qvel.len()];
    let ctrl = vec![0.0; model.actuators.len()];
    for (k, (want, qfrc_constraint)) in steps.iter().enumerate() {
        let witness = model
            .mujoco_native_step_witness_applied(&st, &ctrl, &qfrc)
            .unwrap_or_else(|err| panic!("step {}: {err}", k + 1));
        let next = &witness.next;
        let ok = same(&next.qpos, &want.qpos)
            && same(&next.qvel, &want.qvel)
            && same(&next.act, &want.act)
            && same(&next.qacc_warmstart, &want.qacc_warmstart)
            && same(&witness.qfrc_constraint, qfrc_constraint);
        if !ok {
            eprintln!(
                "step {} mismatch qpos {} qvel {} warm {} qfrc_constraint {}",
                k + 1,
                same(&next.qpos, &want.qpos),
                same(&next.qvel, &want.qvel),
                same(&next.qacc_warmstart, &want.qacc_warmstart),
                same(&witness.qfrc_constraint, qfrc_constraint)
            );
            std::process::exit(1);
        }
        st = next.clone();
    }
    println!(
        "qfrc_applied non-zero, bit-exact for {} steps (qpos qvel act qacc_warmstart qfrc_constraint)",
        steps.len()
    );
}
