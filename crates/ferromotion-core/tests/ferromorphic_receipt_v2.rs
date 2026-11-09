//! Receipt v2 bytes produced by compiling ferromorphic at the clock pin.
//!
//! A text round trip is not envelope interoperability, a joint loop, or an inference claim.
//! `FERROMORPHIC_BIND_VERIFIED` stays false: that repository has no bind.

use ferromotion_core::{
    parse_step_receipt, StepReceipt, EPISODE_ENVELOPE_INTEROPERABLE, FERROMORPHIC_BIND_VERIFIED, FERROMORPHIC_CLOCK_REVISION,
    JOINT_LOOP_EXECUTED,
};

const PIN: &str = "15e219b8f7f1bc76029c359df676cbbcf9817ec0";

const PINNED_SAMPLE: &str = "\
ferromorphic-step 2
epoch 0
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
";

#[test]
fn ferromorphic_receipt_v2_round_trips_both_ways() {
    assert_eq!(FERROMORPHIC_CLOCK_REVISION, PIN);
    const { assert!(!FERROMORPHIC_BIND_VERIFIED); }
    const { assert!(!EPISODE_ENVELOPE_INTEROPERABLE); }
    const { assert!(!JOINT_LOOP_EXECUTED); }

    let sample = ferromorphic::sim::StepReceipt {
        epoch: 0,
        step: 4,
        dt_bits: 0.1_f64.to_bits(),
        mode: "clocked",
        spikes: vec![(4, 0), (4, 1)],
    };
    let text = sample.to_text();
    assert_eq!(text, PINNED_SAMPLE);
    assert_eq!(ferromorphic::sim::StepReceipt::from_text(&text).unwrap(), sample);

    let parsed = parse_step_receipt(&text).expect("ferromotion parses ferromorphic receipt v2");
    assert_eq!(parsed.version, 2);
    assert_eq!(parsed.epoch, sample.epoch);
    assert_eq!(parsed.step, sample.step);
    assert_eq!(parsed.dt_bits, sample.dt_bits);
    assert_eq!(parsed.mode, sample.mode);
    assert_eq!(parsed.spikes.len(), sample.spikes.len());
    assert_eq!(parsed.to_text(), text);
    assert_eq!(ferromorphic::sim::StepReceipt::from_text(&parsed.to_text()).unwrap(), sample);

    let generated = ferromorphic::sim::StepReceipt::from_step(2, 9, 1e-4, ferromorphic::sim::Mode::EventDriven, &[0, 1]);
    let generated_text = generated.to_text();
    assert!(generated_text.starts_with("ferromorphic-step 2\nepoch 2\n"));
    let ours = parse_step_receipt(&generated_text).unwrap();
    assert_eq!(ours.epoch, 2);
    assert_eq!(ours.step, 9);
    assert_eq!(ours.mode, "event");
    assert_eq!(ours.dt_bits, 1e-4_f64.to_bits());
    assert_eq!(ours.spikes.len(), 2);
    assert_eq!(ferromorphic::sim::StepReceipt::from_text(&ours.to_text()).unwrap(), generated);

    let local = StepReceipt {
        version: 2,
        epoch: 1,
        step: 0,
        dt_bits: 0x3eb0c6f7a0b5ed8e,
        mode: "clocked".into(),
        spikes: Vec::new(),
    };
    let via_them = ferromorphic::sim::StepReceipt::from_text(&local.to_text()).unwrap();
    assert_eq!(via_them.epoch, 1);
    assert_eq!(via_them.step, 0);
    assert_eq!(via_them.dt_bits, local.dt_bits);
    assert_eq!(via_them.mode, "clocked");
    assert!(via_them.spikes.is_empty());
    assert_eq!(parse_step_receipt(&via_them.to_text()).unwrap(), local);

    let v1 = "\
ferromorphic-step 1
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
";
    assert!(parse_step_receipt(v1).is_err());
    assert!(ferromorphic::sim::StepReceipt::from_text(v1).is_err());
}
