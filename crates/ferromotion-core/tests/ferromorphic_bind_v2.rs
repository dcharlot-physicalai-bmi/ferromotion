//! `ferromorphic::sim::bind_v2_timestamps` at the pinned git revision.
//!
//! No path override. The three acceptance flags stay false. This file is not a sign-off.

use ferromorphic::NetBuilder;
use ferromorphic::neuron::Lif;
use ferromorphic::sim::{
    ClockSnapshot, EpisodeEnvelope, Mode, Sim, StepReceipt, TimebaseError, bind_v2_timestamps,
    nominal_seconds,
};
use ferromotion_core::{
    EPISODE_ENVELOPE_INTEROPERABLE, FERROMORPHIC_BIND_VERIFIED, FERROMORPHIC_CLOCK_REVISION,
    JOINT_LOOP_EXECUTED,
};

const PIN: &str = "a6722b49ff7dc770b1cb062adac21aa5c7155401";
const PERIOD: u64 = 1_000;

fn flags_stay_false() {
    assert_eq!(FERROMORPHIC_CLOCK_REVISION, PIN);
    const {
        assert!(!JOINT_LOOP_EXECUTED);
    }
    const {
        assert!(!EPISODE_ENVELOPE_INTEROPERABLE);
    }
    const {
        assert!(!FERROMORPHIC_BIND_VERIFIED);
    }
}

fn bits(period_ns: u64) -> u64 {
    nominal_seconds(period_ns)
        .expect("an exact period")
        .to_bits()
}

fn pair(period_ns: u64) -> (EpisodeEnvelope, ClockSnapshot) {
    let dt_bits = bits(period_ns);
    let envelope = EpisodeEnvelope::try_new(1, 1, dt_bits, period_ns, 0).expect("envelope");
    let clock = ClockSnapshot {
        epoch: 1,
        tick: 7,
        dt_bits,
        period_ns: Some(period_ns),
        origin_ns: Some(0),
    };
    (envelope, clock)
}

#[test]
fn a_matching_snapshot_binds() {
    flags_stay_false();
    let (envelope, clock) = pair(PERIOD);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 2, Some(envelope), &clock),
        Ok(())
    );
}

/// Public `Sim` fields, little-endian. `ring`, `inject`, `touched`, and `as_of` are private and
/// have no byte export. `bind_v2_timestamps` is not given `&mut Sim`, so those fields are not
/// written by the call either.
fn sim_public_bytes(sim: &Sim<Lif>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(sim.dt.to_bits().to_le_bytes());
    bytes.extend(sim.t.to_le_bytes());
    bytes.extend(sim.epoch.to_le_bytes());
    match sim.origin_ns {
        Some(v) => {
            bytes.push(1);
            bytes.extend(v.to_le_bytes());
        }
        None => bytes.push(0),
    }
    match sim.period_ns {
        Some(v) => {
            bytes.push(1);
            bytes.extend(v.to_le_bytes());
        }
        None => bytes.push(0),
    }
    bytes.push(match sim.mode {
        Mode::Clocked => 0,
        Mode::EventDriven => 1,
    });
    let ledger = &sim.ledger;
    for value in [
        ledger.syn_ops,
        ledger.syn_fetches,
        ledger.neuron_updates_idle,
        ledger.neuron_updates_driven,
        ledger.spikes_out,
        ledger.reads,
    ] {
        bytes.extend(value.to_le_bytes());
    }
    for neuron in &sim.neurons {
        for value in [
            neuron.tau_m,
            neuron.v_rest,
            neuron.v_th,
            neuron.v_reset,
            neuron.r_m,
            neuron.t_ref,
            neuron.v,
            neuron.refractory,
        ] {
            bytes.extend(value.to_bits().to_le_bytes());
        }
    }
    bytes
}

#[test]
fn a_zero_step_count_checks_and_stamps_nothing() {
    flags_stay_false();
    let dt = nominal_seconds(PERIOD).unwrap();
    let mut builder = NetBuilder::new(2);
    builder.connect(0, 1, 1e-3, 1).unwrap();
    let mut sim = Sim::new(builder.build(), vec![Lif::default(); 2], dt, Mode::Clocked).unwrap();
    sim.episode_start(0, PERIOD).unwrap();
    let tick = sim.t;
    let fired = sim.try_step(&[]).unwrap();
    let receipt = StepReceipt::from_step(sim.epoch, tick, sim.dt, sim.mode, &fired);
    let receipt_bytes = receipt.to_text().into_bytes();
    let state_bytes = sim_public_bytes(&sim);
    let (envelope, clock) = pair(PERIOD);
    let clock_before = clock;
    assert_eq!(clock.dt_bits, sim.dt.to_bits());
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 0, Some(envelope), &clock),
        Ok(())
    );
    assert_eq!(clock, clock_before);
    assert_eq!(receipt.to_text().into_bytes(), receipt_bytes);
    assert_eq!(sim_public_bytes(&sim), state_bytes);
    assert_eq!(StepReceipt::from_text(&receipt.to_text()).unwrap(), receipt);
}

#[test]
fn periods_above_2_pow_53_are_inexact_and_the_bound_is_accepted() {
    flags_stay_false();
    let pow = 1u64 << 53;
    let (envelope, mut clock) = pair(PERIOD);

    clock.period_ns = Some(pow + 1);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::InexactPeriod)
    );
    clock.period_ns = Some(u64::MAX);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::InexactPeriod)
    );
    clock.period_ns = Some(PERIOD);
    assert_eq!(
        bind_v2_timestamps(0, pow + 1, 1, Some(envelope), &clock),
        Err(TimebaseError::InexactPeriod)
    );
    assert_eq!(
        bind_v2_timestamps(0, u64::MAX, 0, Some(envelope), &clock),
        Err(TimebaseError::InexactPeriod)
    );

    let (exact, exact_clock) = pair(pow);
    assert_eq!(
        bind_v2_timestamps(0, pow, 1, Some(exact), &exact_clock),
        Ok(())
    );
    assert_eq!(
        bind_v2_timestamps(0, pow, 0, Some(exact), &exact_clock),
        Ok(())
    );
}

#[test]
fn epoch_origin_period_dt_bits_and_the_t0_grid_are_refused() {
    flags_stay_false();
    let (envelope, mut clock) = pair(PERIOD);

    clock.epoch = 9;
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::StaleEpoch { event: 1, clock: 9 })
    );

    clock.epoch = 1;
    clock.origin_ns = Some(PERIOD);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::AmbiguousEpoch)
    );

    clock.origin_ns = Some(0);
    clock.period_ns = Some(PERIOD * 2);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::AmbiguousEpoch)
    );

    clock.period_ns = Some(PERIOD);
    let flipped = envelope.dt_bits() ^ 1;
    clock.dt_bits = flipped;
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::Mismatch {
            left: flipped,
            right: envelope.dt_bits()
        })
    );

    clock.dt_bits = envelope.dt_bits();
    assert_eq!(
        bind_v2_timestamps(1, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::OffGrid)
    );
}

#[test]
fn a_snapshot_field_changed_after_a_good_bind_is_refused_on_the_next_call() {
    flags_stay_false();
    let (envelope, mut clock) = pair(PERIOD);
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Ok(())
    );
    clock.dt_bits ^= 1;
    assert_eq!(
        bind_v2_timestamps(0, PERIOD, 1, Some(envelope), &clock),
        Err(TimebaseError::Mismatch {
            left: clock.dt_bits,
            right: envelope.dt_bits()
        })
    );
}

#[test]
fn a_forward_product_mismatch_is_refused_before_an_envelope_exists() {
    flags_stay_false();
    // `EpisodeEnvelope`'s fields are private. `try_new` is the only constructor, and it calls
    // `require_forward_product` before returning. `bind_v2_timestamps` therefore cannot be given
    // an envelope whose `dt` bits disagree with the period. The refusal is `DtPeriodMismatch`
    // from `try_new`, not a bind result.
    let bad = 0x3eb0c6f7a0b5ed8d_u64;
    let nominal = bits(PERIOD);
    assert_ne!(bad, nominal);
    assert_eq!(
        EpisodeEnvelope::try_new(1, 1, bad, PERIOD, 0),
        Err(TimebaseError::DtPeriodMismatch {
            dt_bits: bad,
            period_ns: PERIOD,
            nominal_bits: nominal
        })
    );
}
