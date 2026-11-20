//! Contact episode on the real `panda_nohand` meshes plus the scene floor.
//!
//! The contact-free receipt is unchanged. This file does not set a new flag.

use std::fs;
use std::path::PathBuf;

use ferromorphic::NetBuilder;
use ferromorphic::neuron::Lif;
use ferromorphic::readout::{CurrentEncoder, SpikeReadout};
use ferromorphic::sim::{Mode, Sim, StepReceipt, bind_v2_timestamps, nominal_seconds};
use ferromotion_core::{
    EPISODE_ENVELOPE_INTEROPERABLE, EpisodeEnvelope, FERROMORPHIC_BIND_VERIFIED,
    FERROMORPHIC_CLOCK_REVISION, FERROMORPHIC_ENVELOPE_TEXT_INTEROPERABLE, JOINT_LOOP_EXECUTED,
    digest_sha256, tree_from_mjcf,
};

const PERIOD_NS: u64 = 2_000_000;
const N: usize = 16;
const WINDOW: usize = 4;
const NEURONS: usize = 14;
const QPOS_LO: [f64; 7] = [
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
];
const QPOS_HI: [f64; 7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
const TAU_LIM: [f64; 7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/franka_emika_panda")
}

fn load() -> ferromotion_core::MjcfTree {
    let dir = root();
    let xml = fs::read_to_string(dir.join("contact_scene.xml")).unwrap();
    let mut model = tree_from_mjcf(&xml, &|path| {
        let file = dir.join(path);
        if let Ok(bytes) = fs::read(&file) {
            Some(bytes)
        } else {
            fs::read(dir.join("assets").join(path)).ok()
        }
    })
    .unwrap_or_else(|err| panic!("load {err}"));
    assert_eq!(model.joints.len(), 7, "panda_nohand dofs");
    let dt = nominal_seconds(PERIOD_NS).unwrap();
    assert_eq!(model.timestep.to_bits(), dt.to_bits());
    for actuator in &mut model.actuators {
        actuator.gainprm[0] = 0.0;
        actuator.biasprm[0] = 0.0;
        actuator.biasprm[1] = 0.0;
        actuator.biasprm[2] = 0.0;
    }
    model
}

fn ncon_of(step: &ferromotion_core::MjcfStep) -> usize {
    step.forward
        .collision
        .contacts
        .iter()
        .filter(|c| !c.record.exclude)
        .count()
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

struct Trace {
    ticks: Vec<String>,
    ncon: Vec<usize>,
    plant: String,
    state_bytes: String,
    envelope: String,
    spikes: u64,
    first_spike: Option<usize>,
    nonzero_tau: usize,
    max_abs_tau: [f64; 7],
}

fn controller() -> (Sim<Lif>, SpikeReadout, CurrentEncoder) {
    let dt = nominal_seconds(PERIOD_NS).unwrap();
    let mut builder = NetBuilder::new(NEURONS);
    for i in 0..NEURONS {
        builder
            .connect(i as u32, ((i + 1) % NEURONS) as u32, 0.02, 1)
            .unwrap();
    }
    let mut sim = Sim::new(
        builder.build(),
        vec![Lif::default(); NEURONS],
        dt,
        Mode::Clocked,
    )
    .unwrap();
    assert_eq!(sim.episode_start(0, PERIOD_NS).unwrap(), 1);
    let encoder = CurrentEncoder::try_new(
        &vec![5e-9; NEURONS],
        &vec![3e-9; NEURONS],
        &vec![-1e-6; NEURONS],
        &vec![1e-6; NEURONS],
    )
    .unwrap();
    let inputs = WINDOW * NEURONS;
    let mut weights = vec![0.0; 7 * inputs];
    let mut lo = vec![0.0; 7];
    let mut hi = vec![0.0; 7];
    for channel in 0..7 {
        lo[channel] = -TAU_LIM[channel];
        hi[channel] = TAU_LIM[channel];
        for input in 0..inputs {
            if input % 7 == channel {
                weights[channel * inputs + input] = 0.05;
            }
        }
    }
    let readout =
        SpikeReadout::try_new(WINDOW as u64, inputs, &weights, &vec![0.0; 7], &lo, &hi).unwrap();
    (sim, readout, encoder)
}

fn run(with_controller: bool) -> Trace {
    let model = load();
    let (mut sim, readout, encoder) = controller();
    let ours = EpisodeEnvelope::try_new(1, sim.epoch, sim.dt.to_bits(), PERIOD_NS, 0).unwrap();
    let text = ours.to_text();
    let envelope = ferromorphic::sim::EpisodeEnvelope::from_text(&text).unwrap();
    assert_eq!(envelope.to_text().as_bytes(), text.as_bytes());
    let mut data = model.make_data();
    let home = model
        .keyframes
        .iter()
        .position(|k| k.name == "home")
        .unwrap();
    model.reset_data_keyframe(&mut data, home).unwrap();
    assert_eq!(data.qpos.len(), 7);
    let mut history = vec![vec![0u64; NEURONS]; WINDOW];
    let mut ticks = Vec::new();
    let mut ncons = Vec::new();
    let mut spikes = 0u64;
    let mut first_spike = None;
    let mut nonzero_tau = 0usize;
    let mut max_abs_tau = [0.0; 7];
    for k in 0..N {
        let mut observation = Vec::with_capacity(14);
        for (i, q) in data.qpos.iter().enumerate() {
            observation.push(q.clamp(QPOS_LO[i], QPOS_HI[i]));
        }
        observation.extend_from_slice(&data.qvel);
        let amperes = encoder.apply(&observation).unwrap().output;
        let tick = sim.t;
        let fired = sim.try_step(&amperes).unwrap();
        let receipt = StepReceipt::from_step(sim.epoch, tick, sim.dt, sim.mode, &fired);
        spikes += fired.len() as u64;
        if first_spike.is_none() && !fired.is_empty() {
            first_spike = Some(k);
        }
        let mut now = vec![0u64; NEURONS];
        for &id in &fired {
            now[id as usize] += 1;
        }
        history.remove(0);
        history.push(now);
        let counts: Vec<u64> = history.iter().flatten().copied().collect();
        let tau = if with_controller {
            let tau = readout.apply(&counts).unwrap().output;
            for (i, torque) in tau.iter().enumerate() {
                let limited = torque.clamp(-TAU_LIM[i], TAU_LIM[i]);
                assert_eq!(
                    torque.to_bits(),
                    limited.to_bits(),
                    "tick {k} joint {}",
                    i + 1
                );
            }
            tau
        } else {
            vec![0.0; 7]
        };
        let mut any_tau = false;
        for (i, torque) in tau.iter().enumerate() {
            let mag = torque.abs();
            if mag.to_bits() != 0.0f64.to_bits() {
                any_tau = true;
            }
            if mag > max_abs_tau[i] {
                max_abs_tau[i] = mag;
            }
        }
        if any_tau {
            nonzero_tau += 1;
        }
        data.applied.qfrc = tau.clone();
        data.ctrl.fill(0.0);
        let stepped = model.step(&mut data).unwrap();
        let ncon = ncon_of(&stepped);
        assert_eq!(
            bind_v2_timestamps(
                tick * PERIOD_NS,
                PERIOD_NS,
                1,
                Some(envelope),
                &sim.clock_snapshot()
            ),
            Ok(())
        );
        ncons.push(ncon);
        let record = format!(
            "tick {k}\nncon {ncon}\nobs {}\ntau {}\nqpos {}\nqvel {}\n{}",
            bits(&observation),
            bits(&tau),
            bits(&data.qpos),
            bits(&data.qvel),
            receipt.to_text()
        );
        ticks.push(digest_sha256(record.as_bytes()));
    }
    Trace {
        ticks,
        ncon: ncons,
        plant: digest_sha256(format!("{}\n{}", bits(&data.qpos), bits(&data.qvel)).as_bytes()),
        state_bytes: digest_sha256(&sim.state_bytes()),
        envelope: digest_sha256(text.as_bytes()),
        spikes,
        first_spike,
        nonzero_tau,
        max_abs_tau,
    }
}

#[test]
fn panda_contact_episode_touches_and_differs_from_zero_torque() {
    assert_eq!(
        FERROMORPHIC_CLOCK_REVISION,
        "82211dc861af4ecc1afa9603c2caf20d52712b6d"
    );
    const {
        assert!(FERROMORPHIC_BIND_VERIFIED);
    }
    const {
        assert!(JOINT_LOOP_EXECUTED);
    }
    const {
        assert!(FERROMORPHIC_ENVELOPE_TEXT_INTEROPERABLE);
    }
    const {
        assert!(!EPISODE_ENVELOPE_INTEROPERABLE);
    }
    let first = run(true);
    let second = run(true);
    assert_eq!(first.ticks, second.ticks);
    assert_eq!(first.plant, second.plant);
    assert_eq!(first.ncon, second.ncon);
    assert_eq!(first.state_bytes, second.state_bytes);
    assert_eq!(first.spikes, 52);
    assert_eq!(first.first_spike, Some(1));
    assert_eq!(first.nonzero_tau, 15);
    assert_eq!(first.ncon, vec![4; N]);
    let max_bits: Vec<u64> = first.max_abs_tau.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        max_bits,
        vec![
            0x3fc999999999999a,
            0x3fc999999999999a,
            0x3fc999999999999a,
            0x3fb999999999999a,
            0x3fb999999999999a,
            0x3fc3333333333334,
            0x3fc999999999999a,
        ]
    );
    assert_eq!(
        first.envelope,
        "170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a"
    );
    assert_eq!(
        first.state_bytes,
        "b2bfb4081bcd7e2de05fcd249c65b2c5a5cfe7817850eac854467df32ea3a126"
    );
    assert_eq!(
        first.plant,
        "9807515212ffc7c673100e9aa586f66d789504b052178b93b628a5883b131535"
    );
    assert_eq!(
        first.ticks,
        [
            "0aa21599ee8eeefd21101508ce69279a1fa994f04662d264decdc16bcd373574",
            "968d22bd5f41d0d9804f179c71d94b8d4861069812e9d2867982f07ce2868f1e",
            "3c99a3a300e5db8d4d7277d3055fb2be9c9b611ec96d05422d08c3345e55d19e",
            "fd5e15f960c8ba945bfcfe0943b5e9363d53bb4c4b1498b970d538e77c126fd6",
            "ba24a66fa952ee5c8e210b4c4b6c3851f83c03bc5a08cdfb9282cb10ac5ea630",
            "db5beec23e53e50b4738a4200de9d11b9a66a7996a53351812415230453ca478",
            "13354c71f3d9763a582557c373fb37ec56f4c2d473e717ac5f908c7ace7dc726",
            "5a532ec8bc72a70cecefec348c5a3f40cf171858c6b46848e3b2699a0c50eed5",
            "38a21f3a5afa90844cbd478ac339aae73ce332ea14f69316b5d60ad9b8a7067f",
            "a4b573fc3916b708ab21e04462b62226f4a138d8f70beeee2f0a2e637addcddc",
            "372d1ad4cf26e11a16618fee7560bf0a3605967151a18a16eee2ffca9067bcfb",
            "da38b138a4973228b45a8a8492b6341ec0bfb2fc4ead5b006f7418a48c9cc620",
            "4577dcd998b2c5a574af4caadc9b51a65d36157fde48e956410bd79c1aed2768",
            "e6c6a445a3d7f90fcb154f6138667553b1a00fe39261449f49dda00f23a3d360",
            "7e22d7e476354cbac6bb21a1abad974097f82906425ec30ba4963352780521a0",
            "11e43eeca393545d0cc2e9bd27bf774c1966124a53fc4ce47c79ef5fbc6d43fd",
        ]
    );
    let control = run(false);
    let control_again = run(false);
    assert_eq!(control.plant, control_again.plant);
    assert_eq!(control.ncon, vec![4; N]);
    assert_eq!(
        control.plant,
        "455d169e610ce94c850b3609a020bf3bc1d7a0c01abf28ff3789637fb15794c8"
    );
    assert_ne!(control.plant, first.plant);
}
