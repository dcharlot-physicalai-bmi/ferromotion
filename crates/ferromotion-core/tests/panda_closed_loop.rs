//! One executed episode on `franka_emika_panda/panda_nohand.xml`.
//!
//! The plant is [`ferromotion_core::MjcfTree::step`]. The controller is ferromorphic
//! `CurrentEncoder`, `Sim::try_step`, and `SpikeReadout`, then the agreed torque saturation.
//! `JOINT_LOOP_EXECUTED` is this episode, pending bmi-2 review.

use ferromorphic::NetBuilder;
use ferromorphic::neuron::Lif;
use ferromorphic::readout::{CurrentEncoder, SpikeReadout};
use ferromorphic::sim::{Mode, Sim, StepReceipt, bind_v2_timestamps, nominal_seconds};
use ferromotion_core::{
    EPISODE_ENVELOPE_INTEROPERABLE, EpisodeEnvelope, FERROMORPHIC_BIND_VERIFIED,
    FERROMORPHIC_CLOCK_REVISION, JOINT_LOOP_EXECUTED, digest_sha256, tree_from_mjcf,
};

const PIN: &str = "82211dc861af4ecc1afa9603c2caf20d52712b6d";
const MENAGERIE: &str = "4d038b3feae26ec82b46a4d586379114012a8ac7";
const PERIOD_NS: u64 = 2_000_000;
const N: usize = 16;
const WINDOW: usize = 4;
const NEURONS: usize = 14;
const QPOS_LO: [f64; 7] = [
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
];
const QPOS_HI: [f64; 7] = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973];
const TAU_LIM: [f64; 7] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0];

struct EpisodeTrace {
    ticks: Vec<String>,
    state_bytes: String,
    envelope: String,
    plant: String,
    spikes: u64,
    first_spike: Option<usize>,
    nonzero_tau_ticks: usize,
    max_abs_tau: [f64; 7],
    ncon: Vec<usize>,
}

struct ControlTrace {
    plant: String,
    ncon: Vec<usize>,
}

/// A 1 mm cube. The file's bodies state `fullinertia`, so this stand-in is only what the mesh
/// compiler requires in place of the Menagerie STL and OBJ assets. It is not the collision shape
/// of the published meshes.
fn cube_verts() -> [[f32; 3]; 8] {
    let s = 0.001f32;
    [
        [0.0, 0.0, 0.0],
        [s, 0.0, 0.0],
        [s, s, 0.0],
        [0.0, s, 0.0],
        [0.0, 0.0, s],
        [s, 0.0, s],
        [s, s, s],
        [0.0, s, s],
    ]
}

const CUBE_FACES: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [1, 2, 6],
    [1, 6, 5],
    [2, 3, 7],
    [2, 7, 6],
    [3, 0, 4],
    [3, 4, 7],
];

fn tiny_obj() -> Vec<u8> {
    let mut text = String::new();
    for v in cube_verts() {
        text.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
    }
    for f in CUBE_FACES {
        text.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
    }
    text.into_bytes()
}

fn tiny_stl() -> Vec<u8> {
    let verts = cube_verts();
    let mut bytes = vec![0u8; 80];
    bytes.extend_from_slice(&(CUBE_FACES.len() as u32).to_le_bytes());
    for face in CUBE_FACES {
        for _ in 0..3 {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for index in face {
            for coord in verts[index] {
                bytes.extend_from_slice(&coord.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
    }
    bytes
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn load_panda() -> ferromotion_core::MjcfTree {
    let xml = include_str!("fixtures/panda_nohand.xml");
    assert!(
        xml.starts_with("<mujoco model=\"panda nohand\">"),
        "{MENAGERIE}"
    );
    let stl = tiny_stl();
    let obj = tiny_obj();
    let mut model = tree_from_mjcf(xml, &|path| {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".obj") {
            Some(obj.clone())
        } else if lower.ends_with(".stl") {
            Some(stl.clone())
        } else {
            None
        }
    })
    .expect("panda_nohand loads");
    assert_eq!(model.joints.len(), 7);
    assert_eq!(model.actuators.len(), 7);
    for (i, joint) in model.joints.iter().enumerate() {
        assert_eq!(joint.name, format!("joint{}", i + 1));
        let (lo, hi) = joint.range.expect("limited hinge");
        assert_eq!(lo.to_bits(), QPOS_LO[i].to_bits());
        assert_eq!(hi.to_bits(), QPOS_HI[i].to_bits());
    }
    for (i, actuator) in model.actuators.iter().enumerate() {
        let [lo, hi] = actuator.forcerange.expect("forcerange");
        assert_eq!(lo.to_bits(), (-TAU_LIM[i]).to_bits());
        assert_eq!(hi.to_bits(), TAU_LIM[i].to_bits());
    }
    let dt = nominal_seconds(PERIOD_NS).unwrap();
    assert_eq!(model.timestep.to_bits(), dt.to_bits());
    // These general actuators are position servos. The agreed command is joint torque in
    // newton-metres, applied as qfrc_applied after the forcerange saturation. Zeroing the
    // servo gain and bias leaves that torque as the actuator contribution. Inertia, joint
    // damping, armature, gravity, and the implicitfast step stay as the file compiled them.
    for actuator in &mut model.actuators {
        actuator.gainprm[0] = 0.0;
        actuator.biasprm[0] = 0.0;
        actuator.biasprm[1] = 0.0;
        actuator.biasprm[2] = 0.0;
    }
    model
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
    let epoch = sim.episode_start(0, PERIOD_NS).unwrap();
    assert_eq!(epoch, 1);

    // 5e-9 A/rad and a 3e-9 A offset. At the home pose the zero-position joints sit at 3 nA,
    // whose steady potential is above the default LIF threshold. The 0.05 N·m weight stays
    // inside the smallest torque limit, 12 N·m.
    let scale = vec![5e-9; NEURONS];
    let offset = vec![3e-9; NEURONS];
    let lo = vec![-1e-6; NEURONS];
    let hi = vec![1e-6; NEURONS];
    let encoder = CurrentEncoder::try_new(&scale, &offset, &lo, &hi).unwrap();

    let inputs = WINDOW * NEURONS;
    let mut weights = vec![0.0; 7 * inputs];
    let mut bias = vec![0.0; 7];
    let mut tau_lo = vec![0.0; 7];
    let mut tau_hi = vec![0.0; 7];
    for channel in 0..7 {
        bias[channel] = 0.0;
        tau_lo[channel] = -TAU_LIM[channel];
        tau_hi[channel] = TAU_LIM[channel];
        for input in 0..inputs {
            if input % 7 == channel {
                weights[channel * inputs + input] = 0.05;
            }
        }
    }
    let readout =
        SpikeReadout::try_new(WINDOW as u64, inputs, &weights, &bias, &tau_lo, &tau_hi).unwrap();
    (sim, readout, encoder)
}

fn exchanged_envelope(
    sim: &Sim<Lif>,
) -> (EpisodeEnvelope, ferromorphic::sim::EpisodeEnvelope, String) {
    let ours = EpisodeEnvelope::try_new(1, sim.epoch, sim.dt.to_bits(), PERIOD_NS, 0).unwrap();
    let text = ours.to_text();
    let theirs = ferromorphic::sim::EpisodeEnvelope::from_text(&text).unwrap();
    assert_eq!(theirs.to_text().as_bytes(), text.as_bytes());
    let back = EpisodeEnvelope::from_text(&theirs.to_text()).unwrap();
    assert_eq!(back, ours);
    assert_eq!(back.to_text().as_bytes(), text.as_bytes());
    let reverse = theirs.to_text();
    let ours_from_them = EpisodeEnvelope::from_text(&reverse).unwrap();
    assert_eq!(ours_from_them.to_text().as_bytes(), reverse.as_bytes());
    let them_again =
        ferromorphic::sim::EpisodeEnvelope::from_text(&ours_from_them.to_text()).unwrap();
    assert_eq!(them_again.to_text().as_bytes(), reverse.as_bytes());
    (ours, theirs, text)
}

fn run_episode() -> EpisodeTrace {
    let model = load_panda();
    let (mut sim, readout, encoder) = controller();
    let (_ours, envelope, envelope_text) = exchanged_envelope(&sim);
    let mut data = model.make_data();
    let home = model
        .keyframes
        .iter()
        .position(|k| k.name == "home")
        .expect("home key");
    model.reset_data_keyframe(&mut data, home).unwrap();
    assert_eq!(data.qpos.len(), 7);
    assert_eq!(data.qvel.len(), 7);
    assert_eq!(data.ctrl.len(), 7);

    let mut history = vec![vec![0u64; NEURONS]; WINDOW];
    let mut ticks = Vec::with_capacity(N);
    let mut ncon_ticks = Vec::with_capacity(N);
    let mut spikes = 0u64;
    let mut first_spike = None;
    let mut nonzero_tau_ticks = 0usize;
    let mut max_abs_tau = [0.0f64; 7];
    for k in 0..N {
        let mut observation = Vec::with_capacity(14);
        for (i, q) in data.qpos.iter().enumerate() {
            observation.push(q.clamp(QPOS_LO[i], QPOS_HI[i]));
        }
        observation.extend_from_slice(&data.qvel);
        assert_eq!(observation.len(), 14);
        for v in &observation {
            assert!(v.is_finite(), "tick {k}");
        }
        let amperes = encoder.apply(&observation).unwrap().output;
        let tick = sim.t;
        let fired = sim.try_step(&amperes).unwrap();
        let receipt = StepReceipt::from_step(sim.epoch, tick, sim.dt, sim.mode, &fired);
        let mut counts_now = vec![0u64; NEURONS];
        for &id in &fired {
            counts_now[id as usize] += 1;
        }
        spikes += fired.len() as u64;
        if first_spike.is_none() && !fired.is_empty() {
            first_spike = Some(k);
        }
        history.remove(0);
        history.push(counts_now);
        let counts: Vec<u64> = history.iter().flatten().copied().collect();
        let tau = readout.apply(&counts).unwrap().output;
        for (i, torque) in tau.iter().enumerate() {
            let limited = torque.clamp(-TAU_LIM[i], TAU_LIM[i]);
            assert_eq!(
                torque.to_bits(),
                limited.to_bits(),
                "tick {k} joint {}: readout {torque} is outside ±{}",
                i + 1,
                TAU_LIM[i]
            );
        }
        data.applied.qfrc = tau.clone();
        data.ctrl.fill(0.0);
        let stepped = model.step(&mut data).unwrap();
        let ncon = stepped
            .forward
            .collision
            .contacts
            .iter()
            .filter(|c| !c.record.exclude)
            .count();
        assert_eq!(ncon, 0, "tick {k} ncon");
        assert!(
            stepped.forward.collision.contacts.is_empty(),
            "tick {k} detected {}",
            stepped.forward.collision.contacts.len()
        );
        ncon_ticks.push(ncon);
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
            nonzero_tau_ticks += 1;
        }
        for value in data.qpos.iter().chain(data.qvel.iter()) {
            assert!(value.is_finite(), "tick {k}");
        }
        let t0 = tick * PERIOD_NS;
        assert_eq!(
            bind_v2_timestamps(t0, PERIOD_NS, 1, Some(envelope), &sim.clock_snapshot()),
            Ok(())
        );
        let record = format!(
            "tick {k}\nncon {ncon}\nobs {}\namp {}\nspikes {}\ntau {}\nqpos {}\nqvel {}\n{}",
            bits(&observation),
            bits(&amperes),
            fired
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            bits(&tau),
            bits(&data.qpos),
            bits(&data.qvel),
            receipt.to_text()
        );
        ticks.push(digest_sha256(record.as_bytes()));
    }
    EpisodeTrace {
        ticks,
        state_bytes: digest_sha256(&sim.state_bytes()),
        envelope: digest_sha256(envelope_text.as_bytes()),
        plant: digest_sha256(format!("{}\n{}", bits(&data.qpos), bits(&data.qvel)).as_bytes()),
        spikes,
        first_spike,
        nonzero_tau_ticks,
        max_abs_tau,
        ncon: ncon_ticks,
    }
}

fn run_zero_torque() -> ControlTrace {
    let model = load_panda();
    let mut data = model.make_data();
    let home = model
        .keyframes
        .iter()
        .position(|k| k.name == "home")
        .expect("home key");
    model.reset_data_keyframe(&mut data, home).unwrap();
    let mut ncon_ticks = Vec::with_capacity(N);
    for k in 0..N {
        data.applied.qfrc = vec![0.0; 7];
        data.ctrl.fill(0.0);
        let stepped = model.step(&mut data).unwrap();
        let ncon = stepped
            .forward
            .collision
            .contacts
            .iter()
            .filter(|c| !c.record.exclude)
            .count();
        assert_eq!(ncon, 0, "control tick {k} ncon");
        assert!(
            stepped.forward.collision.contacts.is_empty(),
            "control tick {k}"
        );
        ncon_ticks.push(ncon);
        for value in data.qpos.iter().chain(data.qvel.iter()) {
            assert!(value.is_finite(), "control tick {k}");
        }
    }
    ControlTrace {
        plant: digest_sha256(format!("{}\n{}", bits(&data.qpos), bits(&data.qvel)).as_bytes()),
        ncon: ncon_ticks,
    }
}

#[test]
fn panda_nohand_episode_is_byte_identical_on_a_second_run() {
    assert_eq!(FERROMORPHIC_CLOCK_REVISION, PIN);
    const {
        assert!(FERROMORPHIC_BIND_VERIFIED);
    }
    const {
        assert!(JOINT_LOOP_EXECUTED);
    }
    const {
        assert!(!EPISODE_ENVELOPE_INTEROPERABLE);
    }
    let first = run_episode();
    let second = run_episode();
    assert_eq!(first.ticks, second.ticks);
    assert_eq!(first.state_bytes, second.state_bytes);
    assert_eq!(first.envelope, second.envelope);
    assert_eq!(first.plant, second.plant);
    assert_eq!(first.spikes, second.spikes);
    assert_eq!(first.first_spike, second.first_spike);
    assert_eq!(first.nonzero_tau_ticks, second.nonzero_tau_ticks);
    assert_eq!(first.ncon, second.ncon);
    assert_eq!(first.ticks.len(), N);
    assert!(first.spikes > 0, "spike count {}", first.spikes);
    assert_eq!(first.spikes, 52);
    let first_spike = first.first_spike.expect("a tick with a spike");
    assert_eq!(first_spike, 1);
    assert!(
        first.nonzero_tau_ticks > 0,
        "nonzero tau ticks {}",
        first.nonzero_tau_ticks
    );
    assert_eq!(first.nonzero_tau_ticks, 15);
    let max_bits: [u64; 7] = [
        0x3fc999999999999a,
        0x3fc999999999999a,
        0x3fc999999999999a,
        0x3fb999999999999a,
        0x3fb999999999999a,
        0x3fc3333333333334,
        0x3fc999999999999a,
    ];
    for (i, mag) in first.max_abs_tau.iter().enumerate() {
        assert_eq!(mag.to_bits(), max_bits[i], "joint {}", i + 1);
        assert_eq!(second.max_abs_tau[i].to_bits(), max_bits[i]);
    }
    assert!(first.ncon.iter().all(|n| *n == 0));
    let control = run_zero_torque();
    let control_again = run_zero_torque();
    assert_eq!(control.plant, control_again.plant);
    assert_eq!(control.ncon, control_again.ncon);
    assert!(control.ncon.iter().all(|n| *n == 0));
    assert_ne!(control.plant, first.plant);
    assert_eq!(
        control.plant,
        "0041db4cb786ce8a7ad90624804aede87b00eda1da873e6aef9bc10d520432a5"
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
        "b6ed1ec1ca5d2086865d82a056ed16871198f2ebdeb233d6115670b028a3b6e0"
    );
    const TICKS: [&str; N] = [
        "42a48d5af5c79d62600e463e22e4b395dc387eb1a827cac416f7046c1aef6863",
        "07d7e42a0055bb9c121d330a82981dad36060dfcc205dc020daf76681ae8aa83",
        "066bee7c8664e92db77590a09e6c07d9ff3a21f9a1e5d7b4e3a12050086537c2",
        "b2ab3175268745b8caee23e38d1e6fa58376a6c5ccebc4b10103aa0448193925",
        "1a398f6ffe542d075a507ce2f66466854ed0befa4f6749cb4806e999c16e2298",
        "eb8130fc8c878fbdae01aa8a8771a8ca7aeccea7cd2f188890bb14acc20b570a",
        "823c7aeaade6d0776231e9d4603a7ad012babb3b2c9132f81f98c0a2cc6339e5",
        "18a2de87ef8aa2a1fea39e4a25ddfcabb7330fff86fda38011c9dc57a533d4cb",
        "23e7bdcae03b8aa038337ee30c7020fb0a18fa62116f29bf76b6487566ae20c6",
        "49efabe6b7aa84f6b4f6d34726dd4a0617198dfc7de6e805ba0e571f7953cccd",
        "87df866cb055b1e01e0a2e995356c6f1c462adddf5c5bb7603f329613eaafd3c",
        "51628f2ee596f0522f5647b1a0ccdada9db3b2091a1ced557fb03da6a13ed696",
        "35742d9c469b468c254705813e2d9c809e5baeb7fdc95c91bd49e3df65692250",
        "7ce54b5b94d224d6317a44a98af9d19c465a04b16c157bab6f8b70335231450a",
        "d2e919dfa2eb325456d5d6f3b8e3c5923fb1a5a815c6da252b4d48218ce68a31",
        "9518d4d6641cdf55f0aee23405b8be704273e9e85f9cb72d336d1bab6bd24a6d",
    ];
    assert_eq!(first.ticks, TICKS);
    eprintln!(
        "SPIKES {} FIRST {first_spike} TAU_TICKS {}",
        first.spikes, first.nonzero_tau_ticks
    );
    eprintln!("MAX_TAU {:?}", first.max_abs_tau);
    eprintln!("ENVELOPE {}", first.envelope);
    eprintln!("STATE {}", first.state_bytes);
    eprintln!("PLANT {}", first.plant);
    eprintln!("CONTROL {}", control.plant);
    for (i, tick) in first.ticks.iter().enumerate() {
        eprintln!("TICK {i} {tick}");
    }
}
