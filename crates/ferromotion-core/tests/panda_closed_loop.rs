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
        history.remove(0);
        history.push(counts_now);
        let counts: Vec<u64> = history.iter().flatten().copied().collect();
        let mut tau = readout.apply(&counts).unwrap().output;
        for (i, torque) in tau.iter_mut().enumerate() {
            *torque = torque.clamp(-TAU_LIM[i], TAU_LIM[i]);
        }
        data.applied.qfrc = tau.clone();
        data.ctrl.fill(0.0);
        model.step(&mut data).unwrap();
        for value in data.qpos.iter().chain(data.qvel.iter()) {
            assert!(value.is_finite(), "tick {k}");
        }
        let t0 = tick * PERIOD_NS;
        assert_eq!(
            bind_v2_timestamps(t0, PERIOD_NS, 1, Some(envelope), &sim.clock_snapshot()),
            Ok(())
        );
        let record = format!(
            "tick {k}\nobs {}\namp {}\nspikes {}\ntau {}\nqpos {}\nqvel {}\n{}",
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
    assert_eq!(first.ticks.len(), N);
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
        "f0b6b4fb27384ea9a4e0542655028a603843f1f59af19c50fe81980cd849a283",
        "83f9efd472fbf0bd43d08904c11580962ea431ebac6f4e183640c3c997c0d0b2",
        "d11a0f8df3e32895691c2a66b76df76cae9d26e70444f5a66fce5197067eee91",
        "0209e8e7a701d680e84b47ca35483ec3aeadd52c56a7c44a187db1bc6899e967",
        "26208d19502d98025ee87b190f2ce46353a1fa7d330b0e08257905d40d076fde",
        "8fdb5964ddd56242e6610cf655413809d23ae93f2d2e09d4008134563cf9f684",
        "e5e4f23c7d2f314f842f7ae07a8041468d5f19140170fc5bc4ff56847b8e53a2",
        "49cf979a2df2daca9e6d421447fcda78ea14b7e2970fe5c091025dd95a90622a",
        "0d5320f40dc92c1d90876f0b05dd65294e702a942c05c82c02a5c9fb40f9a01a",
        "4f7625d7fc8f30d7173af1b66cc442dcf36b8297a538b621284deae5962278db",
        "acfa06a0f20580e9bbcde7bf422613b34b920beb7d3152845fb987ced8958186",
        "4d2a4ef4b517315322450382fce0690a74cf6b7bebaa11b33af64ab88c639c9a",
        "c9fa92c1ae74ddf07ff8cc32baf5a8b127b2e85e676f7b426c0b320f0c288163",
        "96636dc808673c50532a4cb985afd64102073a33e62dadbdfb7f3089e79067c0",
        "ae78d91d1b6f11f69c1d272b760538949de8e32c97c1e9b0bb0e38baacd0434e",
        "ef87fb1942a177b548a0d6c4b83a4c6bd49bb07a7db7a19af2dce17b8968b0ae",
    ];
    assert_eq!(first.ticks, TICKS);
    eprintln!("ENVELOPE {}", first.envelope);
    eprintln!("STATE {}", first.state_bytes);
    eprintln!("PLANT {}", first.plant);
    for (i, tick) in first.ticks.iter().enumerate() {
        eprintln!("TICK {i} {tick}");
    }
}
