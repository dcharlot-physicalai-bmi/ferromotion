//! Corrected contact episode on the contract-off native stepper.
//!
//! `98ce3db1` stays. This episode does not set a flag. It is pending bmi-2 review.
//!
//! The plant is [`ferromotion_core::MjcfTree::mujoco_native_step`], the stepper behind
//! `menagerie_native_step`, after installing `mujoco-3.13.0-x86_64-contract-off`.

use std::fs;
use std::path::PathBuf;

use ferromorphic::NetBuilder;
use ferromorphic::neuron::Lif;
use ferromorphic::readout::{CurrentEncoder, SpikeReadout};
use ferromorphic::sim::{Mode, Sim, StepReceipt, bind_v2_timestamps, nominal_seconds};
use ferromotion_core::mujoco_actuator::{ActBias, ActGain};
use ferromotion_core::{
    EpisodeEnvelope, FERROMORPHIC_CLOCK_REVISION, MjNativeState, ORACLE_X86_CONTRACT_OFF_ID,
    ORACLE_X86_CONTRACT_OFF_SHA256, digest_sha256, install_policy, reset_policy, tree_from_mjcf,
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
/// Wrist and forearm on the floor. link5–link7 meet the plane; link0 does not.
const REACH: [f64; 7] = [0.0, 0.8, 0.0, -2.2, 0.0, 2.2, 0.0];

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        reset_policy();
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/franka_emika_panda")
}

fn load(disable_contact: bool) -> ferromotion_core::MjcfTree {
    let dir = root();
    let xml = fs::read_to_string(dir.join("contact_reach.xml")).unwrap();
    let mut model = tree_from_mjcf(&xml, &|path| {
        let file = dir.join(path);
        if let Ok(bytes) = fs::read(&file) {
            Some(bytes)
        } else {
            fs::read(dir.join("assets").join(path)).ok()
        }
    })
    .unwrap_or_else(|err| panic!("load {err}"));
    assert_eq!(model.joints.len(), 7);
    let dt = nominal_seconds(PERIOD_NS).unwrap();
    assert_eq!(model.timestep.to_bits(), dt.to_bits());
    if disable_contact {
        for geom in &mut model.geoms {
            geom.params.contype = 0;
            geom.params.conaffinity = 0;
        }
    }
    for actuator in &mut model.actuators {
        actuator.gain = ActGain::Fixed;
        actuator.gainprm[0] = 1.0;
        actuator.bias = ActBias::None;
        actuator.biasprm = [0.0; 10];
        actuator.ctrlrange = None;
        let limit = actuator.forcerange.expect("panda forcerange");
        assert!(limit[0] < 0.0 && limit[1] > 0.0);
    }
    model
}

struct Trace {
    ticks: Vec<String>,
    ncon: Vec<usize>,
    ncon_moving: Vec<usize>,
    nefc: Vec<usize>,
    qfrc: Vec<[f64; 7]>,
    plant: String,
    state_bytes: String,
    envelope: String,
    spikes: u64,
    first_spike: Option<usize>,
    nonzero_tau: usize,
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
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
        &[5e-9; NEURONS],
        &[3e-9; NEURONS],
        &[-1e-6; NEURONS],
        &[1e-6; NEURONS],
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
        SpikeReadout::try_new(WINDOW as u64, inputs, &weights, &[0.0; 7], &lo, &hi).unwrap();
    (sim, readout, encoder)
}

fn run(with_controller: bool, disable_contact: bool) -> Trace {
    let policy = install_policy(
        ORACLE_X86_CONTRACT_OFF_ID,
        ORACLE_X86_CONTRACT_OFF_SHA256,
        "",
    )
    .unwrap();
    assert_eq!(policy.id, ORACLE_X86_CONTRACT_OFF_ID);
    assert_eq!(policy.artifact_sha256, ORACLE_X86_CONTRACT_OFF_SHA256);
    let model = load(disable_contact);
    let (mut sim, readout, encoder) = controller();
    let ours = EpisodeEnvelope::try_new(1, sim.epoch, sim.dt.to_bits(), PERIOD_NS, 0).unwrap();
    let text = ours.to_text();
    let envelope = ferromorphic::sim::EpisodeEnvelope::from_text(&text).unwrap();
    let mut st = MjNativeState {
        time: 0.0,
        qpos: REACH.to_vec(),
        qvel: vec![0.0; 7],
        act: vec![],
        qacc_warmstart: vec![0.0; 7],
        reset: false,
    };
    let mut history = vec![vec![0u64; NEURONS]; WINDOW];
    let mut ticks = Vec::new();
    let mut ncon = Vec::new();
    let mut ncon_moving = Vec::new();
    let mut nefc = Vec::new();
    let mut qfrc_rows = Vec::new();
    let mut spikes = 0u64;
    let mut first_spike = None;
    let mut nonzero_tau = 0usize;
    for k in 0..N {
        let mut observation = Vec::with_capacity(14);
        for (i, q) in st.qpos.iter().enumerate() {
            observation.push(q.clamp(QPOS_LO[i], QPOS_HI[i]));
        }
        observation.extend_from_slice(&st.qvel);
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
        if tau.iter().any(|t| t.to_bits() != 0.0f64.to_bits()) {
            nonzero_tau += 1;
        }
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
        let witness = model
            .mujoco_native_step_witness(&st, &tau)
            .unwrap_or_else(|err| panic!("step {k}: {err}"));
        ncon.push(witness.ncon);
        ncon_moving.push(witness.ncon_moving);
        nefc.push(witness.nefc_contact_moving);
        let mut row = [0.0; 7];
        row.copy_from_slice(&witness.qfrc_constraint);
        qfrc_rows.push(row);
        let record = format!(
            "tick {k}\nncon {}\nncon_moving {}\nnefc {}\nqfrc {}\nobs {}\ntau {}\nqpos {}\nqvel {}\n{}",
            witness.ncon,
            witness.ncon_moving,
            witness.nefc_contact_moving,
            bits(&witness.qfrc_constraint),
            bits(&observation),
            bits(&tau),
            bits(&witness.next.qpos),
            bits(&witness.next.qvel),
            receipt.to_text()
        );
        ticks.push(digest_sha256(record.as_bytes()));
        st = witness.next;
    }
    Trace {
        ticks,
        ncon,
        ncon_moving,
        nefc,
        qfrc: qfrc_rows,
        plant: digest_sha256(format!("{}\n{}", bits(&st.qpos), bits(&st.qvel)).as_bytes()),
        state_bytes: digest_sha256(&sim.state_bytes()),
        envelope: digest_sha256(text.as_bytes()),
        spikes,
        first_spike,
        nonzero_tau,
    }
}

#[test]
fn panda_reach_contact_moves_the_arm() {
    let _guard = Guard;
    assert_eq!(
        FERROMORPHIC_CLOCK_REVISION,
        "82211dc861af4ecc1afa9603c2caf20d52712b6d"
    );
    let first = run(true, false);
    let second = run(true, false);
    assert_eq!(first.ticks, second.ticks);
    assert_eq!(first.plant, second.plant);
    assert_eq!(first.ncon, second.ncon);
    assert_eq!(first.nefc, second.nefc);
    assert!(first.spikes > 0, "spikes {}", first.spikes);
    assert!(first.nonzero_tau > 0);
    assert!(first.nefc.iter().any(|n| *n > 0), "efc {:?}", first.nefc);
    assert!(
        first.ncon_moving.iter().any(|n| *n > 0),
        "moving {:?}",
        first.ncon_moving
    );
    assert!(
        first
            .qfrc
            .iter()
            .any(|row| row.iter().any(|q| q.to_bits() != 0))
    );
    let zero = run(false, false);
    let zero_again = run(false, false);
    assert_eq!(zero.plant, zero_again.plant);
    assert_ne!(zero.plant, first.plant);
    let open = run(true, true);
    let open_again = run(true, true);
    assert_eq!(open.plant, open_again.plant);
    assert!(open.ncon.iter().all(|n| *n == 0));
    assert_ne!(open.plant, first.plant);
    assert_eq!(first.spikes, 45);
    assert_eq!(first.first_spike, Some(1));
    assert_eq!(first.nonzero_tau, 15);
    assert_eq!(
        first.ncon,
        vec![5, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    );
    assert_eq!(first.ncon_moving, first.ncon);
    assert_eq!(
        first.nefc,
        vec![
            20, 20, 20, 20, 16, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12
        ]
    );
    assert_eq!(
        first.plant,
        "5c4c4514c93d3c667deecb6b4f834562ff7b2ad14dc61b6c88943c2f3b443fe7"
    );
    assert_eq!(
        zero.plant,
        "5fd57262dd51c8a559f470e7e5bc0b8a5fc2122bceb001696af4662d77fbe13c"
    );
    assert_eq!(
        open.plant,
        "2e3a8ce12ebd612b665b4f2d3516e1c31dae9b94f202f7bbd63c6bd50544227c"
    );
    assert_eq!(
        first.state_bytes,
        "01e990011ad4d1e59d902d3fc1f480fca708b1cc574970de96f69fbe7a22fe82"
    );
    assert_eq!(
        first.envelope,
        "170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a"
    );
    assert_eq!(
        first.ticks,
        [
            "52094552e834ab5371d8dba92c3c91e77245671045d8369d4877c5082f3f9572",
            "fbfbc3bf6dc5043f5a756ed646a784f0b93764aa2818146ce75678733b83f8fd",
            "48ab9ded3c3388aa6a52a807d7107fa1672e02de1624a6b249c503a4e7d07002",
            "570e32c982417047cd47f882eb13b9c431991529a4fce799ac7f3a1fa478fc6a",
            "72c297b61ef520d56a6b9caa6b71f1447043d296a67675c6a5123102d4b53fc6",
            "f6b60fcdd7228aa89a2245016a46ea046106b66f4a14a243af7bebef198dca28",
            "2e2b5e30402e9601024be406cd18b9894ae565c6ccef801dd457cc1ec036dd25",
            "273eb522ea6643d44a8940387857f5e781483b8e25af3e8348e98c130ad6a4c6",
            "8ea2048ef87fd16dc32c3c714151c199b02b178edfae082f5fdd017dfb855a0c",
            "9de978190bf62a38341a26268d01306ba146b3822c1c9c0ae4d9698f0ee4c25a",
            "dd3324a4db3c9696a5fc9b10cce2a8c755fbdcc5d0b972f297a26f53f27050c2",
            "7e3a005b526836e1532cacb18716ac1a83e2a3b625602ddba6ce6609ba93e441",
            "7d985ab1ecfa907a23b8497b47a68c198f049ad1072c838a2d66d48cc29c5f0a",
            "05baeb7e0e71fb37ffc0a1aababe2d067c9bdbc25a5b60e6ce0c2d4dfa917eb9",
            "0e60bc1ae887a59541a480a0886ecd5ce7fd30d9d61c184c0b370252fb3f86b4",
            "b9410c326d52ca25f70e6435a827723ef38ee3dd1ec9bc575ab3cf20689b34ac",
        ]
    );
}
