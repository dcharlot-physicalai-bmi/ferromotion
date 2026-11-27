//! Corrected contact episode on the contract-off native stepper.
//!
//! `98ce3db1` stays. This episode does not set a flag.
//!
//! The plant is [`ferromotion_core::MjcfTree::mujoco_native_step_witness_applied`]:
//! torque on `qfrc_applied`, actuator control zero, after installing
//! `mujoco-3.13.0-x86_64-contract-off`.

use std::fs;
use std::path::PathBuf;

use ferromorphic::NetBuilder;
use ferromorphic::neuron::Lif;
use ferromorphic::readout::{Clamp, CurrentEncoder, SpikeReadout};
use ferromorphic::sim::{Mode, Sim, StepReceipt, bind_v2_timestamps, nominal_seconds};
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
/// Forty bisections of joint4 between the old reach (`-2.2`, link7 about 60.5 mm
/// into the floor) and `-1.9` (no contact). The other joints stay
/// `0 0.8 0 _ 0 2.2 0`. This is the end whose minimum distance is not deeper
/// than 1 mm.
const JOINT4_BITS: u64 = 0xbfff_5c42_1417_9e64;

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        reset_policy();
    }
}

fn reach() -> [f64; 7] {
    [0.0, 0.8, 0.0, f64::from_bits(JOINT4_BITS), 0.0, 2.2, 0.0]
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
    // Same torque path as `panda_closed_loop`: the command is `qfrc_applied`.
    // Zeroing the position servo's gain and bias leaves that torque as the
    // actuator contribution. `forcerange` stays the file's ±87 / ±12.
    for actuator in &mut model.actuators {
        actuator.gainprm[0] = 0.0;
        actuator.biasprm[0] = 0.0;
        actuator.biasprm[1] = 0.0;
        actuator.biasprm[2] = 0.0;
        let limit = actuator.forcerange.expect("panda forcerange");
        assert!(limit[0] < 0.0 && limit[1] > 0.0);
    }
    model
}

struct Tick {
    hash: String,
    ncon: usize,
    ncon_moving: usize,
    nefc: usize,
    qfrc: [f64; 7],
    min_dist: Option<f64>,
    pairs: Vec<(String, String)>,
    obs_clamps: usize,
    enc_clamps: usize,
    read_clamps: usize,
    forcerange_binds: usize,
    tau_nonzero: bool,
}

struct Trace {
    ticks: Vec<Tick>,
    plant: String,
    state_bytes: String,
    envelope: String,
    spikes: u64,
    first_spike: Option<usize>,
    nonzero_tau: Vec<usize>,
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn clamp_count(sides: &[Clamp]) -> usize {
    sides
        .iter()
        .filter(|side| !matches!(side, Clamp::None))
        .count()
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

fn pair_name(model: &ferromotion_core::MjcfTree, geom: usize) -> String {
    format!("{}/{}", model.geoms[geom].body, model.geoms[geom].name)
}

/// The joint4 value recorded above, recomputed the same way: forty bisections,
/// keeping the end that is not deeper than −1 mm.
fn derived_joint4(model: &ferromotion_core::MjcfTree) -> f64 {
    let mut lo = -2.2f64;
    let mut hi = -1.9f64;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let mut q = reach();
        q[3] = mid;
        let min = min_solver_dist(model, &q);
        // No contact, or a contact no deeper than 1 mm, is the shallow end.
        if min.is_some_and(|d| d < -0.001) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn min_solver_dist(model: &ferromotion_core::MjcfTree, q: &[f64]) -> Option<f64> {
    let hit = model.collide_qpos(q);
    hit.contacts
        .iter()
        .filter(|c| !c.record.exclude)
        .map(|c| c.record.dist)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
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
        qpos: reach().to_vec(),
        qvel: vec![0.0; 7],
        act: vec![],
        qacc_warmstart: vec![0.0; 7],
        reset: false,
    };
    let mut history = vec![vec![0u64; NEURONS]; WINDOW];
    let mut ticks = Vec::new();
    let mut spikes = 0u64;
    let mut first_spike = None;
    let mut nonzero_tau = Vec::new();
    for k in 0..N {
        let mut observation = Vec::with_capacity(14);
        let mut obs_clamps = 0usize;
        for (i, q) in st.qpos.iter().enumerate() {
            let limited = q.clamp(QPOS_LO[i], QPOS_HI[i]);
            if limited.to_bits() != q.to_bits() {
                obs_clamps += 1;
            }
            observation.push(limited);
        }
        observation.extend_from_slice(&st.qvel);
        let encoded = encoder.apply(&observation).unwrap();
        let enc_clamps = clamp_count(&encoded.clamp);
        let tick = sim.t;
        let fired = sim.try_step(&encoded.output).unwrap();
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
        let (tau, read_clamps) = if with_controller {
            let traced = readout.apply(&counts).unwrap();
            (traced.output, clamp_count(&traced.clamp))
        } else {
            (vec![0.0; 7], 0)
        };
        let tau_nonzero = tau.iter().any(|t| t.to_bits() != 0.0f64.to_bits());
        if tau_nonzero {
            nonzero_tau.push(k);
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
            .mujoco_native_step_witness_applied(&st, &[0.0; 7], &tau)
            .unwrap_or_else(|err| panic!("step {k}: {err}"));
        let pairs: Vec<(String, String)> = witness
            .contacts
            .iter()
            .map(|c| (pair_name(&model, c.geom[0]), pair_name(&model, c.geom[1])))
            .collect();
        let min_dist = witness
            .contacts
            .iter()
            .map(|c| c.dist)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        let mut row = [0.0; 7];
        row.copy_from_slice(&witness.qfrc_constraint);
        let pair_text = pairs
            .iter()
            .map(|(a, b)| format!("{a} {b}"))
            .collect::<Vec<_>>()
            .join(",");
        let min_text = min_dist
            .map(|d| format!("{:016x}", d.to_bits()))
            .unwrap_or_default();
        let record = format!(
            "tick {k}\nncon {}\nncon_moving {}\nnefc {}\nmin {min_text}\npairs {pair_text}\nobs_clamp {obs_clamps}\nenc_clamp {enc_clamps}\nread_clamp {read_clamps}\nforcerange {}\nqfrc {}\nobs {}\ntau {}\nqpos {}\nqvel {}\n{}",
            witness.ncon,
            witness.ncon_moving,
            witness.nefc_contact_moving,
            witness.forcerange_binds,
            bits(&witness.qfrc_constraint),
            bits(&observation),
            bits(&tau),
            bits(&witness.next.qpos),
            bits(&witness.next.qvel),
            receipt.to_text()
        );
        ticks.push(Tick {
            hash: digest_sha256(record.as_bytes()),
            ncon: witness.ncon,
            ncon_moving: witness.ncon_moving,
            nefc: witness.nefc_contact_moving,
            qfrc: row,
            min_dist,
            pairs,
            obs_clamps,
            enc_clamps,
            read_clamps,
            forcerange_binds: witness.forcerange_binds,
            tau_nonzero,
        });
        st = witness.next;
    }
    Trace {
        ticks,
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
    let _policy = install_policy(
        ORACLE_X86_CONTRACT_OFF_ID,
        ORACLE_X86_CONTRACT_OFF_SHA256,
        "",
    )
    .unwrap();
    let posed = load(false);
    assert_eq!(derived_joint4(&posed).to_bits(), JOINT4_BITS);
    let start = min_solver_dist(&posed, &reach()).unwrap();
    assert!((-0.001..0.0).contains(&start), "start distance {start}");
    assert_eq!(start.to_bits(), 0xbf50_624d_d2ee_f860);

    let first = run(true, false);
    let second = run(true, false);
    assert_eq!(
        first.ticks.iter().map(|t| &t.hash).collect::<Vec<_>>(),
        second.ticks.iter().map(|t| &t.hash).collect::<Vec<_>>()
    );
    assert_eq!(first.plant, second.plant);
    assert!(first.spikes > 0, "spikes {}", first.spikes);
    assert_eq!(first.nonzero_tau, (1..N).collect::<Vec<_>>());
    let floor_link7 = ("world/floor".to_string(), "link7/geom59".to_string());
    for (k, tick) in first.ticks.iter().enumerate() {
        assert_eq!(tick.obs_clamps, 0, "tick {k} joint-limit clamps");
        assert_eq!(tick.enc_clamps, 0, "tick {k} encoder clamps");
        assert_eq!(tick.read_clamps, 0, "tick {k} readout clamps");
        assert_eq!(tick.forcerange_binds, 0, "tick {k} forcerange");
        assert_eq!(tick.ncon, 3, "tick {k}");
        assert_eq!(tick.ncon_moving, 3, "tick {k}");
        assert_eq!(tick.nefc, 12, "tick {k}");
        assert_eq!(tick.pairs, vec![floor_link7.clone(); 3], "tick {k}");
        let dist = tick.min_dist.unwrap();
        assert!((-0.001..0.0).contains(&dist), "tick {k} dist {dist}");
        if k > 0 {
            let prev = first.ticks[k - 1].min_dist.unwrap();
            assert!(dist > prev, "tick {k} is not separating: {dist} <= {prev}");
        }
        for (joint, force) in tick.qfrc.iter().enumerate() {
            assert_ne!(
                force.to_bits(),
                0.0f64.to_bits(),
                "tick {k} joint {} qfrc_constraint is zero",
                joint + 1
            );
        }
        assert_eq!(tick.tau_nonzero, k > 0, "tick {k} tau");
    }
    let zero = run(false, false);
    let zero_again = run(false, false);
    assert_eq!(zero.plant, zero_again.plant);
    assert_ne!(zero.plant, first.plant);
    assert!(zero.nonzero_tau.is_empty());
    let open = run(true, true);
    let open_again = run(true, true);
    assert_eq!(open.plant, open_again.plant);
    assert!(
        open.ticks
            .iter()
            .all(|t| t.ncon == 0 && t.min_dist.is_none())
    );
    assert_ne!(open.plant, first.plant);
    assert_eq!(first.spikes, 54);
    assert_eq!(first.first_spike, Some(1));
    assert_eq!(
        first.plant,
        "c3b70e92db29d9f112626d68d2d5c387895fb6f72dc95bf98d70262c34d300b2"
    );
    assert_eq!(
        zero.plant,
        "b3dc651ba290426b3e81a70db2e5aa59dbcc9fa787498934c07b5882925e04e8"
    );
    assert_eq!(
        open.plant,
        "a3a1c16ef5627785074a3c219490e1296a8233e62495430b517272f815a23852"
    );
    assert_eq!(
        first.state_bytes,
        "4e49f36ed5148e355c13dd8a58c7d6b48d0062fb4cd56cce059117178ea40e50"
    );
    assert_eq!(
        first.envelope,
        "170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a"
    );
    let min_bits: [u64; 16] = [
        0xbf50_624d_d2ee_f860,
        0xbf50_3799_9123_d0c0,
        0xbf4f_d71e_770f_d700,
        0xbf4f_0be1_726a_3600,
        0xbf4e_1a4d_8ae0_d100,
        0xbf4d_0cf2_f857_8400,
        0xbf4b_ec79_2a1e_0b80,
        0xbf4a_bff4_f35c_bdc0,
        0xbf49_8d25_4e17_8480,
        0xbf48_58a9_a331_cec0,
        0xbf47_2634_45dd_c080,
        0xbf45_f8ae_290b_1b40,
        0xbf44_d258_6386_d280,
        0xbf43_b4e8_efd6_e380,
        0xbf42_a1a4_d6ac_adc0,
        0xbf41_9969_c264_8200,
    ];
    let tick_hash = [
        "59e1f6b5238ea02b4f521bfb804f430a403e6885e7545df393200bc21d0f343e",
        "edeaba3a38f36e7d5623be2dd17a815814c9bad11dae64dd055800776df24a11",
        "79d7fb2c1b08465a658616c672ad720005f535c8a9ccadda08369b4fef7d30ea",
        "6bcb14857a948b97d4bb1d6b9b7f8b47ca6df0d294a4e452461845cb904b5fcb",
        "5f2e3460b0b0013be644d17c907c74512814fe44d19bcd9279e971135dfa2f16",
        "b0838e7e582dd97a02d9b41b3df4bfad062ebe46ad0db01c19d8a69f9d39ac3f",
        "2536a08458cba8cd9d09110bf86d71cfe89f9467bb08b465b659d43428735b18",
        "fa816e4ddb81770d6f854527fa5c6f0e862a25347cff9ade8f9570821ae95705",
        "0c98a2b6f5d653bc93eeb9ad30e7974e2ac635ebd1047dce8b67add333bc6966",
        "fcec610f7c47c3cb6e69d16a1097ef9d4b56e774566692c25b6c6d63ededb2b7",
        "337a1024d6715899539a80a4851d5849d474e3b40a8236632e02b894690a0c56",
        "b212a05de02aad23550cd4395b71d3653d25278edb5957380e03c7df8eb121e8",
        "c849c9347a37532deb40a815734d8b65c3e84123a4d5ceb6bc37da883d3ee5db",
        "905fd8c71e000c874a67854c2347a2312ca2a552fb60b8ab4c8d2f0d8294916c",
        "15d05a124f8b09827aacbc58773d6d4c3859d8b51921be2bd9d74e1866179f4a",
        "b8f43856d97bfe62b2175b0bf36f4a01e989ebc1055b0a3bb2d6146c16943f62",
    ];
    for (k, tick) in first.ticks.iter().enumerate() {
        assert_eq!(tick.min_dist.unwrap().to_bits(), min_bits[k], "tick {k}");
        assert_eq!(tick.hash, tick_hash[k], "tick {k}");
    }
}
