//! **`MjcfTree::sensordata` against MuJoCo's `d.sensordata`, on every Menagerie model that declares
//! sensors.**
//!
//! ```text
//! python scripts/mujoco_sensor_oracle.py <menagerie root> <sensor oracle txt>
//! cargo run --release --example menagerie_sensors -- <menagerie root> <sensor oracle txt> [substring]
//! ```
//!
//! Every sensor is paired with MuJoCo's by index (the sensor list is in file order on both sides), its
//! type checked, and its value compared at each sampled state, per sensor type. A quaternion that is the
//! same rotation with the opposite sign is counted apart: it is the same orientation and a different
//! number, and the number is what a policy reads.

use ferromotion_core::mjcf_sensor::SensorKind;
use ferromotion_core::{tree_from_mjcf, MjcfJointKind};
use std::collections::BTreeMap;

#[derive(Default)]
struct Model {
    rel: String,
    joints: Vec<(String, String, usize)>,
    /// (name, type, dim, adr)
    sensors: Vec<(String, String, usize, usize)>,
    /// (qpos, qvel, sensordata, qacc)
    states: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
}

/// ⛔ `f64::max` DROPS a NaN, so a worst-error fold would read a non-finite value as agreement.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

fn kind_name(k: &SensorKind) -> &'static str {
    use SensorKind::*;
    match k {
        JointPos => "jointpos",
        JointVel => "jointvel",
        TendonPos => "tendonpos",
        TendonVel => "tendonvel",
        ActuatorPos => "actuatorpos",
        ActuatorVel => "actuatorvel",
        BallQuat => "ballquat",
        BallAngVel => "ballangvel",
        FramePos => "framepos",
        FrameXAxis => "framexaxis",
        FrameYAxis => "frameyaxis",
        FrameZAxis => "framezaxis",
        FrameQuat => "framequat",
        FrameLinVel => "framelinvel",
        FrameAngVel => "frameangvel",
        Velocimeter => "velocimeter",
        Gyro => "gyro",
        Magnetometer => "magnetometer",
        JointActFrc => "jointactfrc",
        ActuatorFrc => "actuatorfrc",
        Clock => "clock",
        Accelerometer => "accelerometer",
        Force => "force",
        Torque => "torque",
        Touch => "touch",
        FrameLinAcc => "framelinacc",
        FrameAngAcc => "frameangacc",
        Unsupported(_) => "unsupported",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_sensors <menagerie root> <sensor oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let f = |s: &str| s.parse::<f64>().unwrap();
    let mut models: Vec<Model> = Vec::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        let m = models.last_mut();
        match t[0] {
            "model" => models.push(Model { rel: t[1].to_string(), ..Default::default() }),
            "joint" => m.unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "sensor" => m.unwrap().sensors.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap(), t[4].parse().unwrap())),
            "state" => m.unwrap().states.push(Default::default()),
            "qpos" => m.unwrap().states.last_mut().unwrap().0 = t[1..].iter().map(|x| f(x)).collect(),
            "qvel" => m.unwrap().states.last_mut().unwrap().1 = t[1..].iter().map(|x| f(x)).collect(),
            "sensordata" => m.unwrap().states.last_mut().unwrap().2 = t[1..].iter().map(|x| f(x)).collect(),
            "qacc" => m.unwrap().states.last_mut().unwrap().3 = t[1..].iter().map(|x| f(x)).collect(),
            _ => {}
        }
    }
    // per MuJoCo sensor type: (values compared, within 1e-9, within 1e-6, worst, where, carried)
    let mut by_type: BTreeMap<String, (usize, usize, usize, f64, String, bool)> = BTreeMap::new();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let mut not_carried: BTreeMap<String, usize> = BTreeMap::new();
    let mut quat_sign_flips = 0usize;
    // acceleration-stage values on states where our constrained `qacc` is not MuJoCo's: the sensor reads
    // the solve, so its error there is the solve's (mesh contacts, mostly), counted apart
    let mut solve_off: BTreeMap<String, usize> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let kind = |k: MjcfJointKind| match k {
        MjcfJointKind::Hinge => "hinge",
        MjcfJointKind::Slide => "slide",
        MjcfJointKind::Ball => "ball",
        MjcfJointKind::Free => "free",
    };
    for m in &models {
        if filter.as_ref().is_some_and(|fl| !m.rel.contains(fl.as_str())) {
            continue;
        }
        let path = root.join(&m.rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(e) => {
                *skipped.entry("the loader refuses the model".into()).or_default() += 1;
                notes.push(format!("{}: {e}", m.rel));
                continue;
            }
        };
        let lined_up = t.joints.len() == m.joints.len() && t.joints.iter().zip(&m.joints).zip(t.qposadr()).all(|((a, b), adr)| kind(a.kind) == b.1 && adr == b.2);
        if !lined_up {
            *skipped.entry("the joints do not line up with MuJoCo's".into()).or_default() += 1;
            continue;
        }
        if t.sensors.len() != m.sensors.len() {
            *skipped.entry("a different number of sensors".into()).or_default() += 1;
            notes.push(format!("{}: {} sensors here, {} in MuJoCo", m.rel, t.sensors.len(), m.sensors.len()));
            continue;
        }
        for (s, o) in t.sensors.iter().zip(&m.sensors) {
            if let SensorKind::Unsupported(why) = &s.kind {
                *not_carried.entry(format!("{} ({why})", o.1)).or_default() += m.states.len();
            } else if kind_name(&s.kind) != o.1 {
                notes.push(format!("{}: sensor '{}' is {} here, {} in MuJoCo", m.rel, o.0, kind_name(&s.kind), o.1));
            }
        }
        for (qpos, qvel, want, want_qacc) in &m.states {
            let forward = t.q_from_qpos(qpos, &t.qposadr()).and_then(|q| t.forward_mujoco(&q, qvel, &[], &[])).ok();
            let qacc_ok = forward.as_ref().is_some_and(|fw| fw.qacc.iter().zip(want_qacc).all(|(a, b)| (a - b).abs() <= 1e-6 * b.abs().max(1.0)));
            let got = match t.sensordata_with(qpos, qvel, &[], &[], 0.0, forward.as_ref()) {
                Ok(g) => g,
                Err(e) => {
                    notes.push(format!("{}: sensordata refused: {e}", m.rel));
                    continue;
                }
            };
            for ((s, o), v) in t.sensors.iter().zip(&m.sensors).zip(&got) {
                let acc_stage = matches!(s.kind, SensorKind::Accelerometer | SensorKind::Force | SensorKind::Torque | SensorKind::Touch | SensorKind::FrameLinAcc | SensorKind::FrameAngAcc);
                if acc_stage && v.is_some() && !qacc_ok {
                    *solve_off.entry(o.1.clone()).or_default() += 1;
                    continue;
                }
                let entry = by_type.entry(o.1.clone()).or_insert((0, 0, 0, 0.0, String::new(), false));
                let Some(v) = v else { continue };
                entry.5 = true;
                let w = &want[o.3..o.3 + o.2];
                if v.len() != w.len() {
                    notes.push(format!("{}: sensor '{}' has {} values here, {} in MuJoCo", m.rel, o.0, v.len(), w.len()));
                    continue;
                }
                let err = |v: &[f64]| v.iter().zip(w).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0, nan_max);
                let mut e = err(v);
                if s.kind == SensorKind::FrameQuat || s.kind == SensorKind::BallQuat {
                    let flipped: Vec<f64> = v.iter().map(|x| -x).collect();
                    if err(&flipped) < 1e-9 && e >= 1e-9 {
                        quat_sign_flips += 1;
                        e = err(&flipped);
                    }
                }
                entry.0 += 1;
                entry.1 += usize::from(e < 1e-9);
                entry.2 += usize::from(e < 1e-6);
                if e > entry.3 {
                    entry.3 = e;
                    entry.4 = format!("{} '{}'", m.rel, o.0);
                }
                if e >= 1e-6 && notes.len() < 30 {
                    notes.push(format!("{}: {} '{}' off {e:.2e}: {:?} vs MuJoCo {:?}", m.rel, o.1, o.0, v, w));
                }
            }
        }
    }
    println!("sensor values against MuJoCo 3.13.0, by type (every sensor, every sampled state):");
    for (ty, (n, tight, loose, worst, at, carried)) in &by_type {
        if *carried {
            println!("  {ty:<14} {tight:>5} of {n:<5} within 1e-9, {loose:>5} within 1e-6; worst {worst:.2e} on {at}");
        }
    }
    println!("  framequat/ballquat: {quat_sign_flips} values are the same rotation with the OPPOSITE sign (counted as agreeing above)");
    println!("  acceleration-stage values left out because our constrained qacc is not MuJoCo's there (the solve's error, not the sensor's):");
    for (ty, n) in &solve_off {
        println!("    {n:>5}  {ty}");
    }
    println!("  not carried, by type (sensor values):");
    for (ty, n) in &not_carried {
        println!("    {n:>5}  {ty}");
    }
    println!("  models not compared:");
    for (why, n) in &skipped {
        println!("    {n:>4}  {why}");
    }
    for n in &notes {
        println!("  {n}");
    }
}
