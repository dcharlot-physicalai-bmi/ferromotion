//! **`MjcfTree::step_mujoco` against MuJoCo's own `mj_step`, on every compilable Menagerie model.**
//!
//! Every other sweep checks one state. This one checks that the stages compose over TIME: our loader, our
//! kinematics, our contacts, our rows, our solve and our integrator, stepped from the same start as MuJoCo
//! and compared after steps 1, 5 and 20. The oracle runs MuJoCo with its solver told to converge, which is
//! what `step_mujoco` does, so a model that caps `<option iterations>` is compared as if uncapped.
//!
//! ```text
//! python scripts/mujoco_step_oracle.py <menagerie root> <step oracle txt> 20
//! cargo run --release --example menagerie_step -- <menagerie root> <step oracle txt> [substring]
//! ```
//!
//! ⭐ A model with ACTIVATIONS (the six `ms_human_700` files) starts from random activations under a random
//! control held for the whole run, and its activations are compared with `qpos` and `qvel`.
//!
//! ⛔ A trajectory with contacts is chaotic: two answers that agree to 1e-12 at step 1 can part by far more
//! after twenty steps, because a contact that is barely made in one and barely missed in the other changes
//! everything after it. So step 1 is the claim, and steps 5 and 20 are reported as how far agreement
//! CARRIES, split by whether a mesh (the one collider still known to differ) is in contact at the start.

use ferromotion_core::{tree_from_mjcf, GeomType, MjcfJointKind};
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    capped: bool,
    joints: Vec<(String, String, usize)>,
    start: Vec<f64>,
    startv: Vec<f64>,
    /// the control, held for the whole run (zero unless the model has activations)
    ctrl: Vec<f64>,
    startact: Vec<f64>,
    /// step → (qpos, qvel)
    steps: BTreeMap<usize, (Vec<f64>, Vec<f64>)>,
    /// step → act
    acts: BTreeMap<usize, Vec<f64>>,
}


/// ⛔ `f64::max` DROPS a NaN (`1.0f64.max(NAN) == 1.0`), so a worst-error fold over a state that went
/// non-finite reads as agreement. This one makes any NaN the worst possible error.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        a.max(b)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_step <menagerie root> <step oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap();
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "model" => cases.push(Case { rel: t[1].to_string(), capped: t[4] == "1", ..Default::default() }),
            "joint" => cases.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "start" => cases.last_mut().unwrap().start = t[1..].iter().map(|x| f(x)).collect(),
            "startv" => cases.last_mut().unwrap().startv = t[1..].iter().map(|x| f(x)).collect(),
            "ctrl" => cases.last_mut().unwrap().ctrl = t[1..].iter().map(|x| f(x)).collect(),
            "startact" => cases.last_mut().unwrap().startact = t[1..].iter().map(|x| f(x)).collect(),
            "stepact" => {
                let k: usize = t[1].parse().unwrap();
                cases.last_mut().unwrap().acts.insert(k, t[2..].iter().map(|x| f(x)).collect());
            }
            "step" => {
                let k: usize = t[1].parse().unwrap();
                cases.last_mut().unwrap().steps.entry(k).or_default().0 = t[2..].iter().map(|x| f(x)).collect();
            }
            "stepv" => {
                let k: usize = t[1].parse().unwrap();
                cases.last_mut().unwrap().steps.entry(k).or_default().1 = t[2..].iter().map(|x| f(x)).collect();
            }
            _ => {}
        }
    }
    let last = cases.iter().flat_map(|c| c.steps.keys().copied()).max().unwrap_or(1);
    let mut skip: BTreeMap<String, usize> = BTreeMap::new();
    // per checkpoint: (compared, within 1e-8, within 1e-4, worst, where), with and without a mesh in contact
    type Tally = (usize, usize, usize, f64, String);
    let mut tally: BTreeMap<(usize, bool), Tally> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut stepped = 0usize;
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&c.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(e) => {
                *skip.entry("the loader refuses the model".into()).or_default() += 1;
                notes.push(format!("{}: loader refused: {e}", c.rel));
                continue;
            }
        };
        if !t.actuators_unsupported.is_empty() {
            *skip.entry("an actuator this port does not carry (<plugin>)".into()).or_default() += 1;
            continue;
        }
        // the joints must be MuJoCo's, in MuJoCo's order, or the two qpos vectors are not the same vector
        let kind = |k: MjcfJointKind| match k {
            MjcfJointKind::Hinge => "hinge",
            MjcfJointKind::Slide => "slide",
            MjcfJointKind::Ball => "ball",
            MjcfJointKind::Free => "free",
        };
        let lined_up = t.joints.len() == c.joints.len() && t.joints.iter().zip(&c.joints).zip(t.qposadr()).all(|((a, b), adr)| kind(a.kind) == b.1 && adr == b.2);
        if !lined_up {
            *skip.entry("the joints do not line up with MuJoCo's".into()).or_default() += 1;
            notes.push(format!("{}: joints do not line up", c.rel));
            continue;
        }
        let ctrl = if c.ctrl.is_empty() { vec![0.0; t.actuators.len()] } else { c.ctrl.clone() };
        if t.na() != c.startact.len() {
            *skip.entry("the activations do not line up with MuJoCo's".into()).or_default() += 1;
            notes.push(format!("{}: na {} vs MuJoCo {}", c.rel, t.na(), c.startact.len()));
            continue;
        }
        // ⭐ whether a MESH is in contact at the start: the one collider still known to disagree with
        // MuJoCo's, so the split says whether a divergence can be charged to it
        let mesh_in_contact = t.q_from_qpos(&c.start, &t.qposadr()).ok().is_some_and(|q| {
            t.collide(&q).contacts.iter().any(|con| !con.record.exclude && con.geom.iter().any(|&g| t.geoms[g].kind == GeomType::Mesh))
        });
        let (mut qpos, mut qvel, mut act) = (c.start.clone(), c.startv.clone(), c.startact.clone());
        let mut failed = None;
        for k in 1..=last {
            match t.step_mujoco(&qpos, &qvel, &ctrl, &act) {
                Ok(s) => (qpos, qvel, act) = (s.qpos, s.qvel, s.act),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
            if let Some((wq, wv)) = c.steps.get(&k) {
                let rel = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0, nan_max);
                let e = nan_max(nan_max(rel(&qpos, wq), rel(&qvel, wv)), c.acts.get(&k).map_or(0.0, |wa| rel(&act, wa)));
                let entry = tally.entry((k, mesh_in_contact)).or_insert((0, 0, 0, 0.0, String::new()));
                entry.0 += 1;
                if e < 1e-8 {
                    entry.1 += 1;
                }
                if e < 1e-4 {
                    entry.2 += 1;
                }
                if e > entry.3 {
                    entry.3 = e;
                    entry.4 = format!("{}{}", c.rel, if c.capped { " (its file caps the solver)" } else { "" });
                }
                if k == 1 && e >= 1e-8 && notes.len() < 40 {
                    notes.push(format!("{}: step 1 off {e:.2e}{}", c.rel, if mesh_in_contact { " (mesh in contact)" } else { "" }));
                }
            }
        }
        if let Some(e) = failed {
            let why = if e.contains("collider") {
                "a geom pair this port cannot collide (height field)"
            } else {
                "the step refused the state"
            };
            *skip.entry(why.into()).or_default() += 1;
            notes.push(format!("{}: {e}", c.rel));
            continue;
        }
        stepped += 1;
    }
    println!("models stepped {stepped} of {} ({last} steps each, from the same start as MuJoCo)", cases.len());
    for ((k, mesh), (n, tight, loose, worst, at)) in &tally {
        println!(
            "  after step {k:>2}, {}: {tight} of {n} within 1e-8, {loose} within 1e-4; worst {worst:.2e} on {at}",
            if *mesh { "a MESH in contact at the start" } else { "no mesh in contact       " }
        );
    }
    println!("  not stepped:");
    for (why, n) in &skip {
        println!("    {n:>4}  {why}");
    }
    for n in &notes {
        println!("  {n}");
    }
}
