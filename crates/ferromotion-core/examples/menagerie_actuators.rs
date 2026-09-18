//! **Actuator parity with MuJoCo on all of Menagerie**: the compiled parameters, then the force.
//!
//! An MJCF shortcut tag is sugar — MuJoCo resolves `<position kp="80"/>` into a gain type, a bias type and
//! their parameter vectors before anything runs — so this compares the RESOLVED parameters first and the
//! force second. When they disagree, that ordering says which of the two is wrong, instead of leaving a
//! force error to be blamed on either.
//!
//! ```text
//! cargo run --release --example menagerie_actuators -- <menagerie root> <actuator oracle txt> [substring]
//! ```

use ferromotion_core::{tree_from_mjcf, ActBias, ActGain, MjcfJointKind};
use std::collections::BTreeMap;

#[derive(Default)]
struct OAct {
    name: String,
    /// kept so a disagreement can name the transmission MuJoCo compiled, not only the numbers
    trn: String,
    gain: String,
    bias: String,
    dyn_: String,
    gear: f64,
    gainprm: [f64; 3],
    biasprm: [f64; 3],
    ctrlrange: Option<[f64; 2]>,
    forcerange: Option<[f64; 2]>,
}

#[derive(Default)]
struct OState {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    ctrl: Vec<f64>,
    length: Vec<f64>,
    velocity: Vec<f64>,
    force: Vec<f64>,
    qfrc: Vec<f64>,
}

#[derive(Default)]
struct OModel {
    rel: String,
    joints: Vec<(String, String, usize)>,
    acts: Vec<OAct>,
    states: Vec<OState>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_actuators <menagerie root> <actuator oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s:?}"));
    let mut models: Vec<OModel> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "model" => models.push(OModel { rel: t[1].to_string(), ..Default::default() }),
            "joint" => models.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "actuator" => {
                let lim = |l: &str, a: &str, b: &str| (l == "1").then(|| [f(a), f(b)]);
                models.last_mut().unwrap().acts.push(OAct {
                    name: t[1].to_string(),
                    trn: t[2].to_string(),
                    gain: t[4].to_string(),
                    bias: t[5].to_string(),
                    dyn_: t[6].to_string(),
                    gear: f(t[7]),
                    gainprm: [f(t[8]), f(t[9]), f(t[10])],
                    biasprm: [f(t[11]), f(t[12]), f(t[13])],
                    ctrlrange: lim(t[14], t[15], t[16]),
                    forcerange: lim(t[17], t[18], t[19]),
                });
            }
            "state" => models.last_mut().unwrap().states.push(OState::default()),
            "qpos" | "qvel" | "ctrl" | "act_length" | "act_velocity" | "act_force" | "qfrc_actuator" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
                    "ctrl" => s.ctrl = v,
                    "act_length" => s.length = v,
                    "act_velocity" => s.velocity = v,
                    "act_force" => s.force = v,
                    _ => s.qfrc = v,
                }
            }
            _ => {}
        }
    }

    let (mut seen, mut refused, mut carried, mut params_ok) = (0usize, 0usize, 0usize, 0usize);
    let (mut states, mut state_ok) = (0usize, 0usize);
    let (mut worst_len, mut worst_vel, mut worst_force, mut worst_qfrc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut left_out: BTreeMap<String, usize> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut worst_qfrc_where = String::new();
    for o in &models {
        if filter.as_ref().is_some_and(|fl| !o.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&o.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(e) => {
                refused += 1;
                if notes.len() < 8 {
                    notes.push(format!("refused {}: {e}", o.rel));
                }
                continue;
            }
        };
        seen += 1;
        for (_, why) in &t.actuators_unsupported {
            *left_out.entry(why.clone()).or_default() += 1;
        }
        // which of MuJoCo's actuators we claim: by name, since the ones we leave out shift the indices
        let ours: BTreeMap<&str, usize> = t.actuators.iter().enumerate().map(|(i, a)| (a.name.as_str(), i)).collect();
        // the dof vectors only correspond when every joint is a hinge or a slide in the same order
        let lined_up = t.joints.len() == o.joints.len()
            && t.joints.iter().zip(&o.joints).all(|(a, b)| matches!((a.kind, b.1.as_str()), (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide")));
        for (u, e) in o.acts.iter().enumerate() {
            let Some(&i) = ours.get(e.name.as_str()) else { continue };
            carried += 1;
            let a = &t.actuators[i];
            let gain = match a.gain {
                ActGain::Fixed => "fixed",
                ActGain::Affine => "affine",
            };
            let bias = match a.bias {
                ActBias::None => "none",
                ActBias::Affine => "affine",
            };
            let close = |x: [f64; 3], y: [f64; 3]| (0..3).all(|k| (x[k] - y[k]).abs() <= 1e-9 * y[k].abs().max(1.0));
            let same = gain == e.gain
                && bias == e.bias
                && e.dyn_ == "none"
                && (a.gear - e.gear).abs() < 1e-12
                && close(a.gainprm, e.gainprm)
                && close(a.biasprm, e.biasprm)
                && a.ctrlrange == e.ctrlrange
                && a.forcerange == e.forcerange;
            if same {
                params_ok += 1;
            } else if notes.len() < 8 {
                notes.push(format!(
                    "{} '{}' ({}): gain {gain}/{} bias {bias}/{} gear {}/{} gainprm {:?}/{:?} biasprm {:?}/{:?} ctrl {:?}/{:?} force {:?}/{:?}",
                    o.rel, e.name, e.trn, e.gain, e.bias, a.gear, e.gear, a.gainprm, e.gainprm, a.biasprm, e.biasprm, a.ctrlrange, e.ctrlrange, a.forcerange, e.forcerange
                ));
            }
            // and the force itself, at every sampled state
            if !lined_up {
                continue;
            }
            for s in &o.states {
                let qposadr: Vec<usize> = o.joints.iter().map(|j| j.2).collect();
                let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else { continue };
                states += 1;
                let (l, v) = (a.length(&q), a.velocity(&s.qvel));
                let force = a.force(l, v, s.ctrl[u]);
                let (dl, dv, df) = ((l - s.length[u]).abs(), (v - s.velocity[u]).abs(), (force - s.force[u]).abs());
                worst_len = worst_len.max(dl);
                worst_vel = worst_vel.max(dv);
                worst_force = worst_force.max(df);
                let scale = s.force[u].abs().max(1.0);
                if dl < 1e-9 && dv < 1e-9 && df <= 1e-9 * scale {
                    state_ok += 1;
                } else if notes.len() < 8 {
                    notes.push(format!("{} '{}': length {l} vs {}, velocity {v} vs {}, force {force} vs {}", o.rel, e.name, s.length[u], s.velocity[u], s.force[u]));
                }
                // qfrc_actuator, but only where we carry EVERY actuator — a missing one is a missing term
                if t.actuators_unsupported.is_empty() && u + 1 == o.acts.len() {
                    let mine = ferromotion_core::qfrc_actuator(&t.actuators, &q, &s.qvel, &s.ctrl, &t.dof_actuator_force_range());
                    let d = mine.iter().zip(&s.qfrc).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
                    if d > 1e-6 && notes.len() < 8 {
                        notes.push(format!("{}: qfrc ours {:?} vs MuJoCo {:?}", o.rel, mine, s.qfrc));
                    }
                    if d > worst_qfrc {
                        worst_qfrc = d;
                        worst_qfrc_where = format!("{} ({} actuators, {} dofs)", o.rel, t.actuators.len(), t.tree.joints.len());
                    }
                }
            }
        }
    }
    println!("models {seen} (loader refused {refused})");
    println!("  actuators carried {carried}, compiled parameters identical {params_ok}");
    println!("  force evaluations {states}, matching MuJoCo {state_ok}; worst length {worst_len:.2e}, velocity {worst_vel:.2e}, force {worst_force:.2e}");
    println!("  worst qfrc_actuator, over models where every actuator is carried: {worst_qfrc:.2e} on {worst_qfrc_where}");
    if !left_out.is_empty() {
        println!("  actuators NOT carried, by reason:");
        let mut v: Vec<_> = left_out.iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (why, n) in v {
            println!("    {n:>5}  {why}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
