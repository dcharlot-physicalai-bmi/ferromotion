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
    gainprm: [f64; 10],
    biasprm: [f64; 10],
    ctrlrange: Option<[f64; 2]>,
    forcerange: Option<[f64; 2]>,
    acc0: f64,
}

#[derive(Default)]
struct OState {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    ctrl: Vec<f64>,
    /// one per ACTUATOR, zero where it carries no activation state
    act: Vec<f64>,
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
                    gainprm: std::array::from_fn(|k| f(t[8 + k])),
                    biasprm: std::array::from_fn(|k| f(t[18 + k])),
                    ctrlrange: lim(t[28], t[29], t[30]),
                    forcerange: lim(t[31], t[32], t[33]),
                    acc0: f(t[34]),
                });
            }
            "state" => models.last_mut().unwrap().states.push(OState::default()),
            "qpos" | "qvel" | "ctrl" | "act" | "act_length" | "act_velocity" | "act_force" | "qfrc_actuator" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
                    "ctrl" => s.ctrl = v,
                    "act" => s.act = v,
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
    let mut blocked_models: Vec<String> = Vec::new();
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
        // ⭐ and which MODELS each reason costs: a reason with 1,762 actuators in one model and a reason
        // with 16 across three are not the same size of gap, and only the model count says so
        if !t.actuators_unsupported.is_empty() {
            let mut why: Vec<&str> = t.actuators_unsupported.iter().map(|(_, w)| w.as_str()).collect();
            why.sort_unstable();
            why.dedup();
            blocked_models.push(format!("{}: {}", o.rel, why.join(", ")));
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
                ActGain::Muscle => "muscle",
            };
            let bias = match a.bias {
                ActBias::None => "none",
                ActBias::Affine => "affine",
                ActBias::Muscle => "muscle",
            };
            let dyn_ = match a.dynamics {
                ferromotion_core::mujoco_actuator::ActDyn::None => "none",
                ferromotion_core::mujoco_actuator::ActDyn::Muscle => "muscle",
            };
            let close = |x: [f64; 10], y: [f64; 10]| (0..10).all(|k| (x[k] - y[k]).abs() <= 1e-9 * y[k].abs().max(1.0));
            let same = gain == e.gain
                && bias == e.bias
                && dyn_ == e.dyn_
                // ⛔ `acc0` is a compiled constant like the rest and a muscle's peak force can BE it.
                // 1e-7, not 1e-9. ⛔ `flybody` still misses it at 6e-7, and the cause is NOT conditioning
                // (MuJoCo's own `acc0` reproduces from a dense inverse of its own mass matrix to 1e-15,
                // and `cond(M)` is only 1e3): a fruit fly's principal inertias run down to 7e-14, and
                // MuJoCo diagonalises them with `mjuu_eig3`, which stops on an ABSOLUTE 1e-12. At that
                // scale the stopping rule fires before the first sweep and what comes out depends on the
                // iteration, not on the matrix. It is the documented `eig3` floor at its extreme.
                && (a.acc0 - e.acc0).abs() <= 1e-7 * e.acc0.abs().max(1.0)
                && (a.gear - e.gear).abs() < 1e-12
                && close(a.gainprm, e.gainprm)
                && close(a.biasprm, e.biasprm)
                && a.ctrlrange == e.ctrlrange
                && a.forcerange == e.forcerange;
            if same {
                params_ok += 1;
            } else if notes.len() < 8 {
                notes.push(format!(
                    "{} '{}' ({}): gain {gain}/{} bias {bias}/{} dyn {dyn_}/{} acc0 {:.9}/{:.9} gear {}/{} gainprm {:?}/{:?} biasprm {:?}/{:?} ctrl {:?}/{:?} force {:?}/{:?}",
                    o.rel, e.name, e.trn, e.gain, e.bias, e.dyn_, a.acc0, e.acc0, a.gear, e.gear, a.gainprm, e.gainprm, a.biasprm, e.biasprm, a.ctrlrange, e.ctrlrange, a.forcerange, e.forcerange
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
                // ⛔ the moment for THIS state: a site transmission's turns with the model
                let st = t.actuator_state_at(a, &q, &s.qvel);
                let (l, v) = (st.length, st.velocity);
                let force = a.force(l, v, s.ctrl[u], s.act.get(u).copied().unwrap_or(0.0));
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
                    let mine = t.qfrc_actuator_act(&q, &s.qvel, &s.ctrl, &s.act);
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
    if !blocked_models.is_empty() {
        println!("  models with an actuator not carried: {}", blocked_models.len());
        for b in &blocked_models {
            println!("    {b}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
