//! **Every tendon's length and Jacobian, against MuJoCo's** — `mj_tendon`, fixed and spatial.
//!
//! A fixed tendon's Jacobian is a row of constants and can be checked at load. A spatial one's is the
//! derivative of a path that bends around obstacles, so it has to be checked at STATES: which way round the
//! cable goes, where it leaves the surface, and whether it touches the obstacle at all are all functions of
//! the pose, and none of them is continuous in it.
//!
//! ⛔ MuJoCo stores `ten_J` sparse on anything bigger than a gripper and does not expose its index arrays,
//! so the Jacobian is checked by what it DOES: `ten_velocity = ten_J·q̇` for three random velocities per
//! state. A wrong row survives one random projection with probability zero.
//!
//! ```text
//! cargo run --release --example menagerie_tendons -- <menagerie root> <tendon oracle txt> [substring]
//! ```

use ferromotion_core::{tree_from_mjcf, MjcfJointKind};
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    nv: usize,
    joints: Vec<(String, String, usize)>,
    tendons: Vec<(String, String)>,
    states: Vec<State>,
}

#[derive(Default)]
struct State {
    qpos: Vec<f64>,
    len: Vec<f64>,
    vel: Vec<(Vec<f64>, Vec<f64>)>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_tendons <menagerie root> <tendon oracle txt> [substring]");
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
            "model" => cases.push(Case { rel: t[1].to_string(), nv: t[2].parse().unwrap(), ..Default::default() }),
            "joint" => cases.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "tendon" => cases.last_mut().unwrap().tendons.push((t[1].to_string(), t[2].to_string())),
            "state" => cases.last_mut().unwrap().states.push(State { qpos: t[1..].iter().map(|x| f(x)).collect(), ..Default::default() }),
            "len" => cases.last_mut().unwrap().states.last_mut().unwrap().len = t[1..].iter().map(|x| f(x)).collect(),
            "qvel" => cases.last_mut().unwrap().states.last_mut().unwrap().vel.push((t[1..].iter().map(|x| f(x)).collect(), Vec::new())),
            "tenvel" => cases.last_mut().unwrap().states.last_mut().unwrap().vel.last_mut().unwrap().1 = t[1..].iter().map(|x| f(x)).collect(),
            _ => {}
        }
    }

    let (mut models, mut tendons, mut compared, mut ok_len, mut ok_vel) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut worst_len, mut worst_len_where, mut worst_vel, mut worst_vel_where) = (0.0f64, String::new(), 0.0f64, String::new());
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut bad: BTreeMap<String, usize> = BTreeMap::new();
    let (mut seen_kind, mut ok_kind): (BTreeMap<String, usize>, BTreeMap<String, usize>) = (BTreeMap::new(), BTreeMap::new());
    let mut notes: Vec<String> = Vec::new();
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&c.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            *counts.entry("the loader refuses the model".into()).or_default() += 1;
            continue;
        };
        models += 1;
        tendons += c.tendons.len();
        for (name, why) in &t.tendons_unsupported {
            *counts.entry(why.clone()).or_default() += 1;
            let _ = name;
        }
        // the dof columns only correspond when every joint is a hinge or a slide, in the same order
        let lined_up = t.joints.len() == c.joints.len()
            && t.joints.iter().zip(&c.joints).all(|(a, b)| matches!((a.kind, b.1.as_str()), (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide")));
        if !lined_up {
            *counts.entry("a free or ball joint: the dof columns do not correspond".into()).or_default() += 1;
            continue;
        }
        // ours by name, so a tendon MuJoCo has and we do not simply is not compared
        let ours: BTreeMap<&str, usize> = t.tendons.iter().enumerate().map(|(i, x)| (x.name.as_str(), i)).collect();
        let qposadr: Vec<usize> = c.joints.iter().map(|j| j.2).collect();
        for s in &c.states {
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else { continue };
            let (len, mom) = (t.ten_length(&q), t.ten_moment(&q));
            for (u, (name, kind)) in c.tendons.iter().enumerate() {
                let Some(&i) = ours.get(name.as_str()) else { continue };
                compared += 1;
                let dl = (len[i] - s.len[u]).abs();
                if dl > worst_len {
                    worst_len = dl;
                    worst_len_where = format!("{} / {name} ({kind})", c.rel);
                }
                *seen_kind.entry(kind.clone()).or_default() += 1;
                if dl < 1e-9 {
                    ok_len += 1;
                    *ok_kind.entry(kind.clone()).or_default() += 1;
                } else {
                    *bad.entry(format!("{} ({kind})", c.rel.split('/').next().unwrap_or(""))).or_default() += 1;
                    if notes.len() < 6 {
                        notes.push(format!("{} / {name} ({kind}): length {} vs {}", c.rel, len[i], s.len[u]));
                    }
                }
                if std::env::var("DUMP_PATH").is_ok() && dl > 1e-9 {
                    println!("  {} / {name}: our path", c.rel);
                    for (p, arc) in t.tendon_path_points(i, &q) {
                        println!("      [{:.6}, {:.6}, {:.6}]{}", p[0], p[1], p[2], if arc { "  (arc to next)" } else { "" });
                    }
                }
                let mut worst = 0.0f64;
                for (qv, tv) in &s.vel {
                    let mine: f64 = mom[i].iter().map(|(d, m)| m * qv[*d]).sum();
                    worst = worst.max((mine - tv[u]).abs());
                }
                if worst > worst_vel {
                    worst_vel = worst;
                    worst_vel_where = format!("{} / {name} ({kind})", c.rel);
                }
                if worst < 1e-9 {
                    ok_vel += 1;
                } else if notes.len() < 10 {
                    notes.push(format!("{} / {name} ({kind}): ten_J·qvel off {worst:.2e}", c.rel));
                }
            }
        }
        let _ = c.nv;
    }
    println!("models with tendons: {models} ({tendons} tendons)");
    println!("  tendon-states compared: {compared}");
    println!("  length matching MuJoCo: {ok_len}; worst {worst_len:.2e} on {worst_len_where}");
    for (k, n) in &seen_kind {
        println!("    {k}: {} of {n}", ok_kind.get(k).copied().unwrap_or(0));
    }
    println!("  ten_J·qvel matching, over three random velocities each: {ok_vel}; worst {worst_vel:.2e} on {worst_vel_where}");
    if !counts.is_empty() {
        println!("  not compared:");
        let mut v: Vec<_> = counts.iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (k, n) in v.iter().take(8) {
            println!("    {n:>5}  {k}");
        }
    }
    if !bad.is_empty() {
        println!("  lengths that disagree, by model:");
        let mut v: Vec<_> = bad.iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (k, n) in v.iter().take(10) {
            println!("    {n:>5}  {k}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
