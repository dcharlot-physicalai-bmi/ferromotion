//! **The change of basis a FREE JOINT needs, checked against MuJoCo's mass matrix.**
//!
//! A free joint is six coordinates here and six there, but not the same six. MuJoCo's are the body frame
//! origin's world velocity and an angular velocity; this port's are three world slides and three Euler
//! hinges, `R = Rz(ψ)·Ry(θ)·Rx(φ)`. Everything else in the model corresponds one for one, so the whole
//! difference is a `6 × 6` block `T` with `v_mujoco = T·v_ours`, and the thing that pins it is the MASS
//! MATRIX: `M_ours = Tᵀ·M_mujoco·T` has to hold entry by entry, or the basis is wrong.
//!
//! ⛔ Get this right and three things follow that are impossible without it: `qacc` becomes comparable on a
//! floating base, `dof_armature` can be added where MuJoCo adds it (a diagonal in ITS basis, which is not
//! one in ours), and a contact Jacobian on a legged robot has columns that mean the same thing.
//!
//! ```text
//! cargo run --release --example menagerie_floating -- <menagerie root> <floating oracle txt> [substring]
//! ```

use ferromotion_core::{tree_from_mjcf, MjcfJointKind};
use nalgebra::DMatrix;
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    nv: usize,
    joints: Vec<(String, String, usize)>,
    armature: Vec<f64>,
    invweight0: Vec<f64>,
    qpos0: Vec<f64>,
    states: Vec<State>,
}

#[derive(Default)]
struct State {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    smooth: Vec<f64>,
    xquat: Vec<f64>,
    xpos: Vec<f64>,
    xipos: Vec<f64>,
    subtreecom: Vec<f64>,
    m: Vec<Vec<f64>>,
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
        eprintln!("usage: menagerie_floating <menagerie root> <floating oracle txt> [substring]");
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
            "model" => cases.push(Case { rel: t[1].to_string(), nv: t[3].parse().unwrap(), ..Default::default() }),
            "joint" => cases.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "armature" => cases.last_mut().unwrap().armature = t[1..].iter().map(|x| f(x)).collect(),
            "invweight0" => cases.last_mut().unwrap().invweight0 = t[1..].iter().map(|x| f(x)).collect(),
            "qpos0" => cases.last_mut().unwrap().qpos0 = t[1..].iter().map(|x| f(x)).collect(),
            "state" => cases.last_mut().unwrap().states.push(State::default()),
            "M" => cases.last_mut().unwrap().states.last_mut().unwrap().m.push(t[1..].iter().map(|x| f(x)).collect()),
            "qpos" | "qvel" | "xquat" | "xpos" | "xipos" | "subtreecom" | "smooth" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = cases.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
                    "smooth" => s.smooth = v,
                    "xquat" => s.xquat = v,
                    "xpos" => s.xpos = v,
                    "xipos" => s.xipos = v,
                    "subtreecom" => s.subtreecom = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let (mut seen, mut ok) = (0usize, 0usize);
    let (mut acc_seen, mut acc_ok, mut acc_worst, mut acc_where) = (0usize, 0usize, 0.0f64, String::new());
    let (mut iw_seen, mut iw_ok, mut iw_worst, mut iw_where) = (0usize, 0usize, 0.0f64, String::new());
    let mut acc_skip: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut worst, mut worst_where) = (0.0f64, String::new());
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&c.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            *counts.entry("the loader refuses the model").or_default() += 1;
            continue;
        };
        // one free joint first, then hinges and slides in the same order — anything else is out of scope here
        let lined_up = t.joints.len() == c.joints.len()
            && matches!(t.joints.first().map(|j| j.kind), Some(MjcfJointKind::Free))
            && c.joints.first().is_some_and(|j| j.1 == "free")
            && t.joints[1..].iter().zip(&c.joints[1..]).all(|(a, b)| matches!((a.kind, b.1.as_str()), (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide")));
        if !lined_up {
            *counts.entry("not a single free joint followed by hinges and slides").or_default() += 1;
            continue;
        }
        let nv = c.nv;
        let qposadr: Vec<usize> = c.joints.iter().map(|j| j.2).collect();
        // ⛔ `dof_invweight0` is computed at MuJoCo's OWN qpos0, in MuJoCo's basis, and every constraint row
        // regularises against it. This port's tree drops a free body's initial pose into the joint, so its
        // `q = 0` is the identity orientation whatever the file said — worth knowing how far that goes.
        {
            iw_seen += 1;
            let ours = t.dof_invweight0_mujoco();
            let e = (0..nv).map(|i| (ours[i] - c.invweight0[i]).abs() / c.invweight0[i].abs().max(1e-9)).fold(0.0, nan_max);
            if e < 1e-9 {
                iw_ok += 1;
            } else {
                if e > iw_worst {
                    iw_worst = e;
                    iw_where = c.rel.clone();
                }
                if std::env::var("DUMP_IW").is_ok() && notes.len() < 3 {
                    notes.push(format!("{}: invweight0 ours {:?}\n                theirs {:?}", c.rel, &ours[..8.min(nv)], &c.invweight0[..8.min(nv)]));
                }
            }
        }
        for s in &c.states {
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else {
                *counts.entry("qpos does not map onto our coordinates").or_default() += 1;
                continue;
            };
            seen += 1;
            // ⛔ `MjcfTree::mass_matrix`, not the bare tree one: armature on a free joint is a diagonal in
            // MuJoCo's basis and has to be mapped into this one
            let ours = t.mass_matrix(&q);
            let theirs = DMatrix::from_fn(nv, nv, |r, k| s.m[r][k]);
            // ⛔ armature is NOT stripped any more. `MjcfTree::mass_matrix` puts it where MuJoCo puts it —
            // a diagonal in MuJoCo's basis, mapped here as `Tᵀ·diag(a)·T` — so both matrices carry it and
            // the identity `M_ours = Tᵀ·M_mujoco·T` has to hold with it included. Stripping it as a diagonal
            // from one side and not the other is how a 1.9e-2 error in the joint-joint block appeared while
            // the basis was right all along.
            let (ours_bare, theirs_bare) = (ours.clone(), theirs.clone());
            let _ = &c.armature;
            let tmat = t.free_basis(&q);
            let mapped = tmat.transpose() * &theirs_bare * &tmat;
            let d = (0..nv).map(|r| (0..nv).map(|k| (mapped[(r, k)] - ours_bare[(r, k)]).abs()).fold(0.0, nan_max)).fold(0.0, nan_max);
            let scale = ours_bare.amax().max(1e-12);
            if d / scale > worst {
                worst = d / scale;
                worst_where = c.rel.clone();
            }
            // ⭐ and now the acceleration itself, in MuJoCo's basis. At REST `a_mujoco = T·a_ours` exactly —
            // the velocity-dependent half of the change of basis is `Ṫ·v`, which is zero — so a floating
            // base's `qacc_smooth` becomes comparable entry by entry for the first time.
            if !t.actuators_unsupported.is_empty() || !t.tendons_unsupported.is_empty() {
                *acc_skip.entry("an actuator or tendon this port does not carry").or_default() += 1;
            } else if let Some(a_mj) = t.qacc_smooth_mujoco(&q, &s.qvel, &vec![0.0; t.actuators.len()]) {
                acc_seen += 1;
                let e = (0..nv).map(|i| (a_mj[i] - s.smooth[i]).abs() / s.smooth[i].abs().max(1.0)).fold(0.0, nan_max);
                let moving = s.qvel.iter().any(|v| *v != 0.0);
                if e > acc_worst {
                    acc_worst = e;
                    acc_where = format!("{}{}", c.rel, if moving { " (moving)" } else { " (at rest)" });
                }
                if e < 1e-6 {
                    acc_ok += 1;
                } else {
                    *acc_skip.entry(if moving { "differs while moving" } else { "differs at rest" }).or_default() += 1;
                }
            }
            if d / scale < 1e-9 {
                ok += 1;
            } else if notes.len() < 8 {
                // WHICH block: the base's own 6x6, the coupling to the joints, or the joints themselves
                let blk = |r0: usize, r1: usize, k0: usize, k1: usize| {
                    (r0..r1).flat_map(|r| (k0..k1).map(move |k| (r, k))).map(|(r, k)| (mapped[(r, k)] - ours_bare[(r, k)]).abs()).fold(0.0, nan_max) / scale
                };
                notes.push(format!(
                    "{}: total mass {} vs {}; M off {:.2e} relative — base-base {:.2e} (trans {:.2e}, rot {:.2e}, cross {:.2e}), base-joint {:.2e}, joint-joint {:.2e}",
                    c.rel,
                    ours_bare[(0, 0)],
                    mapped[(0, 0)],
                    d / scale,
                    blk(0, 6, 0, 6),
                    blk(0, 3, 0, 3),
                    blk(3, 6, 3, 6),
                    blk(0, 3, 3, 6),
                    blk(0, 6, 6, nv),
                    blk(6, nv, 6, nv)
                ));
            }
        }
    }
    println!("free-base states whose mass matrix was compared: {seen}");
    println!("  MuJoCo's M mapped into our basis, entry by entry: {ok}");
    println!("  worst relative difference {worst:.2e} on {worst_where}");
    println!("  dof_invweight0 at qpos0, in MuJoCo's basis: {iw_ok} of {iw_seen} models; worst {iw_worst:.2e} on {iw_where}");
    println!("  qacc_smooth in MUJOCO'S basis, at rest AND moving: {acc_ok} of {acc_seen} within 1e-6; worst {acc_worst:.2e} on {acc_where}");
    for (k, n) in &acc_skip {
        println!("    {n:>5}  {k}");
    }
    if !counts.is_empty() {
        println!("  not compared:");
        for (k, n) in &counts {
            println!("    {n:>5}  {k}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
