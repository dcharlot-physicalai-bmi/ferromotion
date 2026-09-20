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

use ferromotion_core::{tree_from_mjcf, tree_mass_matrix, MjcfJointKind};
use nalgebra::{DMatrix, Matrix3, UnitQuaternion, Vector3};
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    nv: usize,
    joints: Vec<(String, String, usize)>,
    armature: Vec<f64>,
    states: Vec<State>,
}

#[derive(Default)]
struct State {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    xquat: Vec<f64>,
    xpos: Vec<f64>,
    xipos: Vec<f64>,
    subtreecom: Vec<f64>,
    m: Vec<Vec<f64>>,
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
            "state" => cases.last_mut().unwrap().states.push(State::default()),
            "M" => cases.last_mut().unwrap().states.last_mut().unwrap().m.push(t[1..].iter().map(|x| f(x)).collect()),
            "qpos" | "qvel" | "xquat" | "xpos" | "xipos" | "subtreecom" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = cases.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
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
        for s in &c.states {
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else {
                *counts.entry("qpos does not map onto our coordinates").or_default() += 1;
                continue;
            };
            seen += 1;
            let ours = tree_mass_matrix(&t.tree.joints, &t.tree.inertia, &t.tree.parent, &q);
            let theirs = DMatrix::from_fn(nv, nv, |r, k| s.m[r][k]);
            // armature is a diagonal in each basis; strip it from both so the map itself is what is tested
            let arm = |m: &DMatrix<f64>, a: &[f64]| {
                let mut m = m.clone();
                for (i, x) in a.iter().enumerate().take(nv) {
                    m[(i, i)] -= x;
                }
                m
            };
            // ⛔ strip it from BOTH matrices at every dof. Away from the base the two bases coincide and both
            // matrices carry the same armature; stripping it from one of them only turned an exact match
            // into a 1.9e-2 error in the joint-joint block and sent me looking at the basis, which was right.
            let ours_bare = arm(&ours, &c.armature);
            let theirs_bare = arm(&theirs, &c.armature);
            let tmat = basis(&s.xquat, &s.xpos, &s.xipos, &s.subtreecom, nv);
            let mapped = tmat.transpose() * &theirs_bare * &tmat;
            let d = (0..nv).map(|r| (0..nv).map(|k| (mapped[(r, k)] - ours_bare[(r, k)]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            let scale = ours_bare.amax().max(1e-12);
            if d / scale > worst {
                worst = d / scale;
                worst_where = c.rel.clone();
            }
            if d / scale < 1e-9 {
                ok += 1;
            } else if notes.len() < 8 {
                // WHICH block: the base's own 6x6, the coupling to the joints, or the joints themselves
                let blk = |r0: usize, r1: usize, k0: usize, k1: usize| {
                    (r0..r1).flat_map(|r| (k0..k1).map(move |k| (r, k))).map(|(r, k)| (mapped[(r, k)] - ours_bare[(r, k)]).abs()).fold(0.0, f64::max) / scale
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

/// `v_mujoco = T·v_ours` for the whole model: identity off the base, and a `6 × 6` block on it.
///
/// Our base is three world slides then yaw, pitch, roll with `R = Rz(ψ)·Ry(θ)·Rx(φ)`, so the Euler rates map
/// to an angular velocity through the columns `[Rxᵀ Ryᵀ ẑ, Rxᵀ ŷ, x̂]` in the BODY frame.
fn basis(xquat: &[f64], xpos: &[f64], xipos: &[f64], subtreecom: &[f64], nv: usize) -> DMatrix<f64> {
    let mut t = DMatrix::identity(nv, nv);
    let quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(xquat[0], xquat[1], xquat[2], xquat[3]));
    let (roll, pitch, _yaw) = quat.euler_angles();
    let rx: Matrix3<f64> = *UnitQuaternion::from_euler_angles(roll, 0.0, 0.0).to_rotation_matrix().matrix();
    let ry: Matrix3<f64> = *UnitQuaternion::from_euler_angles(0.0, pitch, 0.0).to_rotation_matrix().matrix();
    let cols = [rx.transpose() * ry.transpose() * Vector3::z(), rx.transpose() * Vector3::y(), Vector3::x()];
    for (k, col) in cols.iter().enumerate() {
        for r in 0..3 {
            t[(3 + r, 3 + k)] = col[r];
        }
    }
    let _ = (xpos, xipos, subtreecom);
    t
}
