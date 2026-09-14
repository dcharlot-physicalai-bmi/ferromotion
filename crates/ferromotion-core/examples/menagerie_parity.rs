//! **MJCF loader and tree-dynamics parity against MuJoCo, over the whole of MuJoCo Menagerie.**
//!
//! MuJoCo is the physics library the field runs (3.6 M pip installs a month, September 2026), and Menagerie is
//! the asset collection its users load. Whether [`ferromotion_core::tree_from_mjcf`] reads those files, and
//! whether [`ferromotion_core::tree_forward_dynamics`] then moves them the way MuJoCo does, are questions with
//! measurable answers, and this example measures them. For every Menagerie model that MuJoCo itself compiles,
//! MuJoCo's own `mj_kinematics` / `mj_forward` / `mj_fullM` at sampled states is the oracle:
//!
//! * **kinematics** — every body and site pose, to `1e-9`;
//! * **inertia** — mass, centre of mass and inertia tensor of every jointed body that states an `<inertial>`
//!   and has no jointless child welded into it, to `1e-9` relative;
//! * **dynamics** — on fixed-base models whose every massive body states an `<inertial>`: the mass matrix
//!   `M(q)`, the bias force `qfrc_bias` (Coriolis + gravity) and the free acceleration `qacc` under gravity
//!   and joint damping alone, to `1e-8` relative. On models with ball or free joints only the hinge/slide block
//!   of `M` is compared, because MuJoCo's ball/free velocity coordinates differ from the Euler-rate hinges
//!   this loader carries them as.
//!
//! The oracle is produced outside this crate — `scripts/menagerie_oracle.py` over the `mujoco` wheel writes
//! one flat text file per model, with contacts, constraints, springs, actuators and gravity compensation
//! disabled — because the point is to compare against MuJoCo and not against a re-implementation of it. Run as
//!
//! ```text
//! python scripts/menagerie_oracle.py <menagerie_root> <oracle_dir> 4
//! cargo run --release -p ferromotion-core --example menagerie_parity -- <oracle_dir> <menagerie_root>
//! ```
//!
//! # What it measured on 2026-09-14 (MuJoCo 3.13.0, Menagerie at 2026-09-04, 204 compilable models)
//!
//! * kinematics: **201 of 204 models exact** — 3,979 bodies, 3,804 joints, 959 sites, worst error 1.7e-15 m.
//!   The three refusals: `<attach>` (procedural composition) and one OBJ with a 215-vertex face, whose
//!   ear-clipping triangulation is not reproduced (two models share it).
//! * inertia, stated: 2,873 of 3,267 jointed bodies with an `<inertial>` agree to 1e-9 relative; **every** body
//!   outside that tolerance (worst 1.4e-6) states a `fullinertia`, which MuJoCo eigendecomposes with its
//!   Jacobi solver — that solver's precision, not this loader's.
//! * inertia, inferred from geoms by MuJoCo's own rules (density × volume, `legacy` mesh volumes, mesh fitting,
//!   its Jacobi solver ported): **257 of 259 jointed bodies within 1e-5 relative**, worst 2.7e-5 (a fruit-fly
//!   abdomen segment at 1e-13 kg·m²).
//! * dynamics, XML inertials: 84 of 92 fixed-base models within 1e-8 relative on `M`, bias and `qacc`; the
//!   other 8 are the Franka family at 1e-8…1.4e-7, downstream of the `fullinertia` difference above. The
//!   hinge/slide block of `M` matched on 106 of 109 ball/free-jointed models, the other three at ≤1.3e-7.
//! * dynamics, **with MuJoCo's own processed inertials substituted: 201 of 201 models, worst relative
//!   error 2.9e-15 on `M`, 8.9e-15 on the bias, 1.5e-13 on `qacc`.** The tree ABA, RNEA and CRBA here are the
//!   same algorithms MuJoCo runs, to round-off, on every Menagerie model MuJoCo compiles.
//!
//! Each oracle file reads: `model <rel> …`, `body <name> <parent> <mass>`, `joint <name> <type> <body>
//! <qposadr> <dofadr>`, `binertia <body> <mass> <ipos×3> <iquat×4> <inertia×3>`, `site <name> <body>`,
//! `nv <n>`, `gravity <3>`; then per sample `sample <qpos…>`, `xpos <body> <3> <4>`, `sxpos <site> <3>`,
//! `qvel <nv>`, `bias <nv>`, `qacc <nv>`, `M <nv·nv>`.

use ferromotion_core::{tree_forward_dynamics, tree_from_mjcf, tree_inverse_dynamics, tree_mass_matrix, MjcfJointKind, MjcfTree};
use nalgebra::{DMatrix, Matrix3, Point3, UnitQuaternion, Vector3};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

const KIN_TOL: f64 = 1e-9;
const INERTIA_TOL: f64 = 1e-9;
const DYN_TOL: f64 = 1e-8;
const INFERRED_TOL: f64 = 1e-5;

struct Oracle {
    rel: String,
    bodies: Vec<(String, f64)>,
    joints: Vec<(String, String, usize, usize)>, // name, type, qposadr, dofadr
    inertials: HashMap<String, [f64; 11]>,
    sites: Vec<String>,
    nv: usize,
    gravity: Vector3<f64>,
    samples: Vec<Sample>,
}

#[derive(Default)]
struct Sample {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    bias: Vec<f64>,
    qacc: Vec<f64>,
    m: Vec<f64>,
    xpos: HashMap<String, [f64; 7]>,
    sxpos: HashMap<String, [f64; 3]>,
}

fn parse_oracle(text: &str) -> Option<Oracle> {
    let mut o = Oracle { rel: String::new(), bodies: Vec::new(), joints: Vec::new(), inertials: HashMap::new(), sites: Vec::new(), nv: 0, gravity: Vector3::new(0.0, 0.0, -9.81), samples: Vec::new() };
    let nums = |it: std::str::SplitWhitespace| -> Vec<f64> { it.map(|t| t.parse().unwrap()).collect() };
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next()? {
            "model" => o.rel = it.next()?.to_string(),
            "body" => {
                let name = it.next()?.to_string();
                let _parent = it.next()?;
                o.bodies.push((name, it.next()?.parse().ok()?));
            }
            "joint" => {
                let name = it.next()?.to_string();
                let ty = it.next()?.to_string();
                let _body = it.next()?;
                let qa = it.next()?.parse().ok()?;
                o.joints.push((name, ty, qa, it.next()?.parse().ok()?));
            }
            "binertia" => {
                let name = it.next()?.to_string();
                let v = nums(it);
                let mut a = [0.0; 11];
                a.copy_from_slice(&v[..11]);
                o.inertials.insert(name, a);
            }
            "site" => o.sites.push(it.next()?.to_string()),
            "nv" => o.nv = it.next()?.parse().ok()?,
            "gravity" => {
                let v = nums(it);
                o.gravity = Vector3::new(v[0], v[1], v[2]);
            }
            "sample" => o.samples.push(Sample { qpos: nums(it), ..Default::default() }),
            "xpos" => {
                let name = it.next()?.to_string();
                let v = nums(it);
                o.samples.last_mut()?.xpos.insert(name, [v[0], v[1], v[2], v[3], v[4], v[5], v[6]]);
            }
            "sxpos" => {
                let name = it.next()?.to_string();
                let v = nums(it);
                o.samples.last_mut()?.sxpos.insert(name, [v[0], v[1], v[2]]);
            }
            "qvel" => o.samples.last_mut()?.qvel = nums(it),
            "bias" => o.samples.last_mut()?.bias = nums(it),
            "qacc" => o.samples.last_mut()?.qacc = nums(it),
            "M" => o.samples.last_mut()?.m = nums(it),
            _ => {}
        }
    }
    Some(o)
}

fn quat_dist(p: &nalgebra::Isometry3<f64>, x: &[f64]) -> f64 {
    let r = p.rotation.quaternion();
    let (w, i, j, k) = (r.w, r.i, r.j, r.k);
    let d1 = ((w - x[0]).powi(2) + (i - x[1]).powi(2) + (j - x[2]).powi(2) + (k - x[3]).powi(2)).sqrt();
    let d2 = ((w + x[0]).powi(2) + (i + x[1]).powi(2) + (j + x[2]).powi(2) + (k + x[3]).powi(2)).sqrt();
    d1.min(d2)
}

struct Report {
    kin: Result<(f64, f64), String>,
    /// stated <inertial>: bodies compared, bodies exact, worst relative error, its body
    inertia: (usize, usize, f64, String),
    /// inferred from geoms: the same four, at the looser tolerance MuJoCo's eigen-solver leaves
    inferred: (usize, usize, f64, String),
    /// `Err` = skipped with reason; `Ok((what, worst_m, worst_bias, worst_qacc))`
    dyn_: Result<(String, f64, f64, f64), String>,
    /// the same three, with MuJoCo's processed inertials substituted for the XML's
    dyn_mj: Option<(f64, f64, f64)>,
}

fn compare(t: &MjcfTree, o: &Oracle) -> Report {
    let mut rep = Report { kin: Err(String::new()), inertia: (0, 0, 0.0, String::new()), inferred: (0, 0, 0.0, String::new()), dyn_: Err(String::new()), dyn_mj: None };
    // --- addresses: MuJoCo's joints and ours must agree name-for-name
    let by_name: HashMap<&str, (usize, usize)> = o.joints.iter().map(|(n, _, qa, da)| (n.as_str(), (*qa, *da))).collect();
    let mut qadr = Vec::with_capacity(t.joints.len());
    let mut dadr = Vec::with_capacity(t.joints.len());
    for j in &t.joints {
        match by_name.get(j.name.as_str()) {
            Some((qa, da)) => {
                qadr.push(*qa);
                dadr.push(*da);
            }
            None => {
                rep.kin = Err(format!("joint '{}' is not in MuJoCo's model", j.name));
                return rep;
            }
        }
    }
    if t.joints.len() != o.joints.len() {
        rep.kin = Err(format!("{} joints loaded, MuJoCo has {}", t.joints.len(), o.joints.len()));
        return rep;
    }
    let n = t.tree.dof();
    for (b, _) in &o.bodies {
        if t.body_pose(b, &vec![0.0; n]).is_none() {
            rep.kin = Err(format!("body '{b}' is missing"));
            return rep;
        }
    }

    // --- kinematics
    let (mut max_pos, mut max_quat) = (0.0f64, 0.0f64);
    let mut worst = String::new();
    let mut qs = Vec::new();
    for s in &o.samples {
        let q = match t.q_from_qpos(&s.qpos, &qadr) {
            Ok(q) => q,
            Err(e) => {
                rep.kin = Err(e);
                return rep;
            }
        };
        for (b, x) in &s.xpos {
            let p = t.body_pose(b, &q).unwrap();
            let dp = (p.translation.vector - Vector3::new(x[0], x[1], x[2])).norm();
            let dq = quat_dist(&p, &x[3..]);
            if dp > max_pos || dq > max_quat {
                worst = b.clone();
            }
            max_pos = max_pos.max(dp);
            max_quat = max_quat.max(dq);
        }
        for (name, x) in &s.sxpos {
            let Some(p) = t.site_pose(name, &q) else {
                rep.kin = Err(format!("site '{name}' is missing"));
                return rep;
            };
            let dp = (p.translation.vector - Vector3::new(x[0], x[1], x[2])).norm();
            if dp > max_pos {
                worst = format!("site {name}");
            }
            max_pos = max_pos.max(dp);
        }
        qs.push(q);
    }
    rep.kin = if max_pos > KIN_TOL || max_quat > KIN_TOL { Err(format!("worst '{worst}': pos {max_pos:.2e} quat {max_quat:.2e}")) } else { Ok((max_pos, max_quat)) };

    // --- inertia, per jointed body that states an <inertial> and carries no welded child
    let jointed: HashSet<&String> = t.tree.link_names.keys().collect();
    let welded_into: HashSet<usize> = t.body_frames.iter().filter(|(name, _)| !jointed.contains(name)).map(|(_, (idx, _))| *idx).collect();
    let inferred: HashSet<&String> = t.inferred_from_geoms.iter().collect();
    let (mut compared, mut exact, mut worst_rel, mut worst_name) = (0usize, 0usize, 0.0f64, String::new());
    let (mut inf_compared, mut inf_close, mut inf_worst, mut inf_worst_name) = (0usize, 0usize, 0.0f64, String::new());
    for (name, idx) in &t.tree.link_names {
        if welded_into.contains(idx) {
            continue;
        }
        let Some(mi) = o.inertials.get(name) else { continue };
        let Some((_, off)) = t.body_frames.get(name) else { continue };
        let ours = &t.tree.inertia[*idx];
        let com = off * Point3::new(mi[1], mi[2], mi[3]);
        let rq = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(mi[4], mi[5], mi[6], mi[7]));
        let r = *rq.to_rotation_matrix().matrix();
        let tensor = r * Matrix3::from_diagonal(&Vector3::new(mi[8], mi[9], mi[10])) * r.transpose();
        let scale = mi[0].abs().max(1e-12);
        let e_mass = (ours.mass - mi[0]).abs() / scale;
        let e_com = (ours.com - com.coords).norm() / (com.coords.norm() + 1e-3);
        let e_ten = (ours.inertia - tensor).norm() / (tensor.norm() + 1e-12);
        let e = e_mass.max(e_com).max(e_ten);
        if inferred.contains(name) {
            // MuJoCo's mesh inertia goes through its iterative eigen-solver twice; 1e-5 is that solver's precision
            inf_compared += 1;
            if e <= INFERRED_TOL {
                inf_close += 1;
            }
            if e > inf_worst {
                inf_worst = e;
                inf_worst_name = name.clone();
            }
            continue;
        }
        compared += 1;
        if e <= INERTIA_TOL {
            exact += 1;
        }
        if e > worst_rel {
            worst_rel = e;
            worst_name = name.clone();
        }
    }
    rep.inertia = (compared, exact, worst_rel, worst_name);
    rep.inferred = (inf_compared, inf_close, inf_worst, inf_worst_name);

    // --- dynamics
    // a world-fixed body's mass never enters the equations of motion, so only bodies riding on a joint can block
    let massive_uninferred: Vec<&String> = t.no_inertial.iter().filter(|b| !inferred.contains(*b) && t.body_frames.contains_key(*b) && o.bodies.iter().any(|(n, m)| n == *b && *m > 1e-12)).collect();
    if !massive_uninferred.is_empty() {
        rep.dyn_ = Err(format!("{} bodies MuJoCo weighs and this loader could not (e.g. '{}')", massive_uninferred.len(), massive_uninferred[0]));
        return rep;
    }
    if rep.kin.is_err() {
        rep.dyn_ = Err("kinematics did not match".into());
        return rep;
    }
    // MuJoCo's `frictionloss` is a constraint (disabled in the oracle); its damping is passive (kept). Strip
    // friction so both sides model the same forces.
    let joints: Vec<_> = t.tree.joints.iter().map(|j| {
        let mut j = j.clone();
        j.friction = None;
        j
    }).collect();
    let undamped: Vec<_> = joints.iter().map(|j| {
        let mut j = j.clone();
        j.damping = None;
        j
    }).collect();
    let all_1dof = t.joints.iter().all(|j| matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide));
    // dof index in MuJoCo for each of our tree joints (only meaningful for hinge/slide)
    let mut mj_dof = vec![usize::MAX; n];
    for (j, &da) in t.joints.iter().zip(&dadr) {
        if matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) {
            mj_dof[j.first] = da;
        }
    }
    let ours_dofs: Vec<usize> = (0..n).filter(|&i| mj_dof[i] != usize::MAX).collect();
    let (mut worst_m, mut worst_b, mut worst_a) = (0.0f64, 0.0f64, 0.0f64);
    for (s, q) in o.samples.iter().zip(&qs) {
        if s.m.len() != o.nv * o.nv {
            rep.dyn_ = Err("oracle has no dynamics lines".into());
            return rep;
        }
        let m_ours = tree_mass_matrix(&joints, &t.tree.inertia, &t.tree.parent, q);
        let m_ref = DMatrix::from_row_slice(o.nv, o.nv, &s.m);
        let mut scale = 0.0f64;
        let mut dm = 0.0f64;
        for &i in &ours_dofs {
            for &j in &ours_dofs {
                let r = m_ref[(mj_dof[i], mj_dof[j])];
                scale = scale.max(r.abs());
                dm = dm.max((m_ours[(i, j)] - r).abs());
            }
        }
        worst_m = worst_m.max(dm / scale.max(1e-12));
        if all_1dof {
            let mut qd = vec![0.0; n];
            for i in 0..n {
                qd[i] = s.qvel[mj_dof[i]];
            }
            let bias = tree_inverse_dynamics(&undamped, &t.tree.inertia, &t.tree.parent, q, &qd, &vec![0.0; n], o.gravity);
            let qacc = tree_forward_dynamics(&joints, &t.tree.inertia, &t.tree.parent, q, &qd, &vec![0.0; n], o.gravity);
            let (mut sb, mut db, mut sa, mut da) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                sb = sb.max(s.bias[mj_dof[i]].abs());
                db = db.max((bias[i] - s.bias[mj_dof[i]]).abs());
                sa = sa.max(s.qacc[mj_dof[i]].abs());
                da = da.max((qacc[i] - s.qacc[mj_dof[i]]).abs());
            }
            worst_b = worst_b.max(db / sb.max(1.0));
            worst_a = worst_a.max(da / sa.max(1.0));
        }
    }
    let what = if all_1dof { "M+bias+qacc" } else { "M hinge/slide block" };
    rep.dyn_ = Ok((what.to_string(), worst_m, worst_b, worst_a));

    // --- the same dynamics with MuJoCo's OWN processed inertials, so the comparison is of the algorithms alone.
    // Each tree link = its jointed body plus every jointless body welded onto it, composed at the link frame.
    let mut mj_inertia: Vec<Option<(f64, Vector3<f64>, Matrix3<f64>)>> = vec![None; n];
    for (name, (idx, off)) in &t.body_frames {
        let Some(mi) = o.inertials.get(name) else { continue };
        let com = (off * Point3::new(mi[1], mi[2], mi[3])).coords;
        let rq = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(mi[4], mi[5], mi[6], mi[7]));
        let r = *(off.rotation * rq).to_rotation_matrix().matrix();
        let tensor = r * Matrix3::from_diagonal(&Vector3::new(mi[8], mi[9], mi[10])) * r.transpose();
        let slot = &mut mj_inertia[*idx];
        *slot = Some(match slot.take() {
            None => (mi[0], com, tensor),
            Some((m0, c0, i0)) => {
                let m = m0 + mi[0];
                let c = if m > 0.0 { (c0 * m0 + com * mi[0]) / m } else { c0 };
                let shift = |ik: Matrix3<f64>, mk: f64, ck: Vector3<f64>| {
                    let d = ck - c;
                    ik + mk * (Matrix3::identity() * d.dot(&d) - d * d.transpose())
                };
                (m, c, shift(i0, m0, c0) + shift(tensor, mi[0], com))
            }
        });
    }
    let inertia_mj: Vec<_> = mj_inertia
        .into_iter()
        .map(|e| match e {
            Some((mass, com, inertia)) => ferromotion_core::LinkInertia { mass, com, inertia },
            None => ferromotion_core::LinkInertia::zero(),
        })
        .collect();
    let (mut wm, mut wb, mut wa) = (0.0f64, 0.0f64, 0.0f64);
    for (s, q) in o.samples.iter().zip(&qs) {
        let m_ours = tree_mass_matrix(&joints, &inertia_mj, &t.tree.parent, q);
        let m_ref = DMatrix::from_row_slice(o.nv, o.nv, &s.m);
        let (mut scale, mut dm) = (0.0f64, 0.0f64);
        for &i in &ours_dofs {
            for &j in &ours_dofs {
                let r = m_ref[(mj_dof[i], mj_dof[j])];
                scale = scale.max(r.abs());
                dm = dm.max((m_ours[(i, j)] - r).abs());
            }
        }
        wm = wm.max(dm / scale.max(1e-12));
        if all_1dof {
            let qd: Vec<f64> = (0..n).map(|i| s.qvel[mj_dof[i]]).collect();
            let bias = tree_inverse_dynamics(&undamped, &inertia_mj, &t.tree.parent, q, &qd, &vec![0.0; n], o.gravity);
            let qacc = tree_forward_dynamics(&joints, &inertia_mj, &t.tree.parent, q, &qd, &vec![0.0; n], o.gravity);
            let (mut sb, mut db, mut sa, mut da) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                sb = sb.max(s.bias[mj_dof[i]].abs());
                db = db.max((bias[i] - s.bias[mj_dof[i]]).abs());
                sa = sa.max(s.qacc[mj_dof[i]].abs());
                da = da.max((qacc[i] - s.qacc[mj_dof[i]]).abs());
            }
            wb = wb.max(db / sb.max(1.0));
            wa = wa.max(da / sa.max(1.0));
        }
    }
    rep.dyn_mj = Some((wm, wb, wa));
    rep
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_parity <oracle_dir> <menagerie_root>");
        std::process::exit(2);
    }
    let (oracle_dir, root) = (Path::new(&args[1]), Path::new(&args[2]));
    let mut files: Vec<_> = std::fs::read_dir(oracle_dir).expect("oracle dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    let (mut kin_ok, mut kin_bad, mut refused) = (0usize, 0usize, 0usize);
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let (mut bodies_ok, mut sites_ok, mut joints_ok) = (0usize, 0usize, 0usize);
    let (mut in_cmp, mut in_exact, mut in_worst) = (0usize, 0usize, 0.0f64);
    let (mut inf_cmp, mut inf_ok, mut inf_worst) = (0usize, 0usize, 0.0f64);
    let (mut dyn_full, mut dyn_full_ok, mut dyn_block, mut dyn_block_ok) = (0usize, 0usize, 0usize, 0usize);
    let mut dyn_skipped: BTreeMap<String, usize> = BTreeMap::new();
    let (mut worst_m, mut worst_b, mut worst_a) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mj_n, mut mj_ok, mut mj_wm, mut mj_wb, mut mj_wa) = (0usize, 0usize, 0.0f64, 0.0f64, 0.0f64);
    let mut no_inertial_bodies = 0usize;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let Some(o) = parse_oracle(&text) else { continue };
        let model = root.join(&o.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let xml = std::fs::read_to_string(&model).unwrap();
        let resolve = |p: &str| std::fs::read(dir.join(p)).ok();
        let t = match tree_from_mjcf(&xml, &resolve) {
            Ok(t) => t,
            Err(e) => {
                refused += 1;
                let key: String = e.chars().take(70).collect();
                *reasons.entry(key).or_insert(0) += 1;
                println!("REFUSED   {:<48} {e}", o.rel);
                continue;
            }
        };
        no_inertial_bodies += t.no_inertial.len();
        let r = compare(&t, &o);
        let kin = match &r.kin {
            Ok((p, q)) => {
                kin_ok += 1;
                bodies_ok += o.bodies.len();
                sites_ok += o.sites.len();
                joints_ok += o.joints.len();
                format!("kin pos {p:.0e} quat {q:.0e}")
            }
            Err(e) => {
                kin_bad += 1;
                format!("KIN MISMATCH {e}")
            }
        };
        let (c, e, w, wn) = &r.inertia;
        in_cmp += c;
        in_exact += e;
        in_worst = in_worst.max(*w);
        let (ic, ie, iw, iwn) = &r.inferred;
        inf_cmp += ic;
        inf_ok += ie;
        inf_worst = inf_worst.max(*iw);
        let mut inertia = if c == e { format!("inertia {e}/{c}") } else { format!("INERTIA {e}/{c} worst '{wn}' {w:.1e}") };
        if *ic > 0 {
            inertia += &if ic == ie { format!(" inferred {ie}/{ic} ({iw:.0e})") } else { format!(" INFERRED {ie}/{ic} worst '{iwn}' {iw:.1e}") };
        }
        let dynamics = match &r.dyn_ {
            Ok((what, m, b, a)) => {
                let ok = *m <= DYN_TOL && *b <= DYN_TOL && *a <= DYN_TOL;
                if what.starts_with("M+") {
                    dyn_full += 1;
                    dyn_full_ok += ok as usize;
                } else {
                    dyn_block += 1;
                    dyn_block_ok += ok as usize;
                }
                worst_m = worst_m.max(*m);
                worst_b = worst_b.max(*b);
                worst_a = worst_a.max(*a);
                format!("{}{what}: M {m:.0e} bias {b:.0e} qacc {a:.0e}", if ok { "" } else { "DYN MISMATCH " })
            }
            Err(why) => {
                let key: String = why.split(" (e.g.").next().unwrap_or(why).to_string();
                *dyn_skipped.entry(key).or_insert(0) += 1;
                format!("dyn skipped: {why}")
            }
        };
        let with_mj = match r.dyn_mj {
            Some((m, b, a)) => {
                mj_n += 1;
                let ok = m <= DYN_TOL && b <= DYN_TOL && a <= DYN_TOL;
                mj_ok += ok as usize;
                mj_wm = mj_wm.max(m);
                mj_wb = mj_wb.max(b);
                mj_wa = mj_wa.max(a);
                format!(" | with MuJoCo's inertials: M {m:.0e} bias {b:.0e} qacc {a:.0e}")
            }
            None => String::new(),
        };
        println!("{:<48} {kin} | {inertia} | {dynamics}{with_mj}", o.rel);
    }
    println!();
    println!("kinematics: {} oracle models, {kin_ok} exact ({bodies_ok} bodies, {joints_ok} joints, {sites_ok} sites), {kin_bad} mismatched, {refused} refused", files.len());
    println!("inertia:    {in_exact}/{in_cmp} jointed bodies with a stated <inertial> match MuJoCo to 1e-9 (worst rel {in_worst:.1e})");
    println!("            {inf_ok}/{inf_cmp} jointed bodies inferred from geoms match MuJoCo to 1e-5 (worst rel {inf_worst:.1e}); {no_inertial_bodies} bodies state no <inertial>");
    println!("dynamics:   fixed-base M+bias+qacc {dyn_full_ok}/{dyn_full} models; hinge/slide M block on ball/free models {dyn_block_ok}/{dyn_block}; worst rel M {worst_m:.1e} bias {worst_b:.1e} qacc {worst_a:.1e}");
    println!("            with MuJoCo's own processed inertials substituted: {mj_ok}/{mj_n} models; worst rel M {mj_wm:.1e} bias {mj_wb:.1e} qacc {mj_wa:.1e}");
    if !dyn_skipped.is_empty() {
        println!("dynamics skipped:");
        for (r, n) in &dyn_skipped {
            println!("  {n:>3}  {r}");
        }
    }
    if !reasons.is_empty() {
        println!("refusals:");
        for (r, n) in &reasons {
            println!("  {n:>3}  {r}");
        }
    }
}
