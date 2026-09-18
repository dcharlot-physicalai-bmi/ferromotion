//! **End to end: an MJCF file in, an acceleration out, against MuJoCo's own.**
//!
//! Every other sweep here tests one stage against MuJoCo with the other stages' inputs taken FROM MuJoCo —
//! the contact sweep uses the oracle's geom poses, the constraint tests use the oracle's `efc_J` and
//! `efc_aref`. Nothing had checked that the stages compose: load the model with our loader, place it with our
//! kinematics, find the contacts with our collision pipeline, build the rows with our constraint model, solve
//! with our solver, and see whether `qacc` is MuJoCo's.
//!
//! ```text
//! cargo run --release --example menagerie_forward -- <menagerie root> <forward oracle txt> [substring]
//! ```
//!
//! ⛔ It reports what it CANNOT compare as loudly as what it can. A state is only comparable when MuJoCo's
//! own constraint set is one this port claims: contacts and nothing else, `condim` 1 or 3, a pyramidal cone,
//! and no actuator or passive force acting. The rest are counted by what blocks them, because that list is
//! the remaining work and hiding it inside an average would be a lie about coverage.

use ferromotion_core::{
    can_collide, collide_pair_with, contact_jacobian, contact_param, filter_body_pair, margin_and_gap, set_contact, tree_frames, tree_from_mjcf, tree_inverse_dynamics, tree_mass_matrix, CollideOptions, CollisionGeom, ContactRecord, GeomPose, GeomType, InvWeight, MjContact, MjcfJointKind, PairParams, SolImp, SolRef,
};
use nalgebra::{DVector, Vector3};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Default)]
struct OState {
    nefc: usize,
    ncontact: usize,
    ncon: usize,
    actuator: f64,
    passive: f64,
    dims: Vec<usize>,
    qpos: Vec<f64>,
    qfrc_bias: Vec<f64>,
    qacc_smooth: Vec<f64>,
    qacc: Vec<f64>,
}

#[derive(Default)]
struct OModel {
    rel: String,
    cone_pyramidal: bool,
    gravity: Vector3<f64>,
    joints: Vec<(String, String, usize)>,
    states: Vec<OState>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_forward <menagerie root> <forward oracle txt> [substring]");
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
            "model" => models.push(OModel {
                rel: t[1].to_string(),
                cone_pyramidal: t[4] == "0",
                gravity: Vector3::new(f(t[6]), f(t[7]), f(t[8])),
                ..Default::default()
            }),
            "joint" => models.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "state" => models.last_mut().unwrap().states.push(OState {
                nefc: t[2].parse().unwrap(),
                ncontact: t[3].parse().unwrap(),
                ncon: t[4].parse().unwrap(),
                actuator: f(t[5]),
                passive: f(t[6]),
                dims: if t[7] == "-" { Vec::new() } else { t[7].split(',').map(|x| x.parse().unwrap()).collect() },
                ..Default::default()
            }),
            "qpos" | "qfrc_bias" | "qacc_smooth" | "qacc" | "qvel" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qfrc_bias" => s.qfrc_bias = v,
                    "qacc_smooth" => s.qacc_smooth = v,
                    "qacc" => s.qacc = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let mut skip: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut states, mut smooth_ok, mut solved, mut solved_ok) = (0usize, 0usize, 0usize, 0usize);
    let (mut worst_bias, mut worst_smooth, mut worst_qacc) = (0.0f64, 0.0f64, 0.0f64);
    let mut notes: Vec<String> = Vec::new();
    for o in &models {
        if filter.as_ref().is_some_and(|fl| !o.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&o.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(_) => {
                *skip.entry("the loader refuses the model").or_default() += o.states.len();
                continue;
            }
        };
        // ⛔ Only a model whose joints line up one for one can have its `qacc` compared ENTRY BY ENTRY. A
        // free or ball joint is several joints here and one there, so the vectors are not the same vector.
        let lined_up = t.joints.len() == o.joints.len()
            && t.joints.iter().zip(&o.joints).all(|(a, b)| matches!((a.kind, b.1.as_str()), (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide")));
        if !lined_up {
            *skip.entry("a free or ball joint: the dof vectors do not correspond").or_default() += o.states.len();
            continue;
        }
        let qposadr: Vec<usize> = o.joints.iter().map(|j| j.2).collect();
        let (joints, parent, inertia) = (&t.tree.joints, &t.tree.parent, &t.tree.inertia);
        let nv = joints.len();
        // welds, as the contact sweep computes them; with hinges and slides only, a joint IS its body
        let weld_of = |j: Option<usize>| j.map(|k| k + 1).unwrap_or(0);
        let excludes: HashSet<(String, String)> = t.contact_excludes.iter().flat_map(|(a, b)| [(a.clone(), b.clone()), (b.clone(), a.clone())]).collect();
        let pairs: HashMap<(String, String), usize> = t.contact_pairs.iter().enumerate().flat_map(|(i, p)| [((p.geom1.clone(), p.geom2.clone()), i), ((p.geom2.clone(), p.geom1.clone()), i)]).collect();
        for s in &o.states {
            states += 1;
            if s.actuator != 0.0 {
                *skip.entry("an actuator is pushing at ctrl = 0").or_default() += 1;
                continue;
            }
            if s.passive != 0.0 {
                *skip.entry("a passive force (joint spring, damping, fluid)").or_default() += 1;
                continue;
            }
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else {
                *skip.entry("qpos does not map onto our coordinates").or_default() += 1;
                continue;
            };
            // --- the smooth half: mass matrix, bias, and the acceleration with no constraints
            let m = tree_mass_matrix(joints, inertia, parent, &q);
            let bias = tree_inverse_dynamics(joints, inertia, parent, &q, &vec![0.0; nv], &vec![0.0; nv], o.gravity);
            let db = bias.iter().zip(&s.qfrc_bias).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            worst_bias = worst_bias.max(db);
            let Some(minv) = m.clone().try_inverse() else {
                *skip.entry("singular mass matrix").or_default() += 1;
                continue;
            };
            let a0 = &minv * DVector::from_iterator(nv, bias.iter().map(|x| -x));
            let ds = (0..nv).map(|i| (a0[i] - s.qacc_smooth[i]).abs()).fold(0.0, f64::max);
            worst_smooth = worst_smooth.max(ds);
            if ds < 1e-6 {
                smooth_ok += 1;
            } else if notes.len() < 12 {
                notes.push(format!("{}: qacc_smooth off {ds:.2e} (bias off {db:.2e})", o.rel));
            }
            // --- the constrained half, only where MuJoCo's constraint set is one this port claims
            if s.nefc != s.ncontact {
                *skip.entry("a constraint row that is not a contact (dof friction, equality, limit)").or_default() += 1;
                continue;
            }
            if !o.cone_pyramidal {
                *skip.entry("an elliptic cone (the assembled solve here is pyramidal)").or_default() += 1;
                continue;
            }
            if s.dims.iter().any(|&d| d != 1 && d != 3) {
                *skip.entry("condim 4 or 6 in the assembled solve").or_default() += 1;
                continue;
            }
            let frames = tree_frames(&t.tree, &q);
            let mut contacts: Vec<MjContact> = Vec::new();
            let opts = CollideOptions::default();
            let geoms: Vec<Option<CollisionGeom>> = t
                .geoms
                .iter()
                .map(|g| {
                    let world = match g.joint {
                        Some(j) => frames[j] * g.pose,
                        None => g.pose,
                    };
                    let hull = match (&g.mesh, g.kind) {
                        (Some(mn), GeomType::Mesh) => Some(t.mesh_hulls.get(mn)?),
                        _ => None,
                    };
                    Some(CollisionGeom {
                        kind: g.kind,
                        pose: GeomPose { pos: world.translation.vector, mat: *world.rotation.to_rotation_matrix().matrix() },
                        size: g.size,
                        hull,
                    })
                })
                .collect();
            let mut refused = false;
            for i in 0..t.geoms.len() {
                for j in i + 1..t.geoms.len() {
                    let (gi, gj) = (&t.geoms[i], &t.geoms[j]);
                    let (Some(ci), Some(cj)) = (&geoms[i], &geoms[j]) else { continue };
                    if ci.kind == GeomType::HField || cj.kind == GeomType::HField {
                        refused = true;
                        continue;
                    }
                    let pair = pairs.get(&(gi.name.clone(), gj.name.clone())).map(|&k| &t.contact_pairs[k]);
                    let (margin, gap, params) = if let Some(p) = pair {
                        (p.margin, p.gap, PairParams { condim: p.condim, solref: p.solref, solimp: p.solimp, friction: p.friction, adhesion: p.adhesion })
                    } else {
                        if !can_collide(gi.params.contype, gi.params.conaffinity, gj.params.contype, gj.params.conaffinity) {
                            continue;
                        }
                        let (w1, w2) = (weld_of(gi.joint), weld_of(gj.joint));
                        let pw = |j: Option<usize>| j.map(|k| (parent[k] + 1) as usize).unwrap_or(0);
                        if filter_body_pair(w1, pw(gi.joint), usize::from(w1 != 0), w2, pw(gj.joint), usize::from(w2 != 0), true) {
                            continue;
                        }
                        if excludes.contains(&(gi.body.clone(), gj.body.clone())) {
                            continue;
                        }
                        let (m, g) = margin_and_gap(&gi.params, &gj.params);
                        (m, g, contact_param(&gi.params, &gj.params))
                    };
                    let Ok(pre) = collide_pair_with(&opts, margin + gap, ci, cj) else {
                        refused = true;
                        continue;
                    };
                    for p in pre {
                        let c: ContactRecord = set_contact(&p, &params, margin);
                        // a contact inside the gap band is detected but not handed to the solver
                        if c.exclude {
                            continue;
                        }
                        let point = c.pos;
                        let jac = contact_jacobian(joints, parent, &frames, gi.joint, gj.joint, point, &c.frame, 3);
                        let iw = |g: &ferromotion_core::MjcfGeom| match g.joint {
                            Some(k) => ferromotion_core::body_invweight(&minv, joints, parent, &frames, Some(k), ferromotion_core::body_com(&frames, inertia, k)),
                            None => InvWeight { tran: 0.0, rot: 0.0 },
                        };
                        contacts.push(MjContact {
                            jac,
                            dist: c.dist,
                            margin: c.includemargin,
                            condim: c.dim,
                            friction: [c.friction[0], c.friction[1]],
                            solref: SolRef(c.solref[0], c.solref[1]),
                            solimp: SolImp { d0: c.solimp[0], d_width: c.solimp[1], width: c.solimp[2], midpoint: c.solimp[3], power: c.solimp[4] },
                            invweight: [iw(gi), iw(gj)],
                        });
                    }
                }
            }
            if refused {
                *skip.entry("a geom pair this port refuses (height field)").or_default() += 1;
                continue;
            }
            if contacts.len() != s.ncon {
                *skip.entry("we and MuJoCo disagree on how many contacts there are").or_default() += 1;
                if notes.len() < 12 {
                    notes.push(format!("{}: {} contacts, MuJoCo has {}", o.rel, contacts.len(), s.ncon));
                }
                continue;
            }
            solved += 1;
            let qvel = DVector::zeros(nv);
            match ferromotion_core::solve_contacts_mujoco(&m, &a0, &qvel, &contacts, true, 1.0, 1e-12, 4000) {
                Ok(sol) => {
                    let dq = (0..nv).map(|i| (sol.qacc[i] - s.qacc[i]).abs()).fold(0.0, f64::max);
                    worst_qacc = worst_qacc.max(dq);
                    if dq < 1e-6 {
                        solved_ok += 1;
                    } else if notes.len() < 12 {
                        notes.push(format!("{}: qacc off {dq:.2e} with {} contacts", o.rel, contacts.len()));
                    }
                }
                Err(e) => {
                    *skip.entry("the assembled solve refused the rows").or_default() += 1;
                    if notes.len() < 12 {
                        notes.push(format!("{}: solve refused: {e}", o.rel));
                    }
                }
            }
        }
    }
    println!("states {states}");
    println!("  SMOOTH (mass matrix, bias, unconstrained acceleration, all from the file): {smooth_ok} within 1e-6; worst qfrc_bias {worst_bias:.2e}, worst qacc_smooth {worst_smooth:.2e}");
    println!("  CONSTRAINED (contacts found and solved by us): {solved_ok} of {solved} within 1e-6; worst qacc {worst_qacc:.2e}");
    println!("  not compared, by what blocks it:");
    let mut by: Vec<_> = skip.iter().collect();
    by.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    for (k, v) in by {
        println!("    {v:>5}  {k}");
    }
    for n in &notes {
        println!("  {n}");
    }
}
