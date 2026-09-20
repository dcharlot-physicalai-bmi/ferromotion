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
    can_collide, collide_pair_with, contact_jacobian, contact_param, filter_body_pair, margin_and_gap, set_contact, tree_frames, tree_from_mjcf, tree_inverse_dynamics, CollideOptions, CollisionGeom, ContactRecord, GeomPose, GeomType, InvWeight, MjcfJointKind, PairParams, SolImp, SolRef,
};
use nalgebra::{DVector, Vector3};
use std::collections::{BTreeMap, HashMap, HashSet};

/// One of MuJoCo's own contacts: which geoms, where, and — the part a normal-only comparison never sees —
/// the whole `frame`.
#[derive(Clone)]
struct MjCon {
    g1: usize,
    g2: usize,
    dist: f64,
    pos: Vector3<f64>,
    frame: [Vector3<f64>; 3],
    dim: usize,
    margin: f64,
    /// the rows MuJoCo actually built for this contact, in its own order
    d: Vec<f64>,
    aref: Vec<f64>,
}

#[derive(Default)]
struct OState {
    nefc: usize,
    ncontact: usize,
    ncon: usize,
    contacts: Vec<MjCon>,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    bodyacc: Vec<f64>,
    qfrc_bias: Vec<f64>,
    qfrc_passive: Vec<f64>,
    qacc_smooth: Vec<f64>,
    qacc: Vec<f64>,
}

#[derive(Default)]
struct OModel {
    rel: String,
    cone_pyramidal: bool,
    impratio: f64,
    gravity: Vector3<f64>,
    bodies: Vec<String>,
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
                impratio: f(t[5]),
                gravity: Vector3::new(f(t[6]), f(t[7]), f(t[8])),
                ..Default::default()
            }),
            "body" => models.last_mut().unwrap().bodies.push(t[1].to_string()),
            "joint" => models.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "state" => models.last_mut().unwrap().states.push(OState {
                nefc: t[2].parse().unwrap(),
                ncontact: t[3].parse().unwrap(),
                ncon: t[4].parse().unwrap(),
                ..Default::default()
            }),
            "contact" => {
                let v: Vec<f64> = t[4..16].iter().map(|x| f(x)).collect();
                let n: usize = t[25].parse().unwrap();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                s.contacts.push(MjCon {
                    g1: t[1].parse().unwrap(),
                    g2: t[2].parse().unwrap(),
                    dist: f(t[3]),
                    pos: Vector3::new(v[0], v[1], v[2]),
                    frame: [Vector3::new(v[3], v[4], v[5]), Vector3::new(v[6], v[7], v[8]), Vector3::new(v[9], v[10], v[11])],
                    dim: t[16].parse().unwrap(),
                    margin: f(t[17]),
                    d: t[26..26 + n].iter().map(|x| f(x)).collect(),
                    aref: t[26 + n..26 + 2 * n].iter().map(|x| f(x)).collect(),
                });
            }
            "qpos" | "qfrc_bias" | "qfrc_passive" | "qacc_smooth" | "qacc" | "qvel" | "bodyacc" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
                    "qfrc_bias" => s.qfrc_bias = v,
                    "bodyacc" => s.bodyacc = v,
                    "qfrc_passive" => s.qfrc_passive = v,
                    "qacc_smooth" => s.qacc_smooth = v,
                    "qacc" => s.qacc = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let mut skip: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut states, mut smooth_ok, mut smooth_tried, mut solved, mut solved_ok) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut acc_ok, mut acc_tried, mut worst_acc) = (0usize, 0usize, 0.0f64);
    let (mut chain_ok, mut chain_tried, mut worst_chain) = (0usize, 0usize, 0.0f64);
    let (mut solved_nomesh, mut solved_ok_nomesh, mut worst_qacc_nomesh) = (0usize, 0usize, 0.0f64);
    let mut solved_no_worse = 0usize;
    let (mut framed, mut frame_unpaired, mut tangent_differs) = (0usize, 0usize, 0usize);
    let (mut worst_con_normal, mut worst_con_normal_where) = (0.0f64, String::new());
    let (mut worst_con_tangent, mut worst_con_tangent_where) = (0.0f64, String::new());
    let (mut worst_con_d, mut worst_con_d_where, mut worst_con_aref, mut worst_con_aref_where) = (0.0f64, String::new(), 0.0f64, String::new());
    let (mut reversed_pair, mut con_rows_compared) = (0usize, 0usize);
    // `MJ_CONTACT=frame` rebuilds each contact Jacobian from MuJoCo's own frame, `=all` from its frame,
    // witness and depth as well: the difference between the three runs is the attribution.
    let mjsub = std::env::var("MJ_CONTACT").ok();
    let (mut worst_smooth_where, mut worst_qacc_where, mut worst_resid_where) = (String::new(), String::new(), String::new());
    let (mut worst_bias, mut worst_smooth, mut worst_qacc, mut worst_resid, mut worst_passive) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
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
        // ⛔ the joints must CORRESPOND for `qpos` to map at all; whether their coordinates are the same is
        // a separate question, and the answer is no for a free or ball joint — six of ours are three world
        // slides and three Euler hinges, six of MuJoCo's are a world translation and a body-frame angular
        // velocity. Those states are still checked, through the BODY ACCELERATIONS, which no basis owns.
        let kinds_match = t.joints.len() == o.joints.len()
            && t.joints.iter().zip(&o.joints).all(|(a, b)| {
                matches!(
                    (a.kind, b.1.as_str()),
                    (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide") | (MjcfJointKind::Ball, "ball") | (MjcfJointKind::Free, "free")
                )
            });
        if !kinds_match {
            *skip.entry("the joints do not correspond one for one").or_default() += o.states.len();
            continue;
        }
        let elementwise = t.joints.iter().all(|a| matches!(a.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide));
        // ⭐ Only the BASE's six coordinates are in a different basis. Every hinge and slide dof is the same
        // quantity on both sides — the joint angle's second derivative — and sits at the same index, because
        // the joints correspond one for one and a free joint occupies six slots in either convention. So the
        // chain of a legged robot can be compared entry by entry even when its base cannot.
        let shared: Vec<usize> = t.joints.iter().filter(|j| matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide)).map(|j| j.first).collect();
        // ⛔⛔ ARMATURE ON A FREE OR BALL JOINT IS BASIS-DEPENDENT. It is a constant added to the diagonal of
        // the mass matrix, and a diagonal is only a diagonal in one basis: MuJoCo's three body-frame angular
        // velocities and this tree's three Euler-rate hinges are different coordinates, so the same number
        // makes a different matrix. The translations agree; the rotations cannot. Ten Menagerie models do
        // this — every `rainbow_robotics_rby1` variant — and they are exactly the ten whose chain
        // accelerations disagree, at 1.2e-2. No amount of care in the recursion fixes it; only a base that
        // uses MuJoCo's own coordinates would.
        let rotational_armature = t.joints.iter().any(|j| matches!(j.kind, MjcfJointKind::Free | MjcfJointKind::Ball) && j.armature != 0.0);
        let qposadr: Vec<usize> = o.joints.iter().map(|j| j.2).collect();
        let (joints, parent, inertia) = (&t.tree.joints, &t.tree.parent, &t.tree.inertia);
        let nv = joints.len();
        // welds, as the contact sweep computes them; with hinges and slides only, a joint IS its body
        let weld_of = |j: Option<usize>| j.map(|k| k + 1).unwrap_or(0);
        let excludes: HashSet<(String, String)> = t.contact_excludes.iter().flat_map(|(a, b)| [(a.clone(), b.clone()), (b.clone(), a.clone())]).collect();
        let pairs: HashMap<(String, String), usize> = t.contact_pairs.iter().enumerate().flat_map(|(i, p)| [((p.geom1.clone(), p.geom2.clone()), i), ((p.geom2.clone(), p.geom1.clone()), i)]).collect();
        for s in &o.states {
            states += 1;
            if !t.actuators_unsupported.is_empty() {
                *skip.entry("an actuator this port does not carry (muscle, tendon, adhesion, plugin)").or_default() += 1;
                continue;
            }
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else {
                *skip.entry("qpos does not map onto our coordinates").or_default() += 1;
                continue;
            };
            // --- the smooth half: mass matrix, bias, and the acceleration with no constraints
            // ⛔ the tree's own mass matrix does not know that armature on a free or ball joint is a
            // diagonal in MUJOCO's basis; `MjcfTree::mass_matrix` maps it into this one
            let m = t.mass_matrix(&q);
            // ⛔⛔ the oracle's `qvel` is in MUJOCO's basis. On a free or ball base it is not this port's,
            // and feeding it straight in is wrong for every velocity-dependent term there is — the bias,
            // the damping, and every row's `J·q̇`. `T⁻¹` is the identity when there is no such joint.
            let Some(tinv) = t.free_basis(&q).try_inverse() else {
                *skip.entry("gimbal lock: the Euler base's basis map is singular").or_default() += 1;
                continue;
            };
            let qvel: Vec<f64> = (&tinv * DVector::from_row_slice(&s.qvel)).iter().copied().collect();
            let bias = tree_inverse_dynamics(joints, inertia, parent, &q, &qvel, &vec![0.0; nv], o.gravity);
            // ⛔ `qfrc_bias` is a vector in the DOF basis, so it can only be compared where the bases agree
            let db = bias.iter().zip(&s.qfrc_bias).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            if elementwise {
                worst_bias = worst_bias.max(db);
            }
            if std::env::var("DUMP_M").is_ok() {
                println!("M {} state:", o.rel);
                for i in 0..nv {
                    println!("  {:?}", (0..nv).map(|j| m[(i, j)]).collect::<Vec<_>>());
                }
            }
            // the oracle samples with `ctrl = 0`; a position servo still pushes, because its bias is the
            // servo law and the model is not at its setpoint
            // ⛔ compare `qfrc_passive` ITSELF before letting it into the acceleration. This port carries
            // joint damping, joint springs and gravcomp and nothing else — no tendon or fluid force — so
            // a model that uses one of those must be counted as not carried, not reported as a wrong answer.
            // ⛔ in MUJOCO's basis: a generalised force maps by T⁻ᵀ, and a free joint's DAMPING is a per-dof
            // force in MuJoCo's coordinates that does not map at all. With no free or ball joint this is the
            // ordinary `qfrc_passive`.
            let Some(passive) = t.qfrc_passive_mujoco(&q, &s.qvel) else {
                *skip.entry("gimbal lock: the Euler base's basis map is singular").or_default() += 1;
                continue;
            };
            let dp = (0..nv).map(|i| (passive[i] - s.qfrc_passive[i]).abs()).fold(0.0, f64::max);
            if dp > 1e-9 * (0..nv).map(|i| s.qfrc_passive[i].abs()).fold(1.0, f64::max) {
                *skip.entry("a passive force this port does not carry (tendon, fluid)").or_default() += 1;
                worst_passive = worst_passive.max(dp);
                continue;
            }
            let act = t.qfrc_actuator(&q, &qvel, &vec![0.0; t.actuators.len()]);
            let Some(minv) = m.clone().try_inverse() else {
                *skip.entry("singular mass matrix").or_default() += 1;
                continue;
            };
            let a0 = &minv * DVector::from_iterator(nv, (0..nv).map(|i| passive[i] + act[i] - bias[i]));
            let frames_for_acc = tree_frames(&t.tree, &q);
            // ⛔ RELATIVE, against the magnitude of the answer. A joint with a stiff servo accelerates at
            // hundreds of rad/s², so an absolute 1e-6 is a tolerance on the eighth significant digit for one
            // model and on the second for another — the same number meaning two different claims.
            let rel = |a: &[f64], b: &[f64]| (0..a.len()).map(|i| (a[i] - b[i]).abs() / b[i].abs().max(1.0)).fold(0.0, f64::max);
            // ⭐ the basis-free check: every body's PROPER acceleration in the world frame, which is what an
            // accelerometer on it would read. Valid only at rest — at speed the velocity products enter and
            // `J·q̈` is no longer the whole story — and that is stated rather than hidden, because a resting
            // state cannot see a velocity term at all.
            // ⛔ and only where MuJoCo has NO constraint rows: its `bodyacc` is taken at the CONSTRAINED
            // acceleration, so comparing it against our unconstrained `a0` would be comparing two different
            // questions — a robot resting on the floor is not accelerating downward at 9.81.
            if s.qvel.iter().all(|v| *v == 0.0) && s.nefc == 0 && !s.bodyacc.is_empty() && !rotational_armature {
                let mut worst = 0.0f64;
                let mut worst_body = String::new();
                for (b, name) in o.bodies.iter().enumerate() {
                    // ⛔ a body WELDED TO THE WORLD is reported as exactly zero by MuJoCo, not as the
                    // +9.81 its own recursion would give — the static bodies are outside the acceleration
                    // tree. Comparing them would assert a convention, not a dynamic.
                    let Some((j, iso)) = t.body_frames.get(name) else { continue };
                    let (jnt, iso) = (Some(*j), *iso);
                    let world = match jnt {
                        Some(j) => frames_for_acc[j] * iso,
                        None => iso,
                    };
                    let p = world.translation.vector;
                    let jl = ferromotion_core::tree_point_jacobian(joints, parent, &frames_for_acc, jnt, p);
                    let ja = ferromotion_core::tree_angular_jacobian(joints, parent, &frames_for_acc, jnt);
                    // MuJoCo reports PROPER acceleration: the world's own frame carries −gravity, so a body
                    // bolted to it reads +9.81 upward rather than zero
                    let lin = &jl * &a0 - o.gravity;
                    let ang = &ja * &a0;
                    for k in 0..3 {
                        for (mine, theirs, what) in [(ang[k], s.bodyacc[6 * b + k], "ang"), (lin[k], s.bodyacc[6 * b + 3 + k], "lin")] {
                            let e = (mine - theirs).abs() / theirs.abs().max(1.0);
                            if e > worst {
                                worst = e;
                                worst_body = format!("{name} {what}[{k}] {mine} vs {theirs}");
                            }
                        }
                    }
                    if std::env::var("DUMP_ACC").is_ok() && b < 4 {
                        println!("    ours ang {:?} lin {:?}", ang.as_slice(), lin.as_slice());
                    }
                }
                if std::env::var("DUMP_ACC").is_ok() {
                    println!("ACC {} state:", o.rel);
                    for (b, name) in o.bodies.iter().enumerate().take(4) {
                        println!("  {name}: MuJoCo {:?}", &s.bodyacc[6 * b..6 * b + 6]);
                    }
                }
                acc_tried += 1;
                if worst < 1e-6 {
                    acc_ok += 1;
                } else if notes.len() < 12 {
                    notes.push(format!("{}: body acceleration off {worst:.2e}{} — {worst_body}", o.rel, if elementwise { "" } else { " (free/ball base)" }));
                }
                worst_acc = worst_acc.max(worst);
            }
            if !elementwise {
                // ⛔ and only AT REST. The oracle's `qvel` is in MuJoCo's basis, so feeding it to this tree
                // as if it were ours is wrong for the base's six — and at speed those six feed Coriolis
                // terms into every chain dof. A resting state has no such coupling to get wrong.
                let _ = (rotational_armature, &shared);
                // ⭐⭐ the WHOLE acceleration, in MuJoCo's own coordinates — all six of the base's dofs
                // included, at rest and moving. `v_mujoco = T·v_ours`, so `a_mujoco = T·a_ours + Ṫ·v_ours`,
                // and armature and damping on the base go where MuJoCo puts them: on ITS diagonal.
                let Some(a_mj) = t.qacc_smooth_mujoco(&q, &s.qvel, &vec![0.0; t.actuators.len()]) else {
                    *skip.entry("gimbal lock: the Euler base's basis map is singular").or_default() += 1;
                    continue;
                };
                let dq = (0..nv).map(|i| (a_mj[i] - s.qacc_smooth[i]).abs() / s.qacc_smooth[i].abs().max(1.0)).fold(0.0, f64::max);
                chain_tried += 1;
                if dq < 1e-6 {
                    chain_ok += 1;
                } else if notes.len() < 12 {
                    notes.push(format!("{}: floating qacc_smooth off {dq:.2e} in MuJoCo's basis", o.rel));
                }
                worst_chain = worst_chain.max(dq);
            }
            // ⛔ SMOOTH counts the models whose dofs correspond one for one; a free or ball base is counted
            // on the FLOATING BASE line instead, against the same oracle in MuJoCo's coordinates. Counting
            // it here as well would report one comparison twice and, worse, compare `a0` — which is in THIS
            // port's basis — against an answer in MuJoCo's.
            let ds = if elementwise { rel(a0.as_slice(), &s.qacc_smooth) } else { 0.0 };
            // ⭐ separate "our M is wrong" from "inverting M lost digits": put MuJoCo's own answer back
            // through OUR mass matrix. A residual near zero means the matrix agrees and only the solve drifted
            let resid = if !elementwise {
                0.0
            } else {
                let r = &m * DVector::from_row_slice(&s.qacc_smooth) - DVector::from_iterator(nv, (0..nv).map(|i| passive[i] + act[i] - bias[i]));
                let scale = (0..nv).map(|i| (passive[i] + act[i] - bias[i]).abs()).fold(1.0, f64::max);
                r.amax() / scale
            };
            if resid > worst_resid {
                worst_resid = resid;
                worst_resid_where = o.rel.clone();
            }
            if elementwise {
                smooth_tried += 1;
            }
            if ds > worst_smooth {
                worst_smooth = ds;
                worst_smooth_where = o.rel.clone();
            }
            if ds < 1e-6 && elementwise {
                smooth_ok += 1;
            } else if notes.len() < 12 {
                notes.push(format!("{}: qacc_smooth off {ds:.2e} (bias off {db:.2e})", o.rel));
            }
            // --- the constrained half, only where MuJoCo's constraint set is one this port claims
            // ⛔ an equality this port cannot build is a MISSING ROW, not a small error: the solve would be
            // unconstrained where MuJoCo holds a linkage together
            if !t.equalities_unsupported.is_empty() {
                *skip.entry("an equality this port does not build (connect, weld, tendon)").or_default() += 1;
                continue;
            }
            if s.nefc != s.ncontact + t.equalities.len() + s.nefc.saturating_sub(s.ncontact) && false {
                continue;
            }
            let frames = tree_frames(&t.tree, &q);
            let biw = t.body_invweight0();
            let mut contacts: Vec<ferromotion_core::ContactSpec> = Vec::new();
            // which geoms each of our contacts came from, and the frame we gave it — so the tangent pair can
            // be held against MuJoCo's, which a normal-only comparison cannot see
            let mut ourcon: Vec<(usize, usize, Vector3<f64>, [Vector3<f64>; 3], f64)> = Vec::new();
            // ⭐ whether any contact in this state involves a MESH. The witness-point gap measured by
            // `menagerie_contacts` lives only there, so splitting the failures this way turns "the contacts
            // must be it" from an assertion into a count.
            let mut any_mesh = false;
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
            // ⭐ why a pair is NOT here: every gate records the pair it dropped, so a missing contact names
            // the rule that removed it instead of leaving a count to be stared at
            let mut why: BTreeMap<(usize, usize), String> = BTreeMap::new();
            for i in 0..t.geoms.len() {
                for j in i + 1..t.geoms.len() {
                    let (gi, gj) = (&t.geoms[i], &t.geoms[j]);
                    let (Some(ci), Some(cj)) = (&geoms[i], &geoms[j]) else { continue };
                    if ci.kind == GeomType::HField || cj.kind == GeomType::HField {
                        refused = true;
                        continue;
                    }
                    let key = (i.min(j), i.max(j));
                    let pair = pairs.get(&(gi.name.clone(), gj.name.clone())).map(|&k| &t.contact_pairs[k]);
                    let (margin, gap, params) = if let Some(p) = pair {
                        (p.margin, p.gap, PairParams { condim: p.condim, solref: p.solref, solimp: p.solimp, friction: p.friction, adhesion: p.adhesion })
                    } else {
                        if !can_collide(gi.params.contype, gi.params.conaffinity, gj.params.contype, gj.params.conaffinity) {
                            why.insert(key, "contype/conaffinity".into());
                            continue;
                        }
                        let (w1, w2) = (weld_of(gi.joint), weld_of(gj.joint));
                        let pw = |j: Option<usize>| j.map(|k| (parent[k] + 1) as usize).unwrap_or(0);
                        if filter_body_pair(w1, pw(gi.joint), usize::from(w1 != 0), w2, pw(gj.joint), usize::from(w2 != 0), true) {
                            why.insert(key, "the body filter (same weld, both static, or parent and child)".into());
                            continue;
                        }
                        if excludes.contains(&(gi.body.clone(), gj.body.clone())) {
                            why.insert(key, "<contact><exclude>".into());
                            continue;
                        }
                        let (m, g) = margin_and_gap(&gi.params, &gj.params);
                        (m, g, contact_param(&gi.params, &gj.params))
                    };
                    let Ok(pre) = collide_pair_with(&opts, margin + gap, ci, cj) else {
                        refused = true;
                        why.insert(key, "a pair the collider refuses".into());
                        continue;
                    };
                    if pre.is_empty() {
                        why.insert(key, "no contact: the collider found them apart".into());
                    } else {
                        why.insert(key, format!("the collider found {} witness(es) at dist {:?}, includemargin {margin}", pre.len(), pre.iter().map(|p| p.dist).collect::<Vec<_>>()));
                    }
                    for p in pre {
                        let c: ContactRecord = set_contact(&p, &params, margin);
                        // a contact inside the gap band is detected but not handed to the solver
                        if c.exclude {
                            continue;
                        }
                        let point = c.pos;
                        let jac = contact_jacobian(joints, parent, &frames, gi.joint, gj.joint, point, &c.frame, c.dim);
                        // ⛔ at `qpos0`, NOT here: MuJoCo computes `body_invweight0` when it compiles the
                        // model and never recomputes it, so a contact's regularisation carries the reference
                        // pose's inverse weights however far the model has moved since.
                        let iw = |g: &ferromotion_core::MjcfGeom| biw.get(&g.body).copied().unwrap_or(InvWeight::STATIC);
                        if ci.kind == GeomType::Mesh || cj.kind == GeomType::Mesh {
                            any_mesh = true;
                        }
                        ourcon.push((i, j, c.pos, c.frame, c.dist));
                        contacts.push(ferromotion_core::ContactSpec {
                            jac,
                            dist: c.dist,
                            margin: c.includemargin,
                            condim: c.dim,
                            friction: c.friction,
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
            // ⛔ `d.ncon` counts contacts MuJoCo DETECTED, and a contact at `dist >= includemargin` is in
            // that list with `efc_address = -1` and no rows at all. This port drops those, so the number to
            // compare against is the number MuJoCo gave rows to — otherwise a model where the two fingers
            // touch at exactly `dist = 0` reads as sixteen missing contacts when nothing is missing.
            let theirs_with_rows: Vec<&MjCon> = s.contacts.iter().filter(|c| !c.d.is_empty()).collect();
            if contacts.len() != theirs_with_rows.len() {
                *skip.entry("we and MuJoCo disagree on how many contacts there are").or_default() += 1;
                if notes.len() < 12 {
                    notes.push(format!("{}: {} contacts, MuJoCo has {} with rows ({} detected)", o.rel, contacts.len(), theirs_with_rows.len(), s.ncon));
                }
                // ⭐ WHICH pairs, not how many: a count says a collision pipeline disagrees, a pair says
                // where. MuJoCo's own geom ids are in the oracle, and our geom order is MuJoCo's.
                if std::env::var("DUMP_PAIRS").is_ok() {
                    let mut mine: BTreeMap<(usize, usize), i32> = BTreeMap::new();
                    for c in &ourcon {
                        *mine.entry((c.0.min(c.1), c.0.max(c.1))).or_default() += 1;
                    }
                    for c in &theirs_with_rows {
                        *mine.entry((c.g1.min(c.g2), c.g1.max(c.g2))).or_default() -= 1;
                    }
                    for ((a, b), n) in mine.iter().filter(|(_, n)| **n != 0) {
                        let note = why.get(&(*a, *b)).cloned().unwrap_or_else(|| "no record".into());
                        let who = if *n > 0 { format!("only ours — {note}") } else { format!("only MuJoCo's — {note}") };
                        println!("  {}: geom{a} {} + geom{b} {} {} x{}", o.rel, t.geoms[*a].name, t.geoms[*b].name, who, n.abs());
                    }
                }
                continue;
            }
            // ⛔ MuJoCo's contact frame is THREE vectors, and the contact sweep only ever compared the first.
            // Under a pyramidal cone the friction rows are `J_n ± μ_k·J_tk`, so the tangent pair IS part of
            // the law: rotate it about the normal and the pyramid rotates with it. Two ports can agree on
            // every contact position, depth and normal and still be solving different problems.
            let key = |a: usize, b: usize| (a.min(b), a.max(b));
            let mut theirs_by_pair: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for (k, c) in theirs_with_rows.iter().enumerate() {
                theirs_by_pair.entry(key(c.g1, c.g2)).or_default().push(k);
            }
            let mut ours_by_pair: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for (k, c) in ourcon.iter().enumerate() {
                ours_by_pair.entry(key(c.0, c.1)).or_default().push(k);
            }
            let paired: Option<Vec<(usize, usize)>> = (ours_by_pair.len() == theirs_by_pair.len()
                && ours_by_pair.iter().all(|(k, v)| theirs_by_pair.get(k).is_some_and(|w| w.len() == v.len())))
            .then(|| {
                // ⛔ within one geom pair, match by WITNESS POSITION, not by the order the two colliders
                // happened to emit them. On `stanford_tidybot` the same pad pair produces two contacts and
                // the two orders are opposite, which read as a factor-two error in `efc_aref` until the
                // rows were lined up by where the contacts actually are.
                let mut v: Vec<(usize, usize)> = Vec::new();
                for (k, mine) in &ours_by_pair {
                    // ⛔ BEST FIRST, not in our emission order: with three contacts on one geom pair, taking
                    // each of ours in turn and giving it its nearest free partner can chain into a
                    // permutation, and then every row reads as wrong when only the pairing was. On
                    // `google_barkour_v0` that alone reported `efc_D` 4.4e-1 out with the two sets equal.
                    let (mut free_a, mut free_b): (Vec<usize>, Vec<usize>) = (mine.clone(), theirs_by_pair[k].clone());
                    while !free_a.is_empty() && !free_b.is_empty() {
                        let (mut bi, mut bj, mut best) = (0usize, 0usize, f64::INFINITY);
                        for (i, &a) in free_a.iter().enumerate() {
                            for (j, &b) in free_b.iter().enumerate() {
                                let d = (theirs_with_rows[b].pos - ourcon[a].2).norm();
                                if d < best {
                                    (bi, bj, best) = (i, j, d);
                                }
                            }
                        }
                        v.push((free_a.remove(bi), free_b.remove(bj)));
                    }
                }
                v
            });
            match &paired {
                Some(pairs) => {
                    framed += pairs.len();
                    for &(a, b) in pairs {
                        let (mine, theirs) = (&ourcon[a], theirs_with_rows[b]);
                        // MuJoCo may order the pair the other way round, and then its normal is ours negated
                        let flip = if mine.0 == theirs.g1 { 1.0 } else { -1.0 };
                        let dn = (mine.3[0] * flip - theirs.frame[0]).norm();
                        if dn > worst_con_normal {
                            worst_con_normal = dn;
                            worst_con_normal_where = format!("{} geom{}+geom{} (dist {:.3e} vs {:.3e}, witness {:.3e} m apart)", o.rel, theirs.g1, theirs.g2, mine.4, theirs.dist, (mine.2 - theirs.pos).norm());
                        }
                        // the tangent pair spans a plane; what matters is how far it is ROTATED within it
                        let dt = 1.0 - mine.3[1].dot(&theirs.frame[1]).abs();
                        if dt > 1e-9 {
                            tangent_differs += 1;
                        }
                        if dt > worst_con_tangent {
                            worst_con_tangent = dt;
                            worst_con_tangent_where = o.rel.clone();
                        }
                    }
                }
                None => frame_unpaired += 1,
            }
            if let (Some(pairs), Some(mode)) = (&paired, mjsub.as_deref()) {
                for &(a, b) in pairs {
                    let th = theirs_with_rows[b];
                    let dim = contacts[a].condim;
                    let pos = if mode == "all" { th.pos } else { ourcon[a].2 };
                    contacts[a].jac = contact_jacobian(joints, parent, &frames, t.geoms[th.g1].joint, t.geoms[th.g2].joint, pos, &th.frame, dim);
                    if mode == "all" {
                        contacts[a].dist = th.dist;
                        contacts[a].margin = th.margin;
                    }
                }
            }
            solved += 1;
            // the whole row set, in MuJoCo's order: equality, friction, limits, then contacts
            let mut set = t.joint_constraint_rows(&q, &qvel, &t.dof_invweight0());
            let cone = if o.cone_pyramidal { ferromotion_core::Cone::Pyramidal } else { ferromotion_core::Cone::Elliptic };
            match ferromotion_core::contact_rows(&contacts, nv, &qvel, cone, o.impratio, t.timestep) {
                Ok(cr) => {
                    // ⭐ the contact rows themselves, against MuJoCo's own — `efc_D` is where the inverse
                    // weights, the impedance and the cone adjustment all land, and `efc_aref` is where the
                    // reference acceleration does. With the solver and the joint rows already pinned, a
                    // disagreement in the end-to-end acceleration has to show up in one of these two.
                    if let Some(pairs) = &paired {
                        let mut start = vec![0usize; cr.blocks.len() + 1];
                        for (k, b) in cr.blocks.iter().enumerate() {
                            start[k + 1] = start[k] + b.rows();
                        }
                        for &(a, b) in pairs {
                            let th = theirs_with_rows[b];
                            let (lo, hi) = (start[a], start[a + 1]);
                            if th.d.len() != hi - lo {
                                continue;
                            }
                            if std::env::var("DUMP_CON").is_ok() {
                                println!(
                                    "  {} c{a} geoms {}({:?})+{}({:?}) dim {} dist {:.6e} vs {:.6e}\n      invweight {:?} {:?}\n      D    {:?}\n      D mj {:?}\n      aref    {:?}\n      aref mj {:?}",
                                    o.rel, t.geoms[th.g1].name, t.geoms[th.g1].joint, t.geoms[th.g2].name, t.geoms[th.g2].joint, th.dim, ourcon[a].4, th.dist,
                                    contacts[a].invweight[0], contacts[a].invweight[1],
                                    &cr.d[lo..hi], th.d, &cr.aref[lo..hi], th.aref
                                );
                            }
            // ⛔ only where MuJoCo wrote the geom pair in OUR order. Written the other way round its normal
                            // is ours negated and so is its Jacobian difference — the two cancel on the normal
                            // row, but a tangent row comes out negated, so its rows are the same constraint
                            // written differently. Comparing those position by position reports a difference
                            // that is not one, and comparing them as a multiset hides a sign error that would
                            // be one. So they are counted and left out.
                            if ourcon[a].0 != th.g1 {
                                reversed_pair += 1;
                                continue;
                            }
                            con_rows_compared += 1;
                            let sorted = |v: &[f64]| v.to_vec();
                            for (k, (&mine, &dd)) in sorted(&cr.d[lo..hi]).iter().zip(&sorted(&th.d)).enumerate() {
                                let e = (mine - dd).abs() / dd.abs().max(1.0);
                                if e > worst_con_d {
                                    worst_con_d = e;
                                    worst_con_d_where = format!("{} row {k} of a dim-{} contact: {mine:.6e} vs {dd:.6e}", o.rel, th.dim);
                                }
                            }
                            for (k, (&mine, &aa)) in sorted(&cr.aref[lo..hi]).iter().zip(&sorted(&th.aref)).enumerate() {
                                let e = (mine - aa).abs() / aa.abs().max(1.0);
                                if e > worst_con_aref {
                                    worst_con_aref = e;
                                    worst_con_aref_where = format!("{} row {k} of a dim-{} contact: {mine:.6e} vs {aa:.6e}", o.rel, th.dim);
                                }
                            }
                        }
                    }
                    set.append(cr)
                }
                Err(e) => {
                    *skip.entry("a contact row this port cannot build").or_default() += 1;
                    if notes.len() < 12 {
                        notes.push(format!("{}: {e}", o.rel));
                    }
                    continue;
                }
            }
            if set.blocks.iter().map(|b| b.rows()).sum::<usize>() != s.nefc {
                *skip.entry("we and MuJoCo disagree on how many constraint ROWS there are").or_default() += 1;
                continue;
            }
            // ⛔ the solve happens in MUJOCO'S coordinates, because that is where its answer lives. A row's
            // Jacobian maps by `J·T⁻¹`, the mass matrix by `T⁻ᵀ·M·T⁻¹`, and `aref` and `D` do not move at
            // all — `J·q̇` is the same number in either basis. With no free or ball joint `T` is the identity
            // and this is exactly the computation it was before.
            let jac = &set.jac * &tinv;
            let m_mj = tinv.transpose() * &m * &tinv;
            let Some(a0_mj) = t.qacc_smooth_mujoco(&q, &s.qvel, &vec![0.0; t.actuators.len()]).map(DVector::from_vec) else {
                *skip.entry("gimbal lock: the Euler base's basis map is singular").or_default() += 1;
                continue;
            };
            match ferromotion_core::solve_constraints_newton_blocks(&m_mj, &a0_mj, &jac, &set.aref, &set.d, &set.blocks, 1e-13, 200) {
                Ok(sol) => {
                    let dq = (0..nv).map(|i| (sol.qacc[i] - s.qacc[i]).abs() / s.qacc[i].abs().max(1.0)).fold(0.0, f64::max);
                    if !any_mesh {
                        solved_nomesh += 1;
                        worst_qacc_nomesh = worst_qacc_nomesh.max(dq);
                    }
                    if dq > worst_qacc {
                        worst_qacc = dq;
                        worst_qacc_where = format!("{} ({} contacts)", o.rel, contacts.len());
                    }
                    // ⛔⛔ "our cost is no worse" is EVIDENCE OF NOTHING when the rows are our own. The
                    // solve converged, so of course our answer minimises our problem; a wrong contact set
                    // would pass this test every time. It is only evidence where the rows are known to be
                    // MuJoCo's, which here means a state with NO CONTACTS — those rows are the joint rows
                    // `examples/menagerie_rows` checks against MuJoCo, 236 of 236. `umi_gripper`, whose
                    // eight dof-friction rows all sit at their bounds, is that case: run on MuJoCo's OWN
                    // rows by `examples/menagerie_efc` it lands 3.2e-3 away at a cost no worse, so the
                    // difference is MuJoCo's iteration limit and not this port.
                    if dq >= 1e-6 && contacts.is_empty() {
                        let cost_at = |a: &DVector<f64>| {
                            let jar: Vec<f64> = (&jac * a - DVector::from_row_slice(&set.aref)).iter().copied().collect();
                            let u = ferromotion_core::mujoco_constraint_update_blocks(&set.blocks, &set.d, &jar);
                            let da = a - &a0_mj;
                            0.5 * (da.transpose() * &m_mj * &da)[(0, 0)] + u.cost
                        };
                        if cost_at(&sol.qacc) <= cost_at(&DVector::from_row_slice(&s.qacc)) {
                            solved_no_worse += 1;
                        }
                    }
                    if dq < 1e-6 {
                        solved_ok += 1;
                        if !any_mesh {
                            solved_ok_nomesh += 1;
                        }
                    } else if notes.len() < 12 || !any_mesh && notes.len() < 40 {
                        // ⭐ is the constrained answer wrong, or was the UNCONSTRAINED one already? The
                        // smooth comparison above only runs where MuJoCo has no rows at all, so a model
                        // that always has a limit or an equality never reaches it, and a wrong mass matrix
                        // or actuator force there reads as a constraint-solver failure.
                        let smooth = rel(a0_mj.as_slice(), &s.qacc_smooth);
                        notes.push(format!(
                            "{}: qacc off {dq:.2e} with {} contacts{}, {} rows, gradient {:.2e}; unconstrained already off {smooth:.2e}",
                            o.rel,
                            contacts.len(),
                            if any_mesh { " (mesh)" } else { " (NO mesh)" },
                            set.blocks.iter().map(|b| b.rows()).sum::<usize>(),
                            sol.grad_norm
                        ));
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
    println!("  SMOOTH (mass matrix, bias, actuators, unconstrained acceleration — all from the file): {smooth_ok} of {smooth_tried} within 1e-6; worst qfrc_bias {worst_bias:.2e}, worst qacc_smooth (relative) {worst_smooth:.2e} on {worst_smooth_where}");
    println!("    and MuJoCo's own qacc_smooth back through OUR mass matrix: worst residual {worst_resid:.2e} on {worst_resid_where}");
    println!("  CONSTRAINED (contacts found and solved by us): {solved_ok} of {solved} within 1e-6 ({solved_no_worse} more have NO contacts — rows this port checks separately — and reach a cost no worse than MuJoCo's own answer); worst qacc (relative) {worst_qacc:.2e} on {worst_qacc_where}");
    println!("    of those, with NO mesh geom in any contact: {solved_ok_nomesh} of {solved_nomesh}; worst qacc (relative) {worst_qacc_nomesh:.2e}");
    println!("  CONTACT FRAMES (the tangent pair, not only the normal): {framed} contacts paired with MuJoCo's ({frame_unpaired} states could not be paired by geom)");
    println!("    worst normal {worst_con_normal:.2e} on {worst_con_normal_where}; tangent pair rotated on {tangent_differs} of them, worst 1-|t·t'| {worst_con_tangent:.2e} on {worst_con_tangent_where}");
    println!("    worst efc_D (relative) {worst_con_d:.2e} on {worst_con_d_where} ({con_rows_compared} contacts compared row by row; {reversed_pair} left out because MuJoCo wrote the geom pair the other way round)");
    println!("    worst efc_aref (relative) {worst_con_aref:.2e} on {worst_con_aref_where}");
    println!("    worst qfrc_passive on the states it blocked: {worst_passive:.2e}");
    println!("  BODY ACCELERATIONS in the world frame, at rest — basis-free, so a free or ball base is included: {acc_ok} of {acc_tried} within 1e-6; worst {worst_acc:.2e}");
    println!("  FLOATING BASE, every dof in MUJOCO'S coordinates (at rest AND moving): {chain_ok} of {chain_tried} within 1e-6; worst qacc_smooth {worst_chain:.2e}");
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
