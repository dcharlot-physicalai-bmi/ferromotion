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
    contact_jacobian, tree_frames, tree_from_mjcf, tree_inverse_dynamics, GeomType, MjcfJointKind,
};
use nalgebra::{DVector, Vector3};
use std::collections::BTreeMap;

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
    /// the four parts MuJoCo keeps `qfrc_passive` in — a total cannot say which one a port is missing
    qfrc_spring: Vec<f64>,
    qfrc_damper: Vec<f64>,
    qfrc_gravcomp: Vec<f64>,
    qfrc_fluid: Vec<f64>,
    qacc_smooth: Vec<f64>,
    qacc: Vec<f64>,
    /// ⭐ the SAME rows solved to convergence. Equal to `qacc` bit for bit on a model that does not cap
    /// the solver; on an `*_mjx.xml` it is the answer `qacc` was on its way to when MuJoCo stopped.
    qacc_converged: Vec<f64>,
}

#[derive(Default)]
struct OModel {
    rel: String,
    cone_pyramidal: bool,
    impratio: f64,
    gravity: Vector3<f64>,
    /// name, then MuJoCo's compiled `body_invweight0` (translational, rotational), then the inertia it
    /// actually STORES: mass, the three principal values, the quaternion they are diagonal in, and `ipos`
    bodies: Vec<(String, f64, f64, Vec<f64>)>,
    joints: Vec<(String, String, usize)>,
    /// ⛔⛔ `option/iterations` and `option/ls_iterations` are PART OF THE PROBLEM STATEMENT. 29 of
    /// Menagerie's 210 compilable models cap them — every `*_mjx.xml` at 1 or 5 Newton iterations — and
    /// MuJoCo's `qacc` there is one iterate from a zero warm start, 18-74% away from the optimum of the
    /// very rows it built. Comparing a converged port against it reports a physics error that is not one.
    iterations: usize,
    ls_iterations: usize,
    states: Vec<OState>,
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
                iterations: t.get(9).and_then(|x| x.parse().ok()).unwrap_or(usize::MAX),
                ls_iterations: t.get(10).and_then(|x| x.parse().ok()).unwrap_or(usize::MAX),
                ..Default::default()
            }),
            "body" => models.last_mut().unwrap().bodies.push((
                t[1].to_string(),
                t.get(2).map(|x| f(x)).unwrap_or(f64::NAN),
                t.get(3).map(|x| f(x)).unwrap_or(f64::NAN),
                t.get(4..16).map(|v| v.iter().map(|x| f(x)).collect()).unwrap_or_default(),
            )),
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
            "qpos" | "qfrc_bias" | "qfrc_passive" | "qfrc_spring" | "qfrc_damper" | "qfrc_gravcomp" | "qfrc_fluid" | "qacc_smooth" | "qacc" | "qacc_converged" | "qvel" | "bodyacc" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let s = models.last_mut().unwrap().states.last_mut().unwrap();
                match t[0] {
                    "qpos" => s.qpos = v,
                    "qvel" => s.qvel = v,
                    "qfrc_bias" => s.qfrc_bias = v,
                    "bodyacc" => s.bodyacc = v,
                    "qfrc_passive" => s.qfrc_passive = v,
                    "qfrc_spring" => s.qfrc_spring = v,
                    "qfrc_damper" => s.qfrc_damper = v,
                    "qfrc_gravcomp" => s.qfrc_gravcomp = v,
                    "qfrc_fluid" => s.qfrc_fluid = v,
                    "qacc_smooth" => s.qacc_smooth = v,
                    "qacc" => s.qacc = v,
                    "qacc_converged" => s.qacc_converged = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // ⭐ how often MuJoCo itself stopped short of the optimum of its own rows, and by how much
    let (mut trunc_states, mut worst_trunc, mut worst_trunc_where) = (0usize, 0.0f64, String::new());
    let (mut iw0_ok, mut iw0_tried, mut worst_iw0, mut worst_iw0_where) = (0usize, 0usize, 0.0f64, String::new());
    let (mut inertia_ok, mut inertia_tried, mut worst_inertia, mut worst_inertia_where) = (0usize, 0usize, 0.0f64, String::new());
    let (mut inertia_bits, mut inertia_bits_notes) = (0usize, Vec::<String>::new());
    let (mut inertia_permuted, mut inertia_at_eig3_floor, mut inertia_above_floor) = (0usize, 0usize, 0usize);
    // ⛔ how CLOSE the worst case comes to the bound. A bound nothing approaches is not a bound, it is a
    // number chosen to cover the data — this one is reached to within a factor of a few, or it is wrong.
    let mut tightest = 0.0f64;
    // ⛔⛔ the DENOMINATOR, counted from the oracle rather than from whatever reached the gate. A gate
    // whose denominator cannot be derived from the corpus is selecting its own subjects: this one used to
    // sit inside the per-state loop and reported "11,031 of 11,031" while silently skipping every model
    // an earlier check refused — including the one furthest off.
    fn named(o: &OModel) -> impl Iterator<Item = &(String, f64, f64, Vec<f64>)> {
        o.bodies.iter().filter(|(n, ..)| !n.is_empty() && n != "world")
    }
    let bodies_in_corpus: usize = models.iter().map(|o| named(o).count()).sum();
    // and the two things that legitimately remove a body from a gate, counted from the ORACLE so the
    // accounting closes without trusting the port: welded to the world (no dofs, so no inverse weight),
    // and belonging to a model this loader refuses
    let dofless: usize = models.iter().map(|o| named(o).filter(|(.., i)| i.len() == 12 && i[11] != 0.0).count()).sum();
    let mut inertia_notes: Vec<String> = Vec::new();
    let mut refused_bodies = 0usize;
    let (mut worst_ipos, mut worst_ipos_where) = (0.0f64, String::new());
    let (mut passive_gravcomp_only, mut worst_passive_gravcomp) = (0usize, 0.0f64);
    let mut skip: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut states, mut smooth_ok, mut smooth_tried, mut solved, mut solved_ok) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut acc_ok, mut acc_tried, mut worst_acc) = (0usize, 0usize, 0.0f64);
    let (mut chain_ok, mut chain_tried, mut worst_chain) = (0usize, 0usize, 0.0f64);
    let (mut solved_nomesh, mut solved_ok_nomesh, mut worst_qacc_nomesh) = (0usize, 0usize, 0.0f64);
    let mut solved_no_worse = 0usize;
    let (mut framed, mut frame_unpaired, mut tangent_differs) = (0usize, 0usize, 0usize);
    // the library's own `forward_mujoco` against the solve assembled here, and its `<option>` against MuJoCo's
    let (mut lib_compared, mut lib_identical, mut options_differ) = (0usize, 0usize, 0usize);
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
                refused_bodies += o.bodies.iter().filter(|(n, ..)| !n.is_empty() && n != "world").count();
                if notes.len() < 60 {
                    notes.push(format!("{}: loader refused: {}", o.rel, tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()).err().unwrap_or_default()));
                }
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
        // ⛔ the welds as MuJoCo computes them, from the BODY tree — not from the tree dof a geom rides on.
        // A body with two joints has a ride dof whose parent is its own previous dof, and that reads as
        // unrelated to the body it hangs off.
        // ⛔ these two are COMPILED quantities — they do not depend on the state, and comparing them
        // inside the state loop both recomputed them per state and skipped them entirely on a model
        // that some later gate refused. `hello_robot_stretch_3` is blocked on a passive force it has
        // and this port does not, and that silently took its inertia and its inverse weights out of
        // the comparison too — the one model whose mass matrix is furthest off.
        {
            let biw = t.body_invweight0();
            // ⭐ `body_invweight0` against MuJoCo's own, for every body — the number every contact row's
            // `efc_D` divides by. Compared here rather than probed by hand, because it is compiled at
            // `qpos0` and a port that reads the CURRENT state is exact on the reference pose: only a sweep
            // that samples moved states can fail.
            for (name, tran, rot, inert) in &o.bodies {
                if name.is_empty() {
                    continue;
                }
                let Some(w) = biw.get(name) else {
                    // ⛔ a body whose WELD has dofs and still has no inverse weight is a hole in the
                    // gate's subject list, not a convention — named so the denominator closes.
                    if inert.len() == 12 && inert[11] == 0.0 {
                        println!("  ⛔ {} body {name} has dofs and no inverse weight — unaccounted for", o.rel);
                    }
                    continue;
                };
                if tran.is_nan() {
                    continue;
                }
                iw0_tried += 1;
                let e = (w.tran - tran).abs() / tran.abs().max(1e-9);
                let er = (w.rot - rot).abs() / rot.abs().max(1e-9);
                // 1e-7, not 1e-9: on `hello_robot_stretch`, a free base under a long chain, inverting the
                // mass matrix costs about 2e-9 of relative accuracy and every body reads that noise.
                if e.max(er) < 1e-7 {
                    iw0_ok += 1;
                } else if std::env::var("DUMP_IW0").is_ok() {
                    println!("  {} IW0 {name}: tran {:.8e} vs {tran:.8e} (ratio {:.6}), rot {:.8e} vs {rot:.8e}", o.rel, w.tran, tran / w.tran, w.rot);
                }
                if e.max(er) > worst_iw0 {
                    worst_iw0 = e.max(er);
                    worst_iw0_where = format!("{} body {name}: tran {:.8e} vs {tran:.8e}, rot {:.8e} vs {rot:.8e}", o.rel, w.tran, w.rot);
                }
            }
            // ⭐ **the inertia MuJoCo STORES**, body by body: mass, the three principal values, the frame
            // they are diagonal in, and `ipos`. This is upstream of the mass matrix, of every `*_invweight0`
            // and of the fluid model, so a residual here explains several sweep lines at once — and a
            // sweep that only compares the mass matrix cannot say WHICH body it came from.
            for (name, _, _, inert) in &o.bodies {
                if inert.len() != 12 {
                    continue;
                }
                let (Some(&mass), Some(&ipos), Some(&iq)) = (t.body_mass.get(name), t.body_ipos.get(name), t.body_iquat.get(name)) else { continue };
                let Some(&pr) = t.body_iinertia.get(name) else { continue };
                inertia_tried += 1;
                // bit for bit: mass, principal values, the quaternion and ipos, exactly as MuJoCo stores them
                if let Some(qw) = t.body_iquat_wxyz.get(name) {
                    let ours = [mass, pr[0], pr[1], pr[2], qw[0], qw[1], qw[2], qw[3], ipos[0], ipos[1], ipos[2]];
                    if ours.iter().zip(&inert[..11]).all(|(a, b)| a.to_bits() == b.to_bits()) {
                        inertia_bits += 1;
                    } else if inertia_bits_notes.len() < 6 {
                        inertia_bits_notes.push(format!("{} body {name}: ours {ours:?}\n        MuJoCo {:?}", o.rel, &inert[..11]));
                    }
                }
                let scale = inert[1].abs().max(inert[2].abs()).max(inert[3].abs()).max(1e-12);
                let mut e = (mass - inert[0]).abs() / inert[0].abs().max(1e-9);
                // ⭐ `ipos` tracked on its own as well as folded in: it is what `qfrc_gravcomp` and the
                // inertia-box fluid model read, and neither of them touches the inertia TENSOR. A residual
                // in one is a completely different bug from a residual in the other.
                let mut eip = 0.0f64;
                for k in 0..3 {
                    eip = eip.max((ipos[k] - inert[8 + k]).abs() / ipos.norm().max(1e-6));
                }
                e = e.max(eip);
                if eip > worst_ipos {
                    worst_ipos = eip;
                    worst_ipos_where = format!("{} body {name}: {:?} vs {:?}", o.rel, ipos.as_slice(), &inert[8..11]);
                }
                // ⛔ the principal values are a SET paired with a frame, not a sequence: MuJoCo's
                // eigen-solver and this port's can return the same tensor with the axes in a different
                // order, and a component-by-component comparison calls that a failure. It is not — the
                // frame permutes with them and everything downstream reads the pair. `link_gripper_slider`
                // on `hello_robot_stretch` is exactly that: identical tensor, axes 1 and 3 swapped.
                let (mut a, mut b) = (pr, [inert[1], inert[2], inert[3]]);
                a.sort_by(f64::total_cmp);
                b.sort_by(f64::total_cmp);
                for k in 0..3 {
                    e = e.max((a[k] - b[k]).abs() / scale);
                }
                if pr.iter().zip(&inert[1..4]).any(|(x, y)| (x - y).abs() > 1e-9 * scale) {
                    inertia_permuted += 1;
                }
                // the frame, as a rotation: a quaternion and its negation are the same frame, and so is one
                // whose principal axes are permuted when two principal values are equal — compare what the
                // tensor they rebuild says instead
                let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(inert[4], inert[5], inert[6], inert[7]));
                let rt = q.to_rotation_matrix().into_inner();
                let full = |r: &nalgebra::Matrix3<f64>, d: &[f64]| r * nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(d[0], d[1], d[2])) * r.transpose();
                let et = (full(&iq, &pr) - full(&rt, &inert[1..4])).abs().max() / scale;
                e = e.max(et);
                if e < 1e-9 {
                    inertia_ok += 1;
                } else {
                    // ⛔⛔ `mjuu_eig3` has TWO stopping rules and the second is far the coarser. It
                    // stops when the largest off-diagonal falls below an ABSOLUTE 1e-12 — which bounds
                    // the discarded part of the tensor by `1e-12 / scale` — and ALSO the moment the next
                    // Jacobi rotation would have `c > 1 - 1e-12`, i.e. an angle below `√(2e-12)` = 1.4e-6
                    // radians. A residual rotation θ between two axes moves the rebuilt tensor by
                    // `|λi − λj|·θ`, so the second rule allows `√(2e-12)·(λmax − λmin) / scale` — two
                    // hundred times more than the first on a body whose principal values are close.
                    //
                    // `flexiv_rizon4/base` is exactly that: principal values within 4% of each other,
                    // matching MuJoCo's to 2.5e-13, and a rebuilt tensor 3.8e-8 out against an allowance
                    // of 5.9e-8. Reading only the first rule called it a failure.
                    let spread = a[2] - a[0];
                    let floor = (1e-12f64 / scale).max((2e-12f64).sqrt() * spread / scale);
                    if e > 0.0 {
                        tightest = tightest.max(e / floor);
                    }
                    if e <= floor {
                        inertia_at_eig3_floor += 1;
                    } else {
                        inertia_above_floor += 1;
                        if inertia_above_floor <= 6 {
                            inertia_notes.push(format!("{} body {name}: off {e:.2e}, largest principal {scale:.3e}, spread {spread:.3e} (eig3's two floors allow {floor:.2e})", o.rel));
                        }
                    }
                }
                if e >= 1e-9 && e > worst_inertia {
                    worst_inertia = e;
                    worst_inertia_where = format!("{} body {name}: mass {mass:.9} vs {:.9}, principal {pr:?} vs {:?}, tensor off {et:.2e}", o.rel, inert[0], &inert[1..4]);
                }
            }
        }
        for s in &o.states {
            states += 1;
            if !t.actuators_unsupported.is_empty() {
                *skip.entry("an actuator this port does not carry (muscle, tendon, adhesion, plugin)").or_default() += 1;
                if notes.len() < 60 {
                    notes.push(format!("{}: actuator refused: {:?}", o.rel, &t.actuators_unsupported[..t.actuators_unsupported.len().min(3)]));
                }
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
            let db = bias.iter().zip(&s.qfrc_bias).map(|(a, b)| (a - b).abs()).fold(0.0, nan_max);
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
            let dp = (0..nv).map(|i| (passive[i] - s.qfrc_passive[i]).abs()).fold(0.0, nan_max);
            // ⭐ `qfrc_passive` is a SUM of four terms MuJoCo keeps apart, so a residual can be
            // ATTRIBUTED instead of guessed at — and the attribution decides whether the state is
            // comparable at all. A term this port does not carry means the answer would be wrong for a
            // reason the sweep cannot see; a residual in a term it DOES carry is a number, and the state
            // is worth comparing with it named.
            let dgc = t
                .free_basis(&q)
                .try_inverse()
                .map(|tinv| {
                    let g = tinv.transpose() * DVector::from_vec(t.qfrc_gravcomp(&q));
                    (0..nv.min(g.len())).map(|i| (g[i] - s.qfrc_gravcomp.get(i).copied().unwrap_or(0.0)).abs()).fold(0.0, nan_max)
                })
                .unwrap_or(f64::NAN);
            let explained_by_gravcomp = (dp - dgc).abs() <= 0.01 * dp;
            if dp > 1e-9 * (0..nv).map(|i| s.qfrc_passive[i].abs()).fold(1.0, f64::max) && explained_by_gravcomp {
                passive_gravcomp_only += 1;
                worst_passive_gravcomp = worst_passive_gravcomp.max(dp);
            }
            if dp > 1e-9 * (0..nv).map(|i| s.qfrc_passive[i].abs()).fold(1.0, f64::max) && !explained_by_gravcomp {
                *skip.entry("a passive force this port does not carry (tendon, fluid)").or_default() += 1;
                worst_passive = worst_passive.max(dp);
                // ⭐ `qfrc_passive` is a SUM of four terms and MuJoCo keeps them apart, so the residual can
                // be attributed instead of guessed at. `qfrc_gravcomp` is the one this port computes on its
                // own; the rest are named so a missing term is named rather than lumped into "tendon,
                // fluid" whether or not the model has either.
                let part = |v: &Vec<f64>| v.iter().map(|x| x.abs()).fold(0.0, nan_max);
                if notes.len() < 60 {
                    notes.push(format!(
                        "{}: passive off {dp:.2e} — MuJoCo's parts: spring {:.2e}, damper {:.2e}, gravcomp {:.2e} (ours off {dgc:.2e}), fluid {:.2e}",
                        o.rel, part(&s.qfrc_spring), part(&s.qfrc_damper), part(&s.qfrc_gravcomp), part(&s.qfrc_fluid)
                    ));
                }
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
            let rel = |a: &[f64], b: &[f64]| (0..a.len()).map(|i| (a[i] - b[i]).abs() / b[i].abs().max(1.0)).fold(0.0, nan_max);
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
                for (b, (name, _, _, _)) in o.bodies.iter().enumerate() {
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
                    for (b, (name, _, _, _)) in o.bodies.iter().enumerate().take(4) {
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
                let Some(a_mj) = t.qacc_smooth_mujoco(&q, &s.qvel, &vec![0.0; t.actuators.len()], &[]) else {
                    *skip.entry("gimbal lock: the Euler base's basis map is singular").or_default() += 1;
                    continue;
                };
                let dq = (0..nv).map(|i| (a_mj[i] - s.qacc_smooth[i]).abs() / s.qacc_smooth[i].abs().max(1.0)).fold(0.0, nan_max);
                chain_tried += 1;
                if dq < 1e-6 {
                    chain_ok += 1;
                } else if notes.len() < 12 {
                    notes.push(format!("{}: floating qacc_smooth off {dq:.2e} in MuJoCo's basis", o.rel));
                }
                worst_chain = nan_max(worst_chain, dq);
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
            // ⭐ the library's own `mj_collision`, so what this sweep verifies is what a caller gets — with the
            // geoms placed by MuJoCo's own kinematics from MuJoCo's qpos (`MJ_KIN=tree` places them through
            // this crate's tree instead, a few ulps away)
            let collision = if std::env::var("MJ_KIN").as_deref() == Ok("tree") { t.collide_explained(&q) } else { t.collide_qpos_explained(&s.qpos) };
            let refused = !collision.refused.is_empty();
            // ⭐ why a pair is NOT here: every gate records the pair it dropped, so a missing contact names
            // the rule that removed it instead of leaving a count to be stared at
            let why = &collision.why;
            for c in &collision.contacts {
                // a contact inside the gap band is detected but not handed to the solver
                if c.record.exclude {
                    continue;
                }
                let (i, j) = (c.geom[0], c.geom[1]);
                // ⛔ `body_invweight0` at `qpos0`, NOT here: MuJoCo computes it when it compiles the model and
                // never recomputes it, so a contact's regularisation carries the reference pose's inverse
                // weights however far the model has moved since.
                if t.geoms[i].kind == GeomType::Mesh || t.geoms[j].kind == GeomType::Mesh {
                    any_mesh = true;
                }
                ourcon.push((i, j, c.record.pos, c.record.frame, c.record.dist));
                contacts.push(t.contact_spec(c, &frames, &biw));
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
                        // ⛔ this is a COUNT DIFFERENCE on a pair, not a missing pair, and saying "only
                        // ours" invites the second reading — it did, and cost a diagnosis. A pair with
                        // four contacts here and one in MuJoCo is the multi-contact manifold disagreeing,
                        // which is a different bug from a pair one collider never tested.
                        let who = if *n > 0 { format!("{} MORE here", n.abs()) } else { format!("{} MORE in MuJoCo", n.abs()) };
                        println!("  {}: geom{a} {} + geom{b} {}: {who} — {note}", o.rel, t.geoms[*a].name, t.geoms[*b].name);
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
            // ⭐ the cone and impratio the LIBRARY read from the file, checked against the ones MuJoCo compiled:
            // `forward_mujoco` uses the tree's own, so a wrong one there would be invisible if this sweep
            // passed the oracle's values in instead
            let cone = if o.cone_pyramidal { ferromotion_core::Cone::Pyramidal } else { ferromotion_core::Cone::Elliptic };
            if t.cone != cone || t.impratio != o.impratio {
                options_differ += 1;
            }
            // the whole row set, in MuJoCo's order: equality, friction, limits, then contacts — posed by the
            // library, in MuJoCo's coordinates
            let problem = match t.constraint_problem(&q, &s.qvel, &vec![0.0; t.actuators.len()], &[], &contacts) {
                Ok(problem) => {
                    let row0: usize = problem.rows.blocks[..problem.contact_blocks_from].iter().map(|b| b.rows()).sum();
                    let cr = ferromotion_core::AssembledRows {
                        blocks: problem.rows.blocks[problem.contact_blocks_from..].to_vec(),
                        jac: nalgebra::DMatrix::zeros(0, 0),
                        aref: problem.rows.aref[row0..].to_vec(),
                        d: problem.rows.d[row0..].to_vec(),
                    };
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
                    problem
                }
                Err(e) => {
                    *skip.entry("a contact row this port cannot build").or_default() += 1;
                    if notes.len() < 12 {
                        notes.push(format!("{}: {e}", o.rel));
                    }
                    continue;
                }
            };
            if problem.rows.blocks.iter().map(|b| b.rows()).sum::<usize>() != s.nefc {
                *skip.entry("we and MuJoCo disagree on how many constraint ROWS there are").or_default() += 1;
                continue;
            }
            // ⛔ the solve happens in MUJOCO'S coordinates, because that is where its answer lives; the
            // library has already mapped the rows (`J·T⁻¹`) and the mass matrix (`T⁻ᵀ·M·T⁻¹`)
            let (jac, m_mj, a0_mj, set) = (&problem.rows.jac, &problem.m, &problem.a0, &problem.rows);
            // ⭐ and MuJoCo's noslip pass after it, where the model asks for one: the acceleration MuJoCo
            // reports on a noslip model is not the optimum of the main problem
            let solved_rows = ferromotion_core::solve_constraints_newton_blocks(m_mj, a0_mj, jac, &set.aref, &set.d, &set.blocks, 1e-13, 200).and_then(|mut sol| {
                if t.noslip_iterations > 0 {
                    let mut force = sol.force.clone();
                    sol.qacc = ferromotion_core::mujoco_noslip(m_mj, a0_mj, jac, &set.aref, &set.d, &set.blocks, &mut force, t.noslip_iterations, t.noslip_tolerance, t.meaninertia)?;
                    sol.force = force;
                }
                Ok(sol)
            });
            match solved_rows {
                Ok(sol) => {
                    // ⭐ and the ONE-CALL library forward must be this very answer: same contacts, same rows,
                    // same solve. Only where the contacts were not swapped for MuJoCo's (`MJSUB`).
                    if mjsub.is_none() {
                        lib_compared += 1;
                        let lib = if std::env::var("MJ_KIN").as_deref() == Ok("tree") {
                            t.forward_mujoco(&q, &s.qvel, &vec![0.0; t.actuators.len()], &[])
                        } else {
                            let mut d = t.make_data();
                            d.qpos.clone_from(&s.qpos);
                            d.qvel.clone_from(&s.qvel);
                            t.forward(&d)
                        };
                        match lib {
                            Ok(f) if f.qacc.iter().zip(sol.qacc.iter()).all(|(x, y)| x.to_bits() == y.to_bits()) => lib_identical += 1,
                            Ok(f) => notes.push(format!("{}: forward_mujoco differs from the assembled solve by {:.2e}", o.rel, f.qacc.iter().zip(sol.qacc.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, nan_max))),
                            Err(e) => notes.push(format!("{}: forward_mujoco refused a state the sweep solved: {e}", o.rel)),
                        }
                    }
                    // ⛔⛔ compare against the OPTIMUM OF THE ROWS MUJOCO BUILT, not against the iterate it
                    // happened to stop on. On the 29 models that cap `iterations` the two differ, and so
                    // they do on uncapped ones where MuJoCo's default `tolerance` (1e-8) ends the solve
                    // early (`toddlerbot_2xc` 2.33e-4, `i2rt_yam` 5.74e-5): the oracle converges at
                    // tolerance 0. The gap is MuJoCo's stopping rule, not this port's physics, so it is
                    // counted and named below instead of being charged to the contact model.
                    let target: &[f64] = if s.qacc_converged.len() == nv { &s.qacc_converged } else { &s.qacc };
                    let truncated = (s.qacc_converged.len() == nv).then(|| {
                        (0..nv).map(|i| (s.qacc[i] - s.qacc_converged[i]).abs() / s.qacc_converged[i].abs().max(1.0)).fold(0.0, nan_max)
                    });
                    if let Some(g) = truncated.filter(|g| *g >= 1e-6) {
                        trunc_states += 1;
                        if g > worst_trunc {
                            worst_trunc = g;
                            worst_trunc_where = format!("{} (iterations {}, ls_iterations {})", o.rel, o.iterations, o.ls_iterations);
                        }
                    }
                    let dq = (0..nv).map(|i| (sol.qacc[i] - target[i]).abs() / target[i].abs().max(1.0)).fold(0.0, nan_max);
                    if !any_mesh {
                        solved_nomesh += 1;
                        worst_qacc_nomesh = nan_max(worst_qacc_nomesh, dq);
                    }
                    if dq > worst_qacc {
                        worst_qacc = dq;
                        worst_qacc_where = format!("{} ({} contacts)", o.rel, contacts.len());
                    }
                    // ⛔⛔ "our cost is no worse" is EVIDENCE OF NOTHING when the rows are our own. The
                    // solve converged, so of course our answer minimises our problem; a wrong contact set
                    // would pass this test every time. It is only evidence where the rows are known to be
                    // MuJoCo's, which here means a state with NO CONTACTS — those rows are the joint rows
                    // `examples/menagerie_rows` checks against MuJoCo, 236 of 236. And only without noslip:
                    // MuJoCo's answer after a noslip pass is MEANT to leave the optimum of the main problem.
                    //
                    // ⛔⛔ CORRECTED 2026-09-27: this said `umi_gripper`, "whose eight dof-friction rows all
                    // sit at their bounds", lands 3.2e-3 from MuJoCo "at a cost no worse, so the difference
                    // is MuJoCo's iteration limit". It was noslip (`umi_gripper` sets `noslip_iterations` 2);
                    // with `mujoco_noslip` applied the state agrees.
                    if dq >= 1e-6 && contacts.is_empty() && t.noslip_iterations == 0 {
                        let cost_at = |a: &DVector<f64>| {
                            let jar: Vec<f64> = (jac * a - DVector::from_row_slice(&set.aref)).iter().copied().collect();
                            let u = ferromotion_core::mujoco_constraint_update_blocks(&set.blocks, &set.d, &jar);
                            let da = a - a0_mj;
                            0.5 * (da.transpose() * m_mj * &da)[(0, 0)] + u.cost
                        };
                        if cost_at(&sol.qacc) <= cost_at(&DVector::from_row_slice(target)) {
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
    println!("    ⛔ MuJoCo stopped its own solver short of the optimum of the rows it built on {trunc_states} of them (worst {worst_trunc:.2e} on {worst_trunc_where}); those are compared against its CONVERGED answer");
    println!("    the library's one-call MjcfTree::forward (MjcfData) equals this solve BIT FOR BIT on {lib_identical} of {lib_compared}; the tree's <option cone/impratio> differs from MuJoCo's on {options_differ} states");
    println!("  CONTACT FRAMES (the tangent pair, not only the normal): {framed} contacts paired with MuJoCo's ({frame_unpaired} states could not be paired by geom)");
    println!("    worst normal {worst_con_normal:.2e} on {worst_con_normal_where}; tangent pair rotated on {tangent_differs} of them, worst 1-|t·t'| {worst_con_tangent:.2e} on {worst_con_tangent_where}");
    println!("    worst efc_D (relative) {worst_con_d:.2e} on {worst_con_d_where} ({con_rows_compared} contacts compared row by row; {reversed_pair} left out because MuJoCo wrote the geom pair the other way round)");
    println!("    worst efc_aref (relative) {worst_con_aref:.2e} on {worst_con_aref_where}");
    println!("    worst qfrc_passive on the states it blocked: {worst_passive:.2e}");
    println!("    {passive_gravcomp_only} states were let through with a passive residual traced ENTIRELY to qfrc_gravcomp — a term this port carries, worst {worst_passive_gravcomp:.2e}; spring, damper and fluid are exact on them");
    println!(
        "  DENOMINATOR: {bodies_in_corpus} named, non-world bodies in the corpus; {refused_bodies} are in a model the loader refuses, leaving {} — the inertia gate saw {inertia_tried}. {dofless} of those have a WELD WITH NO DOFS and so no inverse weight at all; the inverse-weight gate saw {iw0_tried} of the {} that remain.",
        bodies_in_corpus - refused_bodies,
        bodies_in_corpus - refused_bodies - dofless
    );
    println!("  the inertia MuJoCo STORES (mass, the three principal values, their frame, ipos): {inertia_ok} of {inertia_tried} within 1e-9; worst {worst_inertia:.2e} on {worst_inertia_where}");
    println!("    BIT FOR BIT (mass, body_inertia, body_iquat, body_ipos): {inertia_bits} of {inertia_tried}");
    if std::env::var("MENAGERIE_INERTIA_BITS").is_ok() {
        for n in &inertia_bits_notes {
            println!("      {n}");
        }
    }
    println!("    the inertial POSITION alone, which is what gravcomp and the fluid model read: worst {worst_ipos:.2e} on {worst_ipos_where}");
    println!("    {inertia_permuted} of all {inertia_tried} have the same tensor with the principal axes in a DIFFERENT ORDER — equivalent, because the frame permutes with them, and counted here whether or not they pass");
    println!("    of the rest, {inertia_at_eig3_floor} are inside what mjuu_eig3's two stopping rules allow for their own principal values, and {inertia_above_floor} are not (worst case reaches {tightest:.2} of its own allowance):");
    for n in &inertia_notes {
        println!("      {n}");
    }
    println!("  body_invweight0 (what every contact row's efc_D divides by, compiled at qpos0): {iw0_ok} of {iw0_tried} within 1e-7; worst {worst_iw0:.2e} on {worst_iw0_where}");
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
