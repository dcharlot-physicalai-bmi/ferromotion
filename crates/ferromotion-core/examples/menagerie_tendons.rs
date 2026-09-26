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
use std::collections::{BTreeMap, BTreeSet};

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
    /// MuJoCo's resolved path per tendon: `-1` a site, `-2` a pulley, otherwise the GEOM it wrapped
    wraps: BTreeMap<usize, Vec<(i64, [f64; 3])>>,
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
            // MuJoCo's own path: (which object the point came from, where it is), in order
            "wrap" => {
                let i: usize = t[1].parse().unwrap();
                let pts: Vec<(i64, [f64; 3])> = t[2..].chunks(4).filter(|c| c.len() == 4).map(|c| (c[0].parse().unwrap(), [f(c[1]), f(c[2]), f(c[3])])).collect();
                cases.last_mut().unwrap().states.last_mut().unwrap().wraps.insert(i, pts);
            }
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
    // ⭐ how CLOSE the ones that do not match are: a wrap that picks the other side is out by centimetres,
    // one that differs in the last digits of a tangency is not, and a single pass/fail count hides which
    let mut ladder = [0usize; 4];
    let mut notes: Vec<String> = Vec::new();
    let mut unexplained: BTreeSet<String> = BTreeSet::new();
    let mut wrap_shape: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut skipped_tendons, mut refused_tendons) = (0usize, 0usize);
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(&c.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            *counts.entry("the loader refuses the model".into()).or_default() += 1;
            refused_tendons += c.tendons.len();
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
            // ⛔ in TENDONS, not in models: the two are not the same unit, and an accounting that mixes
            // them cannot close. 22 models is 60 tendons here.
            skipped_tendons += c.tendons.len();
            continue;
        }
        // ours by name, so a tendon MuJoCo has and we do not simply is not compared
        let ours: BTreeMap<&str, usize> = t.tendons.iter().enumerate().map(|(i, x)| (x.name.as_str(), i)).collect();
        let qposadr: Vec<usize> = c.joints.iter().map(|j| j.2).collect();
        for s in &c.states {
            let Ok(q) = t.q_from_qpos(&s.qpos, &qposadr) else { continue };
            let (len, mom) = (t.ten_length(&q), t.ten_moment(&q));
            for (u, (name, kind)) in c.tendons.iter().enumerate() {
                let Some(&i) = ours.get(name.as_str()) else {
                    // ⛔ a tendon MuJoCo has that this port does not carry BY NAME. It is accounted for
                    // only if the loader named a reason; anything else is a hole in the subject list, and
                    // a gate whose denominator cannot be derived from the corpus is selecting its own.
                    if !t.tendons_unsupported.iter().any(|(n, _)| n == name) {
                        unexplained.insert(format!("{} / {name} ({kind})", c.rel));
                    }
                    continue;
                };
                compared += 1;
                let dl = (len[i] - s.len[u]).abs();
                // ⭐⭐ WHERE the two paths part company, by WHAT each of them wrapped and not how many.
                // MuJoCo publishes the resolved path — `wrap_obj` is `-1` for a site and otherwise the
                // GEOM the cable wrapped — and this port now reports the same through `tendon_wraps`, so
                // the two are compared obstacle by obstacle.
                //
                // ⛔ Comparing COUNTS instead called two paths "the same obstacles, a different arc" when
                // they were wrapping different obstacles entirely, with tangent points 9 cm apart. A count
                // cannot tell identity, and the bucket it fills is the one that decides what to fix next.
                if let Some(theirs) = s.wraps.get(&u).filter(|_| dl > 1e-9 && kind == "spatial") {
                    let mut their_seq: Vec<String> = Vec::new();
                    let mut k = 0;
                    while k < theirs.len() {
                        if theirs[k].0 >= 0 {
                            their_seq.push(t.geoms.get(theirs[k].0 as usize).map(|g| g.name.clone()).unwrap_or_default());
                            k += 2;
                        } else {
                            k += 1;
                        }
                    }
                    let ours_seq: Vec<String> = t.tendon_wraps(i, &q).into_iter().map(|(g, _)| g).collect();
                    let only_theirs = their_seq.iter().filter(|g| !ours_seq.contains(g)).count();
                    let only_ours = ours_seq.iter().filter(|g| !their_seq.contains(g)).count();
                    // ⛔ A cable exactly TANGENT to its obstacle (tetheria's sites sit on the pulley's radius)
                    // gets an arc of 0 or a full 2πr depending on the sign of a cross product that is pure
                    // rounding. MuJoCo's own `ten_length` takes both values: perturbing qpos by 1e-12 rad
                    // moves it by exactly 2πr on 1–4% of draws. Named, so it is not mistaken for an arc bug.
                    let full_turn = their_seq.iter().any(|g| t.geoms.iter().find(|x| &x.name == g).is_some_and(|x| (dl - std::f64::consts::TAU * x.size[0]).abs() < 1e-9));
                    *wrap_shape
                        .entry(match (only_theirs, only_ours) {
                            (0, 0) if full_turn => "the SAME obstacles, exactly one full turn (2πr) apart: a tangent cable, decided by rounding",
                            (0, 0) => "the SAME obstacles, a different arc on them",
                            (_, 0) => "MuJoCo wrapped an obstacle this port left alone",
                            (0, _) => "this port wrapped an obstacle MuJoCo left alone",
                            _ => "each wrapped an obstacle the other did not",
                        })
                        .or_default() += 1;
                }
                if dl > worst_len {
                    worst_len = dl;
                    worst_len_where = format!("{} / {name} ({kind})", c.rel);
                }
                *seen_kind.entry(kind.clone()).or_default() += 1;
                for (i, tol) in [1e-9, 1e-6, 1e-4, 1e-2].iter().enumerate() {
                    if dl < *tol {
                        ladder[i] += 1;
                    }
                }
                if dl < 1e-9 {
                    ok_len += 1;
                    *ok_kind.entry(kind.clone()).or_default() += 1;
                } else {
                    *bad.entry(format!("{} ({kind}, {})", c.rel.split('/').next().unwrap_or(""), if len[i] < s.len[u] { "ours SHORTER: we under-wrap" } else { "ours LONGER: we over-wrap" })).or_default() += 1;
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
    let in_corpus: usize = cases.iter().map(|c| c.tendons.len()).sum();
    println!(
        "  DENOMINATOR: {in_corpus} tendons in the corpus; {refused_tendons} in a model the loader refuses, {skipped_tendons} in a model whose dof columns do not correspond, {} unexplained — leaving {} compared at each of 3 states, which is the {compared} tendon-states below",
        unexplained.len(),
        in_corpus - refused_tendons - skipped_tendons
    );
    for u in unexplained.iter().take(8) {
        println!("      {u}");
    }
    println!("  tendon-states compared: {compared}");
    if !wrap_shape.is_empty() {
        println!("  of the spatial ones whose LENGTH disagrees, what the two paths actually did:");
        for (k, n) in &wrap_shape {
            println!("      {n:>5}  {k}");
        }
    }
    println!("  length matching MuJoCo: {ok_len}; worst {worst_len:.2e} on {worst_len_where}");
    for (k, n) in &seen_kind {
        println!("    {k}: {} of {n}", ok_kind.get(k).copied().unwrap_or(0));
    }
    println!("    within 1e-9 {}, 1e-6 {}, 1e-4 {}, 1e-2 {} — of {compared}", ladder[0], ladder[1], ladder[2], ladder[3]);
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
