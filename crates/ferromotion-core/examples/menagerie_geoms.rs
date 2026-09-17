//! **Geom parity with MuJoCo on all of Menagerie**: does the loader place every geom where MuJoCo places
//! it, with the shape and contact parameters MuJoCo gives it?
//!
//! The contact sweep took geom poses FROM the oracle so it could isolate the collision routines. Nothing had
//! checked that this crate can produce those poses itself, which is what a caller holding only an MJCF
//! actually needs. This compares [`ferromotion_core::MjcfTree::geoms`], placed by our own kinematics at
//! MuJoCo's `qpos0`, against MuJoCo's `geom_xpos`, `geom_xmat`, `geom_size` and per-geom contact parameters
//! (`scripts/mujoco_geom_oracle.py`).
//!
//! ```text
//! cargo run --release --example menagerie_geoms -- <menagerie root> <geom oracle txt> [substring]
//! ```

use ferromotion_core::{tree_from_mjcf, MjcfJointKind};
use nalgebra::{Matrix3, Vector3};

#[derive(Default)]
struct OGeom {
    name: String,
    ty: String,
    body: String,
    size: [f64; 3],
    contype: u32,
    conaffinity: u32,
    condim: usize,
    margin: f64,
    gap: f64,
    priority: i32,
    solmix: f64,
    friction: [f64; 3],
    solref: [f64; 2],
    solimp: [f64; 5],
    xpos: Vector3<f64>,
    xmat: Matrix3<f64>,
}

#[derive(Default)]
struct OModel {
    rel: String,
    joints: Vec<(String, String, usize)>,
    qpos0: Vec<f64>,
    geoms: Vec<OGeom>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_geoms <menagerie root> <geom oracle txt> [substring]");
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
            "qpos0" => models.last_mut().unwrap().qpos0 = t[1..].iter().map(|x| f(x)).collect(),
            "geom" => {
                let m: Vec<f64> = t[27..36].iter().map(|x| f(x)).collect();
                models.last_mut().unwrap().geoms.push(OGeom {
                    name: t[1].to_string(),
                    ty: t[2].to_string(),
                    body: t[3].to_string(),
                    size: [f(t[4]), f(t[5]), f(t[6])],
                    contype: t[7].parse::<i64>().unwrap() as u32,
                    conaffinity: t[8].parse::<i64>().unwrap() as u32,
                    condim: t[9].parse().unwrap(),
                    margin: f(t[10]),
                    gap: f(t[11]),
                    priority: t[12].parse().unwrap(),
                    solmix: f(t[13]),
                    friction: [f(t[14]), f(t[15]), f(t[16])],
                    solref: [f(t[17]), f(t[18])],
                    solimp: [f(t[19]), f(t[20]), f(t[21]), f(t[22]), f(t[23])],
                    xpos: Vector3::new(f(t[24]), f(t[25]), f(t[26])),
                    xmat: Matrix3::from_row_slice(&m),
                });
            }
            _ => {}
        }
    }

    let (mut seen, mut refused, mut ngeom, mut count_ok, mut params_ok, mut placed) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut worst_pos, mut worst_rot, mut worst_size) = (0.0f64, 0.0f64, 0.0f64);
    let mut refusals: Vec<String> = Vec::new();
    // one worst example per failure kind, so a single noisy category cannot hide the others
    let mut worst_of: std::collections::BTreeMap<&'static str, (f64, String)> = std::collections::BTreeMap::new();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    // every differing geom, so the tail can be read rather than inferred from one worst example
    let mut diffs: Vec<(f64, String)> = Vec::new();
    let note = |kind: &'static str, mag: f64, what: String, counts: &mut std::collections::BTreeMap<&'static str, usize>, worst_of: &mut std::collections::BTreeMap<&'static str, (f64, String)>| {
        *counts.entry(kind).or_default() += 1;
        let e = worst_of.entry(kind).or_insert((0.0, String::new()));
        if mag >= e.0 {
            *e = (mag, what);
        }
    };
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
                if refusals.len() < 8 {
                    refusals.push(format!("{}: {e}", o.rel));
                }
                continue;
            }
        };
        seen += 1;
        if t.geoms.len() != o.geoms.len() {
            notes.push(format!("{}: {} geoms, MuJoCo has {}", o.rel, t.geoms.len(), o.geoms.len()));
            continue;
        }
        count_ok += 1;

        // MuJoCo's qpos0 in our coordinates; skip a model whose joints we cannot line up one for one
        let names_match = t.joints.len() == o.joints.len()
            && t.joints.iter().zip(&o.joints).all(|(a, b)| {
                matches!(
                    (a.kind, b.1.as_str()),
                    (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide") | (MjcfJointKind::Ball, "ball") | (MjcfJointKind::Free, "free")
                )
            });
        let q = names_match.then(|| t.q_from_qpos(&o.qpos0, &o.joints.iter().map(|j| j.2).collect::<Vec<_>>()).ok()).flatten();
        let frames = q.as_ref().map(|q| ferromotion_core::tree_frames(&t.tree, q));

        for (i, g) in t.geoms.iter().enumerate() {
            ngeom += 1;
            let e = &o.geoms[i];
            let ds = (0..3).map(|k| (g.size[k] - e.size[k]).abs()).fold(0.0, f64::max);
            worst_size = worst_size.max(ds);
            let p = g.params;
            let same = p.condim == e.condim
                && p.contype == e.contype
                && p.conaffinity == e.conaffinity
                && p.priority == e.priority
                && (p.solmix - e.solmix).abs() < 1e-12
                && (p.margin - e.margin).abs() < 1e-12
                && (p.gap - e.gap).abs() < 1e-12
                && p.friction.iter().zip(e.friction).all(|(a, b)| (a - b).abs() < 1e-12)
                && p.solref.iter().zip(e.solref).all(|(a, b)| (a - b).abs() < 1e-12)
                && p.solimp.iter().zip(e.solimp).all(|(a, b)| (a - b).abs() < 1e-12);
            if same {
                params_ok += 1;
            } else {
                note(
                    "contact parameters",
                    1.0,
                    format!(
                        "{} geom {} ({}): condim {}/{} contype {}/{} conaff {}/{} margin {}/{} gap {}/{} prio {}/{} solmix {}/{} friction {:?}/{:?} solref {:?}/{:?} solimp {:?}/{:?}",
                        o.rel, e.name, e.ty, p.condim, e.condim, p.contype, e.contype, p.conaffinity, e.conaffinity, p.margin, e.margin, p.gap, e.gap, p.priority, e.priority, p.solmix, e.solmix, p.friction, e.friction, p.solref, e.solref, p.solimp, e.solimp
                    ),
                    &mut counts,
                    &mut worst_of,
                );
            }
            if ds > 1e-9 {
                let kind = if e.ty == "hfield" { "size (hfield — not carried)" } else { "size" };
                note(kind, ds, format!("{} geom {} ({}): size {:?} vs MuJoCo {:?}", o.rel, e.name, e.ty, g.size, e.size), &mut counts, &mut worst_of);
            }
            let Some(frames) = &frames else { continue };
            let world = match g.joint {
                Some(j) => frames[j] * g.pose,
                None => g.pose,
            };
            placed += 1;
            let dp = (world.translation.vector - e.xpos).norm();
            let dr = (world.rotation.to_rotation_matrix().matrix() - e.xmat).norm();
            worst_pos = worst_pos.max(dp);
            worst_rot = worst_rot.max(dr);
            // a geom that cannot collide is a rendering detail; one that can is physics, so they are counted
            // apart rather than summed into a single number that hides which kind is wrong
            let kind = if e.contype != 0 || e.conaffinity != 0 { "position (COLLIDABLE)" } else { "position (visual only)" };
            if dp > 1e-9 {
                let what = format!("{} geom {} ({}) on {}: position off {dp:.2e} m", o.rel, e.name, e.ty, e.body);
                diffs.push((dp, what.clone()));
                note(kind, dp, what, &mut counts, &mut worst_of);
            }
            let kind = if e.contype != 0 || e.conaffinity != 0 { "rotation (COLLIDABLE)" } else { "rotation (visual only)" };
            if dr > 1e-9 {
                note(kind, dr, format!("{} geom {} ({}) on {}: rotation off {dr:.2e}", o.rel, e.name, e.ty, e.body), &mut counts, &mut worst_of);
            }
        }
    }
    println!("models {seen} (loader refused {refused})");
    for r in &refusals {
        println!("  refused {r}");
    }
    println!("  geom COUNT identical: {count_ok} of {seen}");
    println!("  geoms compared {ngeom}: contact parameters identical {params_ok}, placed at qpos0 {placed}");
    println!("  worst size {worst_size:.2e}, worst world position {worst_pos:.2e} m, worst rotation {worst_rot:.2e}");
    for (kind, n) in &counts {
        let (mag, what) = &worst_of[kind];
        println!("  {n} geoms differ in {kind} (worst {mag:.2e}): {what}");
    }
    for n in notes.iter().take(10) {
        println!("  {n}");
    }
    if std::env::var("VERBOSE").is_ok() {
        diffs.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, what) in diffs.iter().take(40) {
            println!("    {what}");
        }
    }
}
