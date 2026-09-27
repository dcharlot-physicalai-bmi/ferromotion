//! **What the mesh compiler stores, against MuJoCo's own, bit for bit** — `mesh_pos` and `mesh_quat` (the
//! centre of mass and principal frame `mjCMesh::Process` re-expresses every mesh in) and `mesh_vert` (the
//! float vertices every collider reads).
//!
//! ```text
//! python scripts/mujoco_mesh_store_oracle.py <menagerie root> <store oracle txt>
//! cargo run --release --example menagerie_mesh_store -- <menagerie root> <store oracle txt>
//! ```
//!
//! Every other mesh sweep compares within a tolerance, and a tolerance cannot see this: a stored vertex is a
//! double rounded to `f32`, and a double one ulp off can round to the neighbouring float — 6e-8 m on a
//! half-metre part, which is a contact 6e-8 m off with every other input exact.

use ferromotion_core::tree_from_mjcf;
use std::collections::{BTreeMap, BTreeSet};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_mesh_store <menagerie root> <store oracle txt>");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let mut frames: BTreeMap<(String, String), [f64; 7]> = BTreeMap::new();
    let mut verts: BTreeMap<(String, String), Vec<u32>> = BTreeMap::new();
    let hex = |s: &str| -> f64 {
        // C99 hex float: [-]0x1.<hex>p<exp> (or 0x0p+0)
        let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
        let s = s.strip_prefix("0x").unwrap();
        let (mant, exp) = s.split_once('p').unwrap();
        let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
        let mut m = u64::from_str_radix(int, 16).unwrap() as f64;
        let mut scale = 1.0 / 16.0;
        for c in frac.chars() {
            m += c.to_digit(16).unwrap() as f64 * scale;
            scale /= 16.0;
        }
        let v = m * 2f64.powi(exp.parse::<i32>().unwrap());
        if neg { -v } else { v }
    };
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t.first() {
            Some(&"frame") => {
                let v: Vec<f64> = t[3].split_whitespace().map(hex).collect();
                frames.insert((t[1].into(), t[2].into()), v.try_into().unwrap());
            }
            Some(&"vert") => {
                verts.insert((t[1].into(), t[2].into()), t[4].split_whitespace().map(|x| u32::from_str_radix(x, 16).unwrap()).collect());
            }
            _ => {}
        }
    }
    let mut files: Vec<_> = std::fs::read_dir(root).unwrap().flatten().filter(|e| e.path().is_dir()).flat_map(|d| std::fs::read_dir(d.path()).unwrap().flatten().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "xml")).collect();
    files.sort();
    let mut seen = BTreeSet::new();
    let (mut f_same, mut f_diff, mut v_same, mut v_diff, mut v_count_diff) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut pos_same, mut quat_same) = (0usize, 0usize);
    let mut frame_notes = Vec::new();
    let (mut pos_worst, mut quat_worst, mut vert_worst) = (0.0f64, 0.0f64, 0.0f64);
    let mut notes = Vec::new();
    for path in files {
        let dir = path.parent().unwrap().to_path_buf();
        let model = dir.file_name().unwrap().to_string_lossy().to_string();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else { continue };
        for (name, d) in &t.mesh_props {
            let key = (model.clone(), name.clone());
            if !seen.insert(key.clone()) {
                continue;
            }
            if let Some(want) = frames.get(&key) {
                let ours = [d.com.x, d.com.y, d.com.z, d.quat[0], d.quat[1], d.quat[2], d.quat[3]];
                let same = |r: std::ops::Range<usize>| r.clone().all(|k| ours[k].to_bits() == want[k].to_bits());
                pos_same += same(0..3) as usize;
                quat_same += same(3..7) as usize;
                if !same(0..3) && frame_notes.len() < 8 {
                    frame_notes.push(format!("{model}/{name}: pos {:?} vs {:?}", &ours[0..3], &want[0..3]));
                }
                if ours.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()) {
                    f_same += 1;
                } else {
                    f_diff += 1;
                    pos_worst = pos_worst.max((0..3).map(|k| (ours[k] - want[k]).abs()).fold(0.0, f64::max));
                    quat_worst = quat_worst.max((3..7).map(|k| (ours[k] - want[k]).abs()).fold(0.0, f64::max));
                }
            }
            if let (Some(h), Some(want)) = (t.mesh_hulls.get(name), verts.get(&key)) {
                if h.verts.len() * 3 != want.len() {
                    v_count_diff += 1;
                    continue;
                }
                let ours: Vec<u32> = h.verts.iter().flat_map(|v| [v.x as f32, v.y as f32, v.z as f32]).map(f32::to_bits).collect();
                let n = ours.iter().zip(want).filter(|(a, b)| a != b).count();
                if n == 0 {
                    v_same += 1;
                } else {
                    v_diff += 1;
                    let w = ours.iter().zip(want).map(|(a, b)| (f32::from_bits(*a) as f64 - f32::from_bits(*b) as f64).abs()).fold(0.0, f64::max);
                    vert_worst = vert_worst.max(w);
                    if notes.len() < 20 {
                        notes.push(format!("{model}/{name}: {n} of {} coordinates differ, worst {w:.2e}", ours.len()));
                    }
                }
            }
        }
    }
    let unseen: Vec<_> = frames.keys().filter(|k| !seen.contains(*k)).collect();
    let extra = seen.iter().filter(|k| !frames.contains_key(*k)).count();
    let unseen_models: BTreeSet<&str> = unseen.iter().map(|k| k.0.as_str()).collect();
    println!("mesh compiler output against MuJoCo 3.13.0, BIT FOR BIT:");
    println!(
        "  DENOMINATOR: {} meshes in the oracle, {} processed here ({extra} of them missing from the oracle); {} not reached — in a model the loader refuses, or an asset no geom here weighs — in {:?}",
        frames.len(),
        seen.len(),
        unseen.len(),
        unseen_models
    );
    println!("  mesh_pos + mesh_quat: identical {f_same}, different {f_diff} (worst pos {pos_worst:.2e}, quat {quat_worst:.2e})");
    println!("    mesh_pos alone identical {pos_same}, mesh_quat alone identical {quat_same}");
    for n in &frame_notes {
        println!("    {n}");
    }
    println!("  mesh_vert (collision meshes): identical {v_same}, different {v_diff} (worst {vert_worst:.2e} m), vertex count differs {v_count_diff}");
    for n in &notes {
        println!("  {n}");
    }
}
