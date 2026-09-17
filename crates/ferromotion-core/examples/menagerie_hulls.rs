//! **Convex-hull parity with MuJoCo on every Menagerie collision mesh.**
//!
//! The contact sweep (`menagerie_contacts`) compares contacts, which sit downstream of the hull; this
//! compares the hull itself, because a mesh contact's witness points are chosen from hull polygons and a
//! hull that differs by one near-coplanar vertex moves them. MuJoCo builds its hull with **qhull**, which
//! merges facets that are coplanar within a tolerance; `MeshHull` builds one with quickhull and an absolute
//! `1e-9`. `scripts/mujoco_mesh_hull_oracle.py` records, per distinct collision mesh, MuJoCo's vertex count,
//! hull-graph vertex count and merged-polygon count with every polygon's normal and size.
//!
//! ```text
//! cargo run --release --example menagerie_hulls -- <menagerie root> <hull oracle txt>
//! ```
//!
//! Reported per mesh: hull vertices ours vs MuJoCo's, polygons ours vs MuJoCo's, and how many of MuJoCo's
//! polygon normals we reproduce (matched within 1e-6 after sorting, so triangulation order does not count).

use ferromotion_core::tree_from_mjcf;
use nalgebra::Vector3;
use std::collections::{BTreeMap, BTreeSet};

struct OMesh {
    nvert: usize,
    nhull: i64,
    polys: Vec<(usize, Vector3<f64>)>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_hulls <menagerie root> <hull oracle txt>");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let mut oracle: BTreeMap<(String, String), OMesh> = BTreeMap::new();
    let mut cur: Option<(String, String)> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "mesh" => {
                let key = (t[1].to_string(), t[2].to_string());
                oracle.insert(key.clone(), OMesh { nvert: t[3].parse().unwrap(), nhull: t[4].parse().unwrap(), polys: Vec::new() });
                cur = Some(key);
            }
            "poly" => {
                let k = cur.as_ref().unwrap();
                let n: usize = t[1].parse().unwrap();
                let v = Vector3::new(t[2].parse().unwrap(), t[3].parse().unwrap(), t[4].parse().unwrap());
                oracle.get_mut(k).unwrap().polys.push((n, v));
            }
            _ => {}
        }
    }

    // every model, so every mesh the oracle saw is reached; dedup by (dir, mesh name) as the oracle does
    let mut dirs: Vec<_> = std::fs::read_dir(root).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let mut done: BTreeSet<(String, String)> = BTreeSet::new();
    let (mut meshes, mut same_verts, mut same_polys, mut normals_ok, mut normals_total, mut normals_hit) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut worst: Vec<(String, String, usize, i64, usize, usize)> = Vec::new();
    let mut soundness: BTreeMap<(String, String), f64> = BTreeMap::new();
    for dir in &dirs {
        let dirname = dir.file_name().unwrap().to_string_lossy().to_string();
        let mut models: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|x| x == "xml").unwrap_or(false)).collect();
        models.sort();
        for model in &models {
            if oracle.keys().filter(|(d, n)| d == &dirname && !done.contains(&(d.clone(), n.clone()))).count() == 0 {
                break;
            }
            let Ok(xml) = std::fs::read_to_string(model) else { continue };
            let mdir = model.parent().unwrap().to_path_buf();
            let resolve = |p: &str| std::fs::read(mdir.join(p)).ok();
            let Ok(t) = tree_from_mjcf(&xml, &resolve) else { continue };
            for (name, hull) in &t.mesh_hulls {
                let key = (dirname.clone(), name.clone());
                let Some(o) = oracle.get(&key) else { continue };
                if !done.insert(key.clone()) {
                    continue;
                }
                meshes += 1;
                let nh = hull.hull_verts.len();
                let np = hull.polygons.len();
                if o.nhull < 0 || nh as i64 == o.nhull {
                    same_verts += 1;
                }
                if np == o.polys.len() {
                    same_polys += 1;
                }
                // normals: how many of MuJoCo's polygons have one of ours pointing the same way
                let mut hit = 0;
                for (_, n) in &o.polys {
                    if hull.polygons.iter().any(|p| (p.normal - n).norm() < 1e-6) {
                        hit += 1;
                    }
                }
                normals_total += o.polys.len();
                normals_hit += hit;
                if hit == o.polys.len() {
                    normals_ok += 1;
                }
                {
                    // is our hull merely sparser, or does it cut through the mesh? the farthest any mesh
                    // vertex lies outside our hull's faces answers that in metres. ⛔ The face planes are
                    // oriented against the hull's own centroid rather than trusted from the winding: a
                    // normal taken the wrong way round measures the mesh's DIAMETER and reads as a hull
                    // that misses by centimetres, which is exactly what this instrument first reported.
                    let mut centre = Vector3::zeros();
                    for &i in &hull.hull_verts {
                        centre += hull.verts[i];
                    }
                    centre /= hull.hull_verts.len() as f64;
                    let planes: Vec<(Vector3<f64>, f64)> = hull
                        .hull_faces
                        .iter()
                        .filter_map(|f| {
                            let (a, b, c) = (hull.verts[f[0]], hull.verts[f[1]], hull.verts[f[2]]);
                            let nrm = (b - a).cross(&(c - a));
                            let l = nrm.norm();
                            (l > 1e-14).then(|| {
                                let n = nrm / l;
                                let n = if n.dot(&(a - centre)) < 0.0 { -n } else { n };
                                (n, n.dot(&a))
                            })
                        })
                        .collect();
                    let mut out: f64 = 0.0;
                    for v in &hull.verts {
                        let mut h = f64::NEG_INFINITY;
                        for (n, d) in &planes {
                            h = h.max(n.dot(v) - d);
                        }
                        out = out.max(h);
                    }
                    soundness.insert((dirname.clone(), name.clone()), out);
                }
                if nh as i64 != o.nhull || np != o.polys.len() {
                    worst.push((dirname.clone(), name.clone(), o.nvert, o.nhull, nh, np.abs_diff(o.polys.len())));
                }
            }
        }
    }
    println!("collision meshes compared: {meshes} of {} in the oracle", oracle.len());
    println!("  hull vertex count identical: {same_verts}");
    println!("  polygon count identical:     {same_polys}");
    println!("  every MuJoCo polygon normal reproduced: {normals_ok} meshes; {normals_hit} of {normals_total} polygons");
    // the one property a hull OWES regardless of how it is tessellated: it contains every input vertex
    let unsound: Vec<_> = soundness.iter().filter(|&(_, &o)| o > 1e-9).collect();
    let worst_out = soundness.values().copied().fold(0.0, f64::max);
    println!("  SOUND (no mesh vertex outside our own hull): {} of {meshes}; worst excursion {worst_out:.2e} m", meshes - unsound.len());
    for ((d, n), o) in unsound.iter().take(10) {
        println!("    UNSOUND {d}/{n}: {o:.2e} m outside");
    }
    worst.sort_by_key(|w| std::cmp::Reverse(w.5));
    println!("largest polygon-count differences (soundness = how far any mesh vertex lies OUTSIDE our hull):");
    for (d, n, nvert, nhull, ours, diff) in worst.iter().take(20) {
        let out = soundness.get(&(d.clone(), n.clone())).copied().unwrap_or(f64::NAN);
        println!("  {d}/{n}: {nvert} verts, MuJoCo hull {nhull}, ours {ours}, polygons differ by {diff}, worst vertex {out:.2e} outside our hull");
    }
}
