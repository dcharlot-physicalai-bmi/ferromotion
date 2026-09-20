//! **The convex hull's POLYGONS, against MuJoCo's own** — `mjCMesh::MakeGraph`/`MakePolygons`, face by face.
//!
//! The hull sweep compares vertex COUNTS. That is not what the collider reads: `multicontact` looks up the
//! polygons its two support vertices share, and if they share none it returns the single EPA witness instead
//! of a face manifold. A hull with the right vertices and the wrong faces produces one contact where MuJoCo
//! produces four, which is a difference no vertex count can show.
//!
//! ```text
//! cargo run --release --example menagerie_polygons -- <menagerie root> <mesh poly oracle txt> [substring]
//! ```

use ferromotion_core::tree_from_mjcf;
use std::collections::{BTreeMap, BTreeSet};

struct Mesh {
    rel: String,
    name: String,
    nvert: usize,
    polys: Vec<(Vec<f64>, Vec<usize>)>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_polygons <menagerie root> <mesh poly oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap();
    let mut meshes: Vec<Mesh> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "mesh" => meshes.push(Mesh { rel: t[1].to_string(), name: t[2].to_string(), nvert: t[3].parse().unwrap(), polys: Vec::new() }),
            "poly" => {
                let normal = vec![f(t[1]), f(t[2]), f(t[3])];
                let verts: Vec<usize> = t[4..].iter().map(|x| x.parse().unwrap()).collect();
                meshes.last_mut().unwrap().polys.push((normal, verts));
            }
            _ => {}
        }
    }

    let mut by_model: BTreeMap<&str, Vec<&Mesh>> = BTreeMap::new();
    for m in &meshes {
        by_model.entry(m.rel.as_str()).or_default().push(m);
    }
    let (mut seen, mut same_count, mut same_faces) = (0usize, 0usize, 0usize);
    let (mut worst_normal, mut worst_where) = (0.0f64, String::new());
    let (mut shared_faces, mut normal_off, mut normal_flipped) = (0usize, 0usize, 0usize);
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    for (rel, list) in &by_model {
        if filter.as_ref().is_some_and(|fl| !rel.contains(fl.as_str())) {
            continue;
        }
        let model = root.join(rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            *counts.entry("the loader refuses the model").or_default() += 1;
            continue;
        };
        for m in list {
            // only meshes this port actually builds a hull for: a visual-only mesh has no collider
            let Some(hull) = t.mesh_hulls.get(&m.name) else { continue };
            seen += 1;
            if hull.verts.len() != m.nvert {
                *counts.entry("the mesh itself has a different vertex count").or_default() += 1;
                continue;
            }
            if hull.polygons.len() == m.polys.len() {
                same_count += 1;
            }
            // a polygon IS its vertex set: winding and starting point are a convention, membership is not
            let ours: BTreeSet<Vec<usize>> = hull
                .polygons
                .iter()
                .map(|p| {
                    let mut v: Vec<usize> = p.verts.clone();
                    v.sort_unstable();
                    v
                })
                .collect();
            let theirs: BTreeSet<Vec<usize>> = m
                .polys
                .iter()
                .map(|(_, v)| {
                    let mut v = v.clone();
                    v.sort_unstable();
                    v
                })
                .collect();
            if ours == theirs {
                same_faces += 1;
            } else if notes.len() < 10 {
                let only_ours = ours.difference(&theirs).count();
                let only_theirs = theirs.difference(&ours).count();
                notes.push(format!("{rel} / {}: {} polygons here, {} in MuJoCo; {only_ours} only ours, {only_theirs} only MuJoCo's", m.name, hull.polygons.len(), m.polys.len()));
            }
            // where the face sets DO agree, the normals must too
            let mut by_verts: BTreeMap<Vec<usize>, &Vec<f64>> = BTreeMap::new();
            for (n, v) in &m.polys {
                let mut v = v.clone();
                v.sort_unstable();
                by_verts.insert(v, n);
            }
            for p in &hull.polygons {
                let mut v: Vec<usize> = p.verts.clone();
                v.sort_unstable();
                if let Some(n) = by_verts.get(&v) {
                    shared_faces += 1;
                    let d = (0..3).map(|k| (p.normal[k] - n[k]).abs()).fold(0.0, f64::max);
                    let dot: f64 = (0..3).map(|k| p.normal[k] * n[k]).sum();
                    if d > 1e-6 {
                        normal_off += 1;
                    }
                    if dot < -0.99 {
                        normal_flipped += 1;
                    }
                    if d > worst_normal && dot > 0.0 {
                        worst_normal = d;
                        worst_where = format!("{rel} / {}", m.name);
                    }
                }
            }
        }
    }
    println!("collision meshes compared: {seen}");
    println!("  same polygon COUNT as MuJoCo: {same_count}");
    println!("  same polygon SET (every face, by its vertices): {same_faces}");
    println!("  faces both have, by their vertices: {shared_faces}; normals differing by more than 1e-6: {normal_off}, of which OPPOSITE: {normal_flipped}");
    println!("  worst normal disagreement on a shared face that is not simply reversed: {worst_normal:.2e} on {worst_where}");
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
