//! **[`ferromotion_core::qhull::mesh_graph`] against MuJoCo's `mesh_graph` on every Menagerie collision mesh.**
//!
//! ```text
//! python scripts/mujoco_mesh_graph_oracle.py <menagerie root> <graph oracle txt>
//! cargo run --release --example menagerie_qhull -- <menagerie root> <graph oracle txt> [--dump <verts txt>]
//! ```
//!
//! The port is fed each mesh's vertices exactly as MuJoCo feeds qhull — the file's `f32` vertices before
//! `scale`, `refpos` and `refquat` (`mjCMesh::MakeGraph` runs before `ApplyTransformations`) — with the
//! mesh's `maxhullvert`, and its graph is compared integer for integer: the hull's vertices in qhull's
//! order, each vertex's neighbours in qhull's order, and the triangles in qhull's order and orientation.
//! `--dump` also writes those inputs (as `f64` bit patterns) for a reference build of qhull.

use ferromotion_core::qhull::mesh_graph;
use ferromotion_core::tree_from_mjcf;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_qhull <menagerie root> <graph oracle txt> [--dump <verts txt>]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let dump = args.iter().position(|a| a == "--dump").and_then(|i| args.get(i + 1)).cloned();
    let mut oracle: BTreeMap<(String, String), Vec<i32>> = BTreeMap::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        if t.len() == 4 && t[0] == "graph" {
            oracle.insert((t[1].to_string(), t[2].to_string()), t[3].split_whitespace().map(|x| x.parse().unwrap()).collect());
        }
    }
    let mut files: Vec<_> = std::fs::read_dir(root).unwrap().flatten().filter(|e| e.path().is_dir()).flat_map(|d| std::fs::read_dir(d.path()).unwrap().flatten().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "xml")).collect();
    files.sort();
    let mut seen = BTreeSet::new();
    let mut out = String::new();
    let (mut identical, mut differ, mut refused, mut no_oracle) = (0usize, 0usize, 0usize, 0usize);
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    let mut notes = Vec::new();
    let t0 = std::time::Instant::now();
    for path in files {
        let dir = path.parent().unwrap().to_path_buf();
        let model = dir.file_name().unwrap().to_string_lossy().to_string();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else { continue };
        for (name, hull) in &t.mesh_hulls {
            if !seen.insert((model.clone(), name.clone())) {
                continue;
            }
            let pts: Vec<[f64; 3]> = t.mesh_file[name].verts.iter().map(|p| [p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64]).collect();
            if dump.is_some() {
                writeln!(out, "mesh\t{model}\t{name}\t{}\t{}", hull.max_verts.map_or(-1, |c| c as i64), pts.len()).unwrap();
                for p in &pts {
                    writeln!(out, "{:016x} {:016x} {:016x}", p[0].to_bits(), p[1].to_bits(), p[2].to_bits()).unwrap();
                }
            }
            let Some(want) = oracle.get(&(model.clone(), name.clone())) else {
                no_oracle += 1;
                continue;
            };
            match mesh_graph(&pts, hull.max_verts) {
                Ok(g) if &g == want => identical += 1,
                Ok(g) => {
                    differ += 1;
                    if notes.len() < 25 {
                        let first = g.iter().zip(want).position(|(a, b)| a != b).unwrap_or(g.len().min(want.len()));
                        notes.push(format!("{model}/{name}: {} verts {} faces here, MuJoCo {} {}; first difference at [{first}]", g[0], g[1], want[0], want[1]));
                    }
                }
                Err(e) => {
                    refused += 1;
                    *why.entry(e).or_default() += 1;
                }
            }
        }
    }
    if let Some(d) = dump {
        std::fs::write(d, out).unwrap();
    }
    println!("mesh_graph against MuJoCo 3.13.0 ({:.1} s):", t0.elapsed().as_secs_f64());
    println!("  identical {identical}, different {differ}, refused {refused}, no oracle {no_oracle}");
    for (w, n) in &why {
        println!("    {n:>4}  {w}");
    }
    for n in &notes {
        println!("  {n}");
    }
}
