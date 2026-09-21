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
    let (mut hull_vert_same, mut hull_vert_tried, mut hull_vert_diff) = (0usize, 0usize, 0usize);
    let (mut hull_only_ours, mut hull_only_theirs) = (0usize, 0usize);
    let (mut normal_same_line, mut normal_other_line, mut worst_line, mut worst_line_where) = (0usize, 0usize, 0.0f64, String::new());
    let (mut flip_ours_inward, mut flip_theirs_inward, mut flip_ambiguous) = (0usize, 0usize, 0usize);
    let (mut plane_same, mut plane_tried, mut plane_only_ours, mut plane_only_theirs) = (0usize, 0usize, 0usize, 0usize);
    let (mut support_same, mut support_tried, mut worst_support) = (0usize, 0usize, 0.0f64);
    let (mut worst_support_where, mut different_body) = (String::new(), Vec::<String>::new());
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
            if std::env::var("DUMP_HULL").is_ok() {
                let exact = ferromotion_core::try_convex_hull_3d(&hull.verts).map(|h| h.verts.len());
                println!("  {rel} / {}: mesh verts {}, hull verts {}, max_verts {:?}, polygons {}, UNCAPPED hull verts {exact:?}", m.name, hull.verts.len(), hull.hull_verts.len(), hull.max_verts, hull.polygons.len());
            }
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
            // ⭐⭐ the SUPPORTING PLANES, which are merge-invariant: a merged polygon and the triangles it
            // was merged from lie in the same plane, so two hulls of the SAME polytope have the same plane
            // set however their faces are grouped. This is the measurement that separates "we built a
            // different hull" from "we grouped the same hull's faces differently", and nothing about the
            // polygon lists can do it.
            {
                let plane = |n: &[f64], v0: usize| -> [f64; 4] {
                    let p = hull.verts[v0];
                    [n[0], n[1], n[2], n[0] * p.x + n[1] * p.y + n[2] * p.z]
                };
                let mut myplanes: Vec<[f64; 4]> = hull.polygons.iter().map(|p| plane(p.normal.as_slice(), p.verts[0])).collect();
                let theirplanes: Vec<[f64; 4]> = m.polys.iter().map(|(n, v)| plane(n, v[0])).collect();
                // the mesh's own size, so the offset tolerance is relative to the object and not to a metre
                let extent = hull.verts.iter().map(|v| v.amax()).fold(0.0, f64::max).max(1e-9);
                let close = |a: &[f64; 4], b: &[f64; 4]| {
                    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > 0.999_999) && (a[3] - b[3]).abs() < 1e-6 * extent
                };
                let unmatched = |xs: &[[f64; 4]], ys: &[[f64; 4]]| xs.iter().filter(|x| !ys.iter().any(|y| close(x, y))).count();
                let (a, b) = (unmatched(&myplanes, &theirplanes), unmatched(&theirplanes, &myplanes));
                plane_tried += 1;
                if a == 0 && b == 0 {
                    plane_same += 1;
                }
                plane_only_ours += a;
                plane_only_theirs += b;
                myplanes.clear();
                // ⭐⭐⭐ and the SUPPORT FUNCTION, which is the polytope itself: `max over vertices of d·v`
                // over a fixed set of directions is equal for two hulls exactly when they are the same
                // convex body, whatever their faces or their vertex lists say. A plane comparison needs a
                // tolerance on the normal and so cannot tell a different shape from a different
                // triangulation of the same one; this can.
                let ov: BTreeSet<usize> = ours.iter().flatten().copied().collect();
                let tv: BTreeSet<usize> = theirs.iter().flatten().copied().collect();
                let mut worst = 0.0f64;
                let mut seed = 0x9E3779B97F4A7C15u64;
                for _ in 0..256 {
                    let mut d = [0.0f64; 3];
                    for x in d.iter_mut() {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        *x = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                    }
                    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
                    let d = [d[0] / n, d[1] / n, d[2] / n];
                    let sup = |set: &BTreeSet<usize>| {
                        set.iter().map(|&i| d[0] * hull.verts[i].x + d[1] * hull.verts[i].y + d[2] * hull.verts[i].z).fold(f64::NEG_INFINITY, f64::max)
                    };
                    worst = worst.max((sup(&ov) - sup(&tv)).abs() / extent);
                }
                support_tried += 1;
                if worst < 1e-9 {
                    support_same += 1;
                } else {
                    different_body.push(format!("{rel} / {} ({:.2e}, {} hull verts here, {} in MuJoCo, {} shared)", m.name, worst, ov.len(), tv.len(), ov.intersection(&tv).count()));
                }
                if worst > worst_support {
                    worst_support = worst;
                    worst_support_where = format!("{rel} / {}", m.name);
                }
            }
            // the HULL VERTEX SET before the face grouping: if the two hulls keep the same vertices and
            // differ only in how faces are merged, the gap is qhull's coplanar-merge rule and nothing
            // else. If they keep different vertices, it is the hull itself and the merge is downstream.
            let vset = |it: &BTreeSet<Vec<usize>>| -> BTreeSet<usize> { it.iter().flatten().copied().collect() };
            hull_vert_tried += 1;
            if vset(&ours) == vset(&theirs) {
                hull_vert_same += 1;
            } else {
                let (a, b) = (vset(&ours), vset(&theirs));
                hull_only_ours += a.difference(&b).count();
                hull_only_theirs += b.difference(&a).count();
                hull_vert_diff += a.difference(&b).count() + b.difference(&a).count();
            }
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
                        // ⛔⛔ "flipped" is only a convention if both are consistent. An outward normal is
                        // NOT a convention — the polytope determines it — so whichever of the two points
                        // inward is wrong, and on this corpus that is MUJOCO'S, on about 5% of polygons.
                        // Verified with MuJoCo's own data and nothing of ours: of 315,431 polygons across
                        // 804 meshes, 16,164 put one of MuJoCo's OWN hull corners more than 1e-3 of the
                        // part outside the polygon's own plane, worst 2.53 — a whole polytope width, which
                        // only a reversed normal produces. `aloha/d405_solid poly3899` is outward under
                        // `−n` (2.5e-5) and 2.53 out under `+n`; a clean 8-vertex box compiled through the
                        // same path has all six outward, so the convention and the indexing are right and
                        // the reversals are real.
                        // ⛔ the test is SUPPORT, not a centroid: a normal is outward exactly when every
                        // hull vertex is on or behind its own plane. A centroid test needs the centroid to
                        // be interior and says nothing on a sliver.
                        //
                        // ⛔⛔ and the threshold is 1e-3 OF THE PART, not 1e-9. MuJoCo stores mesh vertices
                        // as `float` and its own polygon planes miss its own vertices by up to 2.4e-5 of
                        // the part's size, so at 1e-9 the test reports 11,291 inward normals that are only
                        // storage precision. A normal that is genuinely reversed puts a vertex a whole
                        // polytope width outside its plane, which is nowhere near either number.
                        let v0 = hull.verts[p.verts[0]];
                        let extent = hull.verts.iter().map(|v| v.amax()).fold(0.0, f64::max).max(1e-9);
                        let outward = |nn: [f64; 3]| {
                            let d = nn[0] * v0.x + nn[1] * v0.y + nn[2] * v0.z;
                            let slack = hull.hull_verts.iter().map(|&i| nn[0] * hull.verts[i].x + nn[1] * hull.verts[i].y + nn[2] * hull.verts[i].z - d).fold(f64::NEG_INFINITY, f64::max);
                            slack <= 1e-3 * extent
                        };
                        let (mine_out, theirs_out) = (outward([p.normal.x, p.normal.y, p.normal.z]), outward([n[0], n[1], n[2]]));
                        match (mine_out, theirs_out) {
                            (true, false) => flip_theirs_inward += 1,
                            (false, true) => flip_ours_inward += 1,
                            _ => flip_ambiguous += 1,
                        }
                    }
                    // ⭐ the line the normal lies ON, separately from its sign: a shared face whose
                    // normal points along the same line is a WINDING convention and nothing more, and one
                    // that does not is a different plane through the same vertices — which can only happen
                    // where the polygon is not planar, or where the normal was taken from a degenerate
                    // corner. The two need different fixes, so they are counted apart.
                    if dot.abs() > 0.999_999 {
                        normal_same_line += 1;
                    } else {
                        normal_other_line += 1;
                        if 1.0 - dot.abs() > worst_line {
                            worst_line = 1.0 - dot.abs();
                            worst_line_where = format!("{rel} / {} ({} corners, 1-|n·n'| {:.2e})", m.name, p.verts.len(), 1.0 - dot.abs());
                        }
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
    println!("  of the reversed ones, only OURS points inward: {flip_ours_inward}; only MUJOCO'S does: {flip_theirs_inward} (a MuJoCo defect, not a convention — see the note above); undecided: {flip_ambiguous}");
    println!("  shared faces whose normal lies on the SAME LINE (a winding convention): {normal_same_line}; on a DIFFERENT line (a different plane through the same corners): {normal_other_line}, worst 1-|n·n'| {worst_line:.2e} on {worst_line_where}");
    println!("  same polygon SET (every face, by its vertices): {same_faces}");
    println!("  ⭐⭐ same SUPPORT FUNCTION over 256 directions (the polytope itself, faces and vertex lists aside): {support_same} of {support_tried}; worst {worst_support:.2e} relative to the mesh's own size on {worst_support_where}");
    println!("  ⭐ same set of SUPPORTING PLANES (merge-invariant: the same polytope however its faces are grouped): {plane_same} of {plane_tried}; {plane_only_ours} planes only ours, {plane_only_theirs} only MuJoCo's");
    println!("  same HULL VERTEX SET (the faces aside): {hull_vert_same} of {hull_vert_tried}; {hull_vert_diff} vertices on one hull and not the other ({hull_only_ours} only ours, {hull_only_theirs} only MuJoCo's)");
    println!("  faces both have, by their vertices: {shared_faces}; normals differing by more than 1e-6: {normal_off}, of which OPPOSITE: {normal_flipped}");
    println!("  worst normal disagreement on a shared face that is not simply reversed: {worst_normal:.2e} on {worst_where}");
    if !counts.is_empty() {
        if !different_body.is_empty() {
        println!("  meshes where the two hulls are a DIFFERENT CONVEX BODY, not a different grouping of the same one:");
        for d in different_body.iter().take(12) {
            println!("    {d}");
        }
        if different_body.len() > 12 {
            println!("    … and {} more", different_body.len() - 12);
        }
    }
    println!("  not compared:");
        for (k, n) in &counts {
            println!("    {n:>5}  {k}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
