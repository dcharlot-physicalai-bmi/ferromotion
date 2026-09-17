//! **One collision mesh's hull, checked for the only property a hull owes**: does it contain every input
//! vertex? Scratch instrument for the meshes `menagerie_hulls` reports as unsound.
//!
//! ```text
//! cargo run --release --example hull_one -- <menagerie root> <model.xml> <mesh name>
//! ```

use ferromotion_core::tree_from_mjcf;
use nalgebra::Vector3;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = std::path::Path::new(&a[1]);
    let model = root.join(&a[2]);
    let dir = model.parent().unwrap().to_path_buf();
    let xml = std::fs::read_to_string(&model).expect("model");
    let t = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()).expect("loads");
    let h = t.mesh_hulls.get(&a[3]).expect("mesh hull");
    println!("{} verts, {} hull verts, {} hull faces, {} polygons", h.verts.len(), h.hull_verts.len(), h.hull_faces.len(), h.polygons.len());

    let mut centre = Vector3::zeros();
    for &i in &h.hull_verts {
        centre += h.verts[i];
    }
    centre /= h.hull_verts.len() as f64;
    let planes: Vec<(Vector3<f64>, f64)> = h
        .hull_faces
        .iter()
        .filter_map(|f| {
            let (p, q, r) = (h.verts[f[0]], h.verts[f[1]], h.verts[f[2]]);
            let nrm = (q - p).cross(&(r - p));
            let l = nrm.norm();
            (l > 1e-14).then(|| {
                let n = nrm / l;
                let n = if n.dot(&(p - centre)) < 0.0 { -n } else { n };
                (n, n.dot(&p))
            })
        })
        .collect();
    let flipped = h
        .hull_faces
        .iter()
        .filter(|f| {
            let (p, q, r) = (h.verts[f[0]], h.verts[f[1]], h.verts[f[2]]);
            let nrm = (q - p).cross(&(r - p));
            nrm.norm() > 1e-14 && nrm.dot(&(p - centre)) < 0.0
        })
        .count();
    println!("faces whose winding points INWARD: {flipped} of {}", h.hull_faces.len());

    let mut worst = (0.0f64, 0usize);
    for (i, v) in h.verts.iter().enumerate() {
        let mut d = f64::NEG_INFINITY;
        for (n, dd) in &planes {
            d = d.max(n.dot(v) - dd);
        }
        if d > worst.0 {
            worst = (d, i);
        }
    }
    println!("worst vertex {} at {:?} is {:.3e} m outside", worst.1, h.verts[worst.1].as_slice(), worst.0);
    let on_hull = h.hull_verts.binary_search(&worst.1).is_ok();
    println!("is that vertex in hull_verts? {on_hull}");

    // which FACES are not supporting planes: for a convex hull every vertex is on the inner side of every
    // face, so a face with any vertex outside it is a face the hull should not have
    let mut bad = 0usize;
    let mut worst_face = (0.0f64, 0usize, false);
    for (fi, f) in h.hull_faces.iter().enumerate() {
        let (p, q, r) = (h.verts[f[0]], h.verts[f[1]], h.verts[f[2]]);
        let nrm = (q - p).cross(&(r - p));
        let l = nrm.norm();
        if l <= 1e-14 {
            continue;
        }
        let n = nrm / l;
        let inward = n.dot(&(p - centre)) < 0.0;
        let n = if inward { -n } else { n };
        let d = n.dot(&p);
        let mut mx = 0.0f64;
        for &i in &h.hull_verts {
            mx = mx.max(n.dot(&h.verts[i]) - d);
        }
        if mx > 1e-9 {
            bad += 1;
            if mx > worst_face.0 {
                worst_face = (mx, fi, inward);
            }
        }
    }
    println!("faces that are NOT supporting planes: {bad} of {} (worst {:.3e} m, face {}, winding was inward: {})", h.hull_faces.len(), worst_face.0, worst_face.1, worst_face.2);
}
