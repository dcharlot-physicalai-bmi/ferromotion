//! **3-D triangle-mesh processing** — surface area, enclosed volume and centroid (by the divergence
//! theorem), and an incremental **3-D convex hull**. The collision and sensor layers consume
//! meshes; this is the geometry-processing the tree lacked beyond 2-D hull / OBB. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::Vector3;

/// A 3-D triangle mesh (indexed).
#[derive(Clone, Debug, Default)]
pub struct TriMesh3 {
    pub verts: Vec<Vector3<f64>>,
    pub tris: Vec<[usize; 3]>,
}

impl TriMesh3 {
    /// Total surface area (sum of triangle areas).
    pub fn surface_area(&self) -> f64 {
        self.tris
            .iter()
            .map(|t| {
                let (a, b, c) = (self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]);
                0.5 * (b - a).cross(&(c - a)).norm()
            })
            .sum()
    }

    /// Signed enclosed volume by the divergence theorem: `V = ⅙ Σ (v₀ · (v₁ × v₂))` over triangles
    /// (positive for outward-facing winding).
    pub fn volume(&self) -> f64 {
        let v: f64 = self
            .tris
            .iter()
            .map(|t| {
                let (a, b, c) = (self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]);
                a.dot(&b.cross(&c))
            })
            .sum();
        v / 6.0
    }

    /// Volume-weighted centroid of the enclosed solid.
    pub fn centroid(&self) -> Vector3<f64> {
        let mut c = Vector3::zeros();
        let mut vol = 0.0;
        for t in &self.tris {
            let (a, b, cc) = (self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]);
            let v = a.dot(&b.cross(&cc)) / 6.0;
            c += v * (a + b + cc) / 4.0;
            vol += v;
        }
        if vol.abs() < 1e-30 {
            Vector3::zeros()
        } else {
            c / vol
        }
    }
}

/// Incremental 3-D convex hull of a point set. Returns a closed, outward-wound triangle mesh over a
/// subset of the input vertices. Assumes the points are not all coplanar.
pub fn convex_hull_3d(points: &[Vector3<f64>]) -> TriMesh3 {
    let n = points.len();
    assert!(n >= 4, "need at least 4 points for a 3-D hull");
    let eps = 1e-9;

    // seed: find 4 affinely-independent points
    let p0 = 0;
    let p1 = (1..n).find(|&i| (points[i] - points[p0]).norm() > eps).expect("degenerate: all points equal");
    let e1 = points[p1] - points[p0];
    let p2 = (0..n).find(|&i| (points[i] - points[p0]).cross(&e1).norm() > eps).expect("degenerate: collinear");
    let nrm = (points[p1] - points[p0]).cross(&(points[p2] - points[p0]));
    let p3 = (0..n).find(|&i| (points[i] - points[p0]).dot(&nrm).abs() > eps).expect("degenerate: coplanar");

    // initial tetrahedron faces, oriented outward
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let add_face = |faces: &mut Vec<[usize; 3]>, a: usize, b: usize, c: usize, inside: Vector3<f64>| {
        let n = (points[b] - points[a]).cross(&(points[c] - points[a]));
        if n.dot(&(inside - points[a])) > 0.0 {
            faces.push([a, c, b]); // flip so the normal points away from `inside`
        } else {
            faces.push([a, b, c]);
        }
    };
    let centroid = (points[p0] + points[p1] + points[p2] + points[p3]) / 4.0;
    add_face(&mut faces, p0, p1, p2, centroid);
    add_face(&mut faces, p0, p1, p3, centroid);
    add_face(&mut faces, p0, p2, p3, centroid);
    add_face(&mut faces, p1, p2, p3, centroid);

    let face_normal = |f: &[usize; 3]| (points[f[1]] - points[f[0]]).cross(&(points[f[2]] - points[f[0]]));
    let visible = |f: &[usize; 3], p: Vector3<f64>| face_normal(f).dot(&(p - points[f[0]])) > eps;

    for (i, &p) in points.iter().enumerate() {
        if faces.iter().flatten().any(|&v| v == i) {
            continue;
        }
        // faces visible from p
        let vis: Vec<usize> = (0..faces.len()).filter(|&fi| visible(&faces[fi], p)).collect();
        if vis.is_empty() {
            continue; // inside the current hull
        }
        // horizon: edges on exactly one visible face
        use std::collections::HashMap;
        let mut edge_count: HashMap<(usize, usize), i32> = HashMap::new();
        for &fi in &vis {
            let f = faces[fi];
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        // collect horizon edges *with orientation* from their (single) visible face
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &fi in &vis {
            let f = faces[fi];
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                if edge_count[&(a.min(b), a.max(b))] == 1 {
                    horizon.push((a, b));
                }
            }
        }
        // remove visible faces
        let visset: std::collections::HashSet<usize> = vis.into_iter().collect();
        faces = faces.iter().enumerate().filter(|(fi, _)| !visset.contains(fi)).map(|(_, f)| *f).collect();
        // add new faces from p to each horizon edge (orientation inherited → outward)
        for (a, b) in horizon {
            faces.push([a, b, i]);
        }
    }

    let mut verts = points.to_vec();
    // compact to used vertices
    use std::collections::HashMap;
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut used = Vec::new();
    for f in &faces {
        for &v in f {
            remap.entry(v).or_insert_with(|| {
                used.push(verts[v]);
                used.len() - 1
            });
        }
    }
    let tris = faces.iter().map(|f| [remap[&f[0]], remap[&f[1]], remap[&f[2]]]).collect();
    verts = used;
    TriMesh3 { verts, tris }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// Area / volume / centroid of a unit-cube mesh match the analytic values.
    #[test]
    fn cube_area_volume_centroid() {
        // unit cube [0,1]^3, 12 triangles, outward winding
        let v: Vec<Vector3<f64>> = [
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ]
        .iter()
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
        let f = |q: [usize; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]];
        let mut tris = Vec::new();
        tris.extend(f([0, 3, 2, 1])); // bottom (−z)
        tris.extend(f([4, 5, 6, 7])); // top (+z)
        tris.extend(f([0, 1, 5, 4])); // −y
        tris.extend(f([2, 3, 7, 6])); // +y
        tris.extend(f([1, 2, 6, 5])); // +x
        tris.extend(f([0, 4, 7, 3])); // −x
        let m = TriMesh3 { verts: v, tris };
        eprintln!("cube: area {:.4} vol {:.4} centroid {:?}", m.surface_area(), m.volume(), m.centroid().as_slice());
        assert!((m.surface_area() - 6.0).abs() < 1e-12, "area {}", m.surface_area());
        assert!((m.volume() - 1.0).abs() < 1e-12, "volume {}", m.volume());
        assert!((m.centroid() - Vector3::new(0.5, 0.5, 0.5)).norm() < 1e-12, "centroid off");
    }

    /// 3-D convex hull: the hull of random points inside a cube contains every input point and its
    /// volume approaches the cube's as points fill it; the hull of the 8 cube corners is the cube.
    #[test]
    fn convex_hull_contains_points_and_measures() {
        // hull of the 8 unit-cube corners == cube (volume 1)
        let corners: Vec<Vector3<f64>> = [
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ]
        .iter()
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
        let hull = convex_hull_3d(&corners);
        eprintln!("cube-corner hull: {} tris, volume {:.4}", hull.tris.len(), hull.volume().abs());
        assert!((hull.volume().abs() - 1.0).abs() < 1e-9, "hull volume {}", hull.volume().abs());

        // random cloud → every input point is inside/on the hull
        let mut s = 0xBEEFu64;
        let mut rnd = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            ((z ^ (z >> 31)) as f64) / (u64::MAX as f64)
        };
        let pts: Vec<Vector3<f64>> = (0..200).map(|_| Vector3::new(rnd(), rnd(), rnd())).collect();
        let hull = convex_hull_3d(&pts);
        // signed distance to each face must be ≤ 0 for every point (inside a convex, outward-wound hull)
        let mut worst = f64::NEG_INFINITY;
        for &p in &pts {
            for t in &hull.tris {
                let (a, b, c) = (hull.verts[t[0]], hull.verts[t[1]], hull.verts[t[2]]);
                let nrm = (b - a).cross(&(c - a)).normalize();
                worst = worst.max(nrm.dot(&(p - a)));
            }
        }
        eprintln!("random hull: {} tris, worst point-outside-face signed dist {worst:.2e}", hull.tris.len());
        assert!(worst < 1e-9, "a point lies outside the hull: {worst}");
    }
}
