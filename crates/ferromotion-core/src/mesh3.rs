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
/// subset of the input vertices.
///
/// ⛔ **This PANICS on degenerate input** — fewer than four points, all points equal, all collinear,
/// or all coplanar — and it is public. Prefer [`try_convex_hull_3d`], which returns `None` for each of
/// those and for a non-finite coordinate. This one is kept for callers that have already established
/// their point set is a solid, and it delegates, so the two can never disagree.
///
/// # Panics
///
/// If [`try_convex_hull_3d`] would return `None`.
pub fn convex_hull_3d(points: &[Vector3<f64>]) -> TriMesh3 {
    try_convex_hull_3d(points).expect("convex_hull_3d: the point set is degenerate or non-finite; use try_convex_hull_3d")
}

/// **The convex hull, refusing what it cannot hull.** `None` for fewer than four points, for a
/// non-finite coordinate, and for a point set that is degenerate — all equal, collinear, or coplanar —
/// because a flat point set has no 3-D hull and any answer for it would be a lie with a volume.
///
/// A degenerate part is a real case, not a hypothetical: a mesh exported as a single flat plate, or a
/// decomposition that produced a sliver, arrives here looking like data.
pub fn try_convex_hull_3d(points: &[Vector3<f64>]) -> Option<TriMesh3> {
    let n = points.len();
    if n < 4 || !points.iter().all(|p| p.iter().all(|c| c.is_finite())) {
        return None;
    }
    let eps = 1e-9;
    // every coordinate was just checked finite, so `total_cmp` orders these the way `partial_cmp` would;
    // it is used because it cannot panic, not to define an order for a `NaN`

    // seed: four affinely-independent points, each the most extreme available so a thin part still seeds
    // (the refusals are the same as testing any point: the extreme one passes iff some point passes)
    let p0 = (0..n).min_by(|&a, &b| points[a].x.total_cmp(&points[b].x)).unwrap();
    let p1 = (0..n).max_by(|&a, &b| (points[a] - points[p0]).norm().total_cmp(&(points[b] - points[p0]).norm())).unwrap();
    if (points[p1] - points[p0]).norm() <= eps {
        return None;
    }
    let e1 = points[p1] - points[p0];
    let p2 = (0..n).max_by(|&a, &b| (points[a] - points[p0]).cross(&e1).norm().total_cmp(&(points[b] - points[p0]).cross(&e1).norm())).unwrap();
    if (points[p2] - points[p0]).cross(&e1).norm() <= eps {
        return None;
    }
    let nrm = e1.cross(&(points[p2] - points[p0]));
    let p3 = (0..n).max_by(|&a, &b| (points[a] - points[p0]).dot(&nrm).abs().total_cmp(&(points[b] - points[p0]).dot(&nrm).abs())).unwrap();
    if (points[p3] - points[p0]).dot(&nrm).abs() <= eps {
        return None;
    }

    // Quickhull: every live face keeps the points outside it; the farthest one is added, the faces it sees
    // (found by walking neighbours through a directed-edge table, so the cost is the visible patch, not
    // the hull) are replaced through their horizon, and their outside points are handed to the new faces.
    //
    // ⛔ Handing a deleted face's points to the NEW faces alone loses points, and a lost point is a hole in
    // the hull, not a rounding error: a point can lie outside two faces at once, be booked against the one
    // that is deleted, and be visible from none of the cone's new faces. Measured on Menagerie that left a
    // vertex 5.4 cm outside the hull of a 4 cm part. So the pass below runs to a FIXPOINT: after the queue
    // drains, every point is re-tested against every surviving face, and any point still outside re-enters.
    // The first pass builds essentially the whole hull, the second finds what it dropped, the third confirms
    // nothing is left — and the loop cannot end while a point lies outside. The hull only ever grows, so a
    // point found inside it is inside for good and is never re-tested: that makes the re-tests amortize to a
    // single sweep of points against faces rather than one per pass.
    // ⛔ The plane normal is stored NORMALIZED. With the raw cross product the visibility test scales with
    // twice the triangle's area, so on a finely tessellated part — where facets are square millimetres —
    // a facet is blind to points a real distance outside it, and the "hull" it returns cuts through the
    // mesh. `eps` is a distance in metres only once the normal is a unit vector.
    struct Face {
        v: [usize; 3],
        n: Vector3<f64>,
        d: f64,
        outside: Vec<usize>,
        alive: bool,
    }
    let make = |a: usize, b: usize, c: usize| -> Face {
        let raw = (points[b] - points[a]).cross(&(points[c] - points[a]));
        let l = raw.norm();
        let n = if l > 0.0 { raw / l } else { raw };
        Face { v: [a, b, c], n, d: n.dot(&points[a]), outside: Vec::new(), alive: true }
    };
    let height = |f: &Face, i: usize| f.n.dot(&points[i]) - f.d;
    let mut faces: Vec<Face> = Vec::new();
    use std::collections::HashMap;
    // owner of each directed edge (a, b) among the live faces; the twin (b, a) belongs to the neighbour
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    let centroid = (points[p0] + points[p1] + points[p2] + points[p3]) / 4.0;
    for (a, b, c) in [(p0, p1, p2), (p0, p1, p3), (p0, p2, p3), (p1, p2, p3)] {
        let f = make(a, b, c);
        // orient outward, away from the tetrahedron's centroid
        let f = if f.n.dot(&(centroid - points[a])) > 0.0 { make(a, c, b) } else { f };
        let v = f.v;
        faces.push(f);
        for k in 0..3 {
            owner.insert((v[k], v[(k + 1) % 3]), faces.len() - 1);
        }
    }
    use std::collections::HashSet;
    let mut work: Vec<usize> = Vec::new();
    // a point that has become a hull vertex is on the hull and never outside it again
    let mut settled = vec![false; n];
    for i in [p0, p1, p2, p3] {
        settled[i] = true;
    }
    loop {
        // (re)book every point that still lies outside the hull against the face it is farthest outside of
        let live: Vec<usize> = (0..faces.len()).filter(|&i| faces[i].alive).collect();
        let mut any = false;
        for i in 0..n {
            if settled[i] {
                continue;
            }
            let mut best: Option<(usize, f64)> = None;
            for &k in &live {
                let h = height(&faces[k], i);
                if h > eps && best.map(|(_, b)| h > b).unwrap_or(true) {
                    best = Some((k, h));
                }
            }
            match best {
                Some((k, _)) => {
                    faces[k].outside.push(i);
                    any = true;
                }
                None => settled[i] = true,
            }
        }
        if !any {
            break;
        }
        work.clear();
        work.extend(live);
        run_queue(&mut faces, &mut owner, &mut work, &mut settled, points, eps);
        for f in faces.iter_mut() {
            f.outside.clear();
        }
    }

    // compact to used vertices
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut used = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for f in faces.iter().filter(|f| f.alive) {
        let mut t = [0usize; 3];
        for (k, &v) in f.v.iter().enumerate() {
            t[k] = *remap.entry(v).or_insert_with(|| {
                used.push(points[v]);
                used.len() - 1
            });
        }
        tris.push(t);
    }
    // A hull with no faces, or fewer than four vertices, is not a solid — refuse rather than return an
    // object whose volume is zero and whose support function is a point.
    return (tris.len() >= 4 && used.len() >= 4).then_some(TriMesh3 { verts: used, tris });

    /// Drain the queue: each face with points outside it grows the hull by its farthest one.
    fn run_queue(faces: &mut Vec<Face>, owner: &mut HashMap<(usize, usize), usize>, work: &mut Vec<usize>, settled: &mut [bool], points: &[Vector3<f64>], eps: f64) {
    let make = |a: usize, b: usize, c: usize| -> Face {
        let raw = (points[b] - points[a]).cross(&(points[c] - points[a]));
        let l = raw.norm();
        let n = if l > 0.0 { raw / l } else { raw };
        Face { v: [a, b, c], n, d: n.dot(&points[a]), outside: Vec::new(), alive: true }
    };
    let height = |f: &Face, i: usize| f.n.dot(&points[i]) - f.d;
    while let Some(fi) = work.pop() {
        if !faces[fi].alive || faces[fi].outside.is_empty() {
            continue;
        }
        let p = *faces[fi].outside.iter().max_by(|&&a, &&b| height(&faces[fi], a).total_cmp(&height(&faces[fi], b))).unwrap();
        settled[p] = true;
        // the faces that see p: flood from this one across shared edges
        let mut vis: Vec<usize> = vec![fi];
        let mut seen: HashSet<usize> = HashSet::from([fi]);
        let mut stack = vec![fi];
        while let Some(k) = stack.pop() {
            let v = faces[k].v;
            for e in 0..3 {
                if let Some(&nb) = owner.get(&(v[(e + 1) % 3], v[e])).filter(|&&nb| seen.insert(nb) && height(&faces[nb], p) > eps) {
                    vis.push(nb);
                    stack.push(nb);
                }
            }
        }
        let visset: HashSet<usize> = vis.iter().copied().collect();
        // horizon: directed edges of visible faces whose twin belongs to a face that does not see p
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for &k in &vis {
            let v = faces[k].v;
            for e in 0..3 {
                let (a, b) = (v[e], v[(e + 1) % 3]);
                if owner.get(&(b, a)).map(|nb| !visset.contains(nb)).unwrap_or(true) {
                    horizon.push((a, b));
                }
            }
        }
        let mut orphans: Vec<usize> = Vec::new();
        for &k in &vis {
            faces[k].alive = false;
            let v = faces[k].v;
            for e in 0..3 {
                owner.remove(&(v[e], v[(e + 1) % 3]));
            }
            orphans.extend(faces[k].outside.drain(..).filter(|&i| i != p));
        }
        let first_new = faces.len();
        for (a, b) in horizon {
            faces.push(make(a, b, p));
            let idx = faces.len() - 1;
            owner.insert((a, b), idx);
            owner.insert((b, p), idx);
            owner.insert((p, a), idx);
        }
        for i in orphans {
            if let Some(f) = faces[first_new..].iter_mut().find(|f| height(f, i) > eps) {
                f.outside.push(i);
            }
        }
        work.extend(first_new..faces.len());
    }
    }
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
