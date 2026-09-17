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
    let maxabs = points.iter().flat_map(|p| p.iter().map(|c| c.abs())).fold(0.0, f64::max);
    if maxabs == 0.0 {
        return None;
    }

    // ⛔⛔ **Every yes-or-no question here is answered EXACTLY, not within a tolerance.** Quickhull's
    // correctness rests on one invariant: the faces a new point can see form a single edge-connected patch
    // with a simple horizon. An `eps` breaks that invariant rather than softening it — a band of faces the
    // point sees by less than `eps` acts as a wall, the flood across shared edges finds one component, and
    // the faces of the other components SEE THE POINT AND SURVIVE. Nothing revisits them, because the point
    // is now a hull vertex, so those faces stay in the "hull" cutting through it. Measured on Menagerie that
    // left 397 of 662 collision meshes with a vertex outside their own hull, by as much as 38.8 cm, and on
    // `aloha/angled_extrusion` returned 2,681 hull vertices where qhull finds 245.
    //
    // The cure is not a better tolerance. The predicate is a 3×3 determinant of coordinate differences, so
    // it is exact in integers: scale the coordinates so the largest lands at 2^39, round to integers, and
    // evaluate in `i128`. Differences are then below 2^41, a product of three below 2^123, and the twelve
    // terms below 2^127 — no overflow, no rounding, and the sign is the truth. The quantum is 2^-39 of the
    // part's own size (a micron on a kilometre), far finer than the `f32` the mesh arrived in.
    let e = maxabs.log2().ceil() as i32;
    let scale = 2f64.powi(39 - e);
    let q: Vec<[i128; 3]> = points.iter().map(|p| [(p.x * scale).round() as i128, (p.y * scale).round() as i128, (p.z * scale).round() as i128]).collect();
    // `orient(a, b, c, d)`: the sign of `(b−a) · ((c−a) × (d−a))`. Positive means `d` lies on the side the
    // face's outward normal points to — which is exactly "face `abc` sees `d`".
    let orient = |a: usize, b: usize, c: usize, d: usize| -> i32 {
        let (pa, pb, pc, pd) = (q[a], q[b], q[c], q[d]);
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let w = [pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2]];
        let det = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0]);
        match det.cmp(&0) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
        }
    };

    // Seed: four points that are not coplanar. Each is chosen by the widest `f64` margin so the tetrahedron
    // is fat, then CONFIRMED exactly; if the widest candidate is exactly degenerate the scan falls back to
    // any point that is not, so the refusals mean what they say — no 3-D hull exists.
    let p0 = (0..n).min_by(|&a, &b| (q[a][0], q[a][1], q[a][2]).cmp(&(q[b][0], q[b][1], q[b][2]))).unwrap();
    let far = |f: &dyn Fn(usize) -> f64| -> usize { (0..n).max_by(|&a, &b| f(a).total_cmp(&f(b))).unwrap() };
    let p1 = far(&|i| (points[i] - points[p0]).norm());
    let p1 = if q[p1] != q[p0] { p1 } else { (0..n).find(|&i| q[i] != q[p0])? };
    let cross_q = |i: usize| -> [i128; 3] {
        let (a, b, c) = (q[p0], q[p1], q[i]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    };
    let e1 = points[p1] - points[p0];
    let p2 = far(&|i| (points[i] - points[p0]).cross(&e1).norm());
    let p2 = if cross_q(p2) != [0, 0, 0] { p2 } else { (0..n).find(|&i| cross_q(i) != [0, 0, 0])? };
    // ⛔ "Is this a solid?" is a MODELLING question and takes a tolerance; "which side of this plane is this
    // point on?" is a combinatorial question and must not. The two are asked separately here. A plate that
    // is flat in the real numbers is generally NOT flat on the integer lattice — quantising a plane that the
    // lattice does not contain lifts its points by up to half a quantum, each a different way — so an exact
    // predicate alone would hand back a sliver of 10^-12 m with a volume and a support function, instead of
    // refusing. The thickness is therefore measured in `f64`, relative to the part's own size, first.
    let base = e1.cross(&(points[p2] - points[p0]));
    let nrm = base / base.norm();
    let thickness = (0..n).map(|i| (points[i] - points[p0]).dot(&nrm).abs()).fold(0.0, f64::max);
    if thickness <= 1e-10 * maxabs {
        return None;
    }
    let p3 = far(&|i| (points[i] - points[p0]).dot(&nrm).abs());
    let p3 = if orient(p0, p1, p2, p3) != 0 { p3 } else { (0..n).find(|&i| orient(p0, p1, p2, i) != 0)? };

    // A face keeps the points that lie outside it. The stored plane is `f64` and is used ONLY to rank them
    // — which of a face's outside points is farthest — because that choice affects speed, never correctness.
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
    for (a, b, c, opp) in [(p0, p1, p2, p3), (p0, p1, p3, p2), (p0, p2, p3, p1), (p1, p2, p3, p0)] {
        // wind each seed face so the fourth vertex is INSIDE it
        let f = if orient(a, b, c, opp) > 0 { make(a, c, b) } else { make(a, b, c) };
        let v = f.v;
        faces.push(f);
        for k in 0..3 {
            owner.insert((v[k], v[(k + 1) % 3]), faces.len() - 1);
        }
    }

    // book every point against a face that sees it
    for i in 0..n {
        let mut best: Option<(usize, f64)> = None;
        for k in 0..faces.len() {
            if orient(faces[k].v[0], faces[k].v[1], faces[k].v[2], i) > 0 {
                let h = height(&faces[k], i);
                if best.map(|(_, b)| h > b).unwrap_or(true) {
                    best = Some((k, h));
                }
            }
        }
        if let Some((k, _)) = best {
            faces[k].outside.push(i);
        }
    }

    use std::collections::HashSet;
    let mut work: Vec<usize> = (0..faces.len()).collect();
    while let Some(fi) = work.pop() {
        if !faces[fi].alive || faces[fi].outside.is_empty() {
            continue;
        }
        let p = *faces[fi].outside.iter().max_by(|&&a, &&b| height(&faces[fi], a).total_cmp(&height(&faces[fi], b))).unwrap();
        // the faces that see p, flooded across shared edges — exact, so this IS all of them, and a patch
        let mut vis: Vec<usize> = vec![fi];
        let mut seen: HashSet<usize> = HashSet::from([fi]);
        let mut stack = vec![fi];
        while let Some(k) = stack.pop() {
            let v = faces[k].v;
            for e in 0..3 {
                if let Some(&nb) = owner.get(&(v[(e + 1) % 3], v[e]))
                    && seen.insert(nb)
                    && orient(faces[nb].v[0], faces[nb].v[1], faces[nb].v[2], p) > 0
                {
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
        // a point still outside the hull is outside one of the new faces; one that is not is inside for
        // good, because the hull only ever grows
        for i in orphans {
            if let Some(k) = (first_new..faces.len()).find(|&k| orient(faces[k].v[0], faces[k].v[1], faces[k].v[2], i) > 0) {
                faces[k].outside.push(i);
            }
        }
        work.extend(first_new..faces.len());
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
    (tris.len() >= 4 && used.len() >= 4).then_some(TriMesh3 { verts: used, tris })
}

#[cfg(test)]
mod verification {
    use super::*;

    /// **The hull's one obligation: it contains every point it was given, and every face supports it.**
    ///
    /// The adversarial case is a FINELY TESSELLATED FLAT FACE — the shape of nearly every CAD collision
    /// mesh. Its interior points are exactly coplanar with the face, so a tolerance-based visibility test
    /// either admits them as vertices (and builds slivers whose normals are noise) or splits the visible
    /// region into pieces the flood cannot cross. A tessellated box is therefore the sharpest assertion
    /// available: its hull has EXACTLY eight vertices and twelve triangles, whatever the tessellation.
    #[test]
    fn the_hull_of_a_tessellated_box_has_eight_vertices_and_contains_every_point() {
        let mut shape: Option<(usize, usize)> = None;
        for k in [2usize, 5, 11] {
            let mut pts = Vec::new();
            let step = 1.0 / k as f64;
            for i in 0..=k {
                for j in 0..=k {
                    let (u, w) = (i as f64 * step - 0.5, j as f64 * step - 0.5);
                    for s in [-0.5, 0.5] {
                        pts.push(Vector3::new(u, w, s));
                        pts.push(Vector3::new(u, s, w));
                        pts.push(Vector3::new(s, u, w));
                    }
                }
            }
            let h = try_convex_hull_3d(&pts).expect("a box is a solid");
            eprintln!("  k={k}: {} points -> {} hull vertices, {} tris", pts.len(), h.verts.len(), h.tris.len());
            assert!((h.volume() - 1.0).abs() < 1e-12, "k={k}: the hull of a unit box has volume 1, got {}", h.volume());
            let (out, bad) = worst_excursion(&h, &pts);
            assert_eq!(out, 0.0, "k={k}: a point of the box lies outside its own hull");
            assert_eq!(bad, 0, "k={k}: {bad} faces do not support the hull");
            // ⛔ THE ASSERTION THAT BITES: the answer does not grow with the tessellation. 54 points and 864
            // describe the same box, so they must produce the same hull. A visibility test that chases
            // coplanar points returns hundreds of vertices here and its count climbs with `k`.
            match shape {
                None => shape = Some((h.verts.len(), h.tris.len())),
                Some(s) => assert_eq!((h.verts.len(), h.tris.len()), s, "k={k}: the hull changed with the tessellation"),
            }
            // eight corners, plus at most two seed vertices left lying coplanar on a face — the same
            // artefact that has qhull reporting 252 hull vertices for a part whose exact hull has 245
            assert!(h.verts.len() <= 10, "k={k}: {} hull vertices for a box", h.verts.len());
        }
    }

    /// The same obligation on shapes whose hull is not a handful of corners: a sphere (every point extreme),
    /// a cylinder (two flat caps meeting a curved wall), and a deterministic cloud. Checked by the two
    /// properties that do not depend on the tessellation — no input point outside, no face that fails to
    /// support the hull — because the vertex COUNT is an implementation detail and containment is not.
    #[test]
    fn no_point_lies_outside_the_hull_and_no_face_fails_to_support_it() {
        let mut seed = 0x2f6e2b1u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let sphere: Vec<Vector3<f64>> = (0..2000)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / 2000.0;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let t = i as f64 * 2.399963229728653;
                Vector3::new(r * t.cos(), r * t.sin(), z) * 0.37
            })
            .collect();
        let cylinder: Vec<Vector3<f64>> = (0..64)
            .flat_map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 64.0;
                // the caps are tessellated, so their interiors are exactly coplanar
                (0..=8).flat_map(move |r| {
                    let rr = 0.2 * r as f64 / 8.0;
                    [Vector3::new(rr * a.cos(), rr * a.sin(), -0.5), Vector3::new(rr * a.cos(), rr * a.sin(), 0.5)]
                })
            })
            .collect();
        let cloud: Vec<Vector3<f64>> = (0..3000).map(|_| Vector3::new(rand(), rand(), rand())).collect();
        for (name, pts) in [("sphere", &sphere), ("cylinder", &cylinder), ("cloud", &cloud)] {
            let h = try_convex_hull_3d(pts).unwrap_or_else(|| panic!("{name}: is a solid"));
            let (out, bad) = worst_excursion(&h, pts);
            assert!(out == 0.0, "{name}: a point lies {out:.3e} m outside its own hull");
            assert_eq!(bad, 0, "{name}: {bad} of {} faces do not support the hull", h.tris.len());
            eprintln!("  {name}: {} points -> {} hull vertices, {} faces, sound", pts.len(), h.verts.len(), h.tris.len());
        }
    }

    /// `(farthest any point lies outside the hull, faces that are not supporting planes)`. The face planes
    /// are oriented against the hull's own centroid, never trusted from the winding: a normal taken the
    /// wrong way round measures the part's diameter and turns a correct hull into a failure.
    fn worst_excursion(h: &TriMesh3, pts: &[Vector3<f64>]) -> (f64, usize) {
        let centre = h.verts.iter().sum::<Vector3<f64>>() / h.verts.len() as f64;
        let planes: Vec<(Vector3<f64>, f64)> = h
            .tris
            .iter()
            .filter_map(|t| {
                let (a, b, c) = (h.verts[t[0]], h.verts[t[1]], h.verts[t[2]]);
                let raw = (b - a).cross(&(c - a));
                let l = raw.norm();
                (l > 0.0).then(|| {
                    let n = raw / l;
                    let n = if n.dot(&(a - centre)) < 0.0 { -n } else { n };
                    (n, n.dot(&a))
                })
            })
            .collect();
        // the scale the tolerance is relative to: the hull's own reach from its centroid
        let reach = h.verts.iter().map(|v| (v - centre).norm()).fold(0.0, f64::max);
        let tol = 1e-12 * reach;
        let out = pts.iter().map(|p| planes.iter().map(|(n, d)| n.dot(p) - d).fold(f64::NEG_INFINITY, f64::max)).fold(0.0, f64::max);
        let bad = planes.iter().filter(|(n, d)| h.verts.iter().any(|v| n.dot(v) - d > tol)).count();
        (if out > tol { out } else { 0.0 }, bad)
    }

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
