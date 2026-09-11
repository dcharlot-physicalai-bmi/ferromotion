//! **Solid voxelisation of a triangle mesh** — the interior, not just the skin.
//!
//! # Why this module exists, and what was measured before it
//!
//! ⛔ Approximate convex decomposition was attempted twice in this workspace WITHOUT a volumetric
//! representation, and both attempts were measured to fail. The record is on [`CompoundHull`], whose
//! module carries it in full; the short form is that both partitioned the TRIANGLE LIST and hulled
//! each group:
//!
//! * k-means over triangle centroids made the volume sum **worse than not decomposing** — 1.64x the
//!   mesh's own volume for a single hull, then 3.00x at two parts, 2.41x at three, 2.70x at four. A
//!   group of triangles that are near each other ON THE SURFACE wraps around the solid, so its hull
//!   spans the whole shape.
//! * recursive median split, demanding solid parts, **never opened the channel** of a U-bracket at any
//!   part count from 1 to 24, because splitting two facing walls apart requires a cut whose halves are
//!   each flat, and a 3-D hull of a flat patch does not exist.
//! * the same, allowing flat parts, opened the channel and produced a **hollow shell**: 28 of 32 parts
//!   flat, and a convex CUBE at 12 parts reported 100% of its hull interior excluded.
//!
//! **A surface partition is not a solid decomposition.** V-HACD and CoACD voxelise the interior and
//! work on voxel clusters, which is why they work. This module is that missing piece.
//!
//! # How the interior is found without a ray cast
//!
//! Two passes, and the second is what makes it solid:
//!
//! 1. **Surface**: every voxel a triangle actually overlaps, by the separating-axis test of
//!    Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing* (2001) — 13 axes: the three box normals,
//!    the triangle normal, and the nine edge-pair cross products. Exact, not sampled.
//! 2. **Flood fill from the boundary** through non-surface voxels marks the EXTERIOR. Everything that
//!    is neither surface nor exterior is interior.
//!
//! ⭐ That ordering is why a mesh made by concatenating overlapping primitives voxelises correctly
//! without any repair step: internal faces mark extra surface voxels, which are solid anyway, and the
//! fill still cannot reach the enclosed region. A ray-parity test would have to care about them.
//!
//! ⛔ **It follows that a mesh with a hole in it fills the whole grid**, because the fill leaks inside.
//! [`SolidVoxels::leaked`] reports exactly that rather than returning a confidently wrong volume.
//!
//! # ⛔ The volume is an UPPER bound, and by a measured amount
//!
//! A voxel a triangle merely clips is counted whole, so the solid set always covers the mesh and never
//! misses part of it. That is the right bias for collision — a decomposition built on it cannot leave a
//! piece of the object unrepresented — and it is the wrong one for mass. Measured on a cube:
//!
//! | cells across | 8 | 16 | 32 | 64 |
//! |---|---|---|---|---|
//! | volume error | +42.4% | +20.0% | +9.7% | +4.8% |
//!
//! Each refinement roughly halves it: the error is first order in the cell size, as a one-cell skin
//! over a fixed surface area must be. `a_cube_voxelises_conservatively_and_the_error_is_first_order`
//! asserts the RATE rather than a tolerance, because the rate is the property and any single
//! tolerance is an accident of the resolution chosen.
//!
//! ⭐ **For a true volume use [`TriMesh3::volume`], which integrates it exactly** by the divergence
//! theorem and costs nothing. This grid is for deciding what is solid WHERE, not how much there is.

use crate::mesh3::TriMesh3;
#[cfg(doc)]
use crate::CompoundHull;
use nalgebra::Vector3;

/// A mesh's solid occupancy on a uniform grid.
#[derive(Clone, Debug)]
pub struct SolidVoxels {
    /// World position of the centre of cell `(0, 0, 0)`.
    pub origin: Vector3<f64>,
    /// Edge length of a cell, metres.
    pub cell: f64,
    pub dims: [usize; 3],
    /// `true` where the mesh is solid — surface or interior.
    solid: Vec<bool>,
    /// ⛔ Set when the exterior flood reached a cell that should have been enclosed, i.e. the surface
    /// did not seal. The volume of a leaked grid is meaningless and every consumer must check this.
    pub leaked: bool,
}

/// Triangle versus axis-aligned box, by the separating-axis theorem (Akenine-Möller 2001).
///
/// `half` is the box half-extent and the triangle is given RELATIVE to the box centre. Thirteen axes:
/// three box normals, the triangle normal, and the nine cross products of a triangle edge with a box
/// axis. A test that skipped the nine edge axes would accept a triangle that slices a corner region
/// without touching the box, which is the case a coarse grid hits most often.
fn tri_box_overlap(v: [Vector3<f64>; 3], half: Vector3<f64>) -> bool {
    // 1. the three box normals
    for a in 0..3 {
        let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(l, h), p| (l.min(p[a]), h.max(p[a])));
        if lo > half[a] || hi < -half[a] {
            return false;
        }
    }
    // 2. the triangle normal, as a plane-box test
    let (e0, e1) = (v[1] - v[0], v[2] - v[0]);
    let n = e0.cross(&e1);
    let r = half.x * n.x.abs() + half.y * n.y.abs() + half.z * n.z.abs();
    let d = n.dot(&v[0]);
    if d.abs() > r {
        return false;
    }
    // 3. the nine edge cross products
    let edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
    for e in edges {
        for a in 0..3 {
            let mut axis = Vector3::zeros();
            axis[a] = 1.0;
            let ax = e.cross(&axis);
            if ax.norm_squared() < 1e-24 {
                continue; // degenerate: the edge is parallel to this box axis, no separation possible
            }
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for p in &v {
                let t = ax.dot(p);
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let rr = half.x * ax.x.abs() + half.y * ax.y.abs() + half.z * ax.z.abs();
            if lo > rr || hi < -rr {
                return false;
            }
        }
    }
    true
}

impl SolidVoxels {
    /// Voxelise `mesh` so its longest bounding-box side spans about `cells_across` cells.
    ///
    /// `None` for an empty mesh or a degenerate bound. The grid is padded by one cell on every side so
    /// the exterior flood always has somewhere to start — without that pad a mesh touching the grid
    /// wall would have no seed and the whole grid would read as interior.
    pub fn from_mesh(mesh: &TriMesh3, cells_across: usize) -> Option<SolidVoxels> {
        if mesh.tris.is_empty() || mesh.verts.is_empty() || cells_across == 0 {
            return None;
        }
        let (mut lo, mut hi) = (Vector3::repeat(f64::MAX), Vector3::repeat(f64::MIN));
        for p in &mesh.verts {
            if !p.iter().all(|c| c.is_finite()) {
                return None;
            }
            lo = lo.inf(p);
            hi = hi.sup(p);
        }
        let span = hi - lo;
        let longest = span.x.max(span.y).max(span.z);
        if longest <= 0.0 || !longest.is_finite() {
            return None;
        }
        let cell = longest / cells_across as f64;
        let pad = 1;
        let dim = |s: f64| ((s / cell).ceil() as usize) + 1 + 2 * pad;
        let dims = [dim(span.x), dim(span.y), dim(span.z)];
        let n = dims[0].checked_mul(dims[1])?.checked_mul(dims[2])?;
        if n > 40_000_000 {
            return None; // a grid this size is a caller mistake, not something to spend a minute on
        }
        let origin = lo - Vector3::repeat(pad as f64 * cell);

        // ---- pass 1: surface ----
        let mut surface = vec![false; n];
        let idx = |i: usize, j: usize, k: usize| (k * dims[1] + j) * dims[0] + i;
        let half = Vector3::repeat(cell * 0.5);
        for t in &mesh.tris {
            let tv = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
            let (mut tlo, mut thi) = (Vector3::repeat(f64::MAX), Vector3::repeat(f64::MIN));
            for p in &tv {
                tlo = tlo.inf(p);
                thi = thi.sup(p);
            }
            let c0 = |a: usize| (((tlo[a] - origin[a]) / cell - 0.5).floor().max(0.0)) as usize;
            let c1 = |a: usize| ((((thi[a] - origin[a]) / cell + 0.5).ceil()) as usize).min(dims[a] - 1);
            for k in c0(2)..=c1(2) {
                for j in c0(1)..=c1(1) {
                    for i in c0(0)..=c1(0) {
                        let c = origin + Vector3::new(i as f64, j as f64, k as f64) * cell;
                        if tri_box_overlap([tv[0] - c, tv[1] - c, tv[2] - c], half) {
                            surface[idx(i, j, k)] = true;
                        }
                    }
                }
            }
        }

        // ---- pass 2: flood the exterior, everything else is inside ----
        let mut exterior = vec![false; n];
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();
        let seed = |i: usize, j: usize, k: usize, ex: &mut Vec<bool>, st: &mut Vec<_>| {
            let id = idx(i, j, k);
            if !surface[id] && !ex[id] {
                ex[id] = true;
                st.push((i, j, k));
            }
        };
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    if i == 0 || j == 0 || k == 0 || i + 1 == dims[0] || j + 1 == dims[1] || k + 1 == dims[2] {
                        seed(i, j, k, &mut exterior, &mut stack);
                    }
                }
            }
        }
        while let Some((i, j, k)) = stack.pop() {
            let visit = |ni: usize, nj: usize, nk: usize, st: &mut Vec<(usize, usize, usize)>, ex: &mut Vec<bool>| {
                let id = idx(ni, nj, nk);
                if !surface[id] && !ex[id] {
                    ex[id] = true;
                    st.push((ni, nj, nk));
                }
            };
            if i > 0 { visit(i - 1, j, k, &mut stack, &mut exterior); }
            if j > 0 { visit(i, j - 1, k, &mut stack, &mut exterior); }
            if k > 0 { visit(i, j, k - 1, &mut stack, &mut exterior); }
            if i + 1 < dims[0] { visit(i + 1, j, k, &mut stack, &mut exterior); }
            if j + 1 < dims[1] { visit(i, j + 1, k, &mut stack, &mut exterior); }
            if k + 1 < dims[2] { visit(i, j, k + 1, &mut stack, &mut exterior); }
        }

        let solid: Vec<bool> = (0..n).map(|id| surface[id] || !exterior[id]).collect();
        // ⛔ A sealed mesh leaves an enclosed region the flood cannot reach, so SOME cell must be
        // interior-but-not-surface. If none is, the surface did not seal and the fill leaked.
        let interior_found = (0..n).any(|id| !surface[id] && !exterior[id]);
        Some(SolidVoxels { origin, cell, dims, solid, leaked: !interior_found })
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> bool {
        self.solid[(k * self.dims[1] + j) * self.dims[0] + i]
    }

    /// Number of solid cells.
    pub fn count(&self) -> usize {
        self.solid.iter().filter(|&&b| b).count()
    }

    /// Solid volume, `count · cell³`. ⛔ Meaningless when [`leaked`](Self::leaked) is set.
    pub fn volume(&self) -> f64 {
        self.count() as f64 * self.cell.powi(3)
    }

    /// World centre of every solid cell.
    pub fn centres(&self) -> Vec<Vector3<f64>> {
        let mut out = Vec::with_capacity(self.count());
        for k in 0..self.dims[2] {
            for j in 0..self.dims[1] {
                for i in 0..self.dims[0] {
                    if self.get(i, j, k) {
                        out.push(self.origin + Vector3::new(i as f64, j as f64, k as f64) * self.cell);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_geometry::{primitive_mesh, LinkGeometry};

    /// Translate a mesh and append it, so several primitives become one concave soup.
    fn union(parts: &[(LinkGeometry, Vector3<f64>)]) -> TriMesh3 {
        let mut out = TriMesh3 { verts: Vec::new(), tris: Vec::new() };
        for (g, at) in parts {
            let m = primitive_mesh(g, 24).expect("a primitive meshes");
            let base = out.verts.len();
            out.verts.extend(m.verts.iter().map(|v| v + at));
            out.tris.extend(m.tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        out
    }

    fn cube(side: f64) -> TriMesh3 {
        union(&[(LinkGeometry::Box { size: Vector3::repeat(side) }, Vector3::zeros())])
    }

    /// **The nine edge axes are load-bearing, and no mesh fixture in this module reaches them.**
    ///
    /// ⛔ This test exists because a mutation survived without it. Deleting the nine edge cross
    /// products from [`tri_box_overlap`] — keeping only the three box normals and the triangle plane —
    /// left every other test in this module green. Every fixture here is an axis-aligned box or a
    /// sphere dense enough that some triangle always hits a face test, so the corner-slicing case is
    /// simply unreachable through them.
    ///
    /// It is not a rare case. Searching two million random triangles against a unit box, **110,030 of
    /// them — 5.5% — are separated ONLY by an edge axis**: their bounding box overlaps, their plane
    /// passes within range of the centre, and they still miss. Without those axes a voxeliser marks a
    /// skin of cells the surface never touches, which for a decomposition means parts that reach into
    /// space the object does not occupy.
    ///
    /// The triangle below is the first one that search turned up.
    #[test]
    fn the_edge_axes_reject_a_triangle_the_coarse_tests_accept() {
        let half = Vector3::repeat(1.0);
        // exactly the coarse half of `tri_box_overlap`: box normals and the triangle plane
        let coarse = |v: [Vector3<f64>; 3]| {
            for a in 0..3 {
                let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(l, h), p: &Vector3<f64>| (l.min(p[a]), h.max(p[a])));
                if lo > half[a] || hi < -half[a] {
                    return false;
                }
            }
            let n = (v[1] - v[0]).cross(&(v[2] - v[0]));
            let r = half.x * n.x.abs() + half.y * n.y.abs() + half.z * n.z.abs();
            n.dot(&v[0]).abs() <= r
        };

        let corner_slicer = [
            Vector3::new(2.837505, 1.902975, -2.079223),
            Vector3::new(-0.251445, 0.203314, -1.723676),
            Vector3::new(0.022517, 2.361766, -0.156889),
        ];
        assert!(coarse(corner_slicer), "the fixture is only interesting if the coarse tests ACCEPT it");
        assert!(
            !tri_box_overlap(corner_slicer, half),
            "the edge axes must reject this triangle; without them the voxeliser marks cells nothing touches"
        );

        // the positive control: a triangle straight through the centre must still be accepted, or
        // "rejects" above could be achieved by rejecting everything
        let through = [
            Vector3::new(-2.0, -2.0, 0.0),
            Vector3::new(2.0, -2.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
        ];
        assert!(tri_box_overlap(through, half), "a triangle through the box centre overlaps it");
        eprintln!("  SAT: the corner-slicing triangle is accepted by the coarse tests and rejected by the edge axes");
    }

    /// **The volume is a conservative OVER-estimate and its error is first order in the cell.**
    ///
    /// ⛔ An earlier version of this test asserted the volume against `side³` within a tolerance and
    /// failed at +4.8%. The code was right and the oracle was wrong: a voxel a triangle clips is
    /// counted whole, so a solid voxelisation of anything overstates it by roughly a one-cell skin over
    /// the surface area. That is a FEATURE for collision work — the solid set never misses a piece of
    /// the object — and it means no single tolerance is meaningful, because the number depends entirely
    /// on the resolution asked for.
    ///
    /// What IS meaningful is the rate. A skin one cell thick over a fixed area is `O(h)`, so halving
    /// the cell must halve the error. Measured: 0.4238, 0.1995, 0.0967, 0.0476 — ratios 2.12, 2.06,
    /// 2.03. A surface pass that dropped cells, or double-counted them, would not fall on that line.
    #[test]
    fn a_cube_voxelises_conservatively_and_the_error_is_first_order() {
        let side = 0.4_f64;
        let truth = side.powi(3);
        let mut errs = Vec::new();
        for n in [8usize, 16, 32, 64] {
            let v = SolidVoxels::from_mesh(&cube(side), n).expect("a cube voxelises");
            assert!(!v.leaked, "a closed cube must not leak at n = {n}");
            let rel = v.volume() / truth - 1.0;
            assert!(rel > 0.0, "n = {n}: the voxelisation must OVER-estimate, got {rel:+.4}");
            errs.push(rel);
        }
        let ratios: Vec<f64> = errs.windows(2).map(|w| (w[0] / w[1] * 100.0).round() / 100.0).collect();
        eprintln!("  cube {side} m: errors {:?}, halving ratios {ratios:?}", errs.iter().map(|e| (e * 1e4).round() / 1e4).collect::<Vec<_>>());
        for (i, r) in ratios.iter().enumerate() {
            assert!(
                (1.8..=2.3).contains(r),
                "doubling the resolution must roughly halve the error; step {i} gave a ratio of {r}"
            );
        }
    }

    /// A sphere is the harder convergence case: every voxel on the surface is a partial cell, so the
    /// error is pure discretisation rather than an axis-aligned coincidence.
    #[test]
    fn a_sphere_voxelises_to_its_closed_form_volume() {
        let r = 0.25_f64;
        let truth = 4.0 / 3.0 * core::f64::consts::PI * r.powi(3);
        let m = primitive_mesh(&LinkGeometry::Sphere { radius: r }, 48).expect("a sphere meshes");
        let v = SolidVoxels::from_mesh(&m, 48).expect("it voxelises");
        assert!(!v.leaked, "a closed sphere must not leak");
        let rel = v.volume() / truth - 1.0;
        eprintln!("  sphere r = {r} at 48 cells: voxel volume {:.6} vs 4/3 pi r^3 {:.6}, {:+.2}%", v.volume(), truth, rel * 100.0);
        // conservative, and within the one-cell skin the cube test pins the rate of
        assert!(rel > 0.0, "the voxelisation must over-estimate, got {rel:+.4}");
        assert!(rel < 0.15, "sphere over-estimate {:.2}% is larger than a one-cell skin explains", rel * 100.0);
    }

    /// **THE PROPERTY BOTH FAILED ATTEMPTS LACKED: the channel of a U is EMPTY.**
    ///
    /// ⛔ A surface partition could not express this — the median split never opened the channel at any
    /// part count from 1 to 24, and k-means made the volume worse than not decomposing at all. Here it
    /// is a direct consequence of flood-filling the exterior: the mouth of the U is open to the
    /// outside, so the fill walks in and the cells are never marked solid.
    ///
    /// The mesh is three overlapping boxes concatenated — internal faces and all, with no repair step
    /// — which is the shape a real exporter emits.
    #[test]
    fn the_channel_of_a_u_bracket_is_empty_and_its_volume_is_the_three_boxes() {
        // the same bracket `compound.rs` uses: base slab plus two walls, channel open along +z
        let u = union(&[
            (LinkGeometry::Box { size: Vector3::new(1.2, 0.8, 0.4) }, Vector3::new(0.0, 0.0, 0.0)),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(-0.5, 0.0, 0.5)),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(0.5, 0.0, 0.5)),
        ]);
        let v = SolidVoxels::from_mesh(&u, 48).expect("the bracket voxelises");
        assert!(!v.leaked, "the bracket is sealed by its own walls");

        // the ground truth: three boxes, minus the overlap each wall shares with the slab
        let slab = 1.2 * 0.8 * 0.4;
        let wall = 0.2 * 0.8 * 1.0;
        let overlap = 0.2 * 0.8 * 0.2; // each wall's lower 0.2 m sits inside the slab
        let truth = slab + 2.0 * (wall - overlap);
        let rel = v.volume() / truth - 1.0;
        eprintln!("  U-bracket at 48 cells: voxel volume {:.5} vs three boxes {:.5}, {:+.2}%", v.volume(), truth, rel * 100.0);
        assert!(rel > 0.0, "the voxelisation must over-estimate, got {rel:+.4}");
        assert!(rel < 0.20, "bracket over-estimate {:.2}% is larger than a one-cell skin explains", rel * 100.0);

        // ⭐ and the channel itself: a point in the mouth must be EMPTY. This is the whole claim.
        let probe = Vector3::new(0.0, 0.0, 0.7); // centre of the channel, clear of slab and walls
        let ci = |a: usize| ((probe[a] - v.origin[a]) / v.cell).round() as usize;
        assert!(
            !v.get(ci(0), ci(1), ci(2)),
            "the mouth of the U must be empty; the convex hull is what fills it"
        );
        // a point inside the slab must be solid, or "empty" above means nothing
        let inside = Vector3::new(0.0, 0.0, -0.1);
        let di = |a: usize| ((inside[a] - v.origin[a]) / v.cell).round() as usize;
        assert!(v.get(di(0), di(1), di(2)), "the slab interior must be solid");
    }

    /// ⛔ **An unsealed mesh is reported, not silently filled.** Drop one triangle and the exterior
    /// flood walks in through the hole; every cell then reads exterior, no interior is found, and
    /// `leaked` says so. Without this flag the caller gets a confident volume of roughly zero.
    #[test]
    fn a_mesh_with_a_hole_reports_that_the_fill_leaked() {
        let mut holed = cube(0.4);
        holed.tris.pop();
        holed.tris.pop(); // both triangles of one face, so the hole is a full square
        let v = SolidVoxels::from_mesh(&holed, 24).expect("it still voxelises");
        eprintln!("  holed cube: leaked = {}, volume {:.6} (the sealed cube is 0.064)", v.leaked, v.volume());
        assert!(v.leaked, "a cube missing a face must report a leak");

        let sealed = SolidVoxels::from_mesh(&cube(0.4), 24).expect("sealed");
        assert!(!sealed.leaked, "and the same cube with the face on must not");
    }

    /// The refusals: nothing to voxelise, and a grid a caller would regret asking for.
    #[test]
    fn empty_degenerate_and_absurd_inputs_are_refused() {
        let empty = TriMesh3 { verts: Vec::new(), tris: Vec::new() };
        assert!(SolidVoxels::from_mesh(&empty, 16).is_none(), "an empty mesh");
        assert!(SolidVoxels::from_mesh(&cube(0.4), 0).is_none(), "a zero-resolution request");
        let mut nan = cube(0.4);
        nan.verts[0].x = f64::NAN;
        assert!(SolidVoxels::from_mesh(&nan, 16).is_none(), "a non-finite vertex");
        // a flat mesh has no volume to fill: its bound is degenerate in one axis but not all
        let flat = TriMesh3 {
            verts: vec![Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)],
            tris: vec![[0, 1, 2]],
        };
        let v = SolidVoxels::from_mesh(&flat, 8).expect("a flat mesh still has a grid");
        assert!(v.leaked, "a single triangle seals nothing, so the fill must leak");
    }
}
