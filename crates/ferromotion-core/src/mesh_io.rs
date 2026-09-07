//! **Reading a triangle mesh, and turning it into an inertia** — the step between a robot
//! description and a body that can be simulated.
//!
//! Every model format this crate reads points at geometry it does not carry. A URDF says
//! `<mesh filename="package://arm_description/meshes/link2.stl"/>`; an MJCF says
//! `<mesh file="link2.obj"/>`; a USD stage references a `Mesh` prim. Until now this crate parsed the
//! kinematics and dropped the geometry, which is why [`TriMesh3`]'s own doc line claims "the collision
//! and sensor layers consume meshes" while nothing in the tree could produce one from a file.
//!
//! Two readers, no new dependency, both parsing from a **string or a byte slice** rather than a path —
//! the same discipline [`from_urdf_str`](crate::from_urdf_str) and [`from_mjcf_str`](crate::from_mjcf_str)
//! keep, and the reason this works identically native and in a browser.
//!
//! # The part that matters more than the parsing
//!
//! A reader alone is a mesh nobody consumes. [`solid_inertia`] closes that: it integrates the mesh to
//! a [`LinkInertia`] — the exact type [`forward_dynamics`](crate::forward_dynamics) and the batched GPU
//! kernels already take — so a description carrying meshes and a density produces a body that can be
//! simulated, with no hand-typed inertia tensor anywhere.
//!
//! That integral is exact for a closed, outward-wound mesh, by tetrahedron decomposition about the
//! origin. For a tetrahedron with vertices `p₀…p₃` and signed volume `V`,
//!
//! ```text
//!   ∫ x xᵀ dV = (V/20)·( Σᵢ pᵢpᵢᵀ + (Σᵢ pᵢ)(Σᵢ pᵢ)ᵀ )
//! ```
//!
//! summed over the tetrahedra `(0, a, b, c)` of the triangles. The inertia tensor about the origin is
//! `tr(C)·I − C` scaled by density, and the parallel-axis theorem moves it to the centre of mass.
//!
//! # What a wrong winding does, and why the tests can see it
//!
//! Volume is *signed*. A mesh wound inward reports a negative volume, and a mesh with one flipped
//! triangle reports a volume that is wrong by twice that triangle's tetrahedron. Both are silent: the
//! number is finite and plausible. So the tests here do not only check values, they check
//! **invariants a wrong index cannot satisfy** — that reversing the winding negates the volume
//! exactly, that volume and the inertia eigenvalues are unchanged by a rigid transform of the vertex
//! list, and that a refined tessellation approaches the analytic sphere at the rate its own truncation
//! error predicts. A systematically wrong facet shows up as a wrong convergence *rate*, which a
//! single-value check passes.
//!
//! # Scope
//!
//! ASCII Wavefront OBJ and both STL encodings. This review did not implement GLB, DAE or PLY; GLB is
//! chunked binary with an embedded JSON scene and DAE carries its own unit and up-axis conventions,
//! and each is a module of its own rather than a variant here. Materials, textures, normals, UVs and
//! OBJ's `usemtl`/`o`/`g` grouping are parsed past, not represented: this is geometry for physics.
//!
//! ⛔ **Closedness is not checked, and an open mesh is accepted.** The divergence-theorem volume of an
//! open mesh is finite and often positive, so [`solid_inertia`]'s only structural guard — a positive
//! volume — passes and it returns a body whose mass is simply wrong. **Measured**: the unit cube with
//! its top face removed reports a volume of **0.6667** and a mass to match, because the two removed
//! facets are worth exactly 1/3 of the enclosing integral. Detecting closedness needs an edge-manifoldness pass
//! (every edge shared by exactly two triangles, consistently oriented) which this module does not do;
//! `malformed_input_is_refused_rather_than_read_short` asserts the limitation so it cannot be
//! forgotten. Facets lying in a plane through the origin are worse still: they contribute nothing to
//! the integral, so removing the cube's *bottom* face changes the reported volume by exactly zero.
//!
//! **No asset is vendored.** These functions take bytes the caller supplies, which is the same stance
//! `ferromotion-models` states for robot descriptions — a mesh file carries a licence and this crate
//! ships none.

use crate::mesh3::TriMesh3;
use crate::LinkInertia;
use nalgebra::{Matrix3, Vector3};

/// Read an **ASCII Wavefront OBJ**, keeping only what physics needs: vertices and triangles.
///
/// Handles the forms that appear in real robot assets: `v x y z` (with an optional `w` that is
/// ignored), `f` with 1-based positive indices, **negative** indices counting back from the current
/// vertex list, the `v/vt`, `v//vn` and `v/vt/vn` slash forms, and polygons of more than three
/// vertices, which are fan-triangulated. `vt`, `vn`, `usemtl`, `mtllib`, `o`, `g`, `s` and comments
/// are skipped.
///
/// `None` if a coordinate or index does not parse, if an index is out of range, or if the file
/// declares no triangles — every one of which is a corrupt file rather than an empty one, and the
/// caller should be told the difference.
pub fn from_obj(text: &str) -> Option<TriMesh3> {
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();

    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let x: f64 = it.next()?.parse().ok()?;
                let y: f64 = it.next()?.parse().ok()?;
                let z: f64 = it.next()?.parse().ok()?;
                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    return None;
                }
                verts.push(Vector3::new(x, y, z));
            }
            Some("f") => {
                // Each token is `v`, `v/vt`, `v//vn` or `v/vt/vn`; only the position index is kept.
                let mut poly: Vec<usize> = Vec::new();
                for tok in it {
                    let idx_str = tok.split('/').next()?;
                    let raw: i64 = idx_str.parse().ok()?;
                    let idx = if raw > 0 {
                        (raw as usize).checked_sub(1)?
                    } else if raw < 0 {
                        // negative indices are relative to the END of the current vertex list
                        verts.len().checked_sub(raw.unsigned_abs() as usize)?
                    } else {
                        return None; // 0 is not a valid OBJ index in either convention
                    };
                    if idx >= verts.len() {
                        return None;
                    }
                    poly.push(idx);
                }
                if poly.len() < 3 {
                    return None;
                }
                // fan-triangulate, which is correct for the convex faces robot assets use
                for k in 1..poly.len() - 1 {
                    tris.push([poly[0], poly[k], poly[k + 1]]);
                }
            }
            _ => {}
        }
    }

    (!tris.is_empty()).then_some(TriMesh3 { verts, tris })
}

/// Read an **STL** in either encoding, choosing by content rather than by extension.
///
/// The dispatch is deliberately not "does it start with `solid`": a binary STL's 80-byte header is
/// arbitrary and real exporters have written `solid` into it, which is the classic way an STL reader
/// misreads a file. The test here is structural — a binary STL's length is exactly
/// `84 + 50·count` for the count in its own header — and the ASCII path is the fallback.
pub fn from_stl(bytes: &[u8]) -> Option<TriMesh3> {
    if let Some(m) = from_stl_binary(bytes) {
        return Some(m);
    }
    from_stl_ascii(core::str::from_utf8(bytes).ok()?)
}

/// Read a **binary STL**: an 80-byte header, a `u32` little-endian triangle count, then 50 bytes per
/// triangle (a 3-float normal, three 3-float vertices, a `u16` attribute count).
///
/// The declared count is checked against the file length, so a truncated file is rejected rather than
/// read short. Normals are discarded: they are frequently zero or wrong in exported assets, and the
/// winding carries the orientation this crate needs.
pub fn from_stl_binary(bytes: &[u8]) -> Option<TriMesh3> {
    if bytes.len() < 84 {
        return None;
    }
    let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    // A binary STL is EXACTLY this long. Anything else is a different format or a truncated file, and
    // reading it short is how a mesh silently loses its far side.
    if bytes.len() != 84 + 50 * count || count == 0 {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64;

    let mut soup: Vec<Vector3<f64>> = Vec::with_capacity(count * 3);
    for t in 0..count {
        let base = 84 + 50 * t + 12; // skip the normal
        for v in 0..3 {
            let o = base + 12 * v;
            let p = Vector3::new(f(o), f(o + 4), f(o + 8));
            if !p.iter().all(|c| c.is_finite()) {
                return None;
            }
            soup.push(p);
        }
    }
    Some(weld(&soup))
}

/// Read an **ASCII STL**. Only `vertex x y z` lines are load-bearing; `solid`, `facet normal`,
/// `outer loop`, `endloop`, `endfacet` and `endsolid` are structure this reader does not need to
/// enforce, because a vertex count that is not a multiple of three is itself the error.
pub fn from_stl_ascii(text: &str) -> Option<TriMesh3> {
    let mut soup: Vec<Vector3<f64>> = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("vertex") {
            continue;
        }
        let x: f64 = it.next()?.parse().ok()?;
        let y: f64 = it.next()?.parse().ok()?;
        let z: f64 = it.next()?.parse().ok()?;
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return None;
        }
        soup.push(Vector3::new(x, y, z));
    }
    if soup.is_empty() || !soup.len().is_multiple_of(3) {
        return None;
    }
    Some(weld(&soup))
}

/// Turn a triangle soup into an indexed mesh by welding exactly-equal vertices.
///
/// STL has no shared vertices — every triangle repeats its corners — so a 1000-triangle part arrives
/// as 3000 points where a few hundred are distinct. Welding on the exact bit pattern is the honest
/// choice: it is what the exporter wrote, it never merges two points the file meant to keep apart,
/// and it needs no tolerance the caller would have to justify. Nothing downstream requires a
/// watertight index, and a tolerance-based weld can close a gap the geometry intends.
fn weld(soup: &[Vector3<f64>]) -> TriMesh3 {
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(soup.len() / 3);
    let mut seen: std::collections::HashMap<[u64; 3], usize> = std::collections::HashMap::new();
    let mut idx_of = |p: Vector3<f64>, verts: &mut Vec<Vector3<f64>>| -> usize {
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        *seen.entry(key).or_insert_with(|| {
            verts.push(p);
            verts.len() - 1
        })
    };
    for t in soup.as_chunks::<3>().0 {
        let a = idx_of(t[0], &mut verts);
        let b = idx_of(t[1], &mut verts);
        let c = idx_of(t[2], &mut verts);
        tris.push([a, b, c]);
    }
    TriMesh3 { verts, tris }
}

/// **Scale every vertex**, because a mesh arrives in the units its exporter chose and a robot
/// description states the scale separately (`<mesh scale="0.001 0.001 0.001"/>` is the common case of
/// a part authored in millimetres).
///
/// Non-uniform scale is supported because URDF and MJCF both allow it. Note that it changes the
/// inertia tensor non-trivially, which is why this scales the geometry and lets [`solid_inertia`]
/// integrate the result rather than trying to transform an inertia.
pub fn scale_mesh(mesh: &TriMesh3, scale: Vector3<f64>) -> TriMesh3 {
    TriMesh3 {
        verts: mesh.verts.iter().map(|v| Vector3::new(v.x * scale.x, v.y * scale.y, v.z * scale.z)).collect(),
        tris: mesh.tris.clone(),
    }
}

/// **The mesh's second-moment matrix about the origin**, `C = ∫ x xᵀ dV`, exact for a closed
/// outward-wound mesh.
///
/// Exposed because it is the reusable half: a caller with several meshes in one link frame can sum
/// their volumes and their `C` matrices and take the inertia once, which is not the same as taking
/// each inertia and adding them (those are about different centres).
pub fn second_moment(mesh: &TriMesh3) -> (f64, Matrix3<f64>) {
    let mut vol = 0.0f64;
    let mut c = Matrix3::zeros();
    for t in &mesh.tris {
        let (a, b, d) = (mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]);
        let v = a.dot(&b.cross(&d)) / 6.0; // signed tetrahedron volume against the origin
        vol += v;
        // (V/20)( Σ pᵢpᵢᵀ + (Σ pᵢ)(Σ pᵢ)ᵀ ) over the FOUR vertices; the origin contributes nothing
        // to the first sum and nothing to the second.
        let s = a + b + d;
        let sq = a * a.transpose() + b * b.transpose() + d * d.transpose();
        c += (v / 20.0) * (sq + s * s.transpose());
    }
    (vol, c)
}

/// **A mesh and a density become a body.** Returns the [`LinkInertia`] the dynamics already take:
/// mass, centre of mass, and the inertia tensor **about that centre of mass**.
///
/// `None` unless the density is finite and positive, every vertex is finite, and the mesh encloses a
/// positive volume. A non-positive volume is the diagnostic that matters: it means the mesh is wound
/// inward, is not closed, or is degenerate, and returning a mass for it would hand the caller a body
/// that simulates and is wrong. This crate has been bitten by exactly that shape of failure before,
/// so the refusal is deliberate rather than defensive.
///
/// The result is a *uniform-density* inertia. A real part with a motor at one end is not uniform, and
/// no mesh knows that; if the description states an inertia, prefer the description.
pub fn solid_inertia(mesh: &TriMesh3, density: f64) -> Option<LinkInertia> {
    if !density.is_finite() || density <= 0.0 {
        return None;
    }
    if !mesh.verts.iter().all(|v| v.iter().all(|c| c.is_finite())) || mesh.tris.is_empty() {
        return None;
    }
    let (vol, c) = second_moment(mesh);
    if !vol.is_finite() || vol <= 0.0 {
        return None;
    }
    let mass = density * vol;
    let com = mesh.centroid();
    // inertia about the ORIGIN: I = tr(C)·1 − C, then parallel-axis to the centre of mass
    let i_origin = density * (c.trace() * Matrix3::identity() - c);
    let shift = mass * (com.norm_squared() * Matrix3::identity() - com * com.transpose());
    let inertia = i_origin - shift;
    if !inertia.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(LinkInertia { mass, com, inertia })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    /// A unit cube at the origin corner, written the way an exporter writes one: 8 vertices, 12
    /// outward-wound triangles. Typed here as fixture text so no asset is vendored.
    const CUBE_OBJ: &str = "\
# a unit cube, corner at the origin
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
";

    fn cube() -> TriMesh3 {
        from_obj(CUBE_OBJ).expect("the cube fixture parses")
    }

    /// **The closed-form oracle.** A unit cube's volume, centroid and inertia are textbook, so this
    /// checks the integral against mathematics rather than against another implementation of the same
    /// integral — which is the check this workspace has been burned for skipping.
    #[test]
    fn a_unit_cube_integrates_to_its_closed_form() {
        let m = cube();
        assert_eq!(m.verts.len(), 8, "the cube must weld to 8 distinct vertices");
        assert_eq!(m.tris.len(), 12, "and 12 triangles");

        assert!((m.volume() - 1.0).abs() < 1e-15, "a unit cube encloses exactly 1, got {}", m.volume());
        assert!((m.centroid() - Vector3::new(0.5, 0.5, 0.5)).norm() < 1e-15, "centroid at the middle, got {:?}", m.centroid());
        assert!((m.surface_area() - 6.0).abs() < 1e-14, "six unit faces, got {}", m.surface_area());

        let density = 2700.0; // aluminium, so the numbers are a real part rather than unit-mass
        let li = solid_inertia(&m, density).expect("a closed cube has an inertia");
        assert!((li.mass - density).abs() < 1e-9, "mass is density times unit volume, got {}", li.mass);
        // a cube of side a about its own COM: I = m·a²/6 on each axis, off-diagonals zero
        let want = density / 6.0;
        for i in 0..3 {
            assert!((li.inertia[(i, i)] - want).abs() < 1e-9, "I[{i}{i}] must be m·a²/6 = {want}, got {}", li.inertia[(i, i)]);
            for j in 0..3 {
                if i != j {
                    assert!(li.inertia[(i, j)].abs() < 1e-9, "a cube has no products of inertia, I[{i}{j}] = {}", li.inertia[(i, j)]);
                }
            }
        }
        eprintln!("  unit cube at {density} kg/m³: mass {:.3} kg, I_xx {:.5}, target {:.5}", li.mass, li.inertia[(0, 0)], want);
    }

    /// **Winding is signed, and a flipped facet is silent.** Reversing every triangle must negate the
    /// volume exactly, and `solid_inertia` must REFUSE an inward-wound mesh rather than return a body
    /// with a negative mass. Flipping one facet must move the volume by exactly twice that facet's
    /// tetrahedron, which is the quantity a value-only check cannot see.
    #[test]
    fn winding_is_signed_and_a_single_flipped_facet_is_visible() {
        let m = cube();
        let reversed = TriMesh3 { verts: m.verts.clone(), tris: m.tris.iter().map(|t| [t[0], t[2], t[1]]).collect() };
        assert!((reversed.volume() + 1.0).abs() < 1e-15, "reversing the winding must negate the volume, got {}", reversed.volume());
        assert!(solid_inertia(&reversed, 1000.0).is_none(), "an inward-wound mesh must be refused, not given a negative mass");

        // one facet flipped: the volume changes by exactly 2× that tetrahedron's signed volume
        let mut one = m.clone();
        let t = one.tris[3];
        let (a, b, c) = (one.verts[t[0]], one.verts[t[1]], one.verts[t[2]]);
        let tet = a.dot(&b.cross(&c)) / 6.0;
        one.tris[3] = [t[0], t[2], t[1]];
        let delta = one.volume() - m.volume();
        eprintln!("  one flipped facet: volume {:.6} vs 1.0, delta {:.6}, predicted -2·tet = {:.6}", one.volume(), delta, -2.0 * tet);
        assert!((delta + 2.0 * tet).abs() < 1e-15, "a flipped facet must move the volume by −2·its tetrahedron: {delta} vs {}", -2.0 * tet);
    }

    /// **Rigid-transform invariance**, which is the check an index or stride error cannot pass. Volume
    /// and the inertia *eigenvalues* are properties of the solid, not of the coordinates, so they must
    /// be unchanged by an arbitrary rotation and translation of the vertex list, while the centre of
    /// mass must transform exactly.
    #[test]
    fn volume_and_inertia_eigenvalues_survive_a_rigid_transform() {
        let m = cube();
        let base = solid_inertia(&m, 1000.0).expect("well-posed");
        let mut base_eigs = base.inertia.symmetric_eigenvalues().as_slice().to_vec();
        base_eigs.sort_by(f64::total_cmp);

        let q = UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1);
        let t = Translation3::new(-2.0, 5.0, 0.25);
        let moved = TriMesh3 { verts: m.verts.iter().map(|v| q * v + t.vector).collect(), tris: m.tris.clone() };
        let got = solid_inertia(&moved, 1000.0).expect("well-posed");

        assert!((moved.volume() - m.volume()).abs() < 1e-13, "volume is not a coordinate: {} vs {}", moved.volume(), m.volume());
        assert!((got.mass - base.mass).abs() < 1e-9, "mass moved: {} vs {}", got.mass, base.mass);
        let want_com = q * base.com + t.vector;
        assert!((got.com - want_com).norm() < 1e-12, "the COM must transform exactly: {:?} vs {:?}", got.com, want_com);

        let mut got_eigs = got.inertia.symmetric_eigenvalues().as_slice().to_vec();
        got_eigs.sort_by(f64::total_cmp);
        for (a, b) in got_eigs.iter().zip(&base_eigs) {
            assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "an inertia eigenvalue changed under a rigid motion: {a} vs {b}");
        }
        eprintln!("  rigid transform: eigenvalues {base_eigs:?} -> {got_eigs:?}");
    }

    /// A tessellated sphere, generated here rather than read, so the analytic answer is available.
    /// Latitude-longitude tessellation with `n` bands; the enclosed polyhedron is inscribed, so it
    /// UNDER-estimates the sphere, and the error falls as `1/n²`.
    fn tess_sphere(n: usize, r: f64) -> TriMesh3 {
        use core::f64::consts::PI;
        let mut verts = vec![Vector3::new(0.0, 0.0, r)];
        for i in 1..n {
            let theta = PI * i as f64 / n as f64;
            for j in 0..2 * n {
                let phi = 2.0 * PI * j as f64 / (2 * n) as f64;
                verts.push(Vector3::new(r * theta.sin() * phi.cos(), r * theta.sin() * phi.sin(), r * theta.cos()));
            }
        }
        verts.push(Vector3::new(0.0, 0.0, -r));
        let ring = |i: usize, j: usize| 1 + (i - 1) * 2 * n + j % (2 * n);
        let south = verts.len() - 1;
        let mut tris = Vec::new();
        for j in 0..2 * n {
            tris.push([0, ring(1, j), ring(1, j + 1)]);
        }
        for i in 1..n - 1 {
            for j in 0..2 * n {
                tris.push([ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)]);
                tris.push([ring(i, j), ring(i + 1, j + 1), ring(i, j + 1)]);
            }
        }
        for j in 0..2 * n {
            tris.push([south, ring(n - 1, j + 1), ring(n - 1, j)]);
        }
        TriMesh3 { verts, tris }
    }

    /// **The convergence-RATE oracle.** A value check at one resolution can be passed by a
    /// systematically wrong facet; a wrong winding or a double-counted triangle changes the RATE at
    /// which refinement approaches the analytic sphere. An inscribed lat-long tessellation is
    /// second-order, so halving the band spacing must quarter the error — for volume and for the
    /// inertia `2/5·m·r²` alike.
    #[test]
    fn a_refined_sphere_approaches_its_analytic_form_at_the_predicted_rate() {
        use core::f64::consts::PI;
        let (r, density) = (0.35f64, 1200.0f64);
        let want_vol = 4.0 / 3.0 * PI * r.powi(3);

        let err = |n: usize| -> (f64, f64) {
            let m = tess_sphere(n, r);
            let li = solid_inertia(&m, density).expect("a closed sphere has an inertia");
            let want_i = 0.4 * li.mass * r * r; // 2/5 m r² about any axis through the centre
            let mean_i = (li.inertia[(0, 0)] + li.inertia[(1, 1)] + li.inertia[(2, 2)]) / 3.0;
            ((m.volume() / want_vol - 1.0).abs(), (mean_i / want_i - 1.0).abs())
        };
        let (v8, i8) = err(8);
        let (v16, i16) = err(16);
        let (v32, i32_) = err(32);
        eprintln!("  sphere r={r}: volume error {v8:.3e} -> {v16:.3e} -> {v32:.3e} (ratios {:.2}, {:.2})", v8 / v16, v16 / v32);
        eprintln!("            inertia error {i8:.3e} -> {i16:.3e} -> {i32_:.3e} (ratios {:.2}, {:.2})", i8 / i16, i16 / i32_);

        // inscribed ⇒ always an under-estimate, and second order in the band count
        assert!(v32 < v16 && v16 < v8, "refinement must reduce the volume error monotonically");
        for (a, b) in [(v8, v16), (v16, v32)] {
            assert!((3.0..5.0).contains(&(a / b)), "volume error must fall ~4x per refinement, got {:.2}x", a / b);
        }
        for (a, b) in [(i8, i16), (i16, i32_)] {
            assert!((3.0..5.0).contains(&(a / b)), "inertia error must fall ~4x per refinement, got {:.2}x", a / b);
        }
        assert!(v32 < 1e-2, "and actually be close by the finest level: {v32:.3e}");
    }

    /// **The two encodings must agree with each other and with OBJ**, since all three describe the
    /// same solid. Binary STL is written here byte by byte, which is also the only way to test the
    /// length check without a fixture file.
    #[test]
    fn obj_ascii_stl_and_binary_stl_describe_the_same_solid() {
        let obj = cube();
        // the same 12 triangles as an ASCII STL
        let mut ascii = String::from("solid cube\n");
        for t in &obj.tris {
            ascii.push_str("  facet normal 0 0 0\n    outer loop\n");
            for &i in t {
                let v = obj.verts[i];
                ascii.push_str(&format!("      vertex {} {} {}\n", v.x, v.y, v.z));
            }
            ascii.push_str("    endloop\n  endfacet\n");
        }
        ascii.push_str("endsolid cube\n");
        let from_ascii = from_stl_ascii(&ascii).expect("the ASCII STL parses");

        // and as a binary STL, built to the format's own layout
        let mut bin = vec![0u8; 80];
        bin.extend_from_slice(&(obj.tris.len() as u32).to_le_bytes());
        for t in &obj.tris {
            bin.extend_from_slice(&[0u8; 12]); // normal
            for &i in t {
                let v = obj.verts[i];
                for c in [v.x, v.y, v.z] {
                    bin.extend_from_slice(&(c as f32).to_le_bytes());
                }
            }
            bin.extend_from_slice(&[0u8; 2]); // attribute byte count
        }
        let from_bin = from_stl(&bin).expect("the binary STL parses");

        for (label, m) in [("ascii stl", &from_ascii), ("binary stl", &from_bin)] {
            assert_eq!(m.verts.len(), 8, "{label}: welding must recover 8 distinct vertices from the soup");
            assert_eq!(m.tris.len(), 12, "{label}: 12 triangles");
            assert!((m.volume() - obj.volume()).abs() < 1e-6, "{label}: volume {} vs OBJ {}", m.volume(), obj.volume());
            let a = solid_inertia(m, 2700.0).expect("well-posed");
            let b = solid_inertia(&obj, 2700.0).expect("well-posed");
            assert!((a.mass - b.mass).abs() < 1e-3, "{label}: mass {} vs {}", a.mass, b.mass);
            assert!((a.inertia - b.inertia).amax() < 1e-3, "{label}: inertia differs by {}", (a.inertia - b.inertia).amax());
        }
        // ASCII STL is exact in f64, binary is f32 on the wire, so the ASCII path must be tighter
        assert!((from_ascii.volume() - obj.volume()).abs() < 1e-15, "the ASCII path carries full f64 precision");
        eprintln!("  cube: OBJ {:.9}, ascii STL {:.9}, binary STL {:.9} (f32 on the wire)", obj.volume(), from_ascii.volume(), from_bin.volume());
    }

    /// Non-uniform scale, the `<mesh scale="0.001 0.001 0.001">` case of a part authored in
    /// millimetres, and the reason the inertia is integrated from the scaled geometry rather than
    /// transformed: a non-uniform scale does not act on an inertia tensor the way it acts on a length.
    #[test]
    fn scaling_the_geometry_scales_the_solid_the_way_the_integral_says() {
        let m = cube();
        let uniform = scale_mesh(&m, Vector3::new(0.001, 0.001, 0.001));
        assert!((uniform.volume() - 1e-9).abs() < 1e-24, "a mm-authored unit cube is 1e-9 m³, got {}", uniform.volume());

        let stretched = scale_mesh(&m, Vector3::new(2.0, 1.0, 0.5));
        assert!((stretched.volume() - 1.0).abs() < 1e-15, "this scale preserves volume, got {}", stretched.volume());
        let li = solid_inertia(&stretched, 1.0).expect("well-posed");
        // a box a×b×c about its COM: I_xx = m(b²+c²)/12
        let (a, b, c) = (2.0f64, 1.0f64, 0.5f64);
        for (i, want) in [(0, (b * b + c * c) / 12.0), (1, (a * a + c * c) / 12.0), (2, (a * a + b * b) / 12.0)] {
            assert!((li.inertia[(i, i)] - want).abs() < 1e-12, "stretched box I[{i}{i}] = {} vs closed form {want}", li.inertia[(i, i)]);
        }
        assert!(
            (li.inertia[(0, 0)] - li.inertia[(2, 2)]).abs() > 0.1,
            "the point of this test is that a non-uniform scale makes the axes differ, and they did not"
        );
        eprintln!("  2×1×0.5 box: I = diag({:.5}, {:.5}, {:.5})", li.inertia[(0, 0)], li.inertia[(1, 1)], li.inertia[(2, 2)]);
    }

    /// **A malformed file must be refused, never panic and never read short.** Every case here is one
    /// a real asset pipeline produces: a truncated download, a header count that disagrees with the
    /// payload, an exporter that wrote NaN, an index off the end.
    #[test]
    fn malformed_input_is_refused_rather_than_read_short() {
        // binary STL: the length check is the guard against a truncated file
        let mut good = vec![0u8; 80];
        good.extend_from_slice(&1u32.to_le_bytes());
        good.extend_from_slice(&[0u8; 12]);
        for c in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            good.extend_from_slice(&c.to_le_bytes());
        }
        good.extend_from_slice(&[0u8; 2]);
        assert_eq!(good.len(), 84 + 50, "the fixture must be exactly one triangle long");
        assert!(from_stl_binary(&good).is_some(), "control: a well-formed single-triangle STL reads");

        assert!(from_stl_binary(&good[..good.len() - 10]).is_none(), "a truncated binary STL must be refused");
        // ⛔ AND A FILE THAT IS TOO LONG, which the first version of this battery missed: relaxing the
        // length check from `!=` to `<` survived every other assertion here. An over-long file means
        // the declared count is wrong or it is not a binary STL at all, and reading the first `count`
        // triangles out of it is the same "read short" failure in the other direction — you get a
        // plausible mesh that is missing whatever followed.
        let mut over = good.clone();
        over.extend_from_slice(&[0u8; 7]);
        assert!(from_stl_binary(&over).is_none(), "a binary STL with trailing bytes must be refused, not read as a prefix");
        let mut two_declared_one_present = good.clone();
        two_declared_one_present[80..84].copy_from_slice(&2u32.to_le_bytes());
        assert!(from_stl_binary(&two_declared_one_present).is_none(), "declaring two triangles and carrying one must be refused");
        let mut lying = good.clone();
        lying[80..84].copy_from_slice(&99u32.to_le_bytes());
        assert!(from_stl_binary(&lying).is_none(), "a header count that disagrees with the payload must be refused");
        assert!(from_stl_binary(&[0u8; 20]).is_none(), "a file shorter than the header must be refused");
        let mut zero = good.clone();
        zero[80..84].copy_from_slice(&0u32.to_le_bytes());
        assert!(from_stl_binary(&zero[..84]).is_none(), "zero triangles is not a mesh");

        // ASCII STL
        assert!(from_stl_ascii("solid s\nendsolid s\n").is_none(), "no vertices is not a mesh");
        assert!(from_stl_ascii("vertex 0 0 0\nvertex 1 0 0\n").is_none(), "a vertex count that is not a multiple of 3 must be refused");
        assert!(from_stl_ascii("vertex 0 0 0\nvertex 1 0 0\nvertex nan 1 0\n").is_none(), "a non-finite coordinate must be refused");

        // OBJ
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").is_some(), "control: a one-triangle OBJ reads");
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n").is_none(), "an index off the end must be refused");
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 0\n").is_none(), "index 0 is not valid in either OBJ convention");
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n").is_none(), "a face of two vertices must be refused");
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv inf 1 0\nf 1 2 3\n").is_none(), "a non-finite vertex must be refused");
        assert!(from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n").is_none(), "vertices with no faces is not a mesh");
        assert!(from_obj("").is_none(), "an empty file is not a mesh");

        // and solid_inertia's own refusals
        let m = cube();
        assert!(solid_inertia(&m, 0.0).is_none(), "zero density");
        assert!(solid_inertia(&m, -1.0).is_none(), "negative density");
        assert!(solid_inertia(&m, f64::NAN).is_none(), "non-finite density");
        assert!(solid_inertia(&TriMesh3::default(), 1000.0).is_none(), "an empty mesh has no inertia");
        // ⛔ AN OPEN MESH IS NOT DETECTED, and this asserts the limitation rather than papering over
        // it. The divergence-theorem volume of an open mesh is finite and often positive, so
        // `solid_inertia` returns a body — a wrong one. My first version of this check was
        // `is_none() || volume > 0.0`, which passes trivially and taught me nothing; worse, the facets
        // it removed lay in the z = 0 plane through the ORIGIN, so they contribute zero to the integral
        // and the volume stayed at exactly 1.0. Removing the TOP face is the case that moves.
        let removed = &m.tris[2..4]; // the z = 1 face, which does not pass through the origin
        let lost: f64 = removed
            .iter()
            .map(|t| m.verts[t[0]].dot(&m.verts[t[1]].cross(&m.verts[t[2]])) / 6.0)
            .sum();
        let open = TriMesh3 { verts: m.verts.clone(), tris: [&m.tris[..2], &m.tris[4..]].concat() };
        eprintln!("  open mesh, top face removed: volume {:.4} (closed 1.0, facets worth {lost:.4})", open.volume());
        assert!((open.volume() - (1.0 - lost)).abs() < 1e-15, "the integral must lose exactly the removed tetrahedra: {} vs {}", open.volume(), 1.0 - lost);
        let still = solid_inertia(&open, 1000.0);
        assert!(
            still.is_some(),
            "THE LIMITATION, asserted so it cannot be forgotten: an open mesh with positive volume is \
             ACCEPTED and yields a wrong body. Closedness is not checked, and the volume sign is the \
             only guard. If a future change starts rejecting this, the doc must change with it."
        );
        assert!(
            (still.unwrap().mass - 1000.0 * (1.0 - lost)).abs() < 1e-9,
            "and the mass it yields is the wrong volume times the density, which is the wrong body"
        );
        // the same facets in the z = 0 plane are invisible to the integral, which is why the test above
        // must remove the top face and not the bottom one
        let bottom_open = TriMesh3 { verts: m.verts.clone(), tris: m.tris[2..].to_vec() };
        assert!((bottom_open.volume() - 1.0).abs() < 1e-15, "facets through the origin contribute nothing, so this stays at 1.0");
    }

    /// The OBJ index and face forms real exporters emit, which is what decides whether this reads a
    /// downloaded asset or only a hand-typed one.
    #[test]
    fn the_obj_forms_real_exporters_emit_all_parse_to_the_same_triangle() {
        let want = from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("positive indices");
        for (label, text) in [
            ("negative indices", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"),
            ("v/vt", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n"),
            ("v//vn", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"),
            ("v/vt/vn", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1\n"),
            ("groups and materials", "o part\ng shell\nusemtl steel\nv 0 0 0\nv 1 0 0\nv 0 1 0\ns 1\nf 1 2 3\n"),
            ("trailing w on v", "v 0 0 0 1.0\nv 1 0 0 1.0\nv 0 1 0 1.0\nf 1 2 3\n"),
            ("comments and blank lines", "# header\n\nv 0 0 0\n\nv 1 0 0\nv 0 1 0\n# a face\nf 1 2 3\n"),
        ] {
            let got = from_obj(text).unwrap_or_else(|| panic!("{label} must parse"));
            assert_eq!(got.tris, want.tris, "{label}: triangle indices differ");
            assert_eq!(got.verts.len(), want.verts.len(), "{label}: vertex count differs");
        }
        // a quad must fan-triangulate into exactly two triangles covering the same area
        let quad = from_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n").expect("a quad parses");
        assert_eq!(quad.tris.len(), 2, "a quad fans into two triangles");
        assert!((quad.surface_area() - 1.0).abs() < 1e-15, "and covers the unit square, got {}", quad.surface_area());
    }
}
