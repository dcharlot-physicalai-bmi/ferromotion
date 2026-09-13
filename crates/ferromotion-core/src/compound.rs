//! **A body whose collision geometry is a SET of convex parts** — the thing that lets a real,
//! non-convex robot link reach the contact solvers.
//!
//! [`gjk`](crate::gjk) already answers the convex question, and its [`Support`] trait already accepts
//! an arbitrary point set through [`ConvexPoints`]. What was missing is the level above: a link is not
//! convex. A gripper finger, a motor housing with a mounting flange, a foot with a heel — each is a
//! union of convex pieces, and its convex hull is a different shape that fills in exactly the gaps a
//! robot is built to reach into.
//!
//! # Why one hull is the wrong answer, stated as a number
//!
//! Take a U-shaped bracket and a ball sitting in the mouth of the U. Against the bracket's convex
//! hull the ball is **penetrating**; against the bracket itself it is **free**, and by a wide margin.
//! `the_convex_hull_fills_the_concavity_a_compound_keeps_open` measures both on the same fixture, so
//! the difference is a figure in the test output rather than an argument.
//!
//! # The part that a closest-pair API cannot express
//!
//! ⛔ **The union of convex sets is not convex, so ONE closest pair is not a contact set.** A plate
//! resting across two separated blocks touches both; a foot on a step touches the tread and the riser.
//! A solver handed only the nearest of those loses a constraint and the body sinks through the other.
//!
//! So there are two queries and they answer different questions:
//!
//! * [`compound_distance`] — the single closest pair, which is what a *planner* wants: a clearance.
//! * [`compound_contacts`] — **every** part pair within a margin, which is what a *solver* wants: a
//!   constraint set. `a_plate_on_two_blocks_yields_two_contacts_not_one` asserts that the second
//!   returns two where the first returns one, because that is the whole distinction.
//!
//! # Scope
//!
//! Convex parts, supplied by the caller or taken as the hull of each of several meshes. **This module
//! does not decompose a concave mesh into convex parts** — that is approximate convex decomposition
//! (the V-HACD/CoACD operation). A caller with one concave mesh has three honest options, in order of
//! preference: use the `<collision>` primitives the description already declares (see
//! [`primitive_mesh`](crate::primitive_mesh)), supply the part meshes separately if the exporter wrote
//! them that way, or accept the convex hull and know what it costs — which is the number this
//! module's tests print.
//!
//! # ⛔ A cheap decomposition was attempted twice and MEASURED to fail. Both results, so the next
//! attempt does not repeat them
//!
//! The tempting shortcut is to partition the triangle list and hull each group: no mesh surgery, so no
//! part can come out unclosed. It does not work, and the reason is structural rather than a tuning
//! problem. Measured on a U-shaped bracket of three boxes, with a probe in the mouth of the channel:
//!
//! | attempt | what it did | outcome |
//! |---|---|---|
//! | k-means over triangle centroids | grouped triangles near each other **on the surface** | the parts' volume sum came out **worse than not decomposing**: against a single hull at 1.64x the mesh's own volume integral, k-means gave 3.00x at two parts, 2.41x at three, 2.70x at four, 2.28x at six. A surface-adjacent group wraps around the solid, so its hull spans the whole shape. |
//! | recursive median split, **solid** parts required | split the triangle list at the median centroid along the longest axis | **the channel never opened.** Swept 1, 2, 3, 4, 6, 8, 12, 16 and 24 parts: the excluded fraction of the hull's interior rose 0% → 38.3% and saturated, and the probe was reported INTERSECTING at every single count. Demanding a 3-D hull of both halves refuses exactly the split that separates two facing walls, because each wall's inner face is a flat patch. |
//! | the same, **flat** parts allowed | let a part be a convex polygon, which GJK handles exactly | the channel opens — but the parts degenerate into a **hollow shell**. 28 of 32 parts came out flat for the U, and a *convex cube* at 12 parts reported 100% of its hull interior excluded, meaning the collider fills nothing. A shallow contact works and any penetration passes straight through. |
//!
//! **The lesson is that a surface partition is not a solid decomposition.** V-HACD and CoACD voxelise
//! the *interior* and merge voxel clusters, which is why they work; CoACD then searches cut planes
//! against a collision-aware concavity objective with Monte Carlo tree search. Neither is a
//! triangle-list partition, and no amount of choosing the split better makes one into the other. That
//! is the work, and it is a module of its own with a volumetric representation at its centre — not a
//! function here.
//!
//! ✅ **That module now exists: [`convex_decompose`](crate::acd::convex_decompose).** It voxelises the
//! interior with [`SolidVoxels`](crate::voxel::SolidVoxels) and splits cells, not triangles. On the
//! same bracket with the same probe it returns the three boxes the bracket is made of — `1.000x` its
//! own volume where a single hull is `1.53x`, and the probe `0.314 m` CLEAR where the hull has it
//! `0.380 m` inside. Prefer it to [`CompoundHull::from_mesh_hull`] for any concave mesh.
//!
//! Nothing was shipped from either attempt. The one piece worth keeping is
//! [`try_convex_hull_3d`](crate::try_convex_hull_3d), which came out of needing a hull that refuses a
//! degenerate part instead of panicking.

use crate::gjk::{gjk, ConvexPoints, GjkResult, Support};
use crate::mesh3::{try_convex_hull_3d, TriMesh3};
use crate::{Iso, LinkGeometry};
use nalgebra::{Point3, Vector3};

/// A body's collision geometry as a set of convex parts, each a point set in the body frame.
#[derive(Clone, Debug, Default)]
pub struct CompoundHull {
    pub parts: Vec<ConvexPoints>,
}

/// One part-pair proximity result, carrying **which** parts realised it — a solver needs the indices
/// to attach the constraint to the right piece, and a diagnostic needs them to say where.
///
/// ⛔ **The first version of this carried `distance` and two witness points and nothing else, and it
/// was unusable by a solver.** A [`PgsContact`](crate::PgsContact) needs a contact NORMAL and a gap
/// that goes NEGATIVE on penetration; `distance` is `0` for every intersecting pair, and GJK's witness
/// points are meaningless once the shapes overlap. So the module's claim to produce "the constraint
/// set a solver needs" was false for exactly the pairs a solver exists to resolve. `gap` and `normal`
/// are computed by [`epa`](crate::epa) when the pair intersects and by GJK when it does not.
#[derive(Clone, Copy, Debug)]
pub struct CompoundContact {
    /// GJK distance: `0` when the pair intersects. Kept for a planner, which only wants clearance.
    pub distance: f64,
    /// **Signed** gap: `+distance` when separated, `−depth` when penetrating. This is what a solver
    /// takes as `phi`, and the only field of the three that is meaningful in both cases.
    pub gap: f64,
    /// Unit contact normal, pointing **from part A toward part B**, in both regimes.
    ///
    /// The convention is one statement that holds either way, and it is what the tests check
    /// functionally rather than by asserting a sign: **translating part A by `gap·normal` brings the
    /// pair to exactly touching.** A negative `gap` therefore moves A away from B and separates them;
    /// a positive one moves A toward B and closes the clearance.
    ///
    /// Zero only in the degenerate case where neither GJK nor EPA can produce a direction: exactly
    /// touching, or a pair whose Minkowski difference is flat. A solver must skip a zero normal rather
    /// than normalise it.
    pub normal: Vector3<f64>,
    pub intersecting: bool,
    pub witness_a: Vector3<f64>,
    pub witness_b: Vector3<f64>,
    pub part_a: usize,
    pub part_b: usize,
}

impl CompoundHull {
    /// Wrap already-convex point sets. No hulling is done: if a caller has the parts, their vertices
    /// already define the support function, and re-hulling would only lose the exact vertices.
    pub fn from_convex_parts(parts: Vec<ConvexPoints>) -> Option<CompoundHull> {
        let usable = parts.iter().all(|p| p.pts.len() >= 4 && p.pts.iter().all(|q| q.iter().all(|c| c.is_finite())));
        (!parts.is_empty() && usable).then_some(CompoundHull { parts })
    }

    /// **The convex hull of each mesh**, one part per mesh. This is the honest route for a body an
    /// exporter wrote as several files, which is common — a link's `<collision>` block often lists
    /// three or four meshes precisely because the part is not convex.
    ///
    /// `None` if any mesh has no 3-D hull: a flat plate or a sliver is a real export artifact, and a
    /// part with no volume supports nothing.
    pub fn from_meshes(meshes: &[TriMesh3]) -> Option<CompoundHull> {
        if meshes.is_empty() {
            return None;
        }
        let mut parts = Vec::with_capacity(meshes.len());
        for m in meshes {
            let hull = try_convex_hull_3d(&m.verts)?;
            parts.push(ConvexPoints { pts: hull.verts });
        }
        Some(CompoundHull { parts })
    }

    /// One part, the convex hull of a single mesh. Named separately from [`Self::from_meshes`] so the
    /// **loss is at the call site**: this is the one construction that changes the shape, and a reader
    /// should see that it did.
    pub fn from_mesh_hull(mesh: &TriMesh3) -> Option<CompoundHull> {
        Self::from_meshes(core::slice::from_ref(mesh))
    }

    /// **The decomposition of each mesh**, several parts per mesh — the constructor to reach for when
    /// a link's `<collision>` block names one file and that file is not convex.
    ///
    /// ⛔ [`Self::from_meshes`] hulls. For a bracket, a gripper finger, a channel or a shell that is
    /// not a loss you can accept: the hull fills the very space the part exists to leave empty.
    /// Measured on the U-bracket this module's failure table is written against, a probe in the mouth
    /// of the channel is **0.380 m inside** the hull and **0.314 m clear** of the decomposition.
    ///
    /// Returns one [`AcdReport`](crate::AcdReport) per mesh, in order, so a caller can see what it got — how many parts
    /// each cost and how much slack is left.
    ///
    /// ⚠ `None` if ANY mesh fails to decompose, which for an unsealed mesh it will
    /// ([`SolidVoxels::leaked`](crate::voxel::SolidVoxels)). **There is deliberately no silent
    /// fallback to a hull.** A caller that would rather have the hull than nothing should call
    /// [`Self::from_meshes`] itself, so that the loss appears at the call site — the same reason
    /// [`Self::from_mesh_hull`] is named separately from [`Self::from_meshes`].
    pub fn from_meshes_decomposed(
        meshes: &[TriMesh3],
        opts: &crate::acd::AcdOptions,
    ) -> Option<(CompoundHull, Vec<crate::acd::AcdReport>)> {
        if meshes.is_empty() {
            return None;
        }
        let mut parts = Vec::new();
        let mut reports = Vec::with_capacity(meshes.len());
        for m in meshes {
            let (c, r) = crate::acd::convex_decompose(m, opts)?;
            parts.extend(c.parts);
            reports.push(r);
        }
        Self::from_convex_parts(parts).map(|c| (c, reports))
    }

    /// A compound from the shapes a robot description declares, generated rather than loaded. Each
    /// primitive is already convex, so its generated vertices are used directly and nothing is hulled.
    /// Mesh references are skipped and counted, the same contract
    /// [`primitive_link_inertia`](crate::primitive_link_inertia) uses.
    pub fn from_primitives(shapes: &[(LinkGeometry, Iso)], segments: usize) -> (Option<CompoundHull>, usize) {
        let mut parts = Vec::new();
        let mut skipped = 0usize;
        for (g, tf) in shapes {
            match crate::primitive_mesh(g, segments) {
                Some(m) => parts.push(ConvexPoints {
                    pts: m.verts.iter().map(|v| (tf * Point3::from(*v)).coords).collect(),
                }),
                None => skipped += 1,
            }
        }
        (Self::from_convex_parts(parts), skipped)
    }

    /// Place the whole compound by a rigid transform.
    pub fn transformed(&self, tf: &Iso) -> CompoundHull {
        CompoundHull {
            parts: self
                .parts
                .iter()
                .map(|p| ConvexPoints { pts: p.pts.iter().map(|v| (tf * Point3::from(*v)).coords).collect() })
                .collect(),
        }
    }

    pub fn n_parts(&self) -> usize {
        self.parts.len()
    }

    /// Every vertex of every part, which is what an inertia integral or a bounding volume wants.
    pub fn points(&self) -> impl Iterator<Item = &Vector3<f64>> {
        self.parts.iter().flat_map(|p| p.pts.iter())
    }
}

/// A compound behaves as a support function only if it is convex, which it is not. Implementing
/// [`Support`] would let a caller pass one to `gjk` and get the answer for its **convex hull**
/// silently — the exact error this module exists to prevent — so it is deliberately not implemented.
/// Use [`compound_distance`] or [`compound_contacts`].
impl CompoundHull {
    /// The support function of the compound's convex HULL, named so a caller cannot reach for it by
    /// accident. Useful for a broadphase bound, where an over-estimate of extent is what you want.
    pub fn hull_support(&self, dir: &Vector3<f64>) -> Vector3<f64> {
        let mut best: Option<(f64, Vector3<f64>)> = None;
        for p in &self.parts {
            let s = p.support(dir);
            let d = s.dot(dir);
            if d.is_finite() && best.is_none_or(|(bd, _)| d > bd) {
                best = Some((d, s));
            }
        }
        best.map(|(_, s)| s).unwrap_or_else(Vector3::zeros)
    }
}

/// One part pair, resolved to a signed gap and a normal. GJK answers the separated case; EPA answers
/// the penetrating one, which is the case a solver exists for and the one GJK cannot describe.
fn pair_contact(pa: &ConvexPoints, pb: &ConvexPoints, i: usize, j: usize) -> Option<CompoundContact> {
    let g: GjkResult = gjk(pa, pb);
    if !g.distance.is_finite() {
        return None;
    }
    let (gap, normal) = if g.intersecting {
        match crate::epa::epa(pa, pb) {
            // ⛔ **BOTH terms are negated, for two different reasons.** `gap` is `−depth` because a
            // penetration is a negative gap. `normal` is `−p.normal` because [`Penetration`] points
            // from `B` toward `A` (the direction `A` must move to get clear) and this struct's
            // convention is the other way round, `A` toward `B`. Reversing only one of the two would
            // satisfy the functional test below and still hand a solver the wrong side.
            Some(p) if p.depth.is_finite() && p.normal.iter().all(|c| c.is_finite()) => (-p.depth, -p.normal),
            // EPA refuses a flat Minkowski difference, which is an exactly-touching pair. A zero gap
            // with no direction is the honest answer; inventing a normal here would put a constraint
            // on an axis nothing measured.
            _ => (0.0, Vector3::zeros()),
        }
    } else {
        // ⛔ `witness_b − witness_a`, NOT the other way round: that is the A-toward-B direction this
        // struct documents. The first version used `witness_a − witness_b`, so the two regimes
        // disagreed on direction and the module's stated convention was true of only one of them. A
        // caller mixing a separated and a penetrating pair in one constraint set would have got
        // contradictory normals.
        //
        // ⛔ Chasing that disagreement is what turned up the one BELOW it: [`epa`](crate::epa) was
        // returning the reverse of what [`Penetration`] documents and of what
        // [`sphere_penetration`](crate::sphere_penetration) returns, and the penetrating branch here
        // had been written against the reversed behaviour. Both are fixed; the negation above is the
        // convention change, not a second bug.
        let d = g.witness_b - g.witness_a;
        let nn = d.norm();
        (g.distance, if nn > 1e-12 { d / nn } else { Vector3::zeros() })
    };
    Some(CompoundContact {
        distance: g.distance,
        gap,
        normal,
        intersecting: g.intersecting,
        witness_a: g.witness_a,
        witness_b: g.witness_b,
        part_a: i,
        part_b: j,
    })
}

/// **The closest pair between two compounds**, over all part pairs. What a planner wants: a single
/// clearance number, and which pieces set it.
///
/// `None` if either compound has no parts. If any pair intersects, the returned contact is an
/// intersecting one with distance `0` — but see [`compound_contacts`], because there may be others.
pub fn compound_distance(a: &CompoundHull, b: &CompoundHull) -> Option<CompoundContact> {
    let mut best: Option<CompoundContact> = None;
    for (i, pa) in a.parts.iter().enumerate() {
        for (j, pb) in b.parts.iter().enumerate() {
            let Some(c) = pair_contact(pa, pb, i, j) else { continue };
            // Ordered by the SIGNED gap, so a deeper penetration outranks a shallower one and both
            // outrank any separation. Ordering by `distance` would make every intersecting pair tie at
            // zero and pick an arbitrary one.
            if best.is_none_or(|bc| c.gap < bc.gap) {
                best = Some(c);
            }
        }
    }
    best
}

/// **Every part pair within `margin`** — the constraint set a contact solver needs.
///
/// Returned sorted by **signed gap**, deepest penetration first, so a caller that must truncate drops
/// the least important. An intersecting pair has a negative gap and is therefore always included for
/// any `margin >= 0`, and a **negative** margin selects only pairs penetrating deeper than
/// `|margin|` — the query a diagnostic wants when it is looking for the overlaps that matter.
///
/// ⛔ This is the query that distinguishes a compound from a hull. Taking only
/// [`compound_distance`]'s answer and handing it to a solver loses every simultaneous contact, and a
/// body with one of its two supports missing sinks through the other. `margin` should be at least the
/// solver's own contact-activation distance.
pub fn compound_contacts(a: &CompoundHull, b: &CompoundHull, margin: f64) -> Vec<CompoundContact> {
    // ⛔ A NEGATIVE margin is meaningful and is accepted, which it was not before `gap` was signed.
    // `margin = -0.01` selects only pairs penetrating deeper than a centimetre — the query a
    // diagnostic wants, and the reason a mutation swapping `gap` for `distance` here was invisible:
    // for margin >= 0 the two are equivalent, because an intersecting pair has distance 0 and a
    // negative gap. They differ exactly where a negative margin is allowed.
    if !margin.is_finite() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, pa) in a.parts.iter().enumerate() {
        for (j, pb) in b.parts.iter().enumerate() {
            if let Some(c) = pair_contact(pa, pb, i, j).filter(|c| c.gap <= margin) {
                out.push(c);
            }
        }
    }
    out.sort_by(|x, y| x.gap.total_cmp(&y.gap));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk::{Ball, Cuboid};
    use nalgebra::{Matrix3, Translation3, UnitQuaternion};

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    /// A box as its 8 corners, centred at `c` with half-extents `h`.
    fn box_pts(c: Vector3<f64>, h: Vector3<f64>) -> ConvexPoints {
        let mut pts = Vec::with_capacity(8);
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    pts.push(c + v(sx * h.x, sy * h.y, sz * h.z));
                }
            }
        }
        ConvexPoints { pts }
    }

    /// **The reduction check.** A compound of one part must give exactly what `gjk` gives for that
    /// part, or the compound layer has introduced something of its own.
    #[test]
    fn a_compound_of_one_part_reduces_to_plain_gjk() {
        let a = box_pts(v(0.0, 0.0, 0.0), v(0.5, 0.5, 0.5));
        let b = box_pts(v(2.0, 0.0, 0.0), v(0.5, 0.5, 0.5));
        let direct = gjk(&a, &b);
        let ca = CompoundHull::from_convex_parts(vec![a]).expect("one part");
        let cb = CompoundHull::from_convex_parts(vec![b]).expect("one part");
        let c = compound_distance(&ca, &cb).expect("one pair");

        assert_eq!(c.distance, direct.distance, "the compound must not change the answer for one part");
        assert_eq!(c.intersecting, direct.intersecting);
        assert_eq!((c.part_a, c.part_b), (0, 0));
        // and the analytic answer: two unit boxes 2 apart, centre to centre, gap 1
        assert!((c.distance - 1.0).abs() < 1e-9, "two unit boxes 2 apart leave a gap of 1, got {}", c.distance);
        eprintln!("  one-part reduction: compound {} == gjk {}", c.distance, direct.distance);
    }

    /// A U-shaped bracket: three boxes forming a channel open along +z, with the mouth spanning
    /// x ∈ [−0.4, 0.4] at z > 0.2.
    fn u_bracket() -> CompoundHull {
        CompoundHull::from_convex_parts(vec![
            box_pts(v(0.0, 0.0, 0.0), v(0.6, 0.4, 0.2)),   // base slab
            box_pts(v(-0.5, 0.0, 0.5), v(0.1, 0.4, 0.5)),  // left wall
            box_pts(v(0.5, 0.0, 0.5), v(0.1, 0.4, 0.5)),   // right wall
        ])
        .expect("three parts")
    }

    /// **THE POINT OF THE WHOLE MODULE, as a measured pair of numbers.** A ball sitting in the mouth
    /// of a U is FREE of the bracket and PENETRATING its convex hull. If a compound and a hull ever
    /// agree on this fixture, the compound is not doing its job.
    #[test]
    fn the_convex_hull_fills_the_concavity_a_compound_keeps_open() {
        let u = u_bracket();
        // a ball in the channel, clear of the base and both walls
        let ball = CompoundHull::from_convex_parts(vec![{
            let mut pts = Vec::new();
            // an icosahedron-ish point cloud is unnecessary; a small box stands in for the ball here,
            // and the true Ball is used below through plain gjk for the exact number
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        pts.push(v(sx * 0.15, sy * 0.15, 0.55 + sz * 0.15));
                    }
                }
            }
            ConvexPoints { pts }
        }])
        .expect("one part");

        let compound = compound_distance(&u, &ball).expect("pairs exist");

        // the same query against the bracket's CONVEX HULL: every vertex of every part, hulled
        let hull_pts: Vec<Vector3<f64>> = u.points().copied().collect();
        let hull = try_convex_hull_3d(&hull_pts).expect("the bracket has a hull");
        let against_hull = gjk(&ConvexPoints { pts: hull.verts }, &ball.parts[0]);

        eprintln!(
            "  ball in the mouth of a U: distance to the COMPOUND {:.4} (intersecting {}), to its CONVEX HULL {:.4} (intersecting {})",
            compound.distance, compound.intersecting, against_hull.distance, against_hull.intersecting
        );
        assert!(!compound.intersecting, "the ball must be FREE of the bracket itself");
        assert!(compound.distance > 0.05, "and clear by a real margin, got {:.4}", compound.distance);
        assert!(against_hull.intersecting, "and PENETRATING the convex hull, which is the error this module prevents");
        assert!(
            compound.distance - against_hull.distance > 0.05,
            "the two answers must differ substantially, or this fixture no longer demonstrates the difference"
        );
        // the closest part must be one of the walls or the base, and reported
        assert!(compound.part_a < 3, "the realising part must be identified, got {}", compound.part_a);
    }

    /// **The query a closest-pair API cannot express.** A plate lying across two separated blocks
    /// touches BOTH. `compound_distance` returns one pair by construction; `compound_contacts` must
    /// return two, because a solver given one loses a constraint and the plate rotates into the gap.
    #[test]
    fn a_plate_on_two_blocks_yields_two_contacts_not_one() {
        let blocks = CompoundHull::from_convex_parts(vec![
            box_pts(v(-1.0, 0.0, 0.0), v(0.3, 0.3, 0.3)),
            box_pts(v(1.0, 0.0, 0.0), v(0.3, 0.3, 0.3)),
        ])
        .expect("two blocks");
        // a plate resting on both, its underside exactly at z = 0.3
        let plate = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.35), v(1.5, 0.3, 0.05))])
            .expect("one plate");

        let closest = compound_distance(&blocks, &plate).expect("pairs exist");
        let all = compound_contacts(&blocks, &plate, 1e-6);
        eprintln!(
            "  plate on two blocks: closest pair reports 1 contact at {:.2e}; compound_contacts reports {} at {:?}",
            closest.distance, all.len(),
            all.iter().map(|c| (c.part_a, c.part_b)).collect::<Vec<_>>()
        );

        assert_eq!(all.len(), 2, "both supports must be reported, got {}", all.len());
        assert_ne!(all[0].part_a, all[1].part_a, "and they must be DIFFERENT blocks");
        for c in &all {
            assert!(c.distance <= 1e-6, "both are touching, got {:.3e}", c.distance);
        }
        // the closest-pair query, by construction, gives one of them and no way to know there is another
        assert!(
            [all[0].part_a, all[1].part_a].contains(&closest.part_a),
            "the closest pair must be one of the two, and is therefore an incomplete constraint set"
        );

        // and the margin is load-bearing: raise the plate and both contacts leave together
        let lifted = plate.transformed(&Iso::from_parts(Translation3::new(0.0, 0.0, 0.2), UnitQuaternion::identity()));
        assert_eq!(compound_contacts(&blocks, &lifted, 1e-6).len(), 0, "lifted clear, no contact is within a 1e-6 margin");
        assert_eq!(compound_contacts(&blocks, &lifted, 0.5).len(), 2, "but both are within 0.5");
        eprintln!("  lifted 0.2: 0 contacts within 1e-6, 2 within 0.5 — the margin selects the constraint set");
    }

    /// Rigid invariance: moving both bodies together cannot change the distance between them. This is
    /// the check an index or a frame error cannot pass, and it exercises `transformed`.
    #[test]
    fn moving_both_bodies_together_changes_nothing() {
        let a = u_bracket();
        let b = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 2.0), v(0.2, 0.2, 0.2))]).expect("one part");
        let base = compound_distance(&a, &b).expect("pairs");

        let tf = Iso::from_parts(
            Translation3::new(-3.0, 7.5, 0.25),
            UnitQuaternion::from_euler_angles(0.4, -0.9, 1.3),
        );
        let moved = compound_distance(&a.transformed(&tf), &b.transformed(&tf)).expect("pairs");
        eprintln!("  rigid invariance: {:.9} -> {:.9}", base.distance, moved.distance);
        assert!((moved.distance - base.distance).abs() < 1e-9, "distance moved: {} vs {}", moved.distance, base.distance);
        assert_eq!((moved.part_a, moved.part_b), (base.part_a, base.part_b), "and the same parts must realise it");
    }

    /// A mesh becomes collidable: the box mesh the geometry layer generates, hulled, must give the
    /// analytic box answer against a ball. This is the link that was missing — a loaded or generated
    /// mesh reaching the narrowphase at all.
    #[test]
    fn a_generated_mesh_becomes_collidable_and_matches_the_analytic_box() {
        let mesh = crate::primitive_mesh(&LinkGeometry::Box { size: v(0.4, 0.4, 0.4) }, 8).expect("a box generates");
        let c = CompoundHull::from_mesh_hull(&mesh).expect("a box mesh has a hull");
        assert_eq!(c.n_parts(), 1);
        assert_eq!(c.parts[0].pts.len(), 8, "a box's hull is its 8 corners, got {}", c.parts[0].pts.len());

        // against a ball on the x axis: gap = centre distance − half-extent − radius
        let ball = Ball { center: v(1.5, 0.0, 0.0), radius: 0.25 };
        let g = gjk(&c.parts[0], &ball);
        let want = 1.5 - 0.2 - 0.25;
        eprintln!("  hulled box mesh vs a ball: {:.9}, analytic {:.9}", g.distance, want);
        assert!((g.distance - want).abs() < 1e-6, "{} vs {want}", g.distance);

        // and the same box as an oriented Cuboid must agree, which is two independent shape
        // representations of the same solid rather than the code against itself
        let cube = Cuboid { center: v(0.0, 0.0, 0.0), half: v(0.2, 0.2, 0.2), rot: Matrix3::identity() };
        let g2 = gjk(&cube, &ball);
        assert!((g.distance - g2.distance).abs() < 1e-6, "hulled mesh {} vs analytic cuboid {}", g.distance, g2.distance);

        // the primitive route, straight from the shapes a description declares
        let (from_shapes, skipped) = CompoundHull::from_primitives(
            &[
                (LinkGeometry::Box { size: v(0.4, 0.4, 0.4) }, Iso::identity()),
                (LinkGeometry::Sphere { radius: 0.1 }, Iso::from_parts(Translation3::new(0.0, 0.0, 0.5), UnitQuaternion::identity())),
                (LinkGeometry::Mesh { uri: "x.stl".into(), scale: v(1.0, 1.0, 1.0) }, Iso::identity()),
            ],
            16,
        );
        let from_shapes = from_shapes.expect("two primitives are usable");
        assert_eq!(from_shapes.n_parts(), 2, "the box and the sphere become parts");
        assert_eq!(skipped, 1, "and the mesh reference is counted, not silently dropped");
    }

    /// `hull_support` is the over-estimate a broadphase wants, and it is named so a caller cannot
    /// reach it thinking it is the compound. Checked against the fact that makes it an over-estimate:
    /// it returns a point of the hull, which for a concave body lies outside the body.
    #[test]
    fn hull_support_over_estimates_and_is_not_reachable_as_the_compound() {
        let u = u_bracket();
        // straight up: the walls reach z = 1.0, and that is what the hull support must report
        let up = u.hull_support(&v(0.0, 0.0, 1.0));
        assert!((up.z - 1.0).abs() < 1e-12, "the tallest point is z = 1.0, got {:?}", up);
        // straight down: the base bottom at z = −0.2
        let down = u.hull_support(&v(0.0, 0.0, -1.0));
        assert!((down.z + 0.2).abs() < 1e-12, "the lowest point is z = −0.2, got {:?}", down);
        // it is an over-estimate of the SOLID: the hull's own volume exceeds the parts' total
        let hull = try_convex_hull_3d(&u.points().copied().collect::<Vec<_>>()).expect("hull");
        let parts_volume = 1.2 * 0.8 * 0.4 + 2.0 * (0.2 * 0.8 * 1.0);
        eprintln!("  U bracket: hull volume {:.4}, parts volume {:.4} ({:.1}% larger)", hull.volume(), parts_volume, 100.0 * (hull.volume() / parts_volume - 1.0));
        assert!(hull.volume() > parts_volume * 1.2, "the hull must be substantially larger: {:.4} vs {:.4}", hull.volume(), parts_volume);
    }

    /// ⛔ **Six mutations survived the first version of this test module**, and each one is fixed by a
    /// case below rather than by loosening anything. They are recorded because every one is a fixture
    /// weakness of a kind that recurs: a fixture whose parts are ordered conveniently, a boundary
    /// never sat on, a constructor never called with more than one input, an invariance test whose
    /// symmetry cancels the bug, and a sort whose keys are all equal.
    #[test]
    fn the_fixture_weaknesses_six_mutations_exposed() {
        // 1. THE CLOSEST PAIR MUST BE A MINIMUM, not the first pair examined. Part 0 is deliberately
        //    the FAR one, so returning the first pair gives the wrong answer.
        let ordered = CompoundHull::from_convex_parts(vec![
            box_pts(v(-5.0, 0.0, 0.0), v(0.2, 0.2, 0.2)), // far
            box_pts(v(0.6, 0.0, 0.0), v(0.2, 0.2, 0.2)),  // near
        ])
        .expect("two parts");
        let target = CompoundHull::from_convex_parts(vec![box_pts(v(1.5, 0.0, 0.0), v(0.2, 0.2, 0.2))]).expect("one");
        let c = compound_distance(&ordered, &target).expect("pairs");
        assert_eq!(c.part_a, 1, "the NEAR part is index 1, so a first-pair implementation reports 0");
        assert!((c.distance - 0.5).abs() < 1e-9, "the near gap is 0.5, got {:.6}", c.distance);

        // 2. THE MARGIN IS INCLUSIVE, sat on exactly. Two boxes whose gap is exactly 0.5, tested at
        //    margin 0.5: `<` excludes it, `<=` includes it, and only an exact fixture separates them.
        let gap_half = compound_contacts(&target, &ordered, 0.5);
        assert_eq!(gap_half.len(), 1, "a gap of exactly 0.5 must be INSIDE a margin of 0.5, got {}", gap_half.len());
        assert!(compound_contacts(&target, &ordered, 0.4999999).is_empty(), "and outside a slightly smaller one");

        // 3. `from_meshes` MUST HULL EACH MESH SEPARATELY. Hulling the union is precisely the error
        //    this module exists to prevent, and no test called it with more than one mesh.
        let left = crate::link_geometry::transform_mesh(
            &crate::primitive_mesh(&LinkGeometry::Box { size: v(0.4, 0.4, 0.4) }, 8).expect("box"),
            &Iso::from_parts(Translation3::new(-1.0, 0.0, 0.0), UnitQuaternion::identity()),
        );
        let right = crate::link_geometry::transform_mesh(
            &crate::primitive_mesh(&LinkGeometry::Box { size: v(0.4, 0.4, 0.4) }, 8).expect("box"),
            &Iso::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()),
        );
        let two = CompoundHull::from_meshes(&[left.clone(), right.clone()]).expect("two meshes");
        assert_eq!(two.n_parts(), 2, "two meshes must become TWO parts, not one hull over both");
        // a probe in the gap between them is free of the compound and inside the union's hull
        let probe = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.1, 0.1, 0.1))]).expect("one");
        let vs_compound = compound_distance(&two, &probe).expect("pairs");
        let union_hull = try_convex_hull_3d(&two.points().copied().collect::<Vec<_>>()).expect("hull");
        let vs_union = gjk(&ConvexPoints { pts: union_hull.verts }, &probe.parts[0]);
        eprintln!(
            "  two separated box meshes: probe in the gap is {:.4} from the compound (intersecting {}), {:.4} from the union hull (intersecting {})",
            vs_compound.distance, vs_compound.intersecting, vs_union.distance, vs_union.intersecting
        );
        assert!(!vs_compound.intersecting && vs_compound.distance > 0.5, "free of both boxes, got {:.4}", vs_compound.distance);
        assert!(vs_union.intersecting, "and inside the hull of their union — the error a per-mesh hull avoids");

        // 4. `transformed` MUST APPLY THE ROTATION. Moving BOTH bodies by the same transform cannot
        //    see a dropped rotation, because a pure translation of both preserves distance — the
        //    invariance test's own symmetry cancels the bug. Rotate ONE body instead, about a point
        //    where the rotation genuinely changes the gap.
        let bar = CompoundHull::from_convex_parts(vec![box_pts(v(1.0, 0.0, 0.0), v(0.8, 0.05, 0.05))]).expect("a long bar off-axis");
        let unit = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.1, 0.1, 0.1))]).expect("one");
        // ⛔ The probe must sit OFF the x = y diagonal. My first fixture put a symmetric cube at the
        // origin, and a quarter turn about z maps that configuration onto an equivalent one — the gap
        // came out 0.1000 both before and after, so the fixture could not see the rotation either.
        // At (0.9, 0.3, 0) the bar's long axis points at the probe before the turn and across it after.
        let probe = CompoundHull::from_convex_parts(vec![box_pts(v(0.9, 0.3, 0.0), v(0.1, 0.1, 0.1))]).expect("one");
        let before = compound_distance(&bar, &probe).expect("pairs").distance;
        // a quarter turn about z carries the bar onto the y axis, so it no longer points at the probe
        let spun = bar.transformed(&Iso::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, core::f64::consts::FRAC_PI_2),
        ));
        let after = compound_distance(&spun, &probe).expect("pairs").distance;
        eprintln!("  a bar rotated a quarter turn about z, probe at (0.9, 0.3, 0): gap {before:.4} -> {after:.4}");
        assert!((after - before).abs() > 0.4, "a rotation must change this gap substantially: {before:.4} vs {after:.4}");
        // and the rotated bar's own extent is where the rotation put it
        let tip = spun.hull_support(&v(0.0, 1.0, 0.0));
        assert!(tip.y > 1.7, "the bar's far end must now lie along +y, got {tip:?}");

        // 5. THE SORT IS CLOSEST-FIRST, on keys that actually differ. The plate fixture has both
        //    contacts at the same distance, so any order passes it.
        let staggered = CompoundHull::from_convex_parts(vec![
            box_pts(v(0.0, 0.0, 3.0), v(0.2, 0.2, 0.2)), // 3 away
            box_pts(v(0.0, 0.0, 1.0), v(0.2, 0.2, 0.2)), // 1 away
            box_pts(v(0.0, 0.0, 2.0), v(0.2, 0.2, 0.2)), // 2 away
        ])
        .expect("three parts");
        let all = compound_contacts(&staggered, &unit, 5.0);
        assert_eq!(all.len(), 3, "all three are within 5.0");
        for w in all.windows(2) {
            assert!(w[0].distance <= w[1].distance, "must be sorted closest first, got {:?}", all.iter().map(|c| c.distance).collect::<Vec<_>>());
        }
        assert_eq!(all[0].part_a, 1, "the nearest is part 1, not part 0");
        eprintln!("  staggered parts sorted closest-first: {:?} from parts {:?}",
            all.iter().map(|c| (c.distance * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            all.iter().map(|c| c.part_a).collect::<Vec<_>>());

        // 6. THE COPLANAR CASE IS DOUBLY GUARDED, which is why no single mutation exposes it — and
        //    that is a fact I had to measure rather than assume. `try_convex_hull_3d` refuses a
        //    coplanar point set at the SEED (no fourth affinely-independent point exists, so `p3` is
        //    `None`: verified directly on this grid) AND at the TAIL (a degenerate seed tetrahedron
        //    cannot yield four usable faces over four distinct vertices). Removing either guard alone
        //    leaves the grid refused; removing BOTH lets it through, which is the mutation that
        //    actually fails this assertion. Defence in depth, stated so a later reader does not delete
        //    one of them believing the other is decorative.
        let mut grid = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                grid.push(v(i as f64 * 0.3, j as f64 * 0.3, 0.0));
            }
        }
        assert_eq!(grid.len(), 16);
        assert!(try_convex_hull_3d(&grid).is_none(), "a 16-point coplanar grid has no 3-D hull and must be refused");
        // a coplanar set that is not axis-aligned, so the refusal is not an artifact of z being zero
        let tilted: Vec<Vector3<f64>> = grid.iter().map(|p| v(p.x, p.y, 0.5 * p.x - 0.25 * p.y)).collect();
        assert!(try_convex_hull_3d(&tilted).is_none(), "a TILTED coplanar set must be refused too");
        // and one point lifted off the plane makes it a solid again, so the guard is not simply always-None
        let mut solid = grid.clone();
        solid.push(v(0.45, 0.45, 0.4));
        assert!(try_convex_hull_3d(&solid).is_some(), "lifting one point off the plane must yield a hull");
    }

    /// **A penetrating pair must report a NEGATIVE gap and a usable normal**, which is what a solver
    /// takes and what the first version of this module could not produce. The convention is checked
    /// FUNCTIONALLY: translating part A by `+gap·normal` must actually separate the pair. That is
    /// stronger than asserting a sign, because it fails for a normal that points the wrong way, for one
    /// scaled wrongly, and for a depth that is right in magnitude but measured along the wrong axis.
    #[test]
    fn a_penetrating_pair_reports_a_negative_gap_and_a_normal_that_separates_it() {
        // two unit boxes overlapping by 0.25 along x
        let a = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let b = CompoundHull::from_convex_parts(vec![box_pts(v(0.75, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let c = compound_distance(&a, &b).expect("pairs");

        assert!(c.intersecting, "these overlap");
        assert_eq!(c.distance, 0.0, "GJK reports 0 for any intersecting pair, which is why `gap` exists");
        assert!((c.gap + 0.25).abs() < 1e-6, "the signed gap must be −depth = −0.25, got {:.6}", c.gap);
        assert!((c.normal.norm() - 1.0).abs() < 1e-9, "the normal must be a unit vector, got {:.6}", c.normal.norm());
        assert!(c.normal.x.abs() > 0.99, "the overlap is along x, so the normal must be too: {:?}", c.normal);

        // THE FUNCTIONAL CHECK, and it is the SAME statement the separated test makes: translating A
        // by gap·normal brings the pair to touching. `gap` is negative here, so it moves A away from B
        // by exactly the depth. One statement covering both regimes is what stops the two branches
        // drifting to opposite conventions, which is exactly what they had done.
        let sep = a.transformed(&Iso::from_parts(
            Translation3::from(c.gap * c.normal),
            UnitQuaternion::identity(),
        ));
        let after = compound_distance(&sep, &b).expect("pairs");
        eprintln!(
            "  overlapping boxes: distance {:.3}, gap {:.6}, normal {:?} -> after translating A by gap·normal: gap {:.2e}, intersecting {}",
            c.distance, c.gap, c.normal, after.gap, after.intersecting
        );
        assert!(after.gap.abs() < 1e-6, "after the translation the pair must be exactly touching, got {:.3e}", after.gap);

        // and moving A the WRONG way must drive it deeper, which is what pins the direction
        let deeper = a.transformed(&Iso::from_parts(
            Translation3::from(-c.gap * c.normal),
            UnitQuaternion::identity(),
        ));
        let worse = compound_distance(&deeper, &b).expect("pairs");
        assert!(worse.gap < c.gap - 1e-9, "the opposite translation must deepen the overlap: {:.6} vs {:.6}", worse.gap, c.gap);
    }

    /// A separated pair keeps GJK's answer, and its normal is the witness axis. The two branches must
    /// agree at the boundary, which is the only place a two-branch function can be discontinuous.
    #[test]
    fn a_separated_pair_reports_a_positive_gap_and_the_two_branches_meet_at_touching() {
        let a = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let far = CompoundHull::from_convex_parts(vec![box_pts(v(2.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let c = compound_distance(&a, &far).expect("pairs");
        assert!(!c.intersecting);
        assert!((c.gap - 1.0).abs() < 1e-9, "separated: gap is +distance = 1.0, got {:.6}", c.gap);
        assert!((c.gap - c.distance).abs() < 1e-15, "gap and distance agree when separated");
        assert!((c.normal - Vector3::x()).norm() < 1e-6, "the normal points from A toward B, which is +x here, got {:?}", c.normal);

        // THE SHARED FUNCTIONAL PROPERTY, in the separated regime: translating A by gap·normal brings
        // the pair to touching. The penetrating test checks the same statement, which is what makes it
        // one convention rather than two that happen to agree on a sign.
        let closed = a.transformed(&Iso::from_parts(Translation3::from(c.gap * c.normal), UnitQuaternion::identity()));
        let after = compound_distance(&closed, &far).expect("pairs");
        assert!(after.gap.abs() < 1e-6, "moving A by gap·normal must close a separation to touching, got {:.3e}", after.gap);

        // walk across the boundary: the gap must be continuous and monotone through zero
        let mut prev = f64::INFINITY;
        let mut row = Vec::new();
        for d in [1.2f64, 1.05, 1.0, 0.95, 0.8] {
            let b = CompoundHull::from_convex_parts(vec![box_pts(v(d, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
            let g = compound_distance(&a, &b).expect("pairs").gap;
            row.push((d, (g * 1e6).round() / 1e6));
            assert!(g < prev + 1e-9, "the gap must fall as the boxes approach: {row:?}");
            prev = g;
        }
        eprintln!("  gap across the touching boundary (centre distance, gap): {row:?}");
        // exactly touching at 1.0, and the two branches must not disagree there by more than round-off
        assert!(row[2].1.abs() < 1e-6, "at a centre distance of 1.0 the gap must be ~0, got {}", row[2].1);
        assert!(row[3].1 < 0.0 && row[4].1 < row[3].1, "past touching the gap must go negative and keep falling: {row:?}");
    }

    /// **The ordering is by signed gap**, so a deeper penetration outranks a shallower one and both
    /// outrank any separation. Ordering by `distance` would tie every intersecting pair at zero and
    /// pick an arbitrary one — which is what the first version did.
    #[test]
    fn the_deepest_penetration_outranks_a_shallower_one_and_both_outrank_separation() {
        let probe = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let three = CompoundHull::from_convex_parts(vec![
            box_pts(v(0.0, 3.0, 0.0), v(0.5, 0.5, 0.5)),  // 0: separated by 2.0
            box_pts(v(0.9, 0.0, 0.0), v(0.5, 0.5, 0.5)),  // 1: overlapping by 0.1
            box_pts(v(0.0, 0.0, 0.6), v(0.5, 0.5, 0.5)),  // 2: overlapping by 0.4, the deepest
        ])
        .expect("three parts");

        let closest = compound_distance(&three, &probe).expect("pairs");
        assert_eq!(closest.part_a, 2, "the DEEPEST overlap must win, not whichever zero-distance pair came first");
        assert!((closest.gap + 0.4).abs() < 1e-6, "and its gap is −0.4, got {:.6}", closest.gap);

        let all = compound_contacts(&three, &probe, 0.0);
        assert_eq!(all.len(), 2, "only the two overlapping pairs have gap <= 0");
        // ⛔ A NEGATIVE margin selects only the DEEP overlaps, which is a query the signed gap makes
        // possible and `distance` cannot express — every intersecting pair has distance 0, so a
        // distance-based margin admits both or neither.
        let deep = compound_contacts(&three, &probe, -0.2);
        assert_eq!(deep.len(), 1, "only the 0.4-deep overlap is past a 0.2 penetration threshold, got {}", deep.len());
        assert_eq!(deep[0].part_a, 2, "and it is the deep one");
        assert!(compound_contacts(&three, &probe, -0.5).is_empty(), "nothing penetrates deeper than 0.5");
        eprintln!("  margin -0.2 selects {} of 3 pairs (the deep overlap only); -0.5 selects 0", deep.len());
        assert_eq!((all[0].part_a, all[1].part_a), (2, 1), "sorted deepest first");
        assert!(all[0].gap < all[1].gap, "and their gaps are ordered: {:.4} then {:.4}", all[0].gap, all[1].gap);
        let with_sep = compound_contacts(&three, &probe, 2.5);
        assert_eq!(with_sep.len(), 3, "a wider margin admits the separated pair too");
        assert_eq!(with_sep[2].part_a, 0, "and it sorts LAST, because its gap is the largest");
        eprintln!(
            "  gaps sorted deepest-first: {:?} from parts {:?}",
            with_sep.iter().map(|c| (c.gap * 1e4).round() / 1e4).collect::<Vec<_>>(),
            with_sep.iter().map(|c| c.part_a).collect::<Vec<_>>()
        );
    }

    /// Refusals. Every one is a real export or decomposition artifact rather than a hypothetical.
    #[test]
    fn degenerate_and_empty_compounds_are_refused() {
        assert!(CompoundHull::from_convex_parts(vec![]).is_none(), "no parts is not a body");
        assert!(CompoundHull::from_meshes(&[]).is_none(), "no meshes is not a body");
        // a part with fewer than four points has no volume and supports a point, not a solid
        assert!(CompoundHull::from_convex_parts(vec![ConvexPoints { pts: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)] }]).is_none(), "a triangle is not a 3-D part");
        // a non-finite vertex
        assert!(CompoundHull::from_convex_parts(vec![ConvexPoints { pts: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(f64::NAN, 0.0, 1.0)] }]).is_none(), "a non-finite vertex");
        // a FLAT mesh has no 3-D hull, which is the exporter artifact that used to panic
        let flat = TriMesh3 {
            verts: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(1.0, 1.0, 0.0), v(0.0, 1.0, 0.0)],
            tris: vec![[0, 1, 2], [0, 2, 3]],
        };
        assert!(try_convex_hull_3d(&flat.verts).is_none(), "a coplanar point set has no 3-D hull");
        assert!(CompoundHull::from_mesh_hull(&flat).is_none(), "and a flat mesh cannot become a part");
        // ⛔ the panicking sibling is still reachable and still panics, which is why `from_meshes`
        // uses the refusing one. Asserted so the delegation cannot be quietly reversed.
        assert!(
            std::panic::catch_unwind(|| crate::convex_hull_3d(&flat.verts)).is_err(),
            "convex_hull_3d documents that it panics on a degenerate set; if that changed, its doc must too"
        );

        // margins
        let a = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        let b = CompoundHull::from_convex_parts(vec![box_pts(v(3.0, 0.0, 0.0), v(0.5, 0.5, 0.5))]).expect("one");
        // a negative margin is no longer a refusal: it selects only pairs penetrating deeper than
        // |margin|, and these two are 2.0 apart, so nothing qualifies
        assert!(compound_contacts(&a, &b, -1.0).is_empty(), "separated pairs never satisfy a negative margin");
        // a NaN margin: every `gap <= NaN` comparison is false, so removing the guard happens to give
        // the same empty answer. The guard stays because relying on that is relying on NaN ordering,
        // which is exactly the class of bug this workspace has a gate for — and a NaN margin reaching
        // a solver as "no contacts" rather than as an error is worth refusing explicitly.
        assert!(compound_contacts(&a, &b, f64::NAN).is_empty(), "a non-finite margin selects nothing");
        // ⛔ +INFINITY is refused too, and that is the behaviour rather than my first guess. I asserted
        // it selects everything; `INFINITY.is_finite()` is false, so the guard returns empty. Refusing
        // non-finite input is this crate's posture and a caller wanting every pair should pass a large
        // FINITE margin. Asserting the real answer is also what catches the guard's removal, since an
        // unguarded `gap <= INFINITY` admits all of them.
        assert!(compound_contacts(&a, &b, f64::INFINITY).is_empty(), "+infinity is non-finite and is refused, not treated as universal");
        assert_eq!(compound_contacts(&a, &b, 1e9).len(), 1, "a large FINITE margin is how a caller asks for every pair");
        assert!(compound_contacts(&a, &b, 1.0).is_empty(), "2.0 apart is outside a 1.0 margin");
        assert_eq!(compound_contacts(&a, &b, 2.5).len(), 1, "and inside a 2.5 one");
        assert!(compound_distance(&a, &CompoundHull::default()).is_none(), "an empty compound has no closest pair");
    }

    /// ⭐ **The constructor that closes the failure table at the top of this module.**
    ///
    /// Same bracket, same probe. [`CompoundHull::from_meshes`] gives one part and swallows the
    /// channel; [`CompoundHull::from_meshes_decomposed`] gives the three boxes the bracket is made of
    /// and leaves the mouth open.
    ///
    /// ⛔ The last assertion is the one that matters for a caller's trust: an unsealed mesh returns
    /// `None`. A silent fallback to the hull would put the loss back where nobody can see it.
    #[test]
    fn a_link_declared_as_one_concave_mesh_gets_its_channel_back() {
        use crate::acd::AcdOptions;
        use crate::link_geometry::{primitive_mesh, LinkGeometry};

        let mut bracket = TriMesh3 { verts: Vec::new(), tris: Vec::new() };
        for (g, at) in [
            (LinkGeometry::Box { size: v(1.2, 0.8, 0.4) }, v(0.0, 0.0, 0.0)),
            (LinkGeometry::Box { size: v(0.2, 0.8, 1.0) }, v(-0.5, 0.0, 0.5)),
            (LinkGeometry::Box { size: v(0.2, 0.8, 1.0) }, v(0.5, 0.0, 0.5)),
        ] {
            let m = primitive_mesh(&g, 24).expect("a primitive meshes");
            let base = bracket.verts.len();
            bracket.verts.extend(m.verts.iter().map(|q| q + at));
            bracket.tris.extend(m.tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        let pad = primitive_mesh(&LinkGeometry::Box { size: v(0.3, 0.3, 0.3) }, 24).expect("meshes");
        let probe = CompoundHull::from_convex_parts(vec![box_pts(v(0.0, 0.0, 0.7), v(0.08, 0.08, 0.08))])
            .expect("a probe");

        let hulled = CompoundHull::from_meshes(&[bracket.clone()]).expect("the bracket hulls");
        assert_eq!(hulled.n_parts(), 1, "the control is one part, or nothing below is a comparison");
        let hc = compound_distance(&hulled, &probe).expect("hull vs probe");
        assert!(hc.intersecting, "the hull must swallow the mouth, or the probe is in the wrong place");

        let opts = AcdOptions::default();
        let (split, reports) =
            CompoundHull::from_meshes_decomposed(&[bracket.clone(), pad], &opts).expect("both decompose");
        assert_eq!(reports.len(), 2, "one report per mesh, in order");
        assert!(reports[0].parts >= 2, "the bracket is concave; {} part(s) is not a decomposition", reports[0].parts);
        assert_eq!(reports[1].parts, 1, "the pad is a box and must stay one part");
        assert_eq!(split.n_parts(), reports.iter().map(|r| r.parts).sum::<usize>(), "every part must reach the compound");

        let sc = compound_distance(&split, &probe).expect("parts vs probe");
        eprintln!(
            "  one concave mesh: hull {} part, gap {:+.4} m | decomposed {} parts, gap {:+.4} m",
            hulled.n_parts(), hc.gap, reports[0].parts, sc.gap
        );
        assert!(!sc.intersecting && sc.gap > 0.0, "the mouth must be free once decomposed, got {:+.4}", sc.gap);

        // ⛔ and an unsealed mesh is refused, not quietly hulled.
        //
        // ⚠ A standalone box, not the bracket: dropping the bracket's last two triangles removes a
        // face of a WALL where it is buried inside the slab, and an exterior flood fill cannot reach
        // an internal hole — so that mesh voxelises correctly and does not leak. A hole only matters
        // when it is exposed.
        let mut holed = primitive_mesh(&LinkGeometry::Box { size: v(0.5, 0.5, 0.5) }, 24).expect("meshes");
        let n = holed.tris.len();
        holed.tris.truncate(n - 2);
        assert!(
            CompoundHull::from_meshes_decomposed(&[holed], &opts).is_none(),
            "an unsealed mesh must be refused; falling back to the hull would hide the loss"
        );
    }
}
