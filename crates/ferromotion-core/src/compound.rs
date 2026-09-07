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
//! (the V-HACD/CoACD operation) and it is a module of its own, not a function here. Until it exists,
//! a caller with one concave mesh has three honest options, in order of preference: use the
//! `<collision>` primitives the description already declares (see
//! [`primitive_mesh`](crate::primitive_mesh)), supply the part meshes separately if the exporter wrote
//! them that way, or accept the convex hull and know what it costs — which is the number this
//! module's tests print.

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
#[derive(Clone, Copy, Debug)]
pub struct CompoundContact {
    pub distance: f64,
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

/// **The closest pair between two compounds**, over all part pairs. What a planner wants: a single
/// clearance number, and which pieces set it.
///
/// `None` if either compound has no parts. If any pair intersects, the returned contact is an
/// intersecting one with distance `0` — but see [`compound_contacts`], because there may be others.
pub fn compound_distance(a: &CompoundHull, b: &CompoundHull) -> Option<CompoundContact> {
    let mut best: Option<CompoundContact> = None;
    for (i, pa) in a.parts.iter().enumerate() {
        for (j, pb) in b.parts.iter().enumerate() {
            let g: GjkResult = gjk(pa, pb);
            if !g.distance.is_finite() {
                continue;
            }
            let c = CompoundContact {
                distance: g.distance,
                intersecting: g.intersecting,
                witness_a: g.witness_a,
                witness_b: g.witness_b,
                part_a: i,
                part_b: j,
            };
            if best.is_none_or(|bc| c.distance < bc.distance) {
                best = Some(c);
            }
        }
    }
    best
}

/// **Every part pair within `margin`** — the constraint set a contact solver needs.
///
/// Returned sorted by distance, closest first, so a caller that must truncate drops the least
/// important. An intersecting pair has distance `0` and is always included when `margin >= 0`.
///
/// ⛔ This is the query that distinguishes a compound from a hull. Taking only
/// [`compound_distance`]'s answer and handing it to a solver loses every simultaneous contact, and a
/// body with one of its two supports missing sinks through the other. `margin` should be at least the
/// solver's own contact-activation distance.
pub fn compound_contacts(a: &CompoundHull, b: &CompoundHull, margin: f64) -> Vec<CompoundContact> {
    if !margin.is_finite() || margin < 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, pa) in a.parts.iter().enumerate() {
        for (j, pb) in b.parts.iter().enumerate() {
            let g = gjk(pa, pb);
            if g.distance.is_finite() && g.distance <= margin {
                out.push(CompoundContact {
                    distance: g.distance,
                    intersecting: g.intersecting,
                    witness_a: g.witness_a,
                    witness_b: g.witness_b,
                    part_a: i,
                    part_b: j,
                });
            }
        }
    }
    out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
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
        assert!(compound_contacts(&a, &b, -1.0).is_empty(), "a negative margin selects nothing");
        assert!(compound_contacts(&a, &b, f64::NAN).is_empty(), "a non-finite margin selects nothing");
        assert!(compound_contacts(&a, &b, 1.0).is_empty(), "2.0 apart is outside a 1.0 margin");
        assert_eq!(compound_contacts(&a, &b, 2.5).len(), 1, "and inside a 2.5 one");
        assert!(compound_distance(&a, &CompoundHull::default()).is_none(), "an empty compound has no closest pair");
    }
}
