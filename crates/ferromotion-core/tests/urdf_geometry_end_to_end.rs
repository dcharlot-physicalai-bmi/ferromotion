//! **The geometry route must produce the same body as a stated one, and CI must check it.**
//!
//! `examples/urdf_to_body.rs` demonstrates the whole chain and asserts its own claims — but CI runs
//! `cargo test --workspace`, which *compiles* examples and never runs them, so those assertions are a
//! demonstration and not a gate. This workspace has been bitten by exactly that distinction before: a
//! report that exits 0 is not a gate, and it had printed the bug all along.
//!
//! So the load-bearing claim lives here, where `cargo test` reaches it: a URDF that states only its
//! shapes yields a body, and that body agrees with one whose inertias were computed by hand from the
//! closed forms — to the inscribed-prism deficit of the tessellation and nothing more.
//!
//! # What this gate does and does not cover, measured by mutation
//!
//! `l1` deliberately declares BOTH a visual box and a collision cylinder, of different sizes, so that
//! preferring the visual geometry changes the mass and this gate sees it. A first version gave every
//! link collision geometry only, and the "prefer visual over collision" mutation passed here — which
//! matters, because a real asset usually has both and the visual mesh is the fine one.
//!
//! ⛔ One mutation still passes this gate and is caught elsewhere, recorded so the coverage is not
//! overstated: **dropping the parallel-axis shift in `solid_inertia`**. Every primitive here is
//! centred in its own frame, so its centre of mass is the origin and the shift is identically zero.
//! `mesh_io`'s unit cube sits at the origin CORNER, so its COM is `(0.5, 0.5, 0.5)` and that test is
//! what pins the shift. A gate whose fixtures are all centred cannot see a parallel-axis error.

use ferromotion_core::{
    forward_dynamics, from_urdf_full, geometry_from_urdf, gravity_vector, primitive_link_inertia, LinkInertia,
};
use nalgebra::Vector3;

const DENSITY: f64 = 2700.0;
const SEGMENTS: usize = 256;

/// A two-joint arm: a box base and two cylinders rotated onto the link x axis. No `<inertial>`.
const SHAPES_ONLY: &str = r#"<robot name="arm2">
  <link name="base">
    <collision><origin xyz="0 0 0.04" rpy="0 0 0"/><geometry><box size="0.16 0.16 0.08"/></geometry></collision>
  </link>
  <link name="l1">
    <visual><origin xyz="0.15 0 0" rpy="0 0 0"/><geometry><box size="0.30 0.09 0.09"/></geometry></visual>
    <collision><origin xyz="0.15 0 0" rpy="0 1.5707963267948966 0"/>
      <geometry><cylinder radius="0.035" length="0.30"/></geometry></collision>
  </link>
  <link name="l2">
    <collision><origin xyz="0.10 0 0" rpy="0 1.5707963267948966 0"/>
      <geometry><cylinder radius="0.028" length="0.20"/></geometry></collision>
  </link>
  <link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.08" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="60" velocity="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.30 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="40" velocity="3"/></joint>
  <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/>
    <origin xyz="0.20 0 0" rpy="0 0 0"/></joint>
</robot>"#;

/// A cylinder rotated +π/2 about y: the axial term `mr²/2` lands on the link's xx, the radial
/// `m(3r²+l²)/12` on yy and zz.
fn cylinder_rotated(r: f64, l: f64) -> (f64, [f64; 3]) {
    let m = DENSITY * core::f64::consts::PI * r * r * l;
    (m, [0.5 * m * r * r, m * (3.0 * r * r + l * l) / 12.0, m * (3.0 * r * r + l * l) / 12.0])
}

/// Build the same robot with hand-computed `<inertial>` blocks, which is the independent statement of
/// the same physics that makes this a cross-check rather than a self-comparison.
fn with_stated_inertials() -> String {
    let (m1, i1) = cylinder_rotated(0.035, 0.30);
    let (m2, i2) = cylinder_rotated(0.028, 0.20);
    SHAPES_ONLY
        .replace(
            r#"<link name="l1">"#,
            &format!(
                r#"<link name="l1"><inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="{m1}"/><inertia ixx="{}" ixy="0" ixz="0" iyy="{}" iyz="0" izz="{}"/></inertial>"#,
                i1[0], i1[1], i1[2]
            ),
        )
        .replace(
            r#"<link name="l2">"#,
            &format!(
                r#"<link name="l2"><inertial><origin xyz="0.10 0 0" rpy="0 0 0"/><mass value="{m2}"/><inertia ixx="{}" ixy="0" ixz="0" iyy="{}" iyz="0" izz="{}"/></inertial>"#,
                i2[0], i2[1], i2[2]
            ),
        )
}

fn derived_inertias() -> Vec<LinkInertia> {
    let geo = geometry_from_urdf(SHAPES_ONLY).expect("the description parses");
    ["l1", "l2"]
        .iter()
        .map(|name| {
            let (li, skipped) = primitive_link_inertia(&geo, name, DENSITY, SEGMENTS);
            assert_eq!(skipped, 0, "{name} is all primitives, so nothing may be skipped");
            li.unwrap_or_else(|| panic!("{name} must yield an inertia from its geometry"))
        })
        .collect()
}

/// **The end-to-end claim.** Shapes in, a body out, and that body is the one the closed forms
/// describe — checked on mass, on the inertia tensor, and on the torque the dynamics actually
/// produce, because agreeing on the tensor and disagreeing on the torque would mean the body is
/// assembled wrongly even though each part is right.
#[test]
fn a_shapes_only_urdf_yields_the_same_body_as_a_stated_one() {
    let (robot, stated) = from_urdf_full(SHAPES_ONLY, "base", "tool").expect("the chain builds");
    let (_, hand) = from_urdf_full(&with_stated_inertials(), "base", "tool").expect("the stated version builds");
    let derived = derived_inertias();

    assert_eq!(robot.dof(), 2, "the fixture is a two-joint chain");
    assert_eq!(derived.len(), hand.len(), "one inertia per joint in both routes");

    // The starting point: the description states NOTHING, which is the body this crate could build
    // before the geometry route existed. Asserting it keeps the comparison honest — if a future
    // change starts inferring inertials inside `from_urdf_full`, this test would otherwise silently
    // become a comparison of that inference against itself.
    let stated_mass: f64 = stated.iter().map(|li| li.mass).sum();
    assert_eq!(stated_mass, 0.0, "the shapes-only description must state no mass at all, got {stated_mass}");

    // The inscribed-prism deficit is the only error budget this comparison is allowed.
    let deficit = 1.0 - (SEGMENTS as f64 / (2.0 * core::f64::consts::PI)) * (2.0 * core::f64::consts::PI / SEGMENTS as f64).sin();
    let budget = 4.0 * deficit; // the inertia carries r² terms, so twice the volume error, with margin

    let mut worst_mass = 0.0f64;
    let mut worst_inertia = 0.0f64;
    for (i, name) in ["l1", "l2"].iter().enumerate() {
        assert!(hand[i].mass > 0.1, "{name}: the oracle must state a real mass, got {}", hand[i].mass);
        worst_mass = worst_mass.max((derived[i].mass / hand[i].mass - 1.0).abs());
        assert!(
            (derived[i].com - hand[i].com).norm() < 1e-9,
            "{name}: the centres of mass must coincide, {:?} vs {:?}", derived[i].com, hand[i].com
        );
        for k in 0..3 {
            worst_inertia = worst_inertia.max((derived[i].inertia[(k, k)] / hand[i].inertia[(k, k)] - 1.0).abs());
        }
        // the rotation must have landed the axial term on xx, or the two routes agree by accident
        assert!(
            derived[i].inertia[(0, 0)] < 0.5 * derived[i].inertia[(1, 1)],
            "{name}: a cylinder rotated onto x must have its SMALL term on xx, got diag({:.6}, {:.6}, {:.6})",
            derived[i].inertia[(0, 0)], derived[i].inertia[(1, 1)], derived[i].inertia[(2, 2)]
        );
    }
    eprintln!("  geometry vs hand-computed: worst mass {worst_mass:.3e}, worst inertia {worst_inertia:.3e}, budget {budget:.3e}");
    assert!(worst_mass < budget, "mass disagreement {worst_mass:.3e} exceeds the tessellation budget {budget:.3e}");
    assert!(worst_inertia < budget, "inertia disagreement {worst_inertia:.3e} exceeds the tessellation budget {budget:.3e}");

    // And the torque, which is what a user actually consumes.
    let g = Vector3::new(0.0, 0.0, -9.81);
    let (q, qd) = ([0.0, 0.4], [0.0, 0.0]);
    let td = gravity_vector(&robot, &derived, &q, g);
    let th = gravity_vector(&robot, &hand, &q, g);
    let worst_tau = (0..robot.dof()).map(|i| (td[i] - th[i]).abs()).fold(0.0f64, f64::max);
    eprintln!("  gravity torque: geometry ({:.5}, {:.5}), stated ({:.5}, {:.5}), worst diff {worst_tau:.3e} N·m", td[0], td[1], th[0], th[1]);
    assert!(worst_tau < 1e-3, "the two bodies must produce the same torque, worst {worst_tau:.3e} N·m");
    assert!(
        td[1].abs() > 1e-2,
        "joint 2 must carry a real gravity load at this posture or the torque comparison is vacuous, got {:.3e}", td[1]
    );

    // the derived body integrates
    let qdd = forward_dynamics(&robot, &derived, &q, &qd, &[0.0, 0.0], g);
    assert!(qdd.iter().all(|a| a.is_finite()), "the geometry-derived body must integrate, got {qdd:?}");
    assert!(qdd[1].abs() > 1.0, "and actually accelerate under gravity, got {:.3e}", qdd[1]);
}

/// A link whose collision geometry is a mesh reference must report how many assets it needs rather
/// than silently yielding nothing — the distinction `primitive_link_inertia`'s second return value
/// exists for, checked here because a caller's error handling depends on it.
#[test]
fn a_mesh_only_link_reports_what_it_needs_rather_than_failing_silently() {
    let xml = SHAPES_ONLY.replace(
        r#"<geometry><cylinder radius="0.028" length="0.20"/></geometry>"#,
        r#"<geometry><mesh filename="package://arm2/meshes/l2.stl" scale="0.001 0.001 0.001"/></geometry>"#,
    );
    let geo = geometry_from_urdf(&xml).expect("parses");

    let (li, skipped) = primitive_link_inertia(&geo, "l2", DENSITY, SEGMENTS);
    assert!(li.is_none(), "a mesh-only link cannot be integrated without its asset");
    assert_eq!(skipped, 1, "and the caller must be told ONE asset is missing, not merely handed None");

    // the URI and its scale survive to the caller, which is what makes the asset fetchable
    let m = geo
        .iter()
        .find(|g| matches!(&g.geometry, ferromotion_core::LinkGeometry::Mesh { .. }))
        .expect("the mesh reference is reachable");
    match &m.geometry {
        ferromotion_core::LinkGeometry::Mesh { uri, scale } => {
            assert_eq!(uri, "package://arm2/meshes/l2.stl");
            assert_eq!(*scale, Vector3::new(0.001, 0.001, 0.001), "the mm scale must survive");
            let resolved = ferromotion_core::resolve_uri(uri, &[("arm2", "/opt/share/arm2")]);
            assert_eq!(resolved.as_deref(), Some("/opt/share/arm2/meshes/l2.stl"), "and resolve against a package table");
        }
        other => panic!("expected a mesh, got {other:?}"),
    }

    // l1 is untouched and still fully described, so the failure is scoped to the link that needs an
    // asset rather than poisoning the whole description
    let (li1, skipped1) = primitive_link_inertia(&geo, "l1", DENSITY, SEGMENTS);
    assert!(li1.is_some() && skipped1 == 0, "l1 must be unaffected by l2's missing asset");
    eprintln!("  l2 needs 1 asset and yields nothing; l1 is unaffected and yields {:.4} kg", li1.unwrap().mass);
}
