//! **From a URDF with no inertials to a body that runs dynamics** — the whole chain, end to end.
//!
//! A quickly-written robot description states its shapes and leaves out the mass properties, because
//! the shapes are what a person knows and an inertia tensor is what they would have to look up. Such a
//! description was unusable for dynamics: every algorithm in this crate takes a `LinkInertia` vector,
//! and nothing could produce one from geometry.
//!
//! This runs the chain: parse the URDF, read the geometry `urdf_rs` had already parsed, generate the
//! primitive meshes, integrate each one at a stated density, compose the parts of each link about
//! their combined centre of mass, and hand the result to `gravity_vector` and `forward_dynamics`.
//!
//! **The oracle is a second URDF.** The same robot, same shapes, with `<inertial>` blocks whose
//! numbers were computed by hand from the closed forms — `m(b²+c²)/12` for a box, `mr²/2` and
//! `m(3r²+l²)/12` for a cylinder. If the geometry route is right, the two descriptions must produce
//! the same body and therefore the same torques. That is a cross-check between two independent
//! statements of the same physics, not a comparison of the code against itself.
//!
//! Run: `cargo run -p ferromotion-core --example urdf_to_body`

use ferromotion_core::{
    forward_dynamics, from_urdf_full, geometry_from_urdf, gravity_vector, primitive_link_inertia, LinkInertia,
};
use nalgebra::Vector3;

/// Aluminium, so the numbers are a real part rather than a unit-mass abstraction.
const DENSITY: f64 = 2700.0;
/// Circumferential resolution for the cylinders. The inscribed-prism deficit is
/// `1 − (n/2π)·sin(2π/n)`, so 256 sides is 1.0e-4 of the volume — stated rather than assumed.
const SEGMENTS: usize = 256;

/// Shapes only. No `<inertial>` anywhere: this is the description a person writes.
const SHAPES_ONLY: &str = r#"<robot name="arm3">
  <link name="base">
    <collision><origin xyz="0 0 0.04" rpy="0 0 0"/><geometry><box size="0.16 0.16 0.08"/></geometry></collision>
  </link>
  <link name="l1">
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

/// A box of full extents `a×b×c` at density `rho`: the closed forms, so the fixture below is derived
/// rather than copied from a tool.
fn box_inertial(a: f64, b: f64, c: f64) -> (f64, [f64; 3]) {
    let m = DENSITY * a * b * c;
    (m, [m * (b * b + c * c) / 12.0, m * (a * a + c * c) / 12.0, m * (a * a + b * b) / 12.0])
}

/// A cylinder of radius `r` and length `l` about its own axis (z): `I_zz = mr²/2`,
/// `I_xx = I_yy = m(3r²+l²)/12`. The description rotates it by +π/2 about y, which swaps the roles of
/// x and z, so the fixture states the tensor in the LINK frame — the same thing the geometry route
/// computes by transforming the part.
fn cylinder_inertial_rotated(r: f64, l: f64) -> (f64, [f64; 3]) {
    let m = DENSITY * core::f64::consts::PI * r * r * l;
    let (axial, radial) = (0.5 * m * r * r, m * (3.0 * r * r + l * l) / 12.0);
    // +π/2 about y sends the part's z to the link's x, so the axial term lands on xx
    (m, [axial, radial, radial])
}

fn main() {
    println!("\nFROM A URDF WITH NO INERTIALS TO A BODY THAT RUNS DYNAMICS\n");

    // ---- the geometry route: shapes in, inertias out ----
    let geo = geometry_from_urdf(SHAPES_ONLY).expect("the description parses");
    println!("  the description declares {} shapes across {} links, and no <inertial> at all",
        geo.len(),
        geo.iter().map(|g| g.link.as_str()).collect::<std::collections::BTreeSet<_>>().len());

    let (robot, stated) = from_urdf_full(SHAPES_ONLY, "base", "tool").expect("the chain builds");
    println!("  the chain base -> tool has {} joints\n", robot.dof());

    // `from_urdf_full` reports what the description STATED, which here is nothing: every link's mass
    // is zero, and that is the body this crate could build before the geometry route existed.
    let stated_mass: f64 = stated.iter().map(|li| li.mass).sum();
    println!("  what <inertial> states:        total mass {stated_mass:.4} kg  <- unusable for dynamics");

    // The links along the chain, in the order `from_urdf_full` returns their inertias: ONE PER JOINT,
    // so the moving links only. ⛔ `tool` is not among them — it hangs off a FIXED joint, and
    // `from_urdf_full` folds a fixed child's inertia into its parent rather than giving it a slot. My
    // first version of this example listed it and indexed past the end of the vector. A fixed link
    // carrying geometry must therefore have that geometry folded into its parent too; this fixture
    // gives `tool` none, and the assertion below checks that rather than assuming it.
    let chain_links = ["l1", "l2"];
    assert!(
        !geo.iter().any(|g| g.link == "tool"),
        "this example folds no fixed-link geometry, so `tool` must declare none; it declares some"
    );
    let mut derived: Vec<LinkInertia> = Vec::new();
    for name in chain_links {
        let (li, skipped) = primitive_link_inertia(&geo, name, DENSITY, SEGMENTS);
        let li = li.unwrap_or_else(LinkInertia::zero);
        if skipped > 0 {
            println!("  ⛔ {name}: {skipped} mesh reference(s) skipped — this link needs its asset supplied");
        }
        derived.push(li);
    }
    let derived_mass: f64 = derived.iter().map(|li| li.mass).sum();
    println!("  what the GEOMETRY implies:    total mass {derived_mass:.4} kg at {DENSITY} kg/m³\n");

    println!("  {:<8} {:>10} {:>28} {:>34}", "link", "mass (kg)", "centre of mass (m)", "I diag (kg·m²)");
    for (name, li) in chain_links.iter().zip(&derived) {
        println!(
            "  {name:<8} {:>10.4}  ({:>7.4}, {:>7.4}, {:>7.4})   ({:>9.6}, {:>9.6}, {:>9.6})",
            li.mass, li.com.x, li.com.y, li.com.z,
            li.inertia[(0, 0)], li.inertia[(1, 1)], li.inertia[(2, 2)]
        );
    }

    // ---- the oracle: the same robot with hand-computed inertials in the XML ----
    let (m1, i1) = cylinder_inertial_rotated(0.035, 0.30);
    let (m2, i2) = cylinder_inertial_rotated(0.028, 0.20);
    let (mb, ib) = box_inertial(0.16, 0.16, 0.08);
    let with_inertials = SHAPES_ONLY
        .replace(
            r#"<link name="l1">"#,
            &format!(
                r#"<link name="l1"><inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="{m1}"/>
      <inertia ixx="{}" ixy="0" ixz="0" iyy="{}" iyz="0" izz="{}"/></inertial>"#,
                i1[0], i1[1], i1[2]
            ),
        )
        .replace(
            r#"<link name="l2">"#,
            &format!(
                r#"<link name="l2"><inertial><origin xyz="0.10 0 0" rpy="0 0 0"/><mass value="{m2}"/>
      <inertia ixx="{}" ixy="0" ixz="0" iyy="{}" iyz="0" izz="{}"/></inertial>"#,
                i2[0], i2[1], i2[2]
            ),
        );
    let (_, hand) = from_urdf_full(&with_inertials, "base", "tool").expect("the stated version builds");

    println!("\n  THE ORACLE — the same robot with hand-computed <inertial> blocks, from the closed forms");
    println!("  (box m(b²+c²)/12; cylinder mr²/2 axial and m(3r²+l²)/12 radial, rotated +π/2 about y)");
    println!("  base box, for reference: m {mb:.4} kg, I diag ({:.6}, {:.6}, {:.6})", ib[0], ib[1], ib[2]);
    let mut worst_mass = 0.0f64;
    let mut worst_inertia = 0.0f64;
    for (i, name) in chain_links.iter().enumerate() {
        let dm = (derived[i].mass / hand[i].mass - 1.0).abs();
        let di = (0..3)
            .map(|k| (derived[i].inertia[(k, k)] / hand[i].inertia[(k, k)] - 1.0).abs())
            .fold(0.0f64, f64::max);
        worst_mass = worst_mass.max(dm);
        worst_inertia = worst_inertia.max(di);
        println!("    {name}: mass agrees to {dm:.2e}, worst diagonal inertia to {di:.2e}");
    }
    println!("  worst disagreement anywhere: mass {worst_mass:.2e}, inertia {worst_inertia:.2e}");
    println!("  (the residual is the inscribed-prism deficit at {SEGMENTS} sides, 1 − (n/2π)sin(2π/n) = {:.2e})",
        1.0 - (SEGMENTS as f64 / (2.0 * core::f64::consts::PI)) * (2.0 * core::f64::consts::PI / SEGMENTS as f64).sin());

    // ---- and the body actually runs ----
    let g = Vector3::new(0.0, 0.0, -9.81);
    let q = [0.0, 0.4];
    let qd = [0.0, 0.0];
    let tau_g_derived = gravity_vector(&robot, &derived, &q, g);
    let tau_g_hand = gravity_vector(&robot, &hand, &q, g);
    let qdd = forward_dynamics(&robot, &derived, &q, &qd, &[0.0, 0.0], g);

    println!("\n  AND THE BODY RUNS. Gravity-compensation torque at q = {q:?}:");
    println!("    from the geometry:  ({:>9.5}, {:>9.5}) N·m", tau_g_derived[0], tau_g_derived[1]);
    println!("    from the stated:    ({:>9.5}, {:>9.5}) N·m", tau_g_hand[0], tau_g_hand[1]);
    let worst_tau = (0..robot.dof())
        .map(|i| (tau_g_derived[i] - tau_g_hand[i]).abs())
        .fold(0.0f64, f64::max);
    println!("    worst difference:    {worst_tau:.3e} N·m");
    println!("  free-fall acceleration from the geometry-derived body: ({:>8.4}, {:>8.4}) rad/s²", qdd[0], qdd[1]);

    // The claims above are asserted, so this example fails rather than printing a stale conclusion.
    assert!(stated_mass == 0.0, "the shapes-only description must state no mass, got {stated_mass}");
    assert!(derived_mass > 0.5, "the geometry must imply a real body, got {derived_mass} kg");
    assert!(worst_mass < 1e-3, "the two routes must agree on mass: worst {worst_mass:.3e}");
    assert!(worst_inertia < 1e-3, "and on inertia: worst {worst_inertia:.3e}");
    assert!(worst_tau < 1e-3, "and therefore on torque: worst {worst_tau:.3e} N·m");
    assert!(
        tau_g_derived[1].abs() > 1e-3,
        "joint 2 must carry a real gravity load at this posture, or the comparison above is vacuous"
    );
    assert!(qdd.iter().all(|a| a.is_finite()), "the derived body must integrate");

    println!(
        "\n  WHAT THIS SETTLES. A description that states only its shapes now yields a body, and the\n  \
         body it yields matches one whose inertias were computed by hand from the closed forms to\n  \
         {worst_inertia:.0e} — the inscribed-prism deficit and nothing else. The route needs no mesh file,\n  \
         because a box and a cylinder are generated rather than loaded.\n\n  \
         HONEST SCOPE. Uniform density, which no real part with a motor at one end has: if a\n  \
         description states an inertia, prefer it. Primitives only here — a link whose collision\n  \
         geometry is a mesh reports how many assets it needs and yields nothing until they are\n  \
         supplied, which `primitive_link_inertia`'s second return value is for. Closedness is not\n  \
         checked, so an open mesh handed to `solid_inertia` yields a wrong body rather than a refusal.\n"
    );
}
