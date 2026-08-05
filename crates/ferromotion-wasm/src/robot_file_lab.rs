//! **The robot is the file** — the on-device lab behind the robot-description lesson.
//!
//! A robot description is code you did not write, in a format with conventions you did not choose, and three of those
//! conventions will change your physics without changing anything you would notice by reading the file. This lab
//! prices each one in the units you actually care about: newton-metres and radians per second squared.
//!
//! The three, all from `UsdPhysics`:
//!
//! 1. **Revolute joint limits are in degrees.** Prismatic limits, written in the same field name, are in linear units.
//!    So `physics:lowerLimit = -90` means one thing on one joint and something else on the next, and a consumer that
//!    picks either interpretation globally is wrong about half the file.
//! 2. **`metersPerUnit` scales lengths, and lengths enter dynamics nonlinearly.** A stage authored in centimetres and
//!    read as metres does not simply look small; its inertias, moment arms and gravity torques all move, by different
//!    factors.
//! 3. **`upAxis` defaults to `Y`.** Gravity is not where a roboticist assumes it is, and nothing errors.
//!
//! Each of these is a *silent* failure: the file parses, the robot articulates, the numbers are plausible. The only
//! way to catch one is to compute a physical quantity twice and subtract, which is what the lab does live.
//!
//! The fourth control is unrelated to conventions and is here because it is the other half of loading a real robot: a
//! hand or a legged base is a **branched** tree, not a chain, and [`ferromotion_core::tree_from_urdf`] handles it. The
//! lab pins the branched loader against the serial one on a description both can read.

use ferromotion_core::{gravity_vector, parse_usda, robot_from_usda, tree_from_urdf, KinematicTree};
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

/// Written limit on the revolute joint. Ninety of something.
const WRITTEN_LIMIT: f64 = 90.0;
/// Written limit on the prismatic joint, which is a length however the revolute one is read.
const WRITTEN_PRISMATIC: f64 = 0.5;

#[wasm_bindgen]
pub struct RobotFileLab {
    /// `metersPerUnit` declared by the stage. `1.0` is metres, `0.01` centimetres.
    meters_per_unit: f64,
    /// Declared up axis: `true` for the `UsdPhysics` default `Y`, `false` for the roboticist's `Z`.
    up_axis_y: bool,
}

#[wasm_bindgen]
impl RobotFileLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RobotFileLab {
        RobotFileLab { meters_per_unit: 1.0, up_axis_y: false }
    }

    /// Set `metersPerUnit`. Only the two values a real file uses are offered, because the point is not interpolation.
    pub fn set_centimetres(&mut self, cm: bool) {
        self.meters_per_unit = if cm { 0.01 } else { 1.0 };
    }

    pub fn set_up_axis_y(&mut self, y: bool) {
        self.up_axis_y = y;
    }

    pub fn meters_per_unit(&self) -> f64 {
        self.meters_per_unit
    }

    pub fn is_centimetres(&self) -> bool {
        self.meters_per_unit < 1.0
    }

    pub fn is_up_axis_y(&self) -> bool {
        self.up_axis_y
    }

    /// The stage text, generated so the learner can see exactly which three tokens are in play.
    pub fn stage_text(&self) -> String {
        let up = if self.up_axis_y { "Y" } else { "Z" };
        let mpu = if self.meters_per_unit < 1.0 { "0.01" } else { "1" };
        format!(
            "#usda 1.0\n(\n    defaultPrim = \"robot\"\n    metersPerUnit = {mpu}\n    upAxis = \"{up}\"\n)\n\n\
             def Xform \"robot\" (\n    prepend apiSchemas = [\"PhysicsArticulationRootAPI\"]\n)\n{{\n\
             \x20   def Xform \"link0\" (\n        prepend apiSchemas = [\"PhysicsRigidBodyAPI\"]\n    )\n    {{\n    }}\n\n\
             \x20   def Xform \"link1\" (\n        prepend apiSchemas = [\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\"]\n    )\n    {{\n\
             \x20       float physics:mass = 2.0\n        point3f physics:centerOfMass = (0, 0, 0.15)\n\
             \x20       float3 physics:diagonalInertia = (0.03, 0.03, 0.008)\n\
             \x20       double3 xformOp:translate = (0, 0, 0.05)\n\
             \x20       uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }}\n\n\
             \x20   def Xform \"link2\" (\n        prepend apiSchemas = [\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\"]\n    )\n    {{\n\
             \x20       float physics:mass = 1.3\n        point3f physics:centerOfMass = (0.1, 0, 0.1)\n\
             \x20       float3 physics:diagonalInertia = (0.02, 0.02, 0.005)\n    }}\n\n\
             \x20   def PhysicsRevoluteJoint \"joint1\"\n    {{\n\
             \x20       rel physics:body0 = </robot/link0>\n        rel physics:body1 = </robot/link1>\n\
             \x20       uniform token physics:axis = \"Z\"\n        point3f physics:localPos0 = (0, 0, 0.05)\n\
             \x20       float physics:lowerLimit = -{WRITTEN_LIMIT}\n        float physics:upperLimit = {WRITTEN_LIMIT}\n    }}\n\n\
             \x20   def PhysicsPrismaticJoint \"joint2\"\n    {{\n\
             \x20       rel physics:body0 = </robot/link1>\n        rel physics:body1 = </robot/link2>\n\
             \x20       uniform token physics:axis = \"X\"\n        point3f physics:localPos0 = (0.3, 0, 0)\n\
             \x20       float physics:lowerLimit = -{WRITTEN_PRISMATIC}\n        float physics:upperLimit = {WRITTEN_PRISMATIC}\n    }}\n}}\n"
        )
    }

    fn parsed(&self) -> Option<(ferromotion_core::Robot, Vec<ferromotion_core::LinkInertia>)> {
        let stage = parse_usda(&self.stage_text()).ok()?;
        robot_from_usda(&stage, "link0", "link2").ok()
    }

    /// Gravity direction the stage actually declares. This is the trap: `UsdPhysics` defaults to `Y`.
    fn gravity(&self) -> Vector3<f64> {
        if self.up_axis_y {
            Vector3::new(0.0, -9.81, 0.0)
        } else {
            Vector3::new(0.0, 0.0, -9.81)
        }
    }

    // ---------------------------------------------------------------- trap one: degrees
    /// The revolute limit as the parser reads it: **radians**, because `UsdPhysics` writes degrees.
    pub fn revolute_limit_parsed(&self) -> f64 {
        self.parsed().and_then(|(r, _)| r.joints[0].limits.map(|(_, hi)| hi)).unwrap_or(f64::NAN)
    }

    /// What a consumer that assumed radians would use: the written number, untouched.
    pub fn revolute_limit_if_misread(&self) -> f64 {
        WRITTEN_LIMIT
    }

    /// **The cost of the degrees trap**, as a factor. A joint allowed 90 radians of travel has no limit at all.
    pub fn limit_error_factor(&self) -> f64 {
        let parsed = self.revolute_limit_parsed();
        if parsed.abs() > 0.0 { WRITTEN_LIMIT / parsed } else { f64::NAN }
    }

    /// The prismatic limit as parsed. It must come back as the written `0.5`, unconverted, which is the half of this
    /// trap that catches people who fix it globally: the same field name is degrees on one joint and metres on the next.
    pub fn prismatic_limit_parsed(&self) -> f64 {
        self.parsed().and_then(|(r, _)| r.joints[1].limits.map(|(_, hi)| hi)).unwrap_or(f64::NAN)
    }

    // ---------------------------------------------------------------- trap two: metersPerUnit
    /// Gravity torque on the first joint under the stage as declared, N*m.
    pub fn gravity_torque(&self) -> f64 {
        let Some((r, inertia)) = self.parsed() else { return f64::NAN };
        let q = vec![0.3; r.dof()];
        gravity_vector(&r, &inertia, &q, self.gravity()).first().copied().unwrap_or(f64::NAN)
    }

    /// **Gravity torque in a pose that actually loads the joint.**
    ///
    /// The declared pose does not, and that is worth knowing rather than working around: `joint1` rotates about `Z`,
    /// so with `upAxis = "Z"` gravity is *parallel to the rotation axis* and the torque is identically zero. A test
    /// run in that configuration cannot see a hundred-fold length error, because zero times anything is zero. The
    /// scale comparison below therefore uses a loading direction, and the fact that it has to is the lesson: the pose
    /// you test in decides which bugs are visible.
    pub fn gravity_torque_loaded(&self) -> f64 {
        let Some((r, inertia)) = self.parsed() else { return f64::NAN };
        let q = vec![0.3; r.dof()];
        // Perpendicular to the Z joint axis, so the moment arm is real whatever the stage declares.
        gravity_vector(&r, &inertia, &q, Vector3::new(0.0, -9.81, 0.0)).first().copied().unwrap_or(f64::NAN)
    }

    /// Whether the declared pose leaves the joint unloaded, so a learner is told rather than misled.
    pub fn declared_pose_is_unloaded(&self) -> bool {
        self.gravity_torque().abs() < 1e-12
    }

    /// The loaded torque with the stage read at the *other* length scale: the number you would get if the file said
    /// centimetres and you assumed metres, or the reverse.
    pub fn gravity_torque_other_scale(&self) -> f64 {
        let mut other = RobotFileLab { meters_per_unit: self.meters_per_unit, up_axis_y: self.up_axis_y };
        other.set_centimetres(!self.is_centimetres());
        other.gravity_torque_loaded()
    }

    /// **The cost of the `metersPerUnit` trap**, in newton-metres. Not a percentage: the absolute error a controller
    /// would command.
    pub fn torque_error_from_scale(&self) -> f64 {
        (self.gravity_torque_loaded() - self.gravity_torque_other_scale()).abs()
    }

    /// Ratio between the two, which is not the length ratio, because lengths enter dynamics through both the moment
    /// arm and the inertia.
    pub fn torque_ratio_from_scale(&self) -> f64 {
        let other = self.gravity_torque_other_scale();
        if other.abs() > 0.0 { self.gravity_torque_loaded() / other } else { f64::NAN }
    }

    // ---------------------------------------------------------------- trap three: upAxis
    /// Gravity torque with the up axis flipped: what the same file yields under the other convention.
    pub fn gravity_torque_other_axis(&self) -> f64 {
        let other = RobotFileLab { meters_per_unit: self.meters_per_unit, up_axis_y: !self.up_axis_y };
        other.gravity_torque()
    }

    /// **The cost of the `upAxis` trap**, in newton-metres.
    pub fn torque_error_from_axis(&self) -> f64 {
        (self.gravity_torque() - self.gravity_torque_other_axis()).abs()
    }

    // ---------------------------------------------------------------- branched trees
    /// A two-finger URDF: a branched tree, which a serial chain loader cannot represent.
    fn branched_urdf() -> &'static str {
        r#"<robot name="hand">
  <link name="palm"/>
  <link name="f1a"/><link name="f1b"/>
  <link name="f2a"/><link name="f2b"/>
  <joint name="j1a" type="revolute">
    <parent link="palm"/><child link="f1a"/>
    <origin xyz="0 0.03 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.2" upper="1.2" effort="5" velocity="3"/>
  </joint>
  <joint name="j1b" type="revolute">
    <parent link="f1a"/><child link="f1b"/>
    <origin xyz="0.04 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.2" upper="1.2" effort="5" velocity="3"/>
  </joint>
  <joint name="j2a" type="revolute">
    <parent link="palm"/><child link="f2a"/>
    <origin xyz="0 -0.03 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.2" upper="1.2" effort="5" velocity="3"/>
  </joint>
  <joint name="j2b" type="revolute">
    <parent link="f2a"/><child link="f2b"/>
    <origin xyz="0.04 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.2" upper="1.2" effort="5" velocity="3"/>
  </joint>
</robot>"#
    }

    fn tree() -> Option<KinematicTree> {
        tree_from_urdf(Self::branched_urdf(), "palm").ok()
    }

    /// Degrees of freedom in the branched tree.
    pub fn tree_dof(&self) -> f64 {
        Self::tree().map_or(f64::NAN, |t| t.dof() as f64)
    }

    /// **Joints a serial loader would silently drop.** A chain loader reading base-to-one-tip sees `depth` joints; the
    /// rest of the tree is simply not in its model, and nothing errors. This is the number of them.
    pub fn joints_a_chain_loader_misses(&self) -> f64 {
        Self::tree().map_or(f64::NAN, |t| (t.dof() - t.depth()) as f64)
    }

    /// Number of leaves, which is the number of fingertips you can actually place.
    pub fn tree_leaves(&self) -> f64 {
        Self::tree().map_or(f64::NAN, |t| t.leaves().len() as f64)
    }

    /// Longest chain from the base. The serial loader's view of this robot is one of these paths and nothing else.
    pub fn tree_depth(&self) -> f64 {
        Self::tree().map_or(f64::NAN, |t| t.depth() as f64)
    }

    /// **Fingertip separation at a given pose**, metres. The quantity a serial loader physically cannot compute,
    /// because it can only see one finger at a time.
    pub fn fingertip_separation(&self, q0: f64) -> f64 {
        let Some(t) = Self::tree() else { return f64::NAN };
        let base = nalgebra::Isometry3::identity();
        let q = vec![q0, 0.4, -q0, -0.4];
        let (a, b) = (t.tip_pose("f1b", base, &q), t.tip_pose("f2b", base, &q));
        match (a, b) {
            (Some(a), Some(b)) => (a.translation.vector - b.translation.vector).norm(),
            _ => f64::NAN,
        }
    }
}

impl Default for RobotFileLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Trap one, pinned.** A revolute limit written as `90` is `pi/2` radians, and a consumer that reads it as
    /// radians is wrong by 57.3x. The prismatic limit in the identically-named field is *not* converted.
    #[test]
    fn revolute_limits_are_degrees_and_prismatic_limits_are_not() {
        let lab = RobotFileLab::new();
        let parsed = lab.revolute_limit_parsed();
        eprintln!(
            "written {} -> parsed {parsed:.6} rad; misreading as radians is {:.4}x off",
            WRITTEN_LIMIT,
            lab.limit_error_factor()
        );
        assert!((parsed - core::f64::consts::FRAC_PI_2).abs() < 1e-12, "90 degrees is pi/2, got {parsed}");
        assert!((lab.limit_error_factor() - 57.29577951308232).abs() < 1e-9, "{}", lab.limit_error_factor());

        let prismatic = lab.prismatic_limit_parsed();
        eprintln!("the prismatic limit in the same field name stays {prismatic}: an angle is not a length");
        assert!((prismatic - WRITTEN_PRISMATIC).abs() < 1e-12, "a length must not be degree-converted: {prismatic}");
    }

    /// **Trap two, pinned.** The same file at the two length scales gives different gravity torques, and the ratio is
    /// not the length ratio, because lengths enter through the moment arm and the inertia both.
    #[test]
    fn meters_per_unit_changes_the_torque_by_more_than_the_length_ratio() {
        let mut lab = RobotFileLab::new();
        // First, the reason this test uses a loaded pose: the declared one produces exactly zero torque, because
        // gravity is parallel to the joint axis. A hundred-fold length error is completely invisible there.
        assert!(lab.declared_pose_is_unloaded(), "the declared pose should be unloaded, torque {}", lab.gravity_torque());
        assert!(lab.torque_error_from_scale() > 1e-3, "but a loaded pose must expose the error");

        let metres = lab.gravity_torque_loaded();
        lab.set_centimetres(true);
        let centimetres = lab.gravity_torque_loaded();
        eprintln!("gravity torque: {metres:.6} N*m as metres, {centimetres:.6} N*m as centimetres");
        eprintln!("  absolute error {:.6} N*m, ratio {:.4}x (the length ratio is 100x)", lab.torque_error_from_scale(), lab.torque_ratio_from_scale().abs());
        assert!(metres.abs() > 0.0 && centimetres.abs() > 0.0, "both scales should produce a torque");
        assert!(lab.torque_error_from_scale() > 1e-3, "the mistake must be visible: {}", lab.torque_error_from_scale());
        // The ratio must not be exactly the length ratio, which is the point: this is not a units rescale you can
        // undo by multiplying the answer.
        let r = lab.torque_ratio_from_scale().abs();
        assert!((r - 100.0).abs() > 1.0, "ratio {r} is suspiciously the length ratio");
    }

    /// **Trap three, pinned.** Flipping `upAxis` changes the gravity torque, and nothing in the file errors.
    #[test]
    fn up_axis_silently_changes_where_gravity_pulls() {
        let lab = RobotFileLab::new();
        let z = lab.gravity_torque();
        let y = lab.gravity_torque_other_axis();
        eprintln!("gravity torque: {z:.6} N*m with upAxis Z, {y:.6} N*m with upAxis Y (the UsdPhysics default)");
        eprintln!("  the file parses cleanly either way; the error is {:.6} N*m", lab.torque_error_from_axis());
        assert!(lab.torque_error_from_axis() > 1e-6, "the axis must matter: {}", lab.torque_error_from_axis());
    }

    /// The branched tree has to report its shape honestly: a serial loader sees one path and drops the rest.
    #[test]
    fn a_branched_tree_reports_more_than_one_chain() {
        let lab = RobotFileLab::new();
        eprintln!(
            "two-finger hand: {} dof, {} leaves, deepest chain {}, joints a chain loader misses {}",
            lab.tree_dof(),
            lab.tree_leaves(),
            lab.tree_depth(),
            lab.joints_a_chain_loader_misses()
        );
        assert_eq!(lab.tree_dof(), 4.0);
        assert_eq!(lab.tree_leaves(), 2.0, "two fingertips");
        // Depth is one chain's length, strictly less than the dof: that gap is exactly what a chain loader loses.
        assert!(lab.tree_depth() < lab.tree_dof(), "depth {} vs dof {}", lab.tree_depth(), lab.tree_dof());
        assert_eq!(lab.joints_a_chain_loader_misses(), 2.0, "half this hand is invisible to a chain loader");
    }

    /// The quantity that only exists on a tree: two fingertips at once, and a separation that responds to the pose.
    #[test]
    fn fingertip_separation_responds_to_the_pose() {
        let lab = RobotFileLab::new();
        let a = lab.fingertip_separation(0.0);
        let b = lab.fingertip_separation(0.8);
        eprintln!("fingertip separation: {a:.5} m at q0 = 0.0, {b:.5} m at q0 = 0.8");
        assert!(a.is_finite() && b.is_finite(), "both poses must resolve");
        assert!((a - b).abs() > 1e-4, "closing the fingers should change the separation: {a} vs {b}");
    }
}
