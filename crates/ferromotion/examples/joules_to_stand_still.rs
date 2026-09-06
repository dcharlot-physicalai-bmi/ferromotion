//! **The joule price of standing still — and why the posture barely matters.**
//!
//! Every energy-per-task figure in this workspace prices a MOTION. The published punch bench prices a
//! strike; the cited hardware figure prices a reach at 71.5 ± 48.3 J on a physical 7-DoF arm
//! (`arXiv:2606.15918`). This review did not locate a figure for the other thing a body spends its day
//! doing, which is holding a posture it is already in.
//!
//! # ⛔ Two retractions this file exists to carry
//!
//! **The bird.** A perching bird's digital tendon-locking mechanism does NOT hold it asleep at zero
//! muscular effort. Sleeping European starlings flex knee and ankle only slightly and do not grip a 6 mm
//! perch with the distal two-thirds of the toes; passive leg flexion produces no toe flexion under
//! anaesthesia; anaesthetised starlings cannot stay perched with the mechanism intact; and birds whose
//! digital flexor tendons were severed slept on the perch normally. There is no automatic perching
//! mechanism (Galton & Shepherd, *J Exp Zool A* 317(4):205–215, 2012, <https://doi.org/10.1002/jez.1714>).
//!
//! **The horse, retracted for the same reason by the same experimental design.** The first version of
//! this file retracted the bird and then kept the equine stay apparatus as "the existence proof", which
//! was the identical error twice: an uncited textbook passive-locking story. Electromyography during
//! quiet standing shows the equine hind limb is **actively stabilised**, not passively locked
//! (Schuurman, Kersten & Weijs, *J Anat* 202(4):355–362, 2003,
//! <https://doi.org/10.1046/j.1469-7580.2003.00166.x>).
//!
//! **So this bench has NO biological existence proof for zero-cost posture, and the honest position is
//! that this review did not locate one that survives its own literature.** Latch-mediated spring
//! actuation is well evidenced for RELEASE; a vertebrate holding a posture at zero metabolic cost is not.
//!
//! # What the numbers actually say
//!
//! Copper loss is not the bill. This workspace's own published bench states **4.0 W per actuator for
//! electronics plus holding current** (`P_IDLE_ACT`, TR-2026-41 *Joules per Punch*,
//! <https://physicalai-bmi.org/assets/papers/joules-per-punch>), and a first version of this
//! file used zero, which overstated the importance of posture by a factor of three and inverted its own
//! endurance claim. With the drive electronics counted, the posture-choice lever is almost entirely
//! swamped, and the interesting consequence changes with it:
//!
//! **A latch is not worth having because it saves copper loss. It is worth having because it is the only
//! thing that lets the drive electronics be DE-ENERGISED.** That is a much larger term and a different
//! engineering change.
//!
//! Run: `cargo run -p ferromotion --example joules_to_stand_still`

use ferromotion_control::{Battery, MotorThermal};
use ferromotion_core::{from_urdf_full, gravity_vector};
use nalgebra::Vector3;

/// A three-link arm with stated inertias, roughly the mass distribution of a small collaborative arm:
/// 3.2 kg upper, 2.1 kg fore, 0.9 kg wrist, links 0.35 / 0.30 / 0.10 m, all revolute about z with the
/// links laid along x, so a horizontal posture loads every joint and a folded one does not.
const ARM3: &str = r#"<robot name="arm3">
  <link name="base"/>
  <link name="l1"><inertial><origin xyz="0.175 0 0" rpy="0 0 0"/><mass value="3.2"/>
    <inertia ixx="0.033" ixy="0" ixz="0" iyy="0.033" iyz="0" izz="0.020"/></inertial></link>
  <link name="l2"><inertial><origin xyz="0.150 0 0" rpy="0 0 0"/><mass value="2.1"/>
    <inertia ixx="0.016" ixy="0" ixz="0" iyy="0.016" iyz="0" izz="0.010"/></inertial></link>
  <link name="l3"><inertial><origin xyz="0.050 0 0" rpy="0 0 0"/><mass value="0.9"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/></inertial></link>
  <link name="tip"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="60" velocity="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.35 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="40" velocity="3"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.30 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="15" velocity="3"/></joint>
  <joint name="jt" type="fixed"><parent link="l3"/><child link="tip"/><origin xyz="0.10 0 0" rpy="0 0 0"/></joint>
</robot>"#;

/// Motor torque constant (N·m/A) at the MOTOR shaft, and the gear ratio between motor and joint.
///
/// **The first version of this bench omitted the gearbox** and divided joint torque by a motor-level
/// `kt`, asking a 1 Ω winding to carry 20 A. That is 400 W of copper loss in one joint; the thermal
/// model ran away exactly as it should, reporting a winding at 1.7e27 °C and 2.4 MJ drawn from a pack
/// that holds 864 kJ. A green compile produced it without complaint.
const KT_MOTOR: f64 = 0.10;
const GEAR: [f64; 3] = [120.0, 100.0, 80.0];
/// Winding resistance at 25 °C (Ω) for a small BLDC joint motor.
const R25: f64 = 0.50;
/// **Per-actuator drive electronics plus holding current (W).** Not zero, and not this file's invention:
/// it is `P_IDLE_ACT = 4.0`, declared as "W per actuator: electronics + holding current" by the bench
/// behind TR-2026-41 *Joules per Punch* (<https://physicalai-bmi.org/assets/papers/joules-per-punch>).
/// A sibling bench uses 6.0, so this is the LOW end of the pair. Omitting it is what made the first
/// version of this file overstate the posture lever and claim a 24-hour watch the pack cannot support.
///
/// ⛔ **It is a stated design constant, not a measurement of a physical driver**, and earlier wording
/// here cited it only as "this workspace's own bench", a path a reader of the published crate cannot
/// resolve. The report is the citable artifact; the bench source is not part of this repository.
const P_ELECTRONICS_PER_JOINT_W: f64 = 4.0;
/// Pack: 5 Ah at 48 V. Capacity below is in COULOMBS, which is what `Battery::capacity_c` wants.
const PACK_AH: f64 = 5.0;
const PACK_V: f64 = 48.0;

/// Hold `q` for `minutes`. Returns
/// `(watts_first, watts_last, joules, winding_rise_c, energy_fraction_of_pack, copper_only_j)`.
fn hold(q: &[f64], minutes: f64, label: &str) -> (f64, f64, f64, f64, f64, f64) {
    let (robot, inertia) = from_urdf_full(ARM3, "base", "tip").expect("arm parses");
    // Gravity acts in -y so it loads joints whose axes are z, with the links laid along x.
    let g = Vector3::new(0.0, -9.81, 0.0);
    let tau = gravity_vector(&robot, &inertia, q, g);

    let ambient = 25.0;
    let mut motors: Vec<MotorThermal> = (0..3).map(|_| MotorThermal::new(R25, 8.0, 400.0, 1.2, 1.8, ambient)).collect();
    // Capacity is in COULOMBS: `capacity_c`'s own doc says a 2 Ah cell is 7200 C.
    let mut pack = Battery::lithium(PACK_AH * 3600.0, 0.030, PACK_V);
    // Pack ENERGY, for the fraction reported below. A previous version printed the coulomb-counted SOC
    // drop under a joule label, which understated both figures by about 11% because terminal voltage
    // sags under load. Charge fraction and energy fraction are different quantities.
    let pack_capacity_j = PACK_AH * 3600.0 * PACK_V;

    let dt = 0.01;
    let steps = (minutes * 60.0 / dt) as usize;
    // Joint torque is delivered through the gearbox, so the winding carries tau/(N*kt), not tau/kt.
    let currents: Vec<f64> = (0..3).map(|j| (tau[j] / (GEAR[j] * KT_MOTOR)).abs()).collect();

    // Refuse to integrate a runaway. `equilibrium_rise` returns None when temperature-dependent copper
    // loss outruns the thermal path, which is a real physical outcome and not a number to average.
    for (j, m) in motors.iter().enumerate() {
        match m.equilibrium_rise(currents[j], ambient) {
            Some(r) => assert!(r < 200.0, "joint {j} would settle {r:.0} K above ambient; the parameters are not a joint"),
            None => panic!("joint {j} thermally runs away at {:.2} A; refusing to report an integral of a divergence", currents[j]),
        }
    }

    let electronics_w = P_ELECTRONICS_PER_JOINT_W * 3.0;
    let (mut joules, mut copper_j, mut watts_first, mut watts_last) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..steps {
        // Copper loss at the CURRENT winding temperature, which is the term that grows.
        let p_copper: f64 = (0..3).map(|j| motors[j].copper_loss(currents[j])).sum();
        for j in 0..3 {
            motors[j].step(dt, currents[j], ambient);
        }
        let p_draw = p_copper + electronics_w;
        let v = pack.terminal_voltage(p_draw / PACK_V);
        let i_pack = if v > 1.0 { p_draw / v } else { 0.0 };
        let (delivered, lost) = pack.step(dt, i_pack);
        let p_total = (delivered + lost) / dt;
        joules += p_total * dt;
        copper_j += p_copper * dt;
        if k == 0 {
            watts_first = p_total;
        }
        watts_last = p_total;
    }
    // The winding rise of the HOTTEST joint, which is the one the rise percentage below belongs to. A
    // previous version paired a whole-arm rise percentage with a single joint's temperature.
    let rise = motors.iter().map(|m| m.t_winding).fold(f64::NEG_INFINITY, f64::max) - ambient;
    assert!(joules < pack_capacity_j * 10.0, "{label}: {joules:.0} J is implausible against a {pack_capacity_j:.0} J pack");
    assert!(rise.is_finite() && rise < 200.0, "{label}: winding rose {rise:.1} K, which is a divergence and not a measurement");
    println!(
        "  {label:<26} tau {:>5.2} {:>5.2} {:>5.2} N·m  {watts_first:>6.2} W -> {watts_last:>6.2} W  \
         {joules:>7.0} J  (copper {copper_j:>5.0} J, {:>4.1}%)  winding +{rise:>4.1} °C  {:>5.2}% of pack",
        tau[0], tau[1], tau[2],
        100.0 * copper_j / joules,
        100.0 * joules / pack_capacity_j
    );
    (watts_first, watts_last, joules, rise, joules / pack_capacity_j, copper_j)
}

fn main() {
    println!("\nTHE JOULE PRICE OF STANDING STILL — a 3-link arm holding a posture for 10 minutes\n");
    println!("  Copper loss uses the temperature-corrected resistance R25*(1 + a*(T-Tref)), a = 0.00393 /K.");
    println!(
        "  Drive electronics are counted at {:.1} W per joint, this workspace's own published constant.\n",
        P_ELECTRONICS_PER_JOINT_W
    );

    let horizontal = hold(&[0.0, 0.0, 0.0], 10.0, "arm out horizontal");
    let folded = hold(&[0.0, 2.6, 2.6], 10.0, "arm folded");
    let hanging = hold(&[-std::f64::consts::FRAC_PI_2, 0.0, 0.0], 10.0, "arm hanging straight down");

    // The folded pose sits between the extremes, which is the ordering the copper term produces.
    assert!(
        folded.2 < horizontal.2 && folded.2 > hanging.2,
        "folded must sit between horizontal and hanging: {:.0} / {:.0} / {:.0} J",
        horizontal.2, folded.2, hanging.2
    );

    // The second-order claim, ASSERTED rather than narrated: the same posture costs more as it warms.
    // Without this, removing the temperature-corrected resistance leaves the file's headline sentence
    // contradicting its own output and still exiting 0.
    assert!(
        horizontal.1 > horizontal.0,
        "the hold must RISE as the winding warms; if it does not, R(T) is no longer temperature-corrected"
    );
    let copper_share = 100.0 * horizontal.5 / horizontal.2;

    println!("\n  WHAT THE NUMBERS SAY");
    println!(
        "  1. POSTURE IS ALMOST IRRELEVANT, and the first version of this file said the opposite. Copper\n  \
         loss is {:.1}% of the horizontal hold; the drive electronics are the rest and they do not care\n  \
         what pose the arm is in. Horizontal {:.0} J against hanging {:.0} J is a ratio of {:.2}, not the\n  \
         3.5 this bench reported when it counted copper alone.",
        copper_share, horizontal.2, hanging.2, horizontal.2 / hanging.2
    );
    println!(
        "  2. Holding still is NOT constant: the horizontal copper term rises as its own loss heats the\n  \
         winding by {:.1} °C. That effect is real and it is small next to the constant term.",
        horizontal.3
    );
    let day_j = horizontal.2 * 144.0;
    let pack_j = PACK_AH * 3600.0 * PACK_V;
    println!(
        "  3. ⛔ THE ENDURANCE CLAIM INVERTS. A 24-hour watch costs {:.2} MJ against a {:.0} kJ pack, which\n  \
         is {:.0}% of it: this arm cannot stand still for a day at all, it goes flat after {:.1} h. The\n  \
         earlier version of this file called the same watch \"17% of the pack\" because it omitted the\n  \
         electronics.",
        day_j / 1e6, pack_j / 1e3, 100.0 * day_j / pack_j, pack_j / horizontal.2 * 10.0 / 60.0
    );
    println!(
        "\n  THE FORCING FUNCTION, RESTATED. A latch is not worth having to save copper loss, which is\n  \
         {:.1}% of the bill. It is worth having because holding a pose mechanically is the only thing that\n  \
         permits the drive electronics to be DE-ENERGISED, and that is the term that empties the pack. The\n  \
         bound it moves is endurance, and the change is a mechanism that holds without a powered loop.",
        copper_share
    );
    println!(
        "\n  HONEST SCOPE. Copper loss plus a stated per-joint electronics constant. It omits viscous and\n  \
         Coulomb friction, which are velocity-dependent and therefore contribute nothing at the exactly\n  \
         zero velocity of a static hold, so `arXiv:2606.15915` does NOT support treating this as a lower\n  \
         bound on that account. It also omits gearbox efficiency, any holding brake, and the compute\n  \
         platform, which the companion ledger prices at 15 W. No figure here is a measurement of a\n  \
         physical robot, and this bench has no surviving biological existence proof to compare against.\n"
    );
}
