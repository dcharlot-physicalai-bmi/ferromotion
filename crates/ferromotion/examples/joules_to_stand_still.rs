//! **The joule price of standing still, and why it is not a constant.**
//!
//! Every energy-per-task figure in this workspace prices a MOTION. `research/efa/punch_energetics.rs`
//! prices a strike; the cited hardware numbers price a reach (71.5 ± 48.3 J on a physical 7-DoF arm,
//! arXiv 2606.15918). This review did not locate a figure for the other thing a body spends its day
//! doing, which is holding a posture it is already in.
//!
//! Some earthlings hold one for almost nothing. A horse's stay apparatus locks the limb with ligaments
//! and tendons that do not fatigue, so prolonged standing costs virtually no muscular effort. That is
//! ANATOMY, not a controller, and it is the existence proof this bench is measured against.
//!
//! ⛔ **The bird is NOT a second example, and the first version of this file said it was.** The claim that
//! a perching bird's digital tendon-locking mechanism holds it asleep at zero muscular effort is the
//! textbook story and it was **experimentally refuted**: sleeping European starlings flex knee and ankle
//! only slightly and do not grip a 6 mm perch with the distal two-thirds of the toes; passive leg flexion
//! produces no toe flexion under anaesthesia; anaesthetised starlings cannot stay perched even with the
//! mechanism intact; and birds whose digital flexor tendons were severed slept on the perch normally
//! (Galton & Shepherd, J Exp Zool A 317:262-273, 2012, <https://doi.org/10.1002/jez.1714>). The tendon-locking
//! mechanism is real and does other work; automatic perching during sleep is not what it does.
//!
//! That correction is worth more than the example it damaged, because it is the mechanism robotics
//! imported: avian-inspired perching claws are built on the refuted story. A mechanism can be anatomically
//! real, widely cited, copied into hardware, and still not do the job it is famous for.
//!
//! A robot has no such element. It pays gravity-compensation torque as current, and current as copper
//! loss, for as long as it stands there. That much is obvious. What is not obvious is the SECOND-ORDER
//! term this stack can measure and most cannot:
//!
//! **Copper loss heats the winding, and hot copper has higher resistance, so the SAME posture costs more
//! the longer it is held.** `MotorThermal::copper_loss` uses the temperature-corrected resistance
//! `R₂₅·(1 + α·(T − T_ref))` with `α = 0.00393 /K` for copper. Holding is a RISING power draw against a
//! FALLING pack voltage. A passive lock has neither term, because it draws no current at all.
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
/// that holds 864 kJ. A green compile produced it without complaint. The gearbox is the term that makes
/// a holding torque affordable, and leaving it out is not a small error, it is the whole mechanism.
const KT_MOTOR: f64 = 0.10;
const GEAR: [f64; 3] = [120.0, 100.0, 80.0];
/// Winding resistance at 25 °C (Ω) for a small BLDC joint motor.
const R25: f64 = 0.50;

/// Hold `q` for `minutes` and report what it cost. Returns
/// `(watts_first, watts_last, joules, winding_rise_c, soc_drop_pct)`.
fn hold(q: &[f64], minutes: f64, label: &str) -> (f64, f64, f64, f64, f64) {
    let (robot, inertia) = from_urdf_full(ARM3, "base", "tip").expect("arm parses");
    // Gravity acts in -y so it loads joints whose axes are z, with the links laid along x.
    let g = Vector3::new(0.0, -9.81, 0.0);
    let tau = gravity_vector(&robot, &inertia, q, g);

    let ambient = 25.0;
    // Per-joint winding: R25 Ω, 8 J/K winding, 400 J/K housing, 1.2 K/W to housing, 1.8 K/W out.
    let mut motors: Vec<MotorThermal> = (0..3).map(|_| MotorThermal::new(R25, 8.0, 400.0, 1.2, 1.8, ambient)).collect();
    // Capacity is in COULOMBS: the field is `capacity_c` and its doc says a 2 Ah cell is 7200 C. Passing
    // 5.0 here specified a FIVE-COULOMB pack, which emptied instantly and reported -100% SOC for 1133 J.
    let mut pack = Battery::lithium(5.0 * 3600.0, 0.030, 48.0);

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

    let (mut joules, mut watts_first, mut watts_last) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..steps {
        // Copper loss at the CURRENT winding temperature, which is the term that grows.
        let p_copper: f64 = (0..3).map(|j| motors[j].copper_loss(currents[j])).sum();
        for j in 0..3 {
            motors[j].step(dt, currents[j], ambient);
        }
        // Draw the equivalent current from the pack at its present terminal voltage.
        let v = pack.terminal_voltage(p_copper / 48.0);
        let i_pack = if v > 1.0 { p_copper / v } else { 0.0 };
        let (delivered, lost) = pack.step(dt, i_pack);
        let p_total = (delivered + lost) / dt;
        joules += p_total * dt;
        if k == 0 {
            watts_first = p_total;
        }
        watts_last = p_total;
    }
    let rise = motors.iter().map(|m| m.t_winding).fold(f64::NEG_INFINITY, f64::max) - ambient;
    let soc_drop = (1.0 - pack.soc) * 100.0;
    // A pack holds capacity_c * nominal * 3600 joules. Drawing more than that is a broken bench, not a
    // finding: the first version of this reported 2.4 MJ from an 864 kJ pack and printed it happily.
    let pack_capacity_j = 5.0 * 3600.0 * 48.0;
    assert!(
        joules < pack_capacity_j,
        "{label}: drew {joules:.0} J from a pack holding {pack_capacity_j:.0} J — the bench is wrong, not the robot"
    );
    assert!(rise.is_finite() && rise < 200.0, "{label}: winding rose {rise:.1} K, which is a divergence and not a measurement");
    println!(
        "  {label:<26} tau {:>5.2} {:>5.2} {:>5.2} N·m   {watts_first:>6.2} W -> {watts_last:>6.2} W  \
         ({:>+5.1}%)   {joules:>8.0} J   winding +{rise:>4.1} °C   pack -{soc_drop:.2}%",
        tau[0], tau[1], tau[2],
        100.0 * (watts_last / watts_first - 1.0)
    );
    (watts_first, watts_last, joules, rise, soc_drop)
}

fn main() {
    println!("\nTHE JOULE PRICE OF STANDING STILL — a 3-link arm holding a posture for 10 minutes\n");
    println!("  Copper loss uses the temperature-corrected resistance R25*(1 + a*(T-Tref)), a = 0.00393 /K,");
    println!("  so the same posture draws more power as the winding warms. A passive lock draws none.\n");

    // Horizontal: every link's weight has a moment arm about every proximal joint. Worst case.
    let horizontal = hold(&[0.0, 0.0, 0.0], 10.0, "arm out horizontal");
    // Folded back on itself: the distal links' moments partly oppose, so the hold is cheaper.
    let folded = hold(&[0.0, 2.6, 2.6], 10.0, "arm folded");
    // Straight down: gravity is carried by the structure, not the actuators. The posture a robot
    // should choose if it must idle, and the closest thing it has to a stay apparatus.
    let hanging = hold(&[-std::f64::consts::FRAC_PI_2, 0.0, 0.0], 10.0, "arm hanging straight down");

    println!("\n  WHAT THE NUMBERS SAY");
    println!(
        "  Holding is not free and not constant: the horizontal hold rises {:+.1}% in 10 minutes on the \
         same posture,",
        100.0 * (horizontal.1 / horizontal.0 - 1.0)
    );
    println!("  purely because its own copper loss heated the winding by {:.1} °C.", horizontal.3);
    println!(
        "  Posture choice is the whole variable: {:.0} J horizontal, {:.0} J folded, and EXACTLY {:.0} J hanging,\n  \
         where gravity is carried by the structure and the actuators are asked for nothing. No ratio is quoted\n  \
         against the hanging case because it is an exact zero, not a small number.",
        horizontal.2, folded.2, hanging.2
    );
    println!(
        "\n  THE FORCING FUNCTION. A latch that holds a joint at zero torque removes this integral entirely,\n  \
         and with it the thermal term that makes it grow. The bound it moves is not efficiency, it is DUTY\n  \
         CYCLE. Be honest about the magnitude on THIS arm: {:.0} J per 10 minutes is only {:.2}% of a 5 Ah\n  \
         48 V pack, so a 6 kg three-link arm can afford to stand around. Two things make the term bite. It\n  \
         never stops, so the same rate is {:.0}% of the pack over a 24-hour watch. And it scales: this\n  \
         workspace's own published result (TR-2026-41) measures serial rotary modules multiplying copper as\n  \
         roughly N^(8/3), so the posture bill of a 30-joint body is not a scaled-up version of this one.\n  \
         A latch removes the integral entirely, and with it the thermal term that makes it grow. What becomes\n  \
         possible is a body that can hold a posture indefinitely, which is the precondition for waiting,\n  \
         watching and reacting rather than cycling. Biology solved it with anatomy, not control: the equine\n  \
         stay apparatus holds a standing limb at virtually no muscular effort. Note what the header says\n  \
         about the BIRD: that second example is the refuted one, and it is the one robotics copied.",
        horizontal.2, horizontal.4, horizontal.4 * 144.0
    );
    println!(
        "\n  HONEST SCOPE. Copper loss only. This omits viscous and Coulomb friction, which a cited\n  \
         physics-based model of a 7-DoF arm found DOMINANT at several joints (arXiv 2606.15915), plus drive\n  \
         electronics and holding brakes where fitted. The horizontal figure is therefore a LOWER bound on\n  \
         the price of standing still, and the ratio between postures is the more robust number.\n"
    );
}
