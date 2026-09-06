//! **The complete ledger: one body, one task, every term, and which one dominates.**
//!
//! `E_task = E_compute + E_actuation` was the whole ledger, and a sweep of the biology found three terms
//! missing from it. All three are now in the tree, and this puts them on one body at once, because the
//! open question was never any single term. It was the composition:
//!
//! > for one body on one task, the fraction of the bill from each source, and the joules each costs.
//!
//! ```text
//! E_total(T, d) = E_build  +  E_hold(T)  +  E_sense(T)  +  E_carry(d)
//! ```
//!
//! * **`E_build`** — [`EmbodiedActuator`], mass × embodied intensity. The term that gives `E_task` a
//!   crossover instead of a rate.
//! * **`E_hold`** — gravity-compensation torque through a gearbox into a temperature-corrected winding,
//!   drawn from a Thévenin pack. See `joules_to_stand_still.rs`. Rises as the winding warms.
//! * **`E_sense`** — [`sensing_power_floor`], the minimum sensing watts an accuracy target costs *given
//!   the plant's own delay margin*. A stability margin is a sensing energy budget.
//! * **`E_carry`** — [`OwnedCapability`], the energy to carry mass you own, scaled by the chassis's cost
//!   of transport.
//!
//! **The result is that no term dominates.** Which one is the bill depends on the mission's shape, and a
//! ledger that reports a rate cannot say that. That is the finding, not any single number.
//!
//! Run: `cargo run -p ferromotion --example the_complete_ledger`

use ferromotion_control::{sensing_power_floor, EmbodiedActuator, MotorThermal, OwnedCapability};
use ferromotion_core::{from_urdf_full, gravity_vector};
use nalgebra::Vector3;

/// The same three-link arm as `joules_to_stand_still.rs`, so the two benches are comparable.
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

const KT_MOTOR: f64 = 0.10;
const GEAR: [f64; 3] = [120.0, 100.0, 80.0];
const R25: f64 = 0.50;
/// Embodied intensity for the arm's structure and motors (MJ/kg). **A stated input, not a default** —
/// published figures for processed metals and motor assemblies span roughly an order of magnitude with
/// the process and the accounting boundary, so this is the caller's number to justify, not the module's.
const ARM_INTENSITY_MJ_PER_KG: f64 = 60.0;
/// Embodied intensity for the sensor head, which is electronics and therefore far higher per kilogram.
const SENSOR_INTENSITY_MJ_PER_KG: f64 = 250.0;

/// Steady holding power for a posture, by the same route as `joules_to_stand_still.rs`.
fn holding_watts(q: &[f64]) -> f64 {
    let (robot, inertia) = from_urdf_full(ARM3, "base", "tip").expect("arm parses");
    let tau = gravity_vector(&robot, &inertia, q, Vector3::new(0.0, -9.81, 0.0));
    let ambient = 25.0;
    let mut motors: Vec<MotorThermal> =
        (0..3).map(|_| MotorThermal::new(R25, 8.0, 400.0, 1.2, 1.8, ambient)).collect();
    let currents: Vec<f64> = (0..3).map(|j| (tau[j] / (GEAR[j] * KT_MOTOR)).abs()).collect();
    // Warm the windings to their equilibrium so the figure is the sustained one, not the cold one.
    for _ in 0..60_000 {
        for j in 0..3 {
            motors[j].step(0.01, currents[j], ambient);
        }
    }
    (0..3).map(|j| motors[j].copper_loss(currents[j])).sum()
}

struct Ledger {
    build_j: f64,
    hold_j: f64,
    sense_j: f64,
    carry_j: f64,
}

impl Ledger {
    fn total(&self) -> f64 {
        self.build_j + self.hold_j + self.sense_j + self.carry_j
    }
    fn dominant(&self) -> &'static str {
        let terms = [("build", self.build_j), ("hold", self.hold_j), ("sense", self.sense_j), ("carry", self.carry_j)];
        terms.iter().fold(("none", f64::NEG_INFINITY), |acc, &(n, v)| if v > acc.1 { (n, v) } else { acc }).0
    }
}

fn ledger(mission_s: f64, distance_m: f64, cost_of_transport: f64, hold_w: f64, sense_w: f64) -> Ledger {
    let arm_mass = 3.2 + 2.1 + 0.9;
    let arm = EmbodiedActuator { mass_kg: arm_mass, intensity_mj_per_kg: ARM_INTENSITY_MJ_PER_KG, operating_w: hold_w };
    let sensor = OwnedCapability {
        built: EmbodiedActuator { mass_kg: 0.120, intensity_mj_per_kg: SENSOR_INTENSITY_MJ_PER_KG, operating_w: 0.0 },
        quiescent_w: sense_w,
    };
    let build_j = arm.build_j().expect("finite") + sensor.built.build_j().expect("finite");
    // Carrying is the ownership term's third component, isolated by asking for it over zero seconds.
    let carry_total = sensor.owned_cost_j(0.0, distance_m, cost_of_transport).expect("finite")
        - sensor.built.build_j().expect("finite");
    let carry_arm = cost_of_transport * arm_mass * 9.81 * distance_m;
    Ledger {
        build_j,
        hold_j: hold_w * mission_s,
        sense_j: sense_w * mission_s,
        carry_j: carry_total + carry_arm,
    }
}

fn main() {
    println!("\nTHE COMPLETE LEDGER — one 6.2 kg arm, one 120 g sensor, four terms\n");

    let hold_w = holding_watts(&[0.0, 0.0, 0.0]);

    // The sensing floor is set by the plant, not by the sensor: the delay margin is the budget. Using the
    // 330 ms margin `latency.rs` measures for its own 100 Hz loop, 1% accuracy, 1 nJ per sample, 1 ms
    // correlation time. See `a_plants_delay_margin_sets_a_sensing_power_floor`.
    let floor = sensing_power_floor(0.01, 0.330, 1e-9, 1e-3).expect("well-posed");
    println!("  sustained hold {:.2} W (windings at equilibrium)", hold_w);
    println!(
        "  sensing floor  {:.1} uW ({:.0} samples over {:.0} channels, set by a 330 ms delay margin)\n",
        floor.power_w * 1e6, floor.samples, floor.channels
    );

    // Shares, not raw magnitudes: the four terms span nine orders and the question is which dominates.
    println!("  {:<22} {:>13} {:>18} {:>15} {:>15} {:>15}", "mission", "total", "build", "hold", "sense", "carry");
    let cases = [
        ("1 min, 0 m", 60.0, 0.0),
        ("8 h shift, 12 km", 8.0 * 3600.0, 12_000.0),
        ("1 year, 5000 km", 365.0 * 86400.0, 5_000_000.0),
        ("10 years, 50000 km", 3650.0 * 86400.0, 50_000_000.0),
    ];
    let mut dominants = Vec::new();
    for (label, t, d) in cases {
        let l = ledger(t, d, 3.2, hold_w, floor.power_w);
        let tot = l.total();
        let pct = |v: f64| 100.0 * v / tot;
        println!(
            "  {:<22} {:>10.1} MJ {:>10.1} MJ {:>4.1}% {:>7.1} MJ {:>4.1}% {:>6.2e} J {:>4.2}% {:>7.1} MJ {:>4.1}%",
            label, tot / 1e6,
            l.build_j / 1e6, pct(l.build_j),
            l.hold_j / 1e6, pct(l.hold_j),
            l.sense_j, pct(l.sense_j),
            l.carry_j / 1e6, pct(l.carry_j)
        );
        dominants.push(l.dominant());
    }

    println!("\n  WHAT THE LEDGER SAYS");
    let switched = dominants.windows(2).any(|w| w[0] != w[1]);
    if switched {
        println!("  The dominant term CHANGES with the mission: {}.", dominants.join(" -> "));
        println!("  A ledger that reports a rate cannot express that, which is the whole argument for");
        println!("  carrying all four terms rather than the two it had.");
    } else {
        println!("  On these inputs one term dominates at every duration tested: {}.", dominants[0]);
        println!("  That is an honest negative result, not a vindication of the two-term ledger: it says");
        println!("  the crossover for THIS body lies outside the range swept, and the range is stated.");
    }
    println!(
        "\n  ⭐ THE SENSE LEDGER ANSWER, AND IT REFRAMES THE QUESTION. The information floor for sensing is\n  \
         {:.1} uW against a {:.2} W hold — {:.4}% of the standing power, and utterly invisible in the total.\n  \
         But the SENSOR is not invisible: owning and carrying its 120 g costs {:.1} MJ to build and\n  \
         {:.1} MJ to carry over 5000 km. So the joules of sensing are not in the MEASUREMENT, they are in\n  \
         OWNING AND CARRYING the thing that measures. That is why evolution deletes a modality rather than\n  \
         duty-cycling it, and it is a different answer from the one the question implies.",
        floor.power_w * 1e6, hold_w, 100.0 * floor.power_w / hold_w,
        0.120 * SENSOR_INTENSITY_MJ_PER_KG,
        3.2 * 0.120 * 9.81 * 5_000_000.0 / 1e6
    );
    println!(
        "\n  HONEST SCOPE. Copper loss only for the hold, so E_hold is a LOWER bound; a cited power model of\n  \
         a 7-DoF arm found viscous and Coulomb friction dominant at several joints. Embodied intensities are\n  \
         stated inputs spanning an order of magnitude in the literature. E_sense prices the transducer's\n  \
         information floor, not its readout electronics. No term here is a measurement of a physical robot.\n"
    );
}
