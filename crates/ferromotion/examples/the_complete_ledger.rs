//! **The ledger: one body, one task, five terms, and which one dominates.**
//!
//! `E_task = E_compute + E_actuation` was the whole ledger, and a sweep of the biology found three terms
//! missing from it. All three are now in the tree, and this puts them on one body at once, because the
//! open question was never any single term. It was the composition:
//!
//! > for one body on one task, the fraction of the bill from each source, and the joules each costs.
//!
//! ```text
//! E_total(T, d) = E_build + E_compute(T) + E_hold(T) + E_sense(T) + E_carry(d)
//! ```
//!
//! ⛔ **The first version of this file was titled "the complete ledger", named `E_compute` in its own
//! opening line as the term it was extending, and then omitted it.** At this workspace's own published
//! platform constant (15 W, Orin NX policy-only, from the punch-energetics bench) that is 473 MJ over a
//! year — LARGER than the build term at the nominal intensity, and the largest single term at the low end
//! of the intensity span. It also priced the hold as copper loss alone while the same repo
//! states 4.0 W per actuator for drive electronics, understating the hold sevenfold. Both are counted
//! here, both are stated inputs, and the composition below is what changed as a result.
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
//! ledger that reports a rate cannot say that. That is the finding, not any single number — and counting
//! the two terms the first version omitted turned a two-term contest into a four-way one at one year:
//! at the nominal inputs the one-year split is build 20.8 / compute 24.5 / hold 22.7 / carry 32.1 percent.
//!
//! ⛔ **And that split is a point in a box, not a finding.** Two of the inputs it depends on are not
//! measurements of this body: the embodied intensities, which this file's own note says span roughly an
//! order of magnitude, and the chassis cost of transport, whose sourced legged span is 0.2 to 15. Swept
//! across that box the one-year dominant term is compute, carry OR build depending on the corner, so the
//! file now prints the sweep and says which claim survives it. What survives is that the composition is a
//! CONTEST with no negligible term, and that the two constants deciding it are the body's embodied
//! intensity and its chassis's cost of transport. Those two are the measurements that would make it a
//! number. An earlier version printed the nominal split as four typed literals, which cannot notice that
//! a constant moved; every figure in the output is computed now.
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
/// Per-actuator drive electronics plus holding current (W) — this workspace's own `P_IDLE_ACT`.
const P_ELECTRONICS_PER_JOINT_W: f64 = 4.0;
/// Compute platform draw (W). 15 W is the Orin NX policy-only figure the punch-energetics bench sweeps;
/// that same sweep also runs 25 W for perception plus policy, so this is the LOW end of a stated range.
const P_COMPUTE_W: f64 = 15.0;
/// Embodied intensity for the sensor head, which is electronics and therefore far higher per kilogram.
const SENSOR_INTENSITY_MJ_PER_KG: f64 = 250.0;
/// Cost of transport for the chassis the arm rides on. `OwnedCapability::owned_cost_j` carries the
/// sourced table; ASIMO's 2.0 (54 kg, 1.8 kW at 1.5 m/s, Sakagami et al. 2002) is the mid-legged point.
///
/// ⛔ **An earlier version of this file swept at 3.2 with no source, and that single unsourced number
/// set the term it reported as dominant.** The sourced span for legged systems runs 0.2 to 15, so the
/// composition is now reported ACROSS that span rather than at one point inside it.
const COT_NOMINAL: f64 = 2.0;
/// The chassis span the composition is tested against: every value sourced in `owned_cost_j`'s table.
const COT_SPAN: [(&str, f64); 4] = [("human 0.2", 0.2), ("Cassie 0.7", 0.7), ("ASIMO 2.0", 2.0), ("BigDog 15", 15.0)];
/// The embodied-intensity span this file's own constants declare: "roughly an order of magnitude".
const INTENSITY_SPAN: [(&str, f64); 3] = [("0.3x", 0.3), ("1x", 1.0), ("3x", 3.0)];

/// Steady holding power for a posture, by the same route as `joules_to_stand_still.rs`.
fn holding_watts(q: &[f64]) -> f64 {
    let (robot, inertia) = from_urdf_full(ARM3, "base", "tip").expect("arm parses");
    let tau = gravity_vector(&robot, &inertia, q, Vector3::new(0.0, -9.81, 0.0));
    let ambient = 25.0;
    let mut motors: Vec<MotorThermal> =
        (0..3).map(|_| MotorThermal::new(R25, 8.0, 400.0, 1.2, 1.8, ambient)).collect();
    let currents: Vec<f64> = (0..3).map(|j| (tau[j] / (GEAR[j] * KT_MOTOR)).abs()).collect();
    // Use the ANALYTIC equilibrium rather than a fixed warm-up. A previous version ran 60 000 Euler
    // steps and labelled the result "windings at equilibrium" when the winding was only ~72% of the way
    // there, with the exact answer already available in the crate.
    for (j, m) in motors.iter_mut().enumerate() {
        let rise = m.equilibrium_rise(currents[j], ambient).expect("no runaway at these currents");
        m.t_winding = ambient + rise;
        m.t_housing = ambient + rise * 1.8 / (1.2 + 1.8);
    }
    let copper: f64 = (0..3).map(|j| motors[j].copper_loss(currents[j])).sum();
    copper + P_ELECTRONICS_PER_JOINT_W * 3.0
}

struct Ledger {
    build_j: f64,
    compute_j: f64,
    hold_j: f64,
    sense_j: f64,
    carry_j: f64,
}

impl Ledger {
    fn total(&self) -> f64 {
        self.build_j + self.compute_j + self.hold_j + self.sense_j + self.carry_j
    }
    fn terms(&self) -> [(&'static str, f64); 5] {
        [("build", self.build_j), ("compute", self.compute_j), ("hold", self.hold_j), ("sense", self.sense_j), ("carry", self.carry_j)]
    }
    fn dominant(&self) -> &'static str {
        self.terms().iter().fold(("none", f64::NEG_INFINITY), |acc, &(n, v)| if v > acc.1 { (n, v) } else { acc }).0
    }
    /// The largest single term in joules. Used to say whether anything actually has a majority.
    fn largest(&self) -> f64 {
        self.terms().iter().fold(f64::NEG_INFINITY, |a, &(_, v)| a.max(v))
    }
}

fn ledger(mission_s: f64, distance_m: f64, cost_of_transport: f64, hold_w: f64, sense_w: f64) -> Ledger {
    ledger_at(mission_s, distance_m, cost_of_transport, hold_w, sense_w, 1.0)
}

/// The same ledger with both embodied intensities scaled together, so the composition can be tested
/// against the order-of-magnitude span this file's own constants declare.
fn ledger_at(mission_s: f64, distance_m: f64, cost_of_transport: f64, hold_w: f64, sense_w: f64, intensity_scale: f64) -> Ledger {
    let arm_mass = 3.2 + 2.1 + 0.9;
    let arm = EmbodiedActuator {
        mass_kg: arm_mass,
        intensity_mj_per_kg: ARM_INTENSITY_MJ_PER_KG * intensity_scale,
        operating_w: hold_w,
    };
    let sensor = OwnedCapability {
        built: EmbodiedActuator {
            mass_kg: 0.120,
            intensity_mj_per_kg: SENSOR_INTENSITY_MJ_PER_KG * intensity_scale,
            operating_w: 0.0,
        },
        quiescent_w: sense_w,
    };
    let build_j = arm.build_j().expect("finite") + sensor.built.build_j().expect("finite");
    // Carrying is the ownership term's third component, isolated by asking for it over zero seconds.
    let carry_total = sensor.owned_cost_j(0.0, distance_m, cost_of_transport).expect("finite")
        - sensor.built.build_j().expect("finite");
    let carry_arm = cost_of_transport * arm_mass * 9.81 * distance_m;
    Ledger {
        build_j,
        compute_j: P_COMPUTE_W * mission_s,
        hold_j: hold_w * mission_s,
        sense_j: sense_w * mission_s,
        carry_j: carry_total + carry_arm,
    }
}

fn main() {
    println!("\nTHE LEDGER — one 6.2 kg arm, one 120 g sensor, five terms\n");

    let hold_w = holding_watts(&[0.0, 0.0, 0.0]);

    // The sensing floor is set by the plant, not by the sensor: the delay margin is the budget. Using the
    // 330 ms margin `latency.rs` measures for its own 100 Hz loop, 1% accuracy, 1 nJ per sample, 1 ms
    // correlation time. See `a_plants_delay_margin_sets_a_sensing_power_floor`.
    let floor = sensing_power_floor(0.01, 0.330, 1e-9, 1e-3).expect("well-posed");
    println!(
        "  sustained hold {:.2} W (analytic winding equilibrium + {:.0} W electronics), compute {:.0} W",
        hold_w, P_ELECTRONICS_PER_JOINT_W * 3.0, P_COMPUTE_W
    );
    println!(
        "  sensing floor  {:.1} uW ({:.0} samples over {:.0} channels, set by a 330 ms delay margin)\n",
        floor.power_w * 1e6, floor.samples, floor.channels
    );

    // Shares, not raw magnitudes: the four terms span nine orders and the question is which dominates.
    println!("  {:<20} {:>12} {:>15} {:>14} {:>14} {:>13} {:>14}", "mission", "total", "build", "compute", "hold", "sense", "carry");
    let cases = [
        ("1 min, 0 m", 60.0, 0.0),
        ("8 h shift, 12 km", 8.0 * 3600.0, 12_000.0),
        ("1 year, 5000 km", 365.0 * 86400.0, 5_000_000.0),
        ("10 years, 50000 km", 3650.0 * 86400.0, 50_000_000.0),
    ];
    let mut dominants = Vec::new();
    for (label, t, d) in cases {
        let l = ledger(t, d, COT_NOMINAL, hold_w, floor.power_w);
        let tot = l.total();
        let pct = |v: f64| 100.0 * v / tot;
        println!(
            "  {:<20} {:>9.1} MJ {:>8.1} {:>4.1}% {:>7.1} {:>4.1}% {:>6.1} {:>4.1}% {:>5.1e} {:>4.2}% {:>7.1} {:>4.1}%",
            label, tot / 1e6,
            l.build_j / 1e6, pct(l.build_j),
            l.compute_j / 1e6, pct(l.compute_j),
            l.hold_j / 1e6, pct(l.hold_j),
            l.sense_j, pct(l.sense_j),
            l.carry_j / 1e6, pct(l.carry_j)
        );
        dominants.push(l.dominant());
    }

    // ⛔ The one-year composition used to be four typed literals in this println. Typed literals cannot
    // notice that a constant moved, and two of the constants they depended on were unsourced. They are
    // computed now, and the sweep below reports how far the answer travels inside the stated uncertainty.
    let year = ledger(365.0 * 86400.0, 5_000_000.0, COT_NOMINAL, hold_w, floor.power_w);
    let yr = |v: f64| 100.0 * v / year.total();

    println!("\n  WHAT THE LEDGER SAYS");
    let switched = dominants.windows(2).any(|w| w[0] != w[1]);
    if switched {
        println!("  The dominant term CHANGES with the mission: {}.", dominants.join(" -> "));
        println!("  A ledger that reports a rate cannot express that, which is the argument for carrying");
        println!("  five terms rather than the two it had.");
        println!(
            "  ⭐ AND AT ONE YEAR NO TERM HAS A MAJORITY: build {:.1} / compute {:.1} / hold {:.1} / carry {:.1}",
            yr(year.build_j), yr(year.compute_j), yr(year.hold_j), yr(year.carry_j)
        );
        println!("  percent, largest is {} at {:.1}%. The first version of this file omitted compute and", year.dominant(), yr(year.largest()));
        println!("  understated the hold sevenfold, which made the same mission look like a two-term");
        println!("  contest at 27.7 / 68.2. Counting the missing terms did not weaken the finding.");
    } else {
        println!("  On these inputs one term dominates at every duration tested: {}.", dominants[0]);
        println!("  That is an honest negative result, not a vindication of the two-term ledger: it says");
        println!("  the crossover for THIS body lies outside the range swept, and the range is stated.");
    }
    // HOW FAR THE ANSWER TRAVELS INSIDE THE STATED UNCERTAINTY. Two constants drive the one-year
    // composition and neither is a measurement of this body: the embodied intensities, which the module
    // note says span roughly an order of magnitude, and the chassis cost of transport, whose sourced
    // legged span is 0.2 to 15. Reporting one point inside that box is reporting a choice as a finding.
    println!("\n  SENSITIVITY OF THE ONE-YEAR COMPOSITION (dominant term, its share)");
    print!("  {:<14}", "intensity");
    for (label, _) in COT_SPAN {
        print!(" {label:>16}");
    }
    println!();
    let mut year_dominants = Vec::new();
    for (iscale_label, iscale) in INTENSITY_SPAN {
        print!("  {iscale_label:<14}");
        for (_, cot) in COT_SPAN {
            let l = ledger_at(365.0 * 86400.0, 5_000_000.0, cot, hold_w, floor.power_w, iscale);
            let d = l.dominant();
            year_dominants.push(d);
            print!(" {:>10} {:>4.0}%", d, 100.0 * l.largest() / l.total());
        }
        println!();
    }
    let robust = year_dominants.windows(2).all(|w| w[0] == w[1]);
    if robust {
        println!(
            "  The dominant term is {} at every corner, so the one-year answer survives the whole\n  \
             stated uncertainty. That is the strong form of the claim.",
            year_dominants[0]
        );
    } else {
        let mut seen: Vec<&str> = Vec::new();
        for d in &year_dominants {
            if !seen.contains(d) {
                seen.push(d);
            }
        }
        println!(
            "  ⛔ THE ONE-YEAR DOMINANT TERM IS NOT ROBUST: inside the uncertainty this file itself\n  \
             declares it is any of {}, so the {:.1}% figure above is a point in a box, not a finding.\n  \
             What survives the whole box is the weaker and still useful claim: the composition is a\n  \
             CONTEST, no term is negligible, and the two constants that decide it are the embodied\n  \
             intensity of this body and the cost of transport of the chassis it rides on. Those two are\n  \
             the measurements that would turn this into a number.",
            seen.join(", "), yr(year.largest())
        );
    }

    // THE MISSION-SHAPE CLAIM, CHECKED AT EVERY CORNER RATHER THAN AT THE NOMINAL POINT. The claim that
    // survives the box has to be shown to survive it, so this asserts rather than prints: build is the
    // largest term on the two short missions and is NOT the largest at ten years, for all 12 corners.
    let mut short_build = 0;
    let mut long_not_build = 0;
    for (_, iscale) in INTENSITY_SPAN {
        for (_, cot) in COT_SPAN {
            let minute = ledger_at(60.0, 0.0, cot, hold_w, floor.power_w, iscale);
            let shift = ledger_at(8.0 * 3600.0, 12_000.0, cot, hold_w, floor.power_w, iscale);
            let decade = ledger_at(3650.0 * 86400.0, 50_000_000.0, cot, hold_w, floor.power_w, iscale);
            assert_eq!(minute.dominant(), "build", "a one-minute mission is all build at every corner");
            assert_eq!(shift.dominant(), "build", "an 8 h shift is still build-dominated at every corner");
            assert_ne!(decade.dominant(), "build", "at ten years build is never the largest term");
            short_build += 2;
            long_not_build += 1;
        }
    }
    println!(
        "  Mission shape DOES survive the box: build largest on both short missions at {}/{} corner-missions,\n  and never largest at ten years ({}/{}). The shape is the finding; the split is not.",
        short_build, 2 * INTENSITY_SPAN.len() * COT_SPAN.len(),
        long_not_build, INTENSITY_SPAN.len() * COT_SPAN.len()
    );

    println!(
        "\n  ⭐ THE SENSE LEDGER ANSWER, AND IT REFRAMES THE QUESTION. The information floor for sensing is\n  \
         {:.1} uW against a {:.2} W hold — {:.4}% of the standing power, and utterly invisible in the total.\n  \
         But the SENSOR is not invisible: owning and carrying its 120 g costs {:.1} MJ to build and\n  \
         {:.1} MJ to carry over 5000 km. So the joules of sensing are not in the MEASUREMENT, they are in\n  \
         OWNING AND CARRYING the thing that measures. That is why evolution deletes a modality rather than\n  \
         duty-cycling it, and it is a different answer from the one the question implies.",
        floor.power_w * 1e6, hold_w, 100.0 * floor.power_w / hold_w,
        0.120 * SENSOR_INTENSITY_MJ_PER_KG,
        COT_NOMINAL * 0.120 * 9.81 * 5_000_000.0 / 1e6
    );
    println!(
        "\n  HONEST SCOPE. Copper loss only for the hold, so E_hold is a LOWER bound; a cited power model of\n  \
         a 7-DoF arm found viscous and Coulomb friction dominant at several joints. Embodied intensities are\n  \
         stated inputs spanning an order of magnitude in the literature. E_sense prices the transducer's\n  \
         information floor, not its readout electronics. No term here is a measurement of a physical robot.\n"
    );
}
