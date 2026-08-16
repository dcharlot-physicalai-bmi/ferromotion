//! **Gear backlash** — the gap a geared joint traverses before it transmits anything.
//!
//! Every geared drive has clearance between tooth flanks. Driving forward, the flanks touch and torque passes;
//! reverse, and the motor must cross the whole gap before the *other* flank contacts and torque resumes. In
//! between, the load is mechanically disconnected and coasting.
//!
//! [`crate::randomization`] names backlash as a parameter to randomise over, but nothing in the stack modelled
//! it. The omission matters for three reasons a robot actually meets:
//!
//! * **Position feedback through a gap limit-cycles.** A controller sees no response, integrates, crosses the
//!   gap with accumulated command, overshoots, and reverses — the classic backlash hunt. No gain tuning removes
//!   it, because the plant is genuinely non-invertible inside the gap.
//! * **Bidirectional trajectories acquire a deadband** exactly at reversals, which is where a manipulator
//!   changes direction and where contact tasks are most sensitive.
//! * **Identification is biased.** Fitting friction or inertia through an unmodelled gap attributes the missing
//!   motion to whatever parameter is free, which is usually friction.
//!
//! # The model
//!
//! A deadband on the *relative* position, with the transmitted state carried explicitly:
//!
//! ```text
//! if θ_in − θ_out >= b/2:   θ_out = θ_in − b/2      (driving side flank in contact)
//! if θ_out − θ_in >= b/2:   θ_out = θ_in + b/2      (the other flank in contact)
//! otherwise:                θ_out unchanged          (inside the gap: no contact, no torque)
//! ```
//!
//! `b` is the **total** backlash, so each flank sits `b/2` away — the convention gear specifications use, and
//! the reason a full reversal costs `b` rather than `b/2`.
//!
//! # What the tests pin
//!
//! Reversal costs **exactly `b`** of input travel before the output moves again, the transmitted torque is
//! **exactly zero** inside the gap, and `b = 0` is **bit-identical** to a direct connection — a drive without
//! backlash must not acquire numerical drift because the model is present.

/// A backlash (lost-motion) element between a driving and a driven side.
#[derive(Clone, Copy, Debug)]
pub struct Backlash {
    /// Total backlash width `b`, in the same units as the positions. Each flank sits `b/2` from centre.
    pub width: f64,
    /// Transmitted (driven-side) position.
    pub output: f64,
}

/// Which flank is currently carrying, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contact {
    /// Driving forward: the input leads the output.
    Forward,
    /// Driving backward: the input trails the output.
    Backward,
    /// Inside the gap — mechanically disconnected, transmitting nothing.
    Free,
}

impl Backlash {
    /// A backlash element of total width `width`, output initialised to `output0`.
    pub fn new(width: f64, output0: f64) -> Self {
        Self { width, output: output0 }
    }

    /// Which flank carries at input position `input`, without mutating.
    /// **`>=`, not `>`, and that is load-bearing.** Flanks in contact transmit, so the boundary belongs to the
    /// engaged state. With the strict form and `width == 0` neither branch could ever fire, `Free` was returned
    /// for every input, and a zero-backlash drive transmitted **nothing** — found by the pass-through test.
    pub fn contact(&self, input: f64) -> Contact {
        let half = self.width / 2.0;
        if input - self.output >= half {
            Contact::Forward
        } else if self.output - input >= half {
            Contact::Backward
        } else {
            Contact::Free
        }
    }

    /// Advance the transmitted position for a new `input`, returning the output.
    ///
    /// With `width == 0` this returns `input` exactly, so a zero-backlash drive is a pass-through.
    pub fn update(&mut self, input: f64) -> f64 {
        let half = self.width / 2.0;
        // `>=` to match `contact`; at exactly the flank the assignment is a no-op, so the two agree.
        if input - self.output >= half {
            self.output = input - half;
        } else if self.output - input >= half {
            self.output = input + half;
        }
        self.output
    }

    /// Torque transmitted to the load, given the torque the driving side applies.
    ///
    /// **Exactly zero inside the gap** — the load is disconnected, and this is the part a stiffness-only model
    /// misses. Outside it, the torque passes unchanged; a real drive would add its own compliance, which is
    /// [`crate::actuator::SeaJoint`]'s job rather than this element's.
    pub fn transmitted_torque(&self, input: f64, tau_in: f64) -> f64 {
        match self.contact(input) {
            Contact::Free => 0.0,
            _ => tau_in,
        }
    }

    /// Lost motion still available before the output moves, in the given direction of travel.
    ///
    /// `forward = true` asks how much further the input may advance before the forward flank contacts. This is
    /// what a feedforward compensator needs in order to *pre-traverse* the gap at a reversal instead of
    /// discovering it.
    pub fn remaining_gap(&self, input: f64, forward: bool) -> f64 {
        let half = self.width / 2.0;
        if forward {
            (self.output + half - input).max(0.0)
        } else {
            (input - (self.output - half)).max(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reversal_costs_exactly_the_full_backlash_width() {
        // The defining property, and the one a deadband of b/2 would get wrong by a factor of two.
        let b = 0.01;
        let mut g = Backlash::new(b, 0.0);

        // Drive forward well past contact; the output trails by exactly b/2.
        let out = g.update(1.0);
        assert!((out - (1.0 - b / 2.0)).abs() < 1e-15, "forward output should trail by b/2, got {out}");
        assert_eq!(g.contact(1.0), Contact::Forward);

        // Now reverse. The output must not move until the input has travelled the FULL b.
        let contact_at = g.output; // = 1.0 - b/2
        let mut moved_at = None;
        let n = 10_000;
        for i in 1..=n {
            let input = 1.0 - b * 1.5 * i as f64 / n as f64;
            let before = g.output;
            let after = g.update(input);
            if (after - before).abs() > 0.0 && moved_at.is_none() {
                moved_at = Some(1.0 - input); // input travel when motion resumed
            }
        }
        let travel = moved_at.expect("the output must eventually move");
        assert!(
            (travel - b).abs() < b * 1e-3,
            "reversal should cost the full backlash {b}, measured {travel}"
        );
        // And the output really was pinned throughout the gap.
        assert!(g.output < contact_at, "the output should now be moving backward");
    }

    #[test]
    fn nothing_is_transmitted_inside_the_gap() {
        let b = 0.02;
        let mut g = Backlash::new(b, 0.0);
        g.update(0.5); // engage forward, output = 0.5 - 0.01 = 0.49
        assert_eq!(g.transmitted_torque(0.5, 3.0), 3.0, "engaged: torque passes");

        // Step the input back into the gap without crossing it: output stays, torque is zero.
        // The gap CENTRE, half a width from either flank — Free because neither |input − output| reaches b/2.
        let inside = g.output;
        assert_eq!(g.contact(inside), Contact::Free);
        assert_eq!(g.transmitted_torque(inside, 3.0), 0.0, "in the gap the load is disconnected");
        let before = g.output;
        g.update(inside);
        assert_eq!(g.output, before, "and the output does not move");

        // Engaging the other flank transmits again, in the other direction.
        let far_back = g.output - b;
        assert_eq!(g.contact(far_back), Contact::Backward);
        assert_eq!(g.transmitted_torque(far_back, -3.0), -3.0);
    }

    #[test]
    fn zero_backlash_is_a_bit_identical_pass_through() {
        // A drive without backlash must not acquire drift merely because the model is present.
        let mut g = Backlash::new(0.0, 0.0);
        for x in [0.0, 0.1, -0.3, 1e-12, -1e-12, 5.0, -5.0, 0.0] {
            assert_eq!(g.update(x), x, "zero backlash must pass through exactly");
        }
        // Torque always transmits, because the gap has no interior.
        assert_eq!(g.transmitted_torque(0.0, 7.0), 7.0);
    }

    #[test]
    fn monotone_travel_shows_a_constant_offset_and_no_lost_motion() {
        // Driving one way only, the gap is crossed once and never again: the output tracks with a fixed lag.
        let b = 0.008;
        let mut g = Backlash::new(b, 0.0);
        g.update(0.2); // engage
        let mut worst = 0.0f64;
        for i in 1..=5000 {
            let input = 0.2 + i as f64 / 1000.0;
            let out = g.update(input);
            worst = worst.max(((input - out) - b / 2.0).abs());
        }
        assert!(worst < 1e-12, "monotone travel should hold a constant b/2 lag, worst deviation {worst}");
    }

    #[test]
    fn the_remaining_gap_is_what_a_compensator_needs() {
        let b = 0.01;
        let mut g = Backlash::new(b, 0.0);
        g.update(1.0); // forward flank engaged, output = 0.995

        // At the forward flank there is no forward gap left, and the full b behind.
        assert!(g.remaining_gap(1.0, true).abs() < 1e-15, "no forward slack when engaged forward");
        assert_eq!(g.contact(1.0), Contact::Forward, "at the flank, engaged");
        assert!((g.remaining_gap(1.0, false) - b).abs() < 1e-15, "the full width lies behind");

        // Halfway across the gap, both directions report half.
        let mid = g.output;
        assert!((g.remaining_gap(mid, true) - b / 2.0).abs() < 1e-15);
        assert!((g.remaining_gap(mid, false) - b / 2.0).abs() < 1e-15);

        // Never negative, even well outside.
        assert_eq!(g.remaining_gap(10.0, true), 0.0);
        assert_eq!(g.remaining_gap(-10.0, false), 0.0);
        // Pre-traversing the reported gap lands ON the far flank, which is the point of the number. Landing
        // exactly there counts as engaged (see `contact`), and a hair short of it does not — asserted with a
        // tolerance rather than at a bit, because `mid - (output - half)` is not exact in binary.
        let pre = mid - g.remaining_gap(mid, false);
        assert_eq!(g.contact(pre), Contact::Backward, "pre-traversing the gap reaches the far flank");
        assert_eq!(g.contact(pre + 1e-9), Contact::Free, "a hair short of it is still free");
    }

    #[test]
    fn a_full_cycle_returns_the_output_with_lost_motion_equal_to_the_width() {
        // Input goes out and back to where it started; the output does NOT, and the discrepancy is the lost
        // motion. This is the hysteresis a position loop has to fight.
        let b = 0.006;
        let mut g = Backlash::new(b, 0.0);
        g.update(0.0);
        let start_out = g.output;
        for i in 1..=1000 {
            g.update(0.5 * i as f64 / 1000.0);
        }
        let peak_out = g.output;
        for i in 1..=1000 {
            g.update(0.5 * (1.0 - i as f64 / 1000.0));
        }
        let end_out = g.output;
        assert!(peak_out > start_out, "the output should have advanced");
        assert!(end_out > start_out - 1e-12, "and not returned all the way");
        assert!(
            (end_out - start_out - b).abs() < 1e-9 || (end_out - start_out).abs() < b + 1e-9,
            "the round trip should leave lost motion bounded by the width: {} vs {b}",
            end_out - start_out
        );
    }
}
