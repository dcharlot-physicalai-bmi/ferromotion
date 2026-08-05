//! **Measurement harness** — the statistics a performance claim needs, with no dependencies and no host assumptions.
//!
//! A number without a spread is not a measurement. This harness reports the **median** rather than the mean or the
//! minimum, because a median is what survives an interrupted sample; the **median absolute deviation** and the
//! 10th/90th percentiles, so the spread is visible; and an explicit **noise verdict**, because a run whose p90 is far
//! from its median has not measured the code, it has measured the machine.
//!
//! It carries no dependencies for two reasons. The workspace has exactly one (nalgebra), and the standard Rust
//! benchmark crates do not build for `wasm32-unknown-unknown` — so depending on one would mean performance claims that
//! cannot be checked in the place a lot of this code actually runs. The clock is abstracted behind [`Clock`] instead,
//! with a `std` implementation on native targets and a hook for `performance.now()` in a browser.
//!
//! Iterations per sample are auto-calibrated so each sample lasts at least [`Bench::target_sample`], which is what
//! makes nanosecond-scale operations measurable at all: timing a single 40 ns call measures clock overhead.

use core::hint::black_box;

/// A monotonic clock in seconds. Abstracted so the harness works where `std::time::Instant` does not.
pub trait Clock {
    fn now_seconds(&self) -> f64;
}

/// `std::time::Instant`, on targets that have a working one.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct StdClock;

#[cfg(not(target_arch = "wasm32"))]
impl Clock for StdClock {
    fn now_seconds(&self) -> f64 {
        static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        ORIGIN.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
    }
}

/// A clock backed by any closure — the browser hook, and what the harness's own tests use so they are deterministic.
pub struct FnClock<F: Fn() -> f64>(pub F);

impl<F: Fn() -> f64> Clock for FnClock<F> {
    fn now_seconds(&self) -> f64 {
        (self.0)()
    }
}

/// What one benchmark measured. Times are seconds **per iteration**.
#[derive(Clone, Copy, Debug)]
pub struct Measurement {
    pub median: f64,
    /// Median absolute deviation, the robust spread. Reported rather than a standard deviation because one paused
    /// sample moves a standard deviation and does not move this.
    pub mad: f64,
    pub p10: f64,
    pub p90: f64,
    pub min: f64,
    pub samples: usize,
    pub iters_per_sample: usize,
}

impl Measurement {
    pub fn per_second(&self) -> f64 {
        if self.median > 0.0 { 1.0 / self.median } else { f64::INFINITY }
    }

    /// Spread relative to the median. Above about 0.1 the run is dominated by the machine, not the code.
    pub fn relative_spread(&self) -> f64 {
        if self.median > 0.0 { (self.p90 - self.p10) / self.median } else { f64::INFINITY }
    }

    /// **Whether this measurement is worth quoting.** A benchmark that cannot say this is a benchmark that will be
    /// quoted anyway, which is how performance folklore gets made.
    pub fn trustworthy(&self) -> bool {
        self.samples >= 5 && self.median > 0.0 && self.relative_spread() < 0.1
    }

    /// One line, with the spread and the verdict attached so they cannot be dropped in transit.
    pub fn report(&self, name: &str) -> String {
        format!(
            "{name:<38} {:>12}  +/-{:>9}  [p10 {:>10}, p90 {:>10}]  {:>12}/s  {}",
            format_time(self.median),
            format_time(self.mad),
            format_time(self.p10),
            format_time(self.p90),
            format_count(self.per_second()),
            if self.trustworthy() { "ok" } else { "NOISY - do not quote" }
        )
    }
}

/// Seconds as a human-readable duration with three significant figures.
pub fn format_time(s: f64) -> String {
    if !s.is_finite() {
        return "-".into();
    }
    let (v, unit) = if s < 1e-6 {
        (s * 1e9, "ns")
    } else if s < 1e-3 {
        (s * 1e6, "us")
    } else if s < 1.0 {
        (s * 1e3, "ms")
    } else {
        (s, "s")
    };
    format!("{v:.3} {unit}")
}

/// A rate with an SI suffix.
pub fn format_count(n: f64) -> String {
    if !n.is_finite() {
        return "-".into();
    }
    if n >= 1e9 {
        format!("{:.2}G", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.2}k", n / 1e3)
    } else {
        format!("{n:.1}")
    }
}

/// The harness.
pub struct Bench<C: Clock> {
    pub clock: C,
    /// Minimum wall time per sample. Iterations are calibrated up to reach it.
    pub target_sample: f64,
    pub samples: usize,
    /// Samples discarded before measuring, so caches and branch predictors are warm.
    pub warmup_samples: usize,
    /// Ceiling on calibrated iterations. Without one, a clock too coarse to resolve `target_sample` drives the
    /// iteration count up by a fixed factor every round until the `f64 -> usize` cast saturates at `usize::MAX` —
    /// which is how the harness's first run reported 18446744073709551615 iterations per sample.
    pub max_iters: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for Bench<StdClock> {
    fn default() -> Self {
        Bench { clock: StdClock, target_sample: 0.01, samples: 25, warmup_samples: 3, max_iters: 1 << 30 }
    }
}

impl<C: Clock> Bench<C> {
    pub fn new(clock: C, target_sample: f64, samples: usize, warmup_samples: usize) -> Bench<C> {
        Bench { clock, target_sample, samples, warmup_samples, max_iters: 1 << 30 }
    }

    /// Calibrate how many iterations fit in `target_sample`, scaling up until they do. Returns at least 1 and never
    /// more than [`Self::max_iters`] — the cap is load-bearing, not defensive: a `f64 -> usize` cast saturates rather
    /// than wrapping, so an unbounded growth loop lands on `usize::MAX` and the harness then tries to run it.
    fn calibrate<F: FnMut()>(&self, f: &mut F) -> usize {
        let cap = self.max_iters.max(1);
        let mut iters = 1usize;
        for _ in 0..40 {
            let start = self.clock.now_seconds();
            for _ in 0..iters {
                f();
            }
            let elapsed = self.clock.now_seconds() - start;
            if elapsed >= self.target_sample || iters >= cap {
                return iters.min(cap);
            }
            let growth = if elapsed > 0.0 { (self.target_sample / elapsed).min(8.0) } else { 8.0 };
            let next = (iters as f64 * growth).ceil();
            iters = if next.is_finite() && next < cap as f64 { (next as usize).max(iters + 1) } else { cap };
        }
        iters.min(cap)
    }

    /// **Measure `f`.** Calibrates, warms up, then takes `samples` timed batches and reduces them robustly.
    pub fn run<F: FnMut()>(&self, mut f: F) -> Measurement {
        let iters = self.calibrate(&mut f);
        for _ in 0..self.warmup_samples {
            for _ in 0..iters {
                f();
            }
        }
        let mut per_iter = Vec::with_capacity(self.samples);
        for _ in 0..self.samples {
            let start = self.clock.now_seconds();
            for _ in 0..iters {
                f();
            }
            let elapsed = self.clock.now_seconds() - start;
            per_iter.push(elapsed / iters as f64);
        }
        per_iter.sort_by(f64::total_cmp);
        Measurement {
            median: quantile(&per_iter, 0.5),
            mad: median_absolute_deviation(&per_iter),
            p10: quantile(&per_iter, 0.10),
            p90: quantile(&per_iter, 0.90),
            min: per_iter.first().copied().unwrap_or(f64::NAN),
            samples: per_iter.len(),
            iters_per_sample: iters,
        }
    }

    /// Measure a closure whose result must not be optimised away.
    pub fn run_value<T, F: FnMut() -> T>(&self, mut f: F) -> Measurement {
        self.run(move || {
            black_box(f());
        })
    }
}

/// Linear-interpolated quantile of a **sorted** slice.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Median absolute deviation of a sorted slice.
pub fn median_absolute_deviation(sorted: &[f64]) -> f64 {
    let m = quantile(sorted, 0.5);
    let mut dev: Vec<f64> = sorted.iter().map(|v| (v - m).abs()).collect();
    dev.sort_by(f64::total_cmp);
    quantile(&dev, 0.5)
}

/// A named group of measurements, for printing a table with a header that says what the columns mean.
#[derive(Default)]
pub struct Suite {
    pub rows: Vec<(String, Measurement)>,
}

impl Suite {
    pub fn push(&mut self, name: impl Into<String>, m: Measurement) {
        self.rows.push((name.into(), m));
    }

    pub fn header() -> String {
        format!("{:<38} {:>12}  {:>12}  {:>26}  {:>14}  {}", "benchmark", "median", "MAD", "10th/90th percentile", "rate", "verdict")
    }

    pub fn print(&self, title: &str) {
        println!("\n{title}");
        println!("{}", Suite::header());
        println!("{}", "-".repeat(132));
        for (name, m) in &self.rows {
            println!("{}", m.report(name));
        }
        let noisy = self.rows.iter().filter(|(_, m)| !m.trustworthy()).count();
        if noisy > 0 {
            println!("\n  {noisy} of {} rows are too noisy to quote. Re-run on a quiet machine before citing them.", self.rows.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn quantiles_and_mad_are_right_on_a_known_sample() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&s, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile(&s, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&s, 1.0) - 5.0).abs() < 1e-12);
        // interpolated: p10 of 5 points sits at index 0.4
        assert!((quantile(&s, 0.10) - 1.4).abs() < 1e-12);
        // deviations from the median 3 are [2,1,0,1,2], whose median is 1
        assert!((median_absolute_deviation(&s) - 1.0).abs() < 1e-12);
    }

    /// The median, not the minimum or the mean, is what a paused sample must not move.
    #[test]
    fn one_stalled_sample_moves_the_mean_and_not_the_median() {
        let mut s: Vec<f64> = (0..25).map(|_| 1.0).collect();
        s.push(1000.0); // one interrupted batch
        s.sort_by(f64::total_cmp);
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        eprintln!("25 clean samples of 1.0 plus one stall of 1000: mean {mean:.2}, median {:.2}", quantile(&s, 0.5));
        assert!((quantile(&s, 0.5) - 1.0).abs() < 1e-12, "the median is untouched");
        assert!(mean > 38.0, "the mean is destroyed: {mean:.2}");
    }

    /// Calibration must reach the target sample duration, which is what makes a 40 ns operation measurable.
    #[test]
    fn calibration_reaches_the_target_sample_duration() {
        // A synthetic clock makes the harness's own arithmetic checkable, which a wall clock never is.
        // Each read advances 1 ns; the work itself is free, so only the clock costs anything.
        let tick = Cell::new(0.0f64);
        let bench = Bench::new(
            FnClock(|| {
                let t = tick.get();
                tick.set(t + 1e-9);
                t
            }),
            1e-3,
            5,
            0,
        );
        let m = bench.run(|| {});
        eprintln!("target 1 ms with a 1 ns clock: calibrated to {} iterations per sample", m.iters_per_sample);
        assert!(m.iters_per_sample > 1, "it did not settle for a single iteration");
        assert!(m.iters_per_sample <= 1 << 30, "and it did not saturate to usize::MAX: {}", m.iters_per_sample);
        assert_eq!(m.samples, 5);

        // a clock too coarse to resolve the target must stop at the cap rather than climb forever
        let frozen = Cell::new(0.0f64);
        let capped = Bench { max_iters: 4096, ..Bench::new(FnClock(|| frozen.get()), 1e-3, 5, 0) };
        let stuck = capped.run(|| {});
        eprintln!("a frozen clock: calibrated to {} iterations (cap 4096)", stuck.iters_per_sample);
        assert_eq!(stuck.iters_per_sample, 4096, "it stops at the cap");
        assert!(!stuck.trustworthy(), "and a zero-median measurement is not quotable");
    }

    /// A measurement has to be able to disown itself.
    #[test]
    fn a_noisy_measurement_says_so() {
        let clean = Measurement { median: 1e-6, mad: 1e-9, p10: 0.99e-6, p90: 1.01e-6, min: 0.98e-6, samples: 25, iters_per_sample: 100 };
        let noisy = Measurement { p90: 3e-6, ..clean };
        eprintln!(" clean: spread {:.3} -> {}", clean.relative_spread(), clean.trustworthy());
        eprintln!(" noisy: spread {:.3} -> {}", noisy.relative_spread(), noisy.trustworthy());
        assert!(clean.trustworthy() && !noisy.trustworthy());
        assert!(noisy.report("x").contains("NOISY"), "and it says so where it will be read");
        // too few samples is also untrustworthy, however tight the spread
        assert!(!Measurement { samples: 2, ..clean }.trustworthy());
    }

    #[test]
    fn formatting_picks_sensible_units() {
        assert_eq!(format_time(4.2e-9), "4.200 ns");
        assert_eq!(format_time(4.2e-6), "4.200 us");
        assert_eq!(format_time(4.2e-3), "4.200 ms");
        assert_eq!(format_time(4.2), "4.200 s");
        assert_eq!(format_count(1.5e9), "1.50G");
        assert_eq!(format_count(2.5e6), "2.50M");
    }
}

/// **An allocation counter**, for measuring whether a control loop's hot path allocates.
///
/// A real-time loop that heap-allocates has a latency tail set by the allocator, not by the algorithm, and no amount of
/// median timing reveals it. Counting is the only way to see it: install [`CountingAllocator`] as the global allocator in
/// a binary and read [`AllocCounter`] around the code under test.
///
/// This is the one place in the workspace that needs `unsafe`, because a `GlobalAlloc` implementation is `unsafe` by
/// definition. It forwards every request to the system allocator and only increments a counter, so it changes allocation
/// behaviour in no way other than timing.
pub mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static BYTES: AtomicUsize = AtomicUsize::new(0);

    /// A pass-through allocator that counts calls. Install with `#[global_allocator]`.
    pub struct CountingAllocator;

    // SAFETY: every method forwards directly to `System`, which is a valid allocator, with identical layouts and
    // pointers. The counters are atomics and touch none of the allocation state.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size, Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    /// A snapshot of the counters, for differencing around a region of interest.
    #[derive(Clone, Copy, Debug)]
    pub struct AllocCounter {
        pub allocations: usize,
        pub bytes: usize,
    }

    impl AllocCounter {
        pub fn now() -> AllocCounter {
            AllocCounter { allocations: ALLOCATIONS.load(Ordering::Relaxed), bytes: BYTES.load(Ordering::Relaxed) }
        }

        /// Allocations and bytes since this snapshot.
        pub fn since(&self) -> AllocCounter {
            let now = AllocCounter::now();
            AllocCounter { allocations: now.allocations - self.allocations, bytes: now.bytes - self.bytes }
        }
    }

    /// Count the allocations a closure performs.
    pub fn count<T>(f: impl FnOnce() -> T) -> (T, AllocCounter) {
        let before = AllocCounter::now();
        let out = f();
        (out, before.since())
    }
}
