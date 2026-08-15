# ferromotion-bench

[![crates.io](https://img.shields.io/crates/v/ferromotion-bench.svg)](https://crates.io/crates/ferromotion-bench)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-bench)](https://docs.rs/ferromotion-bench)

**A measurement harness with the statistics a performance claim needs** — no dependencies, no host
assumptions, and it builds for `wasm32`. Part of
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion), usable on its own.

**A number without a spread is not a measurement.** So this reports:

- the **median**, not the mean or the minimum, because a median is what survives an interrupted sample;
- the **median absolute deviation** and the 10th/90th percentiles, so the spread is visible next to the
  number rather than discarded;
- an explicit **noise verdict**, because a run whose p90 sits far from its median has not measured the code,
  it has measured the machine.

**Why zero dependencies.** The workspace has exactly one (`nalgebra`), and the standard Rust benchmark crates
do not build for `wasm32-unknown-unknown` — so depending on one would mean performance claims that cannot be
checked in a browser, which is where a good deal of this code actually runs. The clock is abstracted behind
`Clock` instead: `StdClock` on native targets, and `FnClock` for a hook into `performance.now()`.

Iterations per sample are **auto-calibrated** so every sample lasts at least `target_sample`. That is what
makes nanosecond-scale work measurable at all — timing a single 40 ns call measures clock overhead, not the
call.

```rust
use ferromotion_bench::{Bench, StdClock};

// 50 ms per sample, 20 samples, 3 warmup samples
let b = Bench::new(StdClock, 0.05, 20, 3);
let m = b.run(|| {
    // the work under test
});
println!("{}", b.report("my_kernel"));   // median, MAD, p10/p90, noise verdict
```

Read the spread before the headline. If p90/median is large, the right response is to fix the measurement
environment, not to quote the median — and a ratio between two arms measured in the same noisy run is often
trustworthy where either absolute number is not.

Dual-licensed MIT OR Apache-2.0.
