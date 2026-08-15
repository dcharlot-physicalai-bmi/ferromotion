# ferromotion-viz

[![crates.io](https://img.shields.io/crates/v/ferromotion-viz.svg)](https://crates.io/crates/ferromotion-viz)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-viz)](https://docs.rs/ferromotion-viz)

**Rerun logging for [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion)** — robots,
trajectories and calibration curves streamed into the [rerun.io](https://rerun.io) viewer.

This is deliberately a **companion** crate. The model-based core stays dependency-light and wasm-clean, so
visualization lives here rather than being pulled into every crate that might want it. Native tooling gets
streaming 3-D in a few lines; nothing else in the workspace pays for it.

```rust,no_run
use ferromotion_viz::*;

let (robot, q, waypoint_traj) = ferromotion_viz::doc_fixture();
let rec = rerun::RecordingStreamBuilder::new("my_robot").spawn()?;

log_robot(&rec, "robot", &robot, &q)?;                       // the chain, as points + strip
log_trajectory(&rec, "traj", &robot, &waypoint_traj, 300)?;  // joints + EE path over time
# Ok::<(), Box<dyn std::error::Error>>(())
```

Everything logs through Rerun's standard entity-path and timeline model, so recordings compose with whatever
else the process logs, and `.save(path)` writes an `.rrd` for offline inspection or for attaching to a report.

Because it depends on `rerun`, this crate is **native-only** — reach for it in tooling and tests, not in the
`wasm32` build. Dual-licensed MIT OR Apache-2.0.
