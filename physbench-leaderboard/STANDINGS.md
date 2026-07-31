# Physics-Fidelity Benchmark — Standings

System: frictionless pendulum (energy constant, flow reversible). Verdict = energy drift < 5% AND time-reversibility error < 5%. Scored by CI; do not edit by hand.

| # | model | author | energy drift | reversibility | 1-step RMSE | verdict |
|---|-------|--------|--------------|---------------|-------------|---------|
| 1 | velocity-verlet (structured) | reference | 0.07% | 0.00% | 4.63e-5 | PASS |
| 2 | symplectic (structured) | reference | 2.64% | 0.19% | 3.44e-3 | PASS |
| 3 | structured-net (learned) | reference (trained) | 2.88% | 3.50% | 3.68e-3 | PASS |
| 4 | lossy (wrong invariant) | reference | 30.83% | 240.55% | 4.02e-3 | FAIL |
| 5 | explicit-euler (no structure) | reference | 59.81% | 1342.88% | 3.44e-3 | FAIL |
| 6 | black-box-net (learned) | reference (trained) | 8550.72% | 6444.99% | 2.66e-2 | FAIL |

The tell: a model can be accurate step to step and still violate the invariant over a rollout. Structure, not per-step accuracy, is what earns a PASS. Submit yours — see [README](./README.md).
