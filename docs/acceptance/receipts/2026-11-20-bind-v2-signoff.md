# Sign-off addendum, 2026-11-20

This addendum does not edit the receipt. The receipt is `docs/acceptance/receipts/2026-11-20-bind-v2.md`, receipt commit `cfc22d8e1c38f1bf16c7c06c95d017485652fb71`, code commit `506593c83fe2cc2e0a44e03829c911cef79c0286`, sha256 `52497d20cacdcf1b08ed8b9cdbd1722c79f95569c1a51974346d0f6befb16cb9`.

verify 44755415928 green on 82211dc (comment/doc/fetch-only diff from 1ceec2fc)

Signed off by physical-ai@bmi-2 (ferromorphic owner), 2026-11-20

verify 44742061934 failed in the Zenodo download before any gate ran

Pins: ferromorphic `1ceec2fc884668db16905bed190a8c05873423e7` (tree `9215e24dcdcd5fc2e18cda0e24ebe78c32863d8b`), rustc 1.100.0 (0ef1bc386). The git dependency stays at `1ceec2fc`. `82211dc861af4ecc1afa9603c2caf20d52712b6d` is the verify head. `git diff 1ceec2fc884668db16905bed190a8c05873423e7 82211dc861af4ecc1afa9603c2caf20d52712b6d` changes only comments in `src/sim.rs`, `docs/contracts/ferromotion-clock.md`, and `tools/fetch_heldout_inputs.py`.

Run `44755415928` on `dcharlot-physicalai-bmi/ferromorphic` has `head_sha` `82211dc861af4ecc1afa9603c2caf20d52712b6d`, `conclusion` `success`, and both jobs `nine release gates` and `manifest, mutations, references` completed with `conclusion` `success`.

Run `44742061934` on head `1ceec2fc884668db16905bed190a8c05873423e7` has job `nine release gates` failed at `Held-out tables required by cargo test`. The log is `urllib.error.ContentTooShortError` from `urlretrieve` of Zenodo `indy_20160630_01` (145223876 of 382243800 bytes). The step `Run the nine gates` was skipped.

Scope: tests (a) through (f) verify the bind contract (pure validation, InexactPeriod above 2^53, refusals on epoch, origin, period, dt_bits and the grid, revalidation on the next call, forward-product mismatch refused before an envelope exists). Test (b) proves bind leaves Sim::state_bytes unchanged, which covers clock, mode, exact_timing, ledger, delivery ring, inject, touched and as_of, and not net or neurons.

`FERROMORPHIC_BIND_VERIFIED` is true for that scope. `JOINT_LOOP_EXECUTED` stays false. `EPISODE_ENVELOPE_INTEROPERABLE` stays false.
