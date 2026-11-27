# Panda reach contact, revision 4, 2026-11-27

This file supersedes `docs/acceptance/receipts/2026-11-27-panda-contact-reach-3.md`, sha256 `40a4edcb84d2876baa96bd399fe045c0fdecb9a5c77ead7100142bb7d203a237`. That file is not modified. The episode hashes in it are unchanged.

Commit `f1b28589e2b20175febe9d66dc5ca41e7c250879` adds two facts about the non-zero `qfrc_applied` comparison. This file records them.

## Contact rows

The hinge case stays: two limited hinges, compiler angle left at MuJoCo's degree default, `qfrc_applied` `1.25` and `-0.4`, four steps, no contact. It is still bit-exact.

A second case is the 7-dof Panda in `contact_reach.xml`, with the vendored meshes, the file's position actuators, the floor, and `qfrc_applied`

`0.2 -0.15 0.1 -0.05 0.02 -0.01 0.03`

from pose `0 0.8 0 -1.9600239548543863 0 2.2 0`. Four `mj_step`s. `ncon` is 3 on every step. `ctrl` is left at zero, so the actuators still apply their own bias. Compared with `f64::to_bits` on `qpos`, `qvel`, `act`, `qacc_warmstart`, and `qfrc_constraint`.

oracle-profile run `46027428730`, job `contract-off conformance`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428730/job/140602326835>:

* `libmujoco sha256 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
* `panda.txt step 1 ncon 3` through `panda.txt step 4 ncon 3`
* `qfrc_applied non-zero, bit-exact for 4 steps` for the hinge, then the same line for the Panda
* `step exact 212 of 212; ledger 212` on the menagerie trajectory, whose `qfrc_applied` is zero

## The library hash

`scripts/qfrc_applied_oracle.py` reads `libmujoco.so.3.13.0` next to the imported `mujoco` package and exits unless its sha256 is `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`. The PyPI 3.13.0 wheel's library hashes to `dc9599faeffba8e6ea9f53928de08301ee8c2ba780c4afceadeebe63573c3a1a` and the script refuses it.

The step bits do not distinguish those two libraries. On this host the hinge dump, a sphere on a floor with a non-zero `qfrc_applied`, and the four-step Panda dump were byte-identical under the wheel and under `64156abe`. The sha256 check is what selects the contract-off library.

## Docs job on `8a19270`

Job `140591120180` belongs to run `46024164388`. The Actions API returns `run_id` `46024164388`, name `docs`, conclusion success, and

<https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46024164388/job/140591120180>

The URL in receipt -3 is that job. It is not a separate workflow.

## CI on `f1b2858`

* oracle-profile <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428730>, success
  * `contract-off conformance` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428730/job/140602326835>, success
  * `contract-off native` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428730/job/140602326413>, success
* CI <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768>
  * `integration tests` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768/job/140602327225>, success. Includes `ferromorphic_30812e26_fixtures_match_their_sha256_and_our_verdicts`.
  * `clippy` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768/job/140602326958>, success
  * `docs` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768/job/140602327397>, success
  * `wasm32` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768/job/140602327458>, success
  * `test` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428768/job/140602327459>, failure. Log line: `test result: FAILED. 1014 passed; 3 failed`. The three failures are the published-default goldens.
* menagerie-conformance <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428790>
  * `bit-exact` <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46027428790/job/140602326665>, failure, the published `1 of 212`
