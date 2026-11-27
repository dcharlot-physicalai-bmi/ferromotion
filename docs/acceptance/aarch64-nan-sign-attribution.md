# aarch64 op-vector NaN sign

`docs/acceptance/oracles/aarch64-contract-off-op-vectors.txt` is the output of `scripts/arm_oracle/op-vectors.sh` at `558f3a5fbed63b2599e381c929e297798872592b`, run on `cbio@thor2.local` (Jetson Thor, aarch64). clang 18.1.3 compiled `mju_mulQuat`, `mju_rotVecQuat`, and `mju_mulMatVec3` from the pinned MuJoCo 3.13.0 tree, commit `123347c0eeab7e13c8da0828ab593bbd95bcf335`, tree `c0408a44`, with `-ffp-contract=off`. The file is sha256 `e3f5636da43556256bc63af79f1b31e07e21c7cb54ba188502e3b5d3d239853b`, 144 lines, 37304 bytes. It was transcribed here from the x86 vectors by replacing one integer, then checked against that sha256 before the commit.

Equality in `fp_contract::tests::unfused_sites_match_the_named_oracle_on_random_and_adversarial_vectors` stays bitwise (`f64::to_bits`). On aarch64 the test reads this file. On x86_64 it reads `x86_64-contract-off-op-vectors.txt` (sha256 `c5fb5ce730d4c323d7553c0231db69583cba75ae1bb85137c4283c011a4a72f1`).

## The one different line

Line 17, `rot_vec_quat`. Inputs, as `u64` bit patterns:

`9094988921128908188,4599075939470750515,4598175219545276416,9094988921128908188,4591870180066957722,13822447976325526323,0`

| lane | x86_64 | aarch64 |
|---|---|---|
| 0 | `9092769713007476592` | `9092769713007476592` |
| 1 | `18444492273895866368` (`0xFFF8000000000000`, −qNaN) | `9221120237041090560` (`0x7FF8000000000000`, +qNaN) |
| 2 | `9218868437227405312` | `9218868437227405312` |

Every other line of the two files is identical.

## Generated-NaN sign

A C probe of `inf - inf` and `inf * 0` on Thor2, with `FPCR = 0` (`DN = 0`), writes `0x7ff8000000000000`. The same operations on x86_64 write `0xfff8000000000000`. That is the quiet NaN each CPU generates for an invalid operation. The op bodies and the `-ffp-contract=off` flags are the same on both arches.

With this file in the tree, the aarch64 `fp_contract` test passes, so the Rust unfused `rot_vec_quat` reproduces the aarch64 C bits, including this sign. The same Thor2 run of the library is 1014 passed and the three known published-default failures. `run-rust-212.sh` at `558f3a5` (receipt `876254f7…`) is 212 of 212 steps, native 636 of 636 (`efc_AR` 30 of 30), and model 212 of 212. The normalized native and step logs equal the x86 logs `47956e7bdd28324a64f2afc81679c46094c7d8ceecdaf3a8cb6bc77b5c457db0` and `577978e637fc4cbbd24368f202af17f74adf4536d5cd3c16cc430b1ede79ea5a`. Every `mjData` and `mjModel` field matches. rustc 1.100.0 matches the binary `.comment`.
