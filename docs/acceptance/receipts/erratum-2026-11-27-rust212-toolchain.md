# Erratum: rust212 receipts copied the toolchain line

2026-11-27. The rust212 `RECEIPT.txt` files below are not in this repository. Their `cargo_vv` line says `cargo 1.99.1` because the writer copied stage 2's `provenance/cargo-vv.txt`. That file is the toolchain of an earlier stage, not a reading of the binary the receipt hashed. No receipt file under `docs/acceptance/receipts/` is edited by this note.

The comparisons are unchanged. The same 11 native-field mismatches, with `mj_step` bit-exact on 212 of 212 models, reproduce on both compilers: `7492eadf…` with rustc 1.100.0 and `83919cff…` with rustc 1.99.1.

## Receipts

Each path is under `/home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/` on thor2. The sha256 is of `RECEIPT.txt`. The compiler is the one named in `docs/acceptance/aarch64-native-attribution.md`, read there from the binary and from `cargo-target/.rustc_info.json`, not from `cargo_vv`.

| receipt | sha256 | what ran | compiler that built the binaries | `cargo_vv` |
|---|---|---|---|---|
| `rust212-20261120T135357Z/RECEIPT.txt` | `82263c2b9ec4dd633612e931d9a78c997b9f319d814ac27851726ebbee9bf45e` | stage 5 for `2aeb3611`, 212/212 steps, 11 native fields short of 636 | rustc 1.100.0 (`0ef1bc386`), Thor2's stable toolchain | copied `cargo 1.99.1` |
| `rust212-20261127T131407Z/RECEIPT.txt` | `7492eadfe00d0a9fa941cef92525f91f121b8f2b59cbb9be950ea329d24161e8` | unpatched `2aeb3611`, `parity_mismatch`, 11 unequal tallies, 212 of 212 stepped | rustc 1.100.0 (`0ef1bc386`) | copied `cargo 1.99.1` |
| `rust212-20261127T132120Z/RECEIPT.txt` | `4044aa73e08e6631e1ca8a4d853d9910957068eaa3b10b84e4f7c8f31789736f` | the three-site fix, `parity_accept`, 212 models, 20 steps, 76 tallies | rustc 1.100.0 (`0ef1bc386`) | copied `cargo 1.99.1` |
| `rust212-20261127T132613Z/RECEIPT.txt` | `83919cffae933edb5938696437c1f219d79c2b996995e4168f6c2bc5ecd6d36a` | unpatched `2aeb3611`, the same 11 mismatches as `7492eadf…` | rustc 1.99.1 (`58cf8f9c3`) | copied `cargo 1.99.1` (the release number matches this compiler; the line was still copied) |
| `rust212-20261127T131736Z/RECEIPT.txt` | `f3e5c56bfcb3dba25d8e0b20ee875d13d8752b06ac8ef2f1ea18e85b1d287009` | the three-site fix, `parity_accept`, 212 models, 20 steps, 76 tallies | rustc 1.99.1 (`58cf8f9c3`) | copied `cargo 1.99.1` (the release number matches this compiler; the line was still copied) |

`scripts/arm_oracle/direct-pull/pins.sh` also names `rust212-20261109T215708Z` and the runners refuse to overwrite it. That `RECEIPT.txt` is not in this repository, so this note does not state its `cargo_vv` line.

## EVIDENCE

Binary `.comment` strings, to be filled from thor2. They were not in this repository, and no CI artifact here contains those `menagerie_native` binaries.

For each receipt directory, on the `menagerie_native` (and `menagerie_native_step`) that the receipt hashed:

```text
readelf -p .comment <binary>
```

or, if `readelf` is absent:

```text
objdump -s -j .comment <binary>
```

Paste the `rustc version …` string from `.comment` under the receipt name. A `GCC:` string in the same section is the linker, not the compiler that type-checked the crate.
