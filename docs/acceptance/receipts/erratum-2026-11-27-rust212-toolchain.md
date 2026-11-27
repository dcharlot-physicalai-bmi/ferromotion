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

Read on Thor2 from the example binaries. Each receipt directory is under `/home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/`. The binaries are `<dir>/cargo-target/release/examples/menagerie_native` and `menagerie_native_step`. No receipt file under `docs/acceptance/receipts/` is edited by this note, and none of those receipts records a hash of either binary. The tie from a receipt to these files is the directory name and the timestamp in that name.

Every one of the five `RECEIPT.txt` files says `cargo_vv cargo 1.99.1`. That line was copied. It is not a reading of the binary.

Each `.comment` section also contains `GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` and a `clang version 23.1.0git` string. Those are the linker and the C compiler that built objects linked into the image, not the rustc that type-checked the crate. The rustc string is the one quoted below.

| receipt sha256 | directory | menagerie_native | menagerie_native_step | `.comment` rustc |
|---|---|---|---|---|
| `82263c2b9ec4dd633612e931d9a78c997b9f319d814ac27851726ebbee9bf45e` | `rust212-20261120T135357Z` | `f2969d460813584b718b73a55637b57a6ccc9aa2ccc8a5971a0f035bcd3d2703` | `e1f1f788a839642c62279f94faecef748a7ef58b4e6de77fc4c1f6154bedbfc8` | `rustc version 1.100.0 (0ef1bc386 2026-11-09)` |
| `7492eadfe00d0a9fa941cef92525f91f121b8f2b59cbb9be950ea329d24161e8` | `rust212-20261127T131407Z` | `f2969d460813584b718b73a55637b57a6ccc9aa2ccc8a5971a0f035bcd3d2703` | `e1f1f788a839642c62279f94faecef748a7ef58b4e6de77fc4c1f6154bedbfc8` | `rustc version 1.100.0 (0ef1bc386 2026-11-09)` |
| `f3e5c56bfcb3dba25d8e0b20ee875d13d8752b06ac8ef2f1ea18e85b1d287009` | `rust212-20261127T131736Z` | `b62dd54cddcc10d1086b35b4a2a4d04a2f74803b6570f75f6b8cbf8f3e5bef88` | `f44c1ab06370b03050aaf760e30a9918650d27b2857209af72161a39a4ae41b0` | `rustc version 1.99.1 (58cf8f9c3 2026-10-13)` |
| `4044aa73e08e6631e1ca8a4d853d9910957068eaa3b10b84e4f7c8f31789736f` | `rust212-20261127T132120Z` | `d45363c7520265aeea8aed6503c865406d0b5a7563ff4f16c4beb87f185af459` | `631905caefa587adffad86f84be3f467d33c788829873e85f4633157544113f5` | `rustc version 1.100.0 (0ef1bc386 2026-11-09)` |
| `83919cffae933edb5938696437c1f219d79c2b996995e4168f6c2bc5ecd6d36a` | `rust212-20261127T132613Z` | `cf34ab1f04f2ec3abf7349e786960411ffd825636145e23bbf32b3f95f5efc12` | `f1058c40aacd08eb167b3e50f8a319883062bbb10b2c2457088acb7d6dc5f547` | `rustc version 1.99.1 (58cf8f9c3 2026-10-13)` |

`82263c2b…` and `7492eadf…` name different receipts and different directories. Their `menagerie_native` bytes are the same file, and so are their `menagerie_native_step` bytes.

`cargo_vv cargo 1.99.1` is wrong for `82263c2b…`, `7492eadf…` and `4044aa73…`. Those three binaries were built with rustc 1.100.0 (`0ef1bc386`). It matches the release number of rustc 1.99.1 (`58cf8f9c3`) for `f3e5c56b…` and `83919cff…`, and it was still copied rather than read from the binary.
