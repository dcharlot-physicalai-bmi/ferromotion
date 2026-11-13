# urdf-rs provenance

This directory is crates.io `urdf-rs` **0.10.0** with one dependency change.

* Upstream: <https://github.com/openrr/urdf-rs>
* Published: 2026-09-25. Still the latest crates.io release as of 2026-11-13 (`newest_version` 0.10.0). No newer tag.
* Crate tarball: `https://static.crates.io/crates/urdf-rs/urdf-rs-0.10.0.crate`
* Tarball sha256: `0c1c4d002a2d281d6ae7a1a8b319bddc50506a35731e826fb9dbd8d648c39bcf`
* That digest is the crates.io version checksum for 0.10.0.

`src/`, `README.md`, and `LICENSE` are byte-identical to that tarball. `Cargo.toml` is the tarball's `Cargo.toml.orig` with this diff and nothing else:

```diff
--- Cargo.toml.orig
+++ Cargo.toml
@@ -15,7 +15,7 @@ repository = "https://github.com/openrr/urdf-rs"
 
 # Note: serde is public dependency.
 [dependencies]
-quick-xml = { version = "0.39", features = ["overlapped-lists", "serialize"] }
+quick-xml = { version = "0.42", features = ["overlapped-lists", "serialize"] }
 regex = "1.4.2"
 RustyXML = "0.3.0"
 serde = { version = "1.0.118", features = ["derive"] }
```

Published 0.10.0 requires `quick-xml ^0.39`. That range cannot take 0.41.0, which is the floor for [RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194) and [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195). 0.42.0 is the current quick-xml release and is source-compatible with the calls in this crate (`se::Serializer`, `SeError`, `DeError`).

## Upstream status

No release accepts `quick-xml >= 0.42`. The newest commit on `main` is `f4a97ab207` (2026-09-25), "Measure code coverage and upload to Codecov", which is after the 0.10.0 tag and does not change `quick-xml`.

Open pull request: [openrr/urdf-rs#134](https://github.com/openrr/urdf-rs/pull/134), "Update quick-xml requirement from 0.39 to 0.41" (dependabot, opened against `main`, not merged, last updated 2026-09-28). Its patch is the same one-line bump, to `0.41` rather than `0.42`. Closed predecessor: [#132](https://github.com/openrr/urdf-rs/pull/132) (0.39 to 0.40).

## Exit

Delete `third_party/urdf-rs` and the `[patch.crates-io] urdf-rs` entry when a crates.io `urdf-rs` release depends on `quick-xml >= 0.42`. Until that release exists, this copy stays.
