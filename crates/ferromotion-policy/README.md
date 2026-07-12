# ferromotion-policy

[![crates.io](https://img.shields.io/crates/v/ferromotion-policy.svg)](https://crates.io/crates/ferromotion-policy)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-policy)](https://docs.rs/ferromotion-policy)

On-device runner for exported **learned policies** (RL / VLA), in pure Rust (native + `wasm32`).

The interop tier of [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion): you *run*
trained weights, you don't rewrite training. A small MLP inference engine with the pieces real policies
ship with — observation normalization, tanh-squash, action scaling — plus a JSON checkpoint loader. Small
enough to run in the browser; the same on-device path as an exported RL/VLA policy. (Large transformer
VLAs remain an ONNX / `candle` concern.)

Dual-licensed MIT OR Apache-2.0.
