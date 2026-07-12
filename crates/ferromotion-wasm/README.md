# ferromotion-wasm

[![crates.io](https://img.shields.io/crates/v/ferromotion-wasm.svg)](https://crates.io/crates/ferromotion-wasm)

WebAssembly bindings for [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion) — robot
kinematics, IK, motion retargeting, and trajectory planning **in the browser**, zero install.

Build with [`wasm-pack`](https://github.com/rustwasm/wasm-pack):

```bash
wasm-pack build crates/ferromotion-wasm --target web
```

Then from JavaScript, build a chain by hand or load a URDF, and solve:

```js
const chain = Chain.from_urdf(urdfText, "base_link", "tool");
const q = chain.solve_ik([0.4, 0.1, 0.3, 1, 0, 0, 0], seed);   // pose = [x,y,z, qw,qx,qy,qz]
chain.retarget_step(frames, offsets, targets, prev, 0.01, 0.0); // live teleop/mocap loop
chain.plan_reach(seed, goal, [ox, oy, oz, radius], 44);          // obstacle-avoiding trajectory
```

See `demo/` in the repository for a self-contained page driving an arm with live IK and an
obstacle-avoiding planned trajectory — computed entirely on-device. Dual-licensed MIT OR Apache-2.0.
