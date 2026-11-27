"""Non-zero ``qfrc_applied`` dumps from libmujoco 3.13.0 contract-off.

The loaded ``libmujoco.so.3.13.0`` must hash to
``64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc``.
The PyPI 3.13.0 wheel (sha256 ``dc9599faeffba8e6ea9f53928de08301ee8c2ba780c4afceadeebe63573c3a1a``)
produced the same step bits on the hinge, on a sphere-floor contact, and on the
Panda case below, so the step comparison does not tell the two libraries apart.
This hash check is what refuses the wheel.

    python scripts/qfrc_applied_oracle.py <out dir> <repo root>

Writes ``hinge.xml``, ``hinge.txt``, and ``panda.txt``. Floats are raw
``to_bits`` hex.
"""
import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np

PIN = "64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc"
HINGE = """<mujoco>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body>
      <joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5" damping="0.1"/>
      <geom type="capsule" size="0.02 0.1" mass="0.4"/>
      <body pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0" limited="true" range="-1 1" damping="0.05"/>
        <geom type="capsule" size="0.015 0.08" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def bits(xs) -> str:
    view = np.ascontiguousarray(xs, dtype=np.float64).view(np.uint64)
    return " ".join(f"{int(v):016x}" for v in view.ravel())


def require_pin() -> None:
    so = Path(mujoco.__file__).resolve().parent / "libmujoco.so.3.13.0"
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    if digest != PIN:
        raise SystemExit(f"libmujoco sha256 {digest} is not {PIN} ({so})")
    print(f"libmujoco sha256 {digest}")


def dump(model: mujoco.MjModel, data: mujoco.MjData, steps: int, path: Path) -> None:
    if int(np.count_nonzero(data.qfrc_applied)) == 0:
        raise SystemExit(f"qfrc_applied is zero for {path}")
    lines = [
        "qfrc\t" + bits(data.qfrc_applied),
        "qpos\t" + bits(data.qpos),
        "qvel\t" + bits(data.qvel),
        "act\t" + bits(data.act),
    ]
    for k in range(1, steps + 1):
        mujoco.mj_step(model, data)
        lines.append(f"s\t{k}")
        lines.append("qpos\t" + bits(data.qpos))
        lines.append("qvel\t" + bits(data.qvel))
        lines.append("act\t" + bits(data.act))
        lines.append("warm\t" + bits(data.qacc_warmstart))
        lines.append("qfrc_constraint\t" + bits(data.qfrc_constraint))
        ncon = int(data.ncon)
        print(f"{path.name} step {k} ncon {ncon}")
        if path.name.startswith("panda") and ncon == 0:
            raise SystemExit(f"panda step {k} has no contact")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    out = Path(sys.argv[1])
    root = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    require_pin()

    (out / "hinge.xml").write_text(HINGE)
    hinge = mujoco.MjModel.from_xml_string(HINGE)
    hd = mujoco.MjData(hinge)
    hd.qpos[:] = [0.3, -0.2]
    hd.qvel[:] = [0.05, -0.02]
    hd.qfrc_applied[:] = [1.25, -0.4]
    dump(hinge, hd, 4, out / "hinge.txt")

    panda_xml = root / "crates/ferromotion-core/tests/fixtures/franka_emika_panda/contact_reach.xml"
    panda = mujoco.MjModel.from_xml_path(str(panda_xml))
    pd = mujoco.MjData(panda)
    pd.qpos[:] = [0.0, 0.8, 0.0, -1.9600239548543863, 0.0, 2.2, 0.0]
    pd.qvel[:] = 0.0
    pd.qfrc_applied[:] = [0.2, -0.15, 0.1, -0.05, 0.02, -0.01, 0.03]
    dump(panda, pd, 4, out / "panda.txt")
    print(f"wrote hinge and panda dumps under {out}")


if __name__ == "__main__":
    main()
