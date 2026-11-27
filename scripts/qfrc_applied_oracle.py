"""One step-chain with a non-zero ``qfrc_applied``, from libmujoco 3.13.0.

The process must be the contract-off build (sha256 ``64156abe…``). The oracle-profile
job points the MuJoCo wheel at that library before this runs. Floats are raw
``to_bits`` hex so the Rust side compares bits, not rounded decimals.

    python scripts/qfrc_applied_oracle.py <out.txt> <model.xml>
"""
import sys

import mujoco
import numpy as np

XML = """<mujoco>
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
APPLIED = np.array([1.25, -0.4], dtype=np.float64)
QPOS = np.array([0.3, -0.2], dtype=np.float64)
QVEL = np.array([0.05, -0.02], dtype=np.float64)
N = 4


def bits(xs) -> str:
    view = np.ascontiguousarray(xs, dtype=np.float64).view(np.uint64)
    return " ".join(f"{int(v):016x}" for v in view.ravel())


def main() -> None:
    out, xml_path = sys.argv[1], sys.argv[2]
    open(xml_path, "w").write(XML)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    if int(np.count_nonzero(APPLIED)) == 0:
        raise SystemExit("qfrc_applied fixture is zero")
    data.qpos[:] = QPOS
    data.qvel[:] = QVEL
    data.qfrc_applied[:] = APPLIED
    lines = [
        "qfrc\t" + bits(data.qfrc_applied),
        "qpos\t" + bits(data.qpos),
        "qvel\t" + bits(data.qvel),
        "act\t" + bits(data.act),
    ]
    for k in range(1, N + 1):
        mujoco.mj_step(model, data)
        lines.append(f"s\t{k}")
        lines.append("qpos\t" + bits(data.qpos))
        lines.append("qvel\t" + bits(data.qvel))
        lines.append("act\t" + bits(data.act))
        lines.append("warm\t" + bits(data.qacc_warmstart))
        lines.append("qfrc_constraint\t" + bits(data.qfrc_constraint))
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"qfrc_applied non-zero, {N} steps, nv={model.nv} -> {out}")


if __name__ == "__main__":
    main()
