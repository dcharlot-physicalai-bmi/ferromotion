#!/usr/bin/env python3
"""States for the versioned oracle, from a props file written by oracle_schema_dump.

The draws are the ones in scripts/mujoco_native_oracle.py (seed 20260927, three states)
and scripts/mujoco_native_step_oracle.py (seed 20260928, one start, 20 steps).
Props contain only models the 3.13.0 compiler loaded, in the same sorted order, so a
model the compiler rejects does not consume the generator.
"""
import sys

import numpy as np

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3


def load_props(path):
    models = []
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        key, _, rest = line.partition("\t")
        if key == "model":
            cur = {"rel": rest, "joints": [], "acts": []}
            models.append(cur)
        elif key in ("nq", "nv", "nu", "na", "njnt"):
            cur[key] = int(rest)
        elif key == "qpos0":
            cur["qpos0"] = rest.split() if rest else []
        elif key == "jnt":
            typ, adr, limited, lo, hi = rest.split()
            cur["joints"].append((int(typ), int(adr), int(limited), lo, hi))
        elif key == "actlim":
            limited, lo, hi = rest.split()
            cur["acts"].append((int(limited), lo, hi))
        else:
            raise SystemExit(f"unknown props line {key}")
    return models


def fhex(xs):
    return " ".join(float(x).hex() for x in np.asarray(xs, dtype=np.float64).ravel())


def fromhex_row(words):
    if not words:
        return np.zeros(0, dtype=np.float64)
    return np.array([float.fromhex(w) for w in words], dtype=np.float64)


def native_states(models):
    rng = np.random.default_rng(20260927)
    lines = []
    for model in models:
        q0 = fromhex_row(model["qpos0"])
        lines.append(f"model\t{model['rel']}")
        for s in range(3):
            q = q0.copy()
            v = np.zeros(model["nv"], dtype=np.float64)
            if s > 0:
                for typ, adr, limited, lo_s, hi_s in model["joints"]:
                    if typ == FREE:
                        q[adr : adr + 3] = rng.uniform(-1, 1, 3)
                        q[adr + 3 : adr + 7] = rng.normal(size=4)
                    elif typ == BALL:
                        q[adr : adr + 4] = rng.normal(size=4)
                    else:
                        if limited:
                            lo, hi = float.fromhex(lo_s), float.fromhex(hi_s)
                        else:
                            lo, hi = (-np.pi, np.pi) if typ == HINGE else (-0.5, 0.5)
                        q[adr] = rng.uniform(lo, hi)
                v = rng.uniform(-1, 1, model["nv"])
            u = np.zeros(model["nu"], dtype=np.float64)
            a = np.zeros(model["na"], dtype=np.float64)
            if s > 0:
                for i, (limited, lo_s, hi_s) in enumerate(model["acts"]):
                    if limited:
                        lo, hi = float.fromhex(lo_s), float.fromhex(hi_s)
                    else:
                        lo, hi = -1.0, 1.0
                    span = hi - lo
                    u[i] = rng.uniform(lo - 0.1 * span, hi + 0.1 * span)
                a = rng.uniform(0, 1, model["na"])
            lines.append("qpos\t" + fhex(q))
            lines.append("qvel\t" + fhex(v))
            lines.append("ctrl\t" + fhex(u))
            lines.append("act\t" + fhex(a))
    return lines


def step_states(models):
    rng = np.random.default_rng(20260928)
    lines = []
    for model in models:
        q = fromhex_row(model["qpos0"])
        for typ, adr, limited, lo_s, hi_s in model["joints"]:
            if typ not in (HINGE, SLIDE):
                continue
            if limited:
                lo, hi = float.fromhex(lo_s), float.fromhex(hi_s)
            else:
                lo, hi = -0.5, 0.5
            q[adr] = rng.uniform(lo, hi)
        v = rng.uniform(-0.5, 0.5, model["nv"])
        u = np.zeros(model["nu"], dtype=np.float64)
        for i, (limited, lo_s, hi_s) in enumerate(model["acts"]):
            if limited:
                lo, hi = float.fromhex(lo_s), float.fromhex(hi_s)
            else:
                lo, hi = -1.0, 1.0
            u[i] = rng.uniform(lo, hi)
        a = rng.uniform(0.0, 1.0, model["na"])
        lines.append(f"model\t{model['rel']}")
        lines.append("qpos\t" + fhex(q))
        lines.append("qvel\t" + fhex(v))
        lines.append("act\t" + fhex(a))
        lines.append("ctrl\t" + fhex(u))
    return lines


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in ("native", "step"):
        raise SystemExit("usage: oracle_schema_schedule.py <props> native|step <out>")
    models = load_props(sys.argv[1])
    lines = native_states(models) if sys.argv[2] == "native" else step_states(models)
    with open(sys.argv[3], "w") as f:
        f.write("\n".join(lines) + "\n")
    print(len(models), sys.argv[2], "->", sys.argv[3])


if __name__ == "__main__":
    main()
