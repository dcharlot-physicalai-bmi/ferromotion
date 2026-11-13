#!/usr/bin/env python3
"""Fail unless the 3.13.0 contract-off profile counts are unchanged.

Measured against libmujoco
64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc.
The pin at 575be4e was qacc 148/636, qacc_smooth 575/636, efc_J 382/636,
and 6/212 models bit-exact over 20 steps. The fidelity transcriptions
merged from d14e65f moved those four numbers. M, qLD, xpos, and
qfrc_bias stayed 636/636. The 4.0 gates (d9972b33 and 6f3db612) do not
move any of them: the pre-mechanism binary and the gated binary produce
the same tallies, including 165/212 over 20 steps.
"""
import re
import sys

NATIVE = {
    "M": (636, 636),
    "qLD": (636, 636),
    "xpos": (636, 636),
    "qfrc_bias": (636, 636),
    "qacc": (529, 636),
    "qacc_smooth": (609, 636),
    "efc_J": (572, 636),
}


def native_counts(text):
    if "212 models compared (0 refused)" not in text:
        raise SystemExit("native profile did not compare 212 models with 0 refusals")
    found = {}
    for line in text.splitlines():
        m = re.search(r"mjData\s+(\S+)\s+(\d+) of (\d+)", line)
        if m:
            found[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    missing = [name for name in NATIVE if name not in found]
    if missing:
        raise SystemExit(f"native profile is missing {missing}")
    bad = {name: found[name] for name, want in NATIVE.items() if found[name] != want}
    if bad:
        raise SystemExit(f"native counts changed: {bad}")


def step_count(text):
    m = re.search(
        r"BIT FOR BIT at every one of (\d+) steps: (\d+) of (\d+) models stepped \((\d+) the loader refuses\)",
        text,
    )
    if not m:
        raise SystemExit("step profile summary line is missing")
    steps, exact, compared, refused = (int(m.group(i)) for i in range(1, 5))
    if (steps, exact, compared, refused) != (20, 165, 212, 0):
        raise SystemExit(
            f"step count changed: {exact} of {compared} over {steps} steps, refused {refused}"
        )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: oracle_313_profile_check.py <native.txt> <step.txt>")
    native_counts(open(sys.argv[1]).read())
    step_count(open(sys.argv[2]).read())
    print("3.13.0 contract-off profile counts unchanged: 20-step 165/212; M qLD xpos qfrc_bias 636/636; qacc 529/636; qacc_smooth 609/636; efc_J 572/636")


if __name__ == "__main__":
    main()
