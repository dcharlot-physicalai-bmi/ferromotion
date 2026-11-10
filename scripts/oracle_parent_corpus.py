#!/usr/bin/env python3
"""Full-corpus native and step dumps of a commit and its parent, then the diff.

Skips dumps that already exist. Build progress goes to stderr.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_commit_build import build, resolve

MENAGERIE = os.environ.get("MENAGERIE", "/tmp/mujoco_menagerie")
NATIVE_SCHED = os.environ.get("NATIVE_SCHED", "/tmp/oracle-dumps/3.13.0/native-schedule.txt")
STEP_SCHED = os.environ.get("STEP_SCHED", "/tmp/oracle-dumps/3.13.0/step-schedule.txt")
DUMP_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_schema_dump.c")
OUT = os.environ.get("CAND_OUT", "/tmp/cand-dumps")
DIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_schema_diff.py")


def dump(commit):
    sha, src, lib, plug = build(commit)
    short = sha[:12]
    out = os.path.join(OUT, short)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "sha"), "w").write(sha + "\n")
    exe = os.path.join(OUT, f"dump-{short}")
    if not os.path.isfile(exe):
        subprocess.run(
            ["cc", "-O2", f"-I{os.path.join(src, 'include')}", DUMP_SRC, lib, "-o", exe, f"-Wl,-rpath,{os.path.dirname(lib)}", "-lm"],
            check=True,
        )
    native = os.path.join(out, "native.txt")
    step = os.path.join(out, "step.txt")
    if not os.path.isfile(native) or os.path.getsize(native) < 1000:
        print(f"DUMP native {short}", flush=True)
        subprocess.run([exe, "native", MENAGERIE, plug, NATIVE_SCHED, native], check=True)
    if not os.path.isfile(step) or os.path.getsize(step) < 1000:
        print(f"DUMP step {short}", flush=True)
        subprocess.run([exe, "step", MENAGERIE, plug, STEP_SCHED, step], check=True)
    return short


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: oracle_parent_corpus.py <commit>...")
    for spec in sys.argv[1:]:
        sha = resolve(spec)
        parent = resolve(sha + "^")
        print(f"PAIR {sha[:12]} parent {parent[:12]}", flush=True)
        pid = dump(parent)
        cid = dump(sha)
        for kind in ("native", "step"):
            dest = os.path.join(OUT, f"{cid}-vs-parent-{kind}.txt")
            with open(dest, "w") as f:
                subprocess.run(
                    ["python3", DIFF, os.path.join(OUT, pid, f"{kind}.txt"), os.path.join(OUT, cid, f"{kind}.txt")],
                    check=True,
                    stdout=f,
                )
            print(f"=== {cid} {kind} ===", flush=True)
            print(open(dest).read(), flush=True)


if __name__ == "__main__":
    main()
