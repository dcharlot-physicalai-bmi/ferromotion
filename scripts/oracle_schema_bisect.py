#!/usr/bin/env python3
"""Bisect Menagerie field changes between two MuJoCo tags.

The search commits are those that touch src/, include/, or plugin/. A commit
that does not change those trees cannot change the compiled library. The build
directory is reused. The dumper is relinked against that library.

Usage:
  oracle_schema_bisect.py <old_tag> <new_tag> <schedule> <fields-file> <out.json>

fields-file: one "m body_inertia" or "d qacc" per line. State 0 for d fields.
The schedule is a native schedule containing only the sentinel models.
"""
import json
import os
import subprocess
import sys

FULL = os.environ.get("MUJOCO_FULL", "/tmp/mujoco-full")
WORK = os.environ.get("MUJOCO_BISECT_WORK", "/tmp/oracle-bisect")
MENAGERIE = os.environ.get("MENAGERIE", "/tmp/mujoco_menagerie")
DUMP_SRC = os.environ.get("DUMP_SRC", "/agent/repos/ferromotion/scripts/oracle_schema_dump.c")


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:])
        raise SystemExit(f"command failed {p.returncode}: {' '.join(cmd[:6])}")
    return p


def src_commits(old, new):
    # First-parent only. A full rev-list includes side-branch trees from months
    # earlier (one between 3.14 and 3.15 still has no m->nout), which the
    # dumper cannot compile. The mainline tree at each first-parent commit is
    # the state that reached the next tag.
    p = run(["git", "-C", FULL, "rev-list", "--first-parent", "--reverse", f"{old}..{new}", "--", "src", "include", "plugin"])
    commits = [c for c in p.stdout.split() if c]
    # The old tag is the baseline even if it is not in the range.
    base = run(["git", "-C", FULL, "rev-parse", old]).stdout.strip()
    return [base] + commits


def checkout(commit):
    run(["git", "-C", FULL, "checkout", "-f", "--detach", commit])


def build(commit):
    # Clean tree per commit. The shared incremental tree archived lodepng with
    # CMAKE_C_COMPILER_AR-NOTFOUND. See oracle_commit_build.py.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from oracle_commit_build import build as clean_build
    _sha, src, lib, plug = clean_build(commit)
    return src, lib, plug


def dump_signature(commit, schedule, fields):
    src, lib, plug = build(commit)
    out = f"/tmp/bisect-{commit[:12]}.txt"
    exe = f"/tmp/bisect-dump-{commit[:12]}"
    inc = os.path.join(src, "include")
    subprocess.run(
        ["cc", "-O2", f"-I{inc}", DUMP_SRC, lib, "-o", exe, f"-Wl,-rpath,{os.path.dirname(lib)}", "-lm"],
        check=True,
    )
    subprocess.run([exe, "native", MENAGERIE, plug, schedule, out], check=True)
    want = set(fields)
    sig = {}
    cur = None
    state = -1
    seen = 0
    for line in open(out):
        line = line.rstrip("\n")
        if not line:
            continue
        p = line.split("\t")
        if p[0] == "model":
            cur = p[1]
            seen = 0
            state = -1
            sig.setdefault(cur, {})
            continue
        if p[0] == "qpos":
            seen += 1
            state = seen - 1
            continue
        if p[0] == "load_error":
            sig[cur]["load_error"] = p[1] if len(p) > 1 else ""
            continue
        if p[0] == "m":
            key = f"m {p[1]}"
            if key in want:
                sig[cur][key] = p[2] if len(p) > 2 else ""
        elif p[0] == "d" and state in (0, 1):
            key = f"d {p[1]}@{state}"
            if key in want:
                sig[cur][key] = p[2] if len(p) > 2 else ""
    return sig


def dump_release_fields(path, fields):
    want = set(fields)
    sig = {}
    cur = None
    state = -1
    seen = 0
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        p = line.split("\t")
        if p[0] == "model":
            cur = p[1]
            seen = 0
            state = -1
            sig[cur] = {}
            continue
        if cur is None:
            continue
        if p[0] == "qpos":
            seen += 1
            state = seen - 1
            continue
        if p[0] == "m":
            key = f"m {p[1]}"
            if key in want:
                sig[cur][key] = p[2] if len(p) > 2 else ""
        elif p[0] == "d" and state in (0, 1):
            key = f"d {p[1]}@{state}"
            if key in want:
                sig[cur][key] = p[2] if len(p) > 2 else ""
    return sig


def changed_fields(a, b, fields):
    out = []
    models = sorted(set(a) | set(b))
    for field in fields:
        rels = [m for m in models if a.get(m, {}).get(field) != b.get(m, {}).get(field)]
        if rels:
            out.append({"field": field, "models": rels})
    return out


def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: oracle_schema_bisect.py <old> <new> <schedule> <fields> <out.json>")
    old, new, schedule, fields_path, out_path = sys.argv[1:]
    fields = [ln.strip() for ln in open(fields_path) if ln.strip() and not ln.startswith("#")]
    commits = src_commits(old, new)
    print(f"commits {len(commits)} fields {len(fields)}", flush=True)
    cache = {}

    def sig(i):
        if i not in cache:
            print(f"build {i} {commits[i][:12]}", flush=True)
            cache[i] = dump_signature(commits[i], schedule, fields)
        return cache[i]

    # Endpoints first so a failed build is obvious.
    sig(0)
    sig(len(commits) - 1)
    # The new-tag library must reproduce the release dump on the sentinels.
    # A stale object would make the search attribute a field to the wrong commit.
    release = f"/tmp/oracle-dumps/{new}/native.txt"
    if os.path.exists(release):
        rel = dump_release_fields(release, fields)
        got = sig(len(commits) - 1)
        common = {m: rel[m] for m in got if m in rel}
        bad = changed_fields(common, got, fields)
        if bad:
            raise SystemExit(f"endpoint {new} does not match the release dump: {bad}")
        print(f"endpoint {new} matches the release dump", flush=True)
    found = []

    def walk(lo, hi):
        delta = changed_fields(sig(lo), sig(hi), fields)
        if not delta:
            return
        if hi == lo + 1:
            found.append({
                "commit": commits[hi],
                "parent": commits[lo],
                "fields": delta,
            })
            print(f"change {commits[hi][:12]} {[d['field'] for d in delta]}", flush=True)
            return
        mid = (lo + hi) // 2
        walk(lo, mid)
        walk(mid, hi)

    walk(0, len(commits) - 1)
    doc = {"old": old, "new": new, "n_src_commits": len(commits), "changes": found}
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path} changes {len(found)}", flush=True)


if __name__ == "__main__":
    main()
