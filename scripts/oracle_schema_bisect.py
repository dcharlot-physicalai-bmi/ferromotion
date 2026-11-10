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
    p = run(["git", "-C", FULL, "rev-list", "--reverse", f"{old}..{new}", "--", "src", "include", "plugin"])
    commits = [c for c in p.stdout.split() if c]
    # The old tag is the baseline even if it is not in the range.
    base = run(["git", "-C", FULL, "rev-parse", old]).stdout.strip()
    return [base] + commits


def checkout(commit):
    run(["git", "-C", FULL, "checkout", "-f", "--detach", commit])


def build(commit):
    checkout(commit)
    build_dir = os.path.join(WORK, "build")
    if not os.path.exists(os.path.join(build_dir, "build.ninja")) and not os.path.exists(os.path.join(build_dir, "Makefile")):
        raise SystemExit(f"missing configured build at {build_dir}")
    libdir = os.path.join(build_dir, "lib")
    if os.path.isdir(libdir):
        for name in os.listdir(libdir):
            if name.startswith("libmujoco.so"):
                os.remove(os.path.join(libdir, name))
    targets = ["mujoco"]
    for name, path in (
        ("actuator", "plugin/actuator/CMakeLists.txt"),
        ("elasticity", "plugin/elasticity/CMakeLists.txt"),
        ("sensor", "plugin/sensor/CMakeLists.txt"),
        ("sdf_plugin", "plugin/sdf/CMakeLists.txt"),
    ):
        if os.path.exists(os.path.join(FULL, path)):
            targets.append(name)
    p = run(["cmake", "--build", build_dir, "--target", *targets, "-j", os.environ.get("JOBS", "4")], check=False)
    if p.returncode != 0:
        import shutil
        shutil.rmtree(os.path.join(build_dir, "_deps"), ignore_errors=True)
        cache = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.exists(cache):
            os.remove(cache)
        # A CMake change needs a reconfigure. Keep the same prefix and flags.
        flags = "-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx"
        env = os.environ.copy()
        env["LDFLAGS"] = "-fuse-ld=gold"
        env["LIBRARY_PATH"] = "/usr/lib/gcc/x86_64-linux-gnu/14" + (
            ":" + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else ""
        )
        cfg = [
            "cmake", "-S", FULL, "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=/usr/lib/llvm-18/bin/clang",
            "-DCMAKE_CXX_COMPILER=/usr/lib/llvm-18/bin/clang++",
            f"-DCMAKE_C_FLAGS_RELEASE={flags}",
            f"-DCMAKE_CXX_FLAGS_RELEASE={flags}",
            f"-DCMAKE_C_FLAGS_DEBUG={flags}",
            f"-DCMAKE_CXX_FLAGS_DEBUG={flags}",
            "-DMUJOCO_BUILD_EXAMPLES=OFF",
            "-DMUJOCO_BUILD_SIMULATE=OFF",
            "-DMUJOCO_BUILD_TESTS=OFF",
            "-DMUJOCO_ENABLE_AVX=OFF",
            "-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF",
            "-DMUJOCO_ENABLE_LTO=OFF",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
        ]
        c = subprocess.run(cfg, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if c.returncode != 0:
            sys.stderr.write(c.stdout[-4000:])
            raise SystemExit(f"reconfigure failed at {commit}")
        p = run(["cmake", "--build", build_dir, "--target", *targets, "-j", os.environ.get("JOBS", "4")])
    lib = None
    for dirpath, _, files in os.walk(libdir):
        for name in files:
            if name.startswith("libmujoco.so.") and not os.path.islink(os.path.join(dirpath, name)):
                lib = os.path.join(dirpath, name)
    if lib is None:
        raise SystemExit(f"no libmujoco at {commit}")
    plug = os.path.join(build_dir, "lib", "mujoco_plugin")
    if not os.path.isdir(plug):
        plug = os.path.join(build_dir, "lib")
    return lib, plug


def dump_signature(commit, schedule, fields):
    lib, plug = build(commit)
    out = f"/tmp/bisect-{commit[:12]}.txt"
    exe = f"/tmp/bisect-dump-{commit[:12]}"
    inc = os.path.join(FULL, "include")
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
