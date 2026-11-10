#!/usr/bin/env python3
"""Clean per-commit contract-off build.

The incremental tree failed when a later configure regenerated FetchContent
link scripts. Clang 18 has no llvm-ar on this image, so CMake recorded
CMAKE_C_COMPILER_AR-NOTFOUND and cmake_link_script tried to exec that name
while archiving liblodepng.a, libccd.a, libqhullstatic_r.a, and libtinyxml2.a.
The error string is "Error running link command: No such file or directory".

Each commit gets its own source worktree and build directory. The archiver
is /usr/bin/ar, which is what the 3.13.0 oracle's link.txt already invoked.
ccache covers the compiler only.
"""
import os
import subprocess
import sys

FULL = os.environ.get("MUJOCO_FULL", "/tmp/mujoco-full")
ROOT = os.environ.get("ORACLE_COMMIT_ROOT", "/tmp/oracle-commits")
FLAGS = "-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx"


def run(cmd, cwd=None, env=None):
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-6000:])
        raise SystemExit(f"failed {p.returncode}: {' '.join(cmd[:8])}")
    return p.stdout


def resolve(commit):
    return run(["git", "-C", FULL, "rev-parse", commit]).strip()


def worktree(commit):
    sha = resolve(commit)
    src = os.path.join(ROOT, "src", sha[:12])
    if not os.path.isdir(os.path.join(src, ".git")) and not os.path.isfile(os.path.join(src, ".git")):
        os.makedirs(os.path.dirname(src), exist_ok=True)
        run(["git", "-C", FULL, "worktree", "add", "--detach", src, sha])
    return sha, src


def build(commit):
    sha, src = worktree(commit)
    work = os.path.join(ROOT, "build", sha[:12])
    bld = os.path.join(work, "build")
    libdir = os.path.join(bld, "lib")
    marker = os.path.join(work, "built")
    if os.path.isfile(marker):
        lib = open(marker).read().strip()
        if os.path.isfile(lib):
            plug = os.path.join(libdir, "mujoco_plugin")
            if not os.path.isdir(plug):
                plug = libdir
            return sha, src, lib, plug
    os.makedirs(work, exist_ok=True)
    env = os.environ.copy()
    env["LDFLAGS"] = "-fuse-ld=gold"
    env["LIBRARY_PATH"] = "/usr/lib/gcc/x86_64-linux-gnu/14" + (
        ":" + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else ""
    )
    env.setdefault("CCACHE_DIR", "/tmp/ccache")
    cmake = [
        "cmake", "-S", src, "-B", bld,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=/usr/lib/llvm-18/bin/clang",
        "-DCMAKE_CXX_COMPILER=/usr/lib/llvm-18/bin/clang++",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_C_COMPILER_AR:FILEPATH=/usr/bin/ar",
        "-DCMAKE_CXX_COMPILER_AR:FILEPATH=/usr/bin/ar",
        "-DCMAKE_C_COMPILER_RANLIB:FILEPATH=/usr/bin/ranlib",
        "-DCMAKE_CXX_COMPILER_RANLIB:FILEPATH=/usr/bin/ranlib",
        f"-DCMAKE_C_FLAGS_RELEASE={FLAGS}",
        f"-DCMAKE_CXX_FLAGS_RELEASE={FLAGS}",
        f"-DCMAKE_C_FLAGS_DEBUG={FLAGS}",
        f"-DCMAKE_CXX_FLAGS_DEBUG={FLAGS}",
        "-DMUJOCO_BUILD_EXAMPLES=OFF",
        "-DMUJOCO_BUILD_SIMULATE=OFF",
        "-DMUJOCO_BUILD_TESTS=OFF",
        "-DMUJOCO_ENABLE_AVX=OFF",
        "-DMUJOCO_ENABLE_AVX_INTRINSICS=OFF",
        "-DMUJOCO_ENABLE_LTO=OFF",
        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
    ]
    run(cmake, env=env)
    targets = ["mujoco"]
    for name, path in (
        ("actuator", "plugin/actuator/CMakeLists.txt"),
        ("elasticity", "plugin/elasticity/CMakeLists.txt"),
        ("sensor", "plugin/sensor/CMakeLists.txt"),
        ("sdf_plugin", "plugin/sdf/CMakeLists.txt"),
    ):
        if os.path.isfile(os.path.join(src, path)):
            targets.append(name)
    run(["cmake", "--build", bld, "--target", *targets, "-j", os.environ.get("JOBS", "4")], env=env)
    lib = None
    for dirpath, _, files in os.walk(libdir):
        for name in files:
            path = os.path.join(dirpath, name)
            if name.startswith("libmujoco.so.") and not os.path.islink(path):
                lib = path
    if lib is None:
        raise SystemExit(f"no libmujoco for {sha}")
    # A poisoned link script must not ship.
    for dirpath, _, files in os.walk(bld):
        if "link.txt" in files:
            text = open(os.path.join(dirpath, "link.txt")).read()
            if "NOTFOUND" in text:
                raise SystemExit(f"archiver NOTFOUND in {dirpath}/link.txt")
    plug = os.path.join(libdir, "mujoco_plugin")
    if not os.path.isdir(plug):
        plug = libdir
    open(marker, "w").write(lib)
    print(f"built {sha[:12]} {lib}", file=sys.stderr, flush=True)
    return sha, src, lib, plug


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: oracle_commit_build.py <commit>")
    sha, src, lib, plug = build(sys.argv[1])
    print(sha)
    print(src)
    print(lib)
    print(plug)
