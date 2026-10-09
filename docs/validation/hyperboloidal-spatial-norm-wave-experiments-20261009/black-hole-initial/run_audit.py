"""Scratch-only detached wormhole audit; never builds or runs native athena."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[3]
work = Path(__file__).resolve().parent
includes = ["src", "build-layer-release", "build-layer-release/kokkos"]
for part in ["core", "containers", "algorithms", "simd"]:
    includes.extend([f"build-layer-release/kokkos/{part}/src", f"kokkos/{part}/src"])
common = ["/usr/bin/c++", "-std=c++17", "-DKOKKOS_DEPENDENCE"]
common += [f"-I{x}" for x in includes]
for part in ["desul", "mdspan"]:
    common.extend(["-isystem", f"kokkos/tpls/{part}/include"])
libs = [f"build-layer-release/kokkos/{part}/src/libkokkos{part}.a"
        for part in ["containers", "algorithms", "core", "simd"]]
critical = [root / "src/z4c/hyperboloidal" / p for p in
            ["cartesian_patch.hpp", "conformal_rhs.hpp", "layer_reference.hpp",
             "layer_gauge.hpp", "reference_gauge.hpp", "tensor_geometry.hpp"]]
critical = [p for p in critical if p.exists()]
scratch = [work / p for p in ["preferred_snapshot.hpp", "frozen_kernel.cpp", "kernel_norm.cpp", "spatial_norm.hpp", "frozen_outer_oracle.py", "norm_leading.py", "norm_second.py", "norm_oracle.py", "run_audit.py"]] + [work.parent / p for p in ["detached_height.hpp", "detached_wormhole.hpp"]]
def hashes(paths):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}
frozen_index = root / "build-layer-research/detached-wormhole/gauge-gate/frozen-index.json"
frozen = json.loads(frozen_index.read_text())
def verify_frozen():
    return all(hashlib.sha256((root/p).read_bytes()).hexdigest() == v["sha256"] for p,v in frozen["files"].items())
assert verify_frozen()
before = hashes(critical + scratch)
results = []
start_all = time.monotonic()
def run(label, command):
    start = time.monotonic()
    # macOS LeakSanitizer is unsupported; address/undefined sanitizers stay on.
    result = subprocess.run(command, cwd=root,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "halt_on_error=1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (work / f"{label}.log").write_text(result.stdout)
    (work / f"{label}.stderr").write_text(result.stderr)
    results.append({"label": label, "seconds": time.monotonic()-start,
        "returncode": result.returncode, "command": command, "output": result.stdout[:20000], "stderr": result.stderr})
    print(label, "PASS" if not result.returncode else "FAIL", result.stdout.strip()[:1000], result.stderr.strip(), flush=True)
    if result.returncode:
        save()
        raise RuntimeError(label)

def save():
    after = hashes(critical + scratch)
    receipt = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "seconds": time.monotonic()-start_all, "source_sha256_before": before,
        "source_sha256_after": after, "sources_unchanged": before == after, "prior_69_files_unchanged": verify_frozen(),
        "results": results, "native_evolution_run": False,
        "sanitizers": "address,undefined; LeakSanitizer disabled on macOS"}
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")

for mode, flags in [("release", ["-O3", "-DNDEBUG"]),
                    ("debug", ["-O0", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])]:
    for target in ["kernel_norm"]:
        binary = work / f"{target}-{mode}"
        run(f"build-{target}-{mode}", common + flags + [str(work/f"{target}.cpp"), "-o", str(binary)] + libs)
        run(f"test-{target}-{mode}", [str(binary)])
        results[-1]["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
for proof in ["norm_leading", "norm_second", "norm_oracle"]:
    run(proof, [sys.executable, str(work/(proof+".py"))])
save()
print("Saved receipt", work/"receipt.json", flush=True)
