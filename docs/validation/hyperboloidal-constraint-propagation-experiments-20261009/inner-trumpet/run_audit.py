"""Ignored scratch audit: exact local slices; no athena build or BH evolution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
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
critical = sorted((root/"src/z4c/hyperboloidal").glob("*.hpp"))
critical += [root/"tst/hyperboloidal/kernel_symbol.cpp",
             root/"tst/hyperboloidal/check_kernel_symbol.py"]
scratch = [work/p for p in ["local_trumpet.hpp", "kernel_inner.cpp", "proof.py",
           "oracle.py", "principal_snapshot.cpp", "principal_inner.cpp", "run_audit.py"]]
def hashes(paths):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}
frozen_indices = [root/"build-layer-research/detached-wormhole"/name/"frozen-index.json"
                  for name in ["gauge-gate", "spatialnorm-gate", "general-spatialnorm-gate"]]
frozen = {}
for index in frozen_indices:
    frozen.update(json.loads(index.read_text())["files"])
def verify_frozen():
    return all(hashlib.sha256((root/p).read_bytes()).hexdigest() == v["sha256"]
               for p, v in frozen.items())
assert verify_frozen()
original = (root/"tst/hyperboloidal/kernel_symbol.cpp").read_text()
adapted = original.replace("bool physical_lapse, double matrix[20][20]) {",
    "bool physical_lapse, double matrix[20][20], double eta_inner=0) {")
adapted = adapted.replace("gauge.physical_trace_lapse = physical_lapse;",
    "gauge.physical_trace_lapse = physical_lapse;\n  gauge.shift_inner = eta_inner;")
adapted = adapted[:-3]+"\n  return 0;\n}\n"
assert adapted == (work/"principal_snapshot.cpp").read_text()
before = hashes(critical+scratch)
results = []
start_all = time.monotonic()
def save():
    after = hashes(critical+scratch)
    receipt = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=root, text=True).strip(), "seconds": time.monotonic()-start_all,
        "source_sha256_before": before, "source_sha256_after": after,
        "sources_unchanged": before == after,
        "prior_69_34_27_frozen_files_unchanged": verify_frozen(),
        "frozen_indices": hashes(frozen_indices), "results": results,
        "native_evolution_run": False, "sanitizers": "address,undefined; leak disabled on macOS"}
    (work/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
def run(label, command):
    start = time.monotonic()
    result = subprocess.run(command, cwd=root,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0", "UBSAN_OPTIONS": "halt_on_error=1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (work/f"{label}.log").write_text(result.stdout)
    (work/f"{label}.stderr").write_text(result.stderr)
    results.append({"label": label, "seconds": time.monotonic()-start,
        "returncode": result.returncode, "command": command,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "output": result.stdout[:1000], "stderr": result.stderr})
    print(label, "PASS" if not result.returncode else "FAIL",
          result.stdout.strip()[:150], result.stderr.strip()[:1000], flush=True)
    if result.returncode:
        save()
        raise RuntimeError(label)

for mode, flags in [("release", ["-O3", "-DNDEBUG"]),
                   ("debug", ["-O0", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])]:
    for target in ["kernel_inner", "principal_inner"]:
        binary = work/f"{target}-{mode}"
        run(f"build-{target}-{mode}", common+flags+[str(work/f"{target}.cpp"), "-o", str(binary)]+libs)
        run(f"test-{target}-{mode}", [str(binary)])
        results[-1]["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    run(f"basis-{mode}", [sys.executable, str(root/"tst/hyperboloidal/check_kernel_symbol.py"),
                         str(work/f"principal_inner-{mode}")])
for script in ["proof", "oracle"]:
    run(script, [sys.executable, str(work/f"{script}.py")])
save()
print("Saved receipt", work/"receipt.json", flush=True)
