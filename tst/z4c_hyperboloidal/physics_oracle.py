#!/usr/bin/env python3
"""Build/run the new point math against the read-only N12 implementation.

The old bulk operator body is extracted at test time, not rewritten as a second
copy of the new implementation. Only its storage/jet construction is adapted.
Requires an existing native AthenaK Serial CMake build (for Kokkos/config.hpp).
"""
import argparse
import pathlib
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--legacy", type=pathlib.Path, required=True)
parser.add_argument("--build", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
legacy, build, output = args.legacy.resolve(), args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
source = (legacy / "src/z4c/hyperboloidal/hyperbolic_composite_device.hpp").read_text()
start = source.index("    KOKKOS_INLINE_FUNCTION void operator()(int local) const {")
end = source.index("\n  };", start)
operator = source[start:end]
(output / "physics_legacy_operator.hpp").write_text(operator + "\n")
start = source.index("    KOKKOS_INLINE_FUNCTION int Offset(int axis, int amount) const {")
end = source.index("    KOKKOS_INLINE_FUNCTION Real PolicyPhysicalValue", start)
(output / "physics_legacy_sampler.hpp").write_text(source[start:end] + "\n")
flags = (build / "src/CMakeFiles/athena.dir/flags.make").read_text()
includes = re.search(r"^CXX_INCLUDES = (.*)$", flags, re.M).group(1)
options = ["BROAD_SMOOTH_A", "CONFIGURED_FORMULATION", "DIRECT_OUTER",
           "INNER_TELEGRAPHER", "NORMALIZED_INNER_SHIFT", "RESTORE_INNER_DAMPING",
           "SHORT_INNER_GAUGE", "SMOOTH_GAUGE_BLEND", "SPHERICAL_POLICY"]
defines = " ".join("-DHYPERBOLIC_" + option + "=1" for option in options)
libraries = " ".join(str(build / ("kokkos/" + sub + "/src/libkokkos" + name + ".a"))
                     for sub, name in [("containers", "containers"), ("algorithms", "algorithms"),
                                       ("core", "core"), ("simd", "simd")])
src = root / "tst/z4c_hyperboloidal/physics_oracle.cpp"
exe = output / "physics_oracle"
# Paths here are supplied by the developer and written as quoted Make variables.
makefile = output / "physics_oracle.mk"
makefile.write_text(
    "all:\n\t/usr/bin/c++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra "
    + defines + " -DHYPERBOLIC_EXPERIMENTAL_TIMESTEP=0 " + includes
    + f' -I"{legacy / "src"}" -I"{output}" "{src}" {libraries} -o "{exe}"\n')
subprocess.run(["make", "-j64", "-f", str(makefile)], cwd=output, check=True)
subprocess.run([str(exe)], cwd=output, check=True)
