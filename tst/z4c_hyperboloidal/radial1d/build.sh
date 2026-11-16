#!/bin/sh
# Assessment-only build of the 1D radial driver. Sources come from THIS
# worktree; Kokkos headers, config.hpp and the Kokkos static library are read
# (never written) from an existing local build tree given as $1.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
kokkos_tree=${1:?usage: build.sh EXISTING_REPO_WITH_build-binary [OUTPUT] [SOURCE]}
out=${2:-$here/radial_a}
source_file=${3:-$here/radial_a.cpp}
b=$kokkos_tree/build-binary
/usr/bin/c++ -ffp-contract=off -O3 -DNDEBUG -std=c++17 -arch arm64 -DKOKKOS_DEPENDENCE \
  -I"$root/src" -I"$b" -I"$b/kokkos" -I"$b/kokkos/core/src" -I"$kokkos_tree/kokkos/core/src" \
  -I"$b/kokkos/containers/src" -I"$kokkos_tree/kokkos/containers/src" \
  -I"$b/kokkos/algorithms/src" -I"$kokkos_tree/kokkos/algorithms/src" \
  -I"$b/kokkos/simd/src" -I"$kokkos_tree/kokkos/simd/src" \
  -isystem "$kokkos_tree/kokkos/tpls/desul/include" -isystem "$kokkos_tree/kokkos/tpls/mdspan/include" \
  -isystem /opt/homebrew/Cellar/open-mpi/5.0.10/include \
  "$source_file" -o "$out" \
  "$b/kokkos/containers/src/libkokkoscontainers.a" "$b/kokkos/core/src/libkokkoscore.a" \
  /opt/homebrew/Cellar/open-mpi/5.0.10/lib/libmpi.dylib
