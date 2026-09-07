# Intrinsic geometry and finite-radius state map

This is the first compiled foundation of `intrinsic_clean_v1`, with layout
version 1 and 50 components. It is not selected by the evolution parser and is
not accepted by any restart reader. The existing 55-field state remains intact.

Offsets follow the pinned candidate oracle: w=0, rho=1, chart=2, beta=7, K=10,
Ahat=11, Z=16, C=19, p=20, l=23, S=26, B=41. Chart order is (a,c,b,d,e);
Ahat order is (xx,xy,xz,yy,yz), with zz=-xx-yy. S is direction-major 3x5;
B is derivative-direction-major 3x3. The standalone compiled fixture asserts
all legacy offsets against the actual PcGh class.

`intrinsic_geometry.hpp` computes T, its explicit inverse, g, g inverse, Ahat,
A, Q=J*S, the full chart Jacobian J and its Hessian. There are no divisions by
w, rho or alpha. Exponentials of chart coordinates can overflow; finite stored
coordinates do not imply acceptable conditioning. This helper does not clip,
project, or enforce an evolution health policy.

`intrinsic_state_map.hpp` explicitly maps all 50 components into the old 55
slots, including L=2*l. The reverse uses a Cholesky factor and the inverse
chart tangent map. It accepts only positive w/rho, finite data, SPD g with
determinant one, and tangent A and Q within the declared tolerance. It rejects
invalid input without changing its output array. Accepting a nonzero floating
tolerance loses trace/determinant errors of that order; the mathematical map
is bijective only on the exact algebraic constraint submanifold. It does not
convert arbitrary off-algebraic legacy states or establish restart compatibility.
It does preserve independent spatial reduction errors within that submanifold.

The independent checker constructs explicit symbolic metric entries, then
obtains first and second derivatives through SymPy rather than the C++ dT/ddT
loops. It proves exact determinant and tangent trace identities and compares
all 351 geometry entries, 55 forward-map entries and 50 reverse-map entries.
Half the reverse inputs use independent random SPD eigensystems and symmetric
tensors made trace-free against their inverse metric. Their chart tangent
inverse is computed from differentiated scalar Cholesky entries, not the
C++ matrix formula. All 100 valid states carry arbitrary GH/curvature/gradient
components. Six invalid states test w/rho, SPD, determinant, A trace and Q trace.
The frozen normalized tolerance is 2e-12; CPU error is 8.14e-16.

Reproduction (use absolute paths for the two externally built dependencies):

```sh
cmake -S analysis/pc_gh_clean_reduction/compiled -B /path/to/new-map-build \
  -DKokkos_DIR=/path/to/athena-build/kokkos \
  -DATHENA_CONFIG_DIR=/path/to/athena-build
cmake --build /path/to/new-map-build -j2
/path/to/python analysis/pc_gh_clean_reduction/check_intrinsic_map.py \
  --binary /path/to/new-map-build/intrinsic_map --output /path/to/new-results
```

The first build failed because the supplied config directory incorrectly ended
in `/src`; config.hpp is in the Athena build root. Correcting that argument
resolved compilation. This build failure and both CPU result sets are retained.
The second set adds explicit compiled determinant/inverse/A-trace/Q-trace checks.
CUDA uses the same headers and checker with a source manifest, links the already
qualified Kokkos CUDA build, and separately records the legacy ABI header source.
No complete RHS, characteristic or evolution validation follows from this map.

`check_intrinsic_exact.py` separately proves the explicit two-sided triangular
and metric inverses and five-coordinate tangent inverse exactly. CUDA now passes
the same 106-case checker at 6.36e-16. NumPy/SymPy platform differences slightly
change generated input bytes; a separate cross-backend run on identical CUDA
input bytes agrees at 4.11e-16. All GPU work in this checkpoint is complete.
