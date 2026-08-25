# Ref-GH gamma0/gamma2 subsidiary gates

This compact bundle records the local uniform-grid constraint-damping matrix
implemented at `8101cfbc`, with portable CSV output at `6ebac8cd`. The starting
documentation checkpoint was `b45e168c`; Kokkos is
`6739bc623081648af9e752b616d9671527922cbf`.

Current executable SHA-256 values:

- Kokkos Serial: `00e38ebe726bc60bb490a36db562195a93947c1a64ec573e078371b48bec2bcc`
- Kokkos OpenMP: `ca4c20d5af25655ee8808ce08454177985e86f67f95aedb97094fe1adc471cbe`

## Gamma0 transverse GH mode

The new `problem/perturbation=gh_transverse` mode plants the transverse
linearized constraint eigenmode described by Eq. (21) of Lindblom et al.
(2006). For wave number `k`, fourth-order modified wave number `k_h`, and
`omega_h^2=k_h^2-gamma0^2/4`, the nonzero perturbations are chosen so that

```text
C_y = A cos(k x)
C_y L2(t) = C_y L2(0) exp[-(gamma0/2 + lambda_KO)t].
```

The `32x8x8`, `t=0.2`, RK4 matrix uses `gamma0=0.25,0.5,1.0` and
`diss=0,0.02`. All six trajectories pass with maximum absolute growth-factor
error `5.35e-7`. The largest reduction norm is `7.19e-18`; the curl norm is
zero. Thus the test seeds a GH constraint without a material reduction/curl
seed.

## Gamma2 reduction and curl matrix

The deterministic random-state test uses an `8^3` periodic grid, RK4,
`t=0.2`, `gamma2=0,0.5,1.0`, and `diss=0,0.02`. Against the independently
measured `gamma2=0` KO baseline, it checks every history time against

```text
C_L2(gamma2,diss,t) = C_L2(0,diss,t) exp(-gamma2 t).
```

Both reduction and curl trajectories pass. Maximum absolute growth-factor
errors are `7.07e-7` and `2.87e-6`, respectively. With KO off, the final
growth factors at `gamma2=0.5/1.0` are `0.9048375/0.8187312` for reduction
and `0.9048382/0.8187316` for curl. With KO on, the separately measured final
KO factors are `0.9720309` and `0.9718944`; their products with
`exp(-gamma2 t)` reproduce the combined runs.

## Reproduction

```text
# Gamma2 matrix: use tst/inputs/ref_gh_stability.athinput with
problem/perturbation=random
ref_gh/gamma2=0,0.5,1
ref_gh/diss=0,0.02

python3 scripts/ref_gh/analyze_gamma2_subsidiary_matrix.py RUN_ROOT \
  --output-json analysis.json --output-csv analysis.csv

# Gamma0 matrix: use the same input with
problem/perturbation=gh_transverse problem/amp=1e-8
mesh/nx1=32 mesh/nx2=8 mesh/nx3=8
meshblock/nx1=32 meshblock/nx2=8 meshblock/nx3=8
ref_gh/gamma0=0.25,0.5,1 ref_gh/gamma2=0
ref_gh/diss=0,0.02

python3 scripts/ref_gh/analyze_gamma0_transverse_mode.py RUN_ROOT \
  --output-json analysis.json --output-csv analysis.csv
```

The raw `.hst` and final `.dat` files are retained beside the analyses. These
are local CPU, linearized/robust uniform-grid tests. They do not establish
nonlinear black-hole constraint behavior, SMR behavior, device portability,
q control, or trumpet stability.
