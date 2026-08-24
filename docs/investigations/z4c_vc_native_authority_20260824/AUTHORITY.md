# Source and run authority

The campaign began from reviewed numerical commit
`6dd20656a305f2543bbbd7001550c6ac67019180` and reviewed source tree
`551b16fab36ec1d4ce913b39a6478384723aa382` on
`codex/z4c-vc-derefine-slot-repair-20260824`. The initial numerical-source
diff against that commit was empty.

Phase 2 exposed nondeterministic CUDA atomic accumulation only in secondary
Cartoon regional history diagnostics. Commit
`d63519328214a6315a9cc1f7d5e4a1aa4bca21b0` replaces those unordered atomics
with a dedicated Kokkos sum reduction and adds a source regression guard. It
does not change evolved variables, equations, AMR decisions, transfer, RK,
gauge, KO, CFL, or output cadence. All production evidence in this directory
after the discovery run uses source tree
`9fa84d4b79c2d50ce935f5416fba6d57f99aa5b4`.

The Perlmutter CUDA executable SHA-256 is
`3a395bfdaf217d617fee43d2cbcd38e7a13c2a0f4207e3a764c3513eb8c0405f`.
The focused one-A100 suite passed 20/20 tests. In the post-repair aggregate
host run, 140/141 enabled tests passed and the AMR-history integration test hit
its fixed 600-second timeout under concurrent load; that sole test then passed
in isolation in 177.96 seconds. Two CUDA-only tests were disabled in the host
build.

The fixed production configuration is native vertex centering, Cartoon SO(2),
O4 finite differences, q6 vertex transfer, RK4, CFL 0.15, KO 0.02,
max-domain-|K|-scaled telegraph lapse with tau=kappa=1, Gamma-driver shift with
eta=2, no Z4 damping, no chi floor, pre-collapsed lapse, `dchi_max=0.01`, and
the repaired default derefinement factor 0.25.

The fresh N256 authority through the early gate is
`evidence/perlmutter/runs/n256_native_record_t2p5_v2_dethist/n256_native_authority.jsonl`
with SHA-256
`fd08e6b32b094ef9e8e928a7ad8e061edcc6e67617609952f0aefa47e4b0f694`.

The failed later extension is preserved separately. It is not a qualified
authority for production replay.
