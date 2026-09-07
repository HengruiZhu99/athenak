# Intrinsic subsidiary checks

The exact arbitrary-function checks and independent differentiation of the
compiled CPU/CUDA point kernel pass. Production equations are unchanged.
The test is a local continuum check, not a numerical evolution.

For each of 12 smooth cubic local fields, all 50 state components vary and the
independent p,l,S,B fields generally differ from the exact primary gradients.
Curvature, C/Z and shift gradients are nonzero. The lapse alpha=rho*w and its
first/second derivatives use the true product rule. The rate is
lambda=alpha*(1+0.2*x-0.15*y+0.1*z), so its spatial derivative is nonzero.
Gauge-switch samples cover the inner, transition and outer regions.

The checker evaluates the compiled full RHS at 444 points and forms the ten
configuration time derivatives directly from its primary rows, including
alpha_t=rho*w_t+w*rho_t. Fourth-order independent spatial differentiation then
constructs E_t=G_t-d(x_t) and Omega_t=d(G_t). Targets use exact polynomial jets:
Lie_beta(E)-lambda*E and Lie_beta(Omega)-lambda*Omega-dlambda wedge E.
It does not reuse the compiled source Jet outputs as the derivative oracle.
All 30 reductions and 30 independent curl components enter the max-error test.
The omission controls for stretching and rate-gradient forcing exceed the
frozen 1e-5 discrimination threshold in every case.

At h=0.02,0.01,0.005, maximum CPU reduction residuals are
3.617e-7,2.275e-8,1.424e-9; curl residuals are
2.031e-6,1.267e-7,7.913e-9. Both orders approach four, matching the independent
differentiation rule, not an evolved solution's convergence rate. CUDA agrees
at these orders; the complete 128-output backend difference is 5.471e-16.
The authorized direct A100 batch took 0.37 seconds with over 36 GiB free.

The exact SymPy script separately proves the reduction and curl identities for
an arbitrary scalar potential/source, arbitrary one-form G, shift and variable
rate. It verifies Cartan's i_beta dG term with its sign intact. An arbitrary
metric component as a function of five chart coordinates proves
rawcurl(Q)_ij = J*Omega_ij + H[(d_i s)*E_j-(d_j s)*E_i].
The canceled primary-gradient product uses Hessian symmetry. These statements
apply componentwise; no curvature or GH vanishing assumption is used. Smoothness
and commuting continuum derivatives are required. No discrete interface or
puncture-uniform estimate follows automatically.

## Reproduction and evidence

Use numpy and sympy in an isolated Python environment:

```sh
python -W error analysis/pc_gh_clean_reduction/check_intrinsic_subsidiary.py \
  --binary /absolute/path/to/intrinsic_rhs \
  --output /absolute/path/to/new-output-directory
python analysis/pc_gh_clean_reduction/check_subsidiary_exact.py \
  --output /absolute/path/to/new-exact-result.json
```

For a remote backend, copy generated input.txt and states.json into a new run,
execute `intrinsic_rhs input.txt output.txt kokkos.txt` and save run.log and
binary-sha256.txt. Copy the batch back and add `--replay /absolute/path/to/batch`
to the first command. It checks identical regenerated input bytes and states,
analyzes the saved output without executing the supplied local binary, and
reports the saved remote executable hash. Existing outputs are never overwritten.

Compact results, component errors, negative-control strengths, convergence plot,
exact proof results and input/output/source/build hashes are in
qualification-runs-20260907/pcgh-clean-reduction/intrinsic-subsidiary-001/.
Large raw inputs, outputs and polynomial coefficients remain at manifest paths.
The production kernel is the previously checked d8f3110 construction; no
reference-repository code is imported or changed by these checks.

Full Fourier/transient and coupled RK/KO analysis remain, along with intrinsic
mesh evolution, halo/transfer/restart integration and all physical gates.
