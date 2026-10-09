The saved-data readbacks pass; they do not repair the failed propagation or
establish continuum constraint stability. No kernel query, compilation,
eigenvalue, exponential or growth execution was performed. All three released
runs used the exact source reviewed in root-source-review.json, produced empty
stderr and kept every input hash unchanged. Launch HEAD was ecab931ffc8fb0f5c1f70c9678563c82b6fbc090.

The failed-growth readback has840 column/point records. It retains the original
SciPy matmul failure at t=.25 and only the saved t=0 state; propagation remains
failed. The four selected columns are diagnosed from their saved actual Jv,
not replaced by lambda*v. Saved actual Jv matches the pinned total matrix action
to3.08e-15 scaled; independent scalar point contractions agree within1.17e-15.
Physical seed amplitudes and energy-normalized modal columns are retained.

| Saved lambda | Peak sample C(v) | Peak sample C(actual Jv) |
|---|---:|---:|
|2.049333+6.984849i|2.402878|17.491215|
|-0.281972+2.394510i|3.818196|9.205882|
|-0.752420+11.552581i|1.479345|17.126457|
|-0.885354+6.376097i|6.779227|43.639724|

For the first column, H/M/Z/Theta sample peaks are1.980210/1.360680/.031233/
.028137; the corresponding actual-Jv peaks are14.414502/9.904768/.227350/
.204819. The physical seed alpha and beta columns have exactly zero initial
constraints, while their projected total-matrix constraint-rate sample peaks
are1.467766 and.121307. These measured values do not classify a continuum or
subsidiary eigenmode. The diagnostic C(Jv)-lambda*C(v) scaled residual4.07e-13
is finite-matrix linear algebra only.

N12 and N16 each have294 point/witness records; all84 initial gauge vectors are
exactly zero. The table compares their bulk projected-rate peaks with the frozen
N8 analytic-point readback at the same21 centers. Values are Euclidean summaries
of the raw Cartesian physical8 components H,Mxyz,Zxyz,Theta_physical. They are
neither an integrated energy nor a geometric norm of spatial covectors.

| Gauge seed envelope | N8 bulk | N12 bulk | N16 bulk |
|---|---:|---:|---:|
|constant alpha|986.112865|265.078053|170.303922|
|rho^3 beta|2651.108425|565.639419|123.620486|
|exp(-8rho) alpha|14.311411|1.326608|3.483039|
|exp(-8rho) beta|5.899415|.471758|.479171|

N12 SAT peaks in the same order are.021094/.090403/.000165/.000122; N16
SAT peaks are.013983/.368324/.000107/.000488. Bulk, SAT and total are retained
separately. No smallness demand or CPBC conclusion is imposed. Polynomial gauge
seeds compare the same envelope; exponential seeds are N-specific interpolants.
Their scalar envelope interpolation errors are displayed separately. The mixed
degree behavior supplies no order or convergence theorem.

The positive-Omega continuum gauge rate zero is derived Einstein-sector ADM
tangency from the preceding source/rate audit. It is not a numerical pass of
either failed ordinary-FD sequence. Both FD stops remain failed, and the generic
nongauge comparator C_ref[L_actual Phi(X)] remains unresolved. The present
analytic maps calculate C_ref[Phi(Y)] and cannot supply missing higher spatial
derivatives of the unprojected RHS.

The exact command, sources, data, authorization, matrix T/Riesz/Cholesky checks
and complete physical8 vectors are in each fresh run receipt and cases.jsonl.
quick-summary.json contains exact saved summaries; stage-inventory.json records
artifact hashes. Every NPZ/NPY is large_payload regardless of size. The original
970a117d index and its compact-verifier failure remain unchanged. Finite-pulse
stability and later single-BH wormhole-to-trumpet formation with the Minkowski
hyperboloidal reference remain unresolved.
