# Phase-3 residual gauge-coupling red checkpoint

This compact artifact preserves the first perturbed-state source-unit result
after adding the direct residual gauge-driver equations and an oracle-only
residual Einstein gauge-source implementation.

The Release/Serial build completed. The source-unit executable then exited
with status 1 at the new hard comparison. The already-established general
driver algebra still passed at `1.38778e-16`, and the exact matched static
`q=1` residual driver and source remained bitwise zero over the expanded
sample matrix. The perturbed comparison failed at sample 1322, decoded as
the analytic backend, `q=1`, `q_dot=q_ddot=0`, `r=0.8M`, first angular
direction, and an off-diagonal Einstein gauge-source category. Its maximum
conditioned error was `0.103832`, against the unchanged
`1024*epsilon_binary64 = 2.27374e-13` tolerance.

This is a red diagnostic checkpoint, not repaired-GH qualification. The
legacy full reconstruction is cancellation-sensitive, so the failure does
not by itself identify whether the new residual source or the binary64 legacy
comparison is wrong. An independent high-precision or generated residual
oracle is required before production dispatch. No production task graph,
evolution, or performance path calls the new residual source.
