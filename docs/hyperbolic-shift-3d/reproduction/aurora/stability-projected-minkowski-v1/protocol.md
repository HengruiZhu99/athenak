# Projection-only duration discriminator

The unchanged-driver resolution controls are complete. They show increasing
Minkowski growth with spatial refinement and essentially unchanged growth when
the n16 timestep is halved. This next test isolates the already-implemented
final-stage algebraic projection from the cross-face angular candidate.

Use the frozen source-v5 projected executable, SHA256
`362093fc8c998576c7d88a14afdc2c5ebcdcb6a68f408b7fdaf92ad62b7b1486`,
whose combined numerical-source/launcher manifest is
`8d0eba426c84cdf9997864cf7471a44914cc590b9a4d4ea2efe022b944287c93`.
Its twelve-case preflight passed, including native CPU/GPU comparisons,
algebraic-projection checks and bitwise direct/restarted V2 checkpoints.
The duration admission permits testing; it does not assert stability.

Start fresh analytic Minkowski data at n16 on the same 19-block Cartesian
core and spherical shells, dt=.0005, and target coordinate t=1. Record every
.05. Use one Aurora debug node under CompactBinaryMerger with twelve ranks
and distinct GPU tiles. No puncture or pulse evolution is part of this plan.

The candidate normalizes det(g) and removes tr_g(A) after each accepted RK
step, then rebuilds null closure and dependent D. It retains the original
face-local angular scri stencil and disabled J. It also uses the previously
validated staged-jet sampling optimization; that optimization can change
roundoff seeding, so compare growth rates and sustained errors, not just
failure times. This is not a bitwise arithmetic-only projection comparison.

Keep the original stability limits: completed/current diagnostics, positive
chi/lapse, state deviation <=1e-8, H/M L2 <=1e-7, both null residuals <=1e-12,
and recorded CFL <=.5. Additionally require final determinant and trace
residuals <=1e-12 and the reported final_stage_standard/V2 scheme. Preserve
any failed-stage capture and distinguish walltime bounds from endpoint
completion. No failed or scientifically rejected state is restarted.

The numerical child has 1,020 seconds inside a 20-minute allocation. Reserve
130 MB for outputs plus 100 MB for atomic checkpoint work under the fixed
5 GB campaign cap. Larger-block communication efficiency is investigated
separately; this test retains the preflight-qualified n16 grid.

A pass at t=1 is a bounded Minkowski result only. The requested puncture
transition to 100 M, further Minkowski duration and eventual convergence
still need their own evidence. A failure requires another controlled
boundary test rather than relabeling the endpoint or relaxing tolerances.
