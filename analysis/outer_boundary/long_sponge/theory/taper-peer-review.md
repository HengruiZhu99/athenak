# Independent algebra check

Reviewer: `/root/boundary_audit`, 2026-09-20.

The reviewer independently checked the stated frozen-alpha/chi/beta, spatially varying sigma reduction. Momentum damping is +2 partial_i(sigma Theta); differentiating Q_t cancels its spatial gradient terms against Theta_t and leaves -2(D0 sigma)Q. Differentiating Theta_t gives -alpha chi div(sigma Q)-2(D0 sigma)Theta. A stationary layer with nonzero normal shift has D0 sigma=-beta dot grad(sigma).

This checks the analytic reduction, not a curved-background taper implementation or evolution.
