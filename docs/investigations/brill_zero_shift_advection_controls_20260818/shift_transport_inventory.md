# Shift-transport inventory

This inventory separates the actual contraction `beta^j d_j u` from
derivatives of beta and from gauge sources.

| Evolved field | Pure transport | Separate non-transport shift terms | Z arm | U2 arm |
|---|---|---|---|---|
| `chi` | scalar advection | `chi div(beta)` | zero | O2 transport only |
| `Khat`, `Theta` | scalar advection | none | zero | O2 transport only |
| `gtilde_ij`, `Atilde_ij` | tensor advection | Lie deformation from `d beta` and `div(beta)` | zero | O2 transport; deformation stays O6 |
| `Gamma^i` | vector advection | `Gamma*d beta`, inverse-metric `d d beta`, divergence terms | zero | O2 transport; beta derivatives stay O6 |
| `alpha` | scalar advection | lapse/telegraph sources | zero | O2 transport only |
| `beta^i` | vector self-advection | Gamma-driver, eta damping, harmonic-shift sources | prescribed exactly zero | O2 transport; driver unchanged |
| `B_i` | vector advection when telegraph lapse is active | telegraph damping/lapse gradient | retained as lapse flux | O2 transport; telegraph sources unchanged |

There is no separate Gamma-driver auxiliary `B^i` in this source.  The three
`z4c_B*` slots are the telegraph-lapse flux when telegraph lapse is enabled.
The consistent prescribed-shift invariant is therefore bitwise `beta^i=0`;
`B_i=0` is additionally enforced only when those slots are not the active
telegraph-lapse state.  Zeroing an active telegraph `B_i` would change the lapse
prescription and violate the controlled comparison.

All centered first/second geometric derivatives retain the configured O6
provider.  Only explicit `*Advective` calls dispatch to `Lx<2>` in U2.
