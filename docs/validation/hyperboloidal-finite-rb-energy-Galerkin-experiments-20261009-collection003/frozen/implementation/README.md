# Fresh finite-radius implementation preparation

This prefix contains source-only preparation for the reviewed total-J finite-radius control. No scientific executable, actual radial operator, eigensystem or propagation has been generated here. The source-derivative/mass/operator release is still pending the final addendum review. Eigenvalues and propagation require a separate later release.

`point_energy.hpp` implements the explicitly declared component packing, kernel tensor chart, point energy/strong/weak source densities, independent symmetric volume production and incoming penalty work. It has not been compiled or numerically exercised. Its point inputs must come from the separately gated actual full22 continuum action, complete reference normalization, first-order configuration derivative and coefficient derivatives. It does not provide these inputs or a radial discretization.

The adjacent held tree retains the original preparation receipt and the final addendum with fixed manufactured fields, thresholds and source scope. It is read-only input to this fresh implementation. All existing basis/angular/core/scalar and Cartesian histories remain unchanged.
