#ifndef OUTPUTS_PDF_ACCUMULATE_HPP_
#define OUTPUTS_PDF_ACCUMULATE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pdf_accumulate.hpp
//! \brief CUDA-portable compile-time policies for production PDF accumulation.

#include <cmath>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs.hpp"

namespace pdf {

template <typename SizeView>
struct CartesianPdfCellMeasure {
  SizeView size;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const int m, const int /*i*/) const {
    return size(m).dx1 * size(m).dx2 * size(m).dx3;
  }
};

template <typename SizeView>
struct CartoonPdfCellMeasure {
  SizeView size;
  int nx1;
  int is;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const int m, const int i) const {
    const Real rho = CellCenterX(i - is, nx1, size(m).x1min, size(m).x1max);
    // The full signed-rho plane is evolved.  Only its positive half owns unique
    // cylindrical fine cells; the negative partner contributes zero measure.
    return CartoonCylindricalFineCellMeasure(rho, size(m).dx1, size(m).dx2);
  }
};

// The cell-measure type is selected once by the host.  Keeping the Kokkos lambda in a
// named function template avoids nvcc's prohibition on extended lambdas nested inside a
// host generic lambda while retaining one shared binning/scatter implementation and no
// per-cell runtime symmetry branch.
template <typename CellMeasure>
void AccumulatePdfHistogram(const CellMeasure cell_measure,
                            const DvceArray5D<Real> outvars,
                            const DvceArray5D<Real> state,
                            const Kokkos::View<Real*> bins,
                            const Kokkos::View<Real*> bins2,
                            const PDFData::ScatterResult scatter,
                            const int nmb, const int ks, const int ke,
                            const int js, const int je, const int is, const int ie,
                            const int nbin, const int nbin2, const int dimension,
                            const Real step_size, const Real step_size2,
                            const bool logscale, const bool logscale2,
                            const bool mass_weighted) {
  par_for("pdf", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x_val = outvars(0, m, k, j, i);
    int x_bin = -1;
    if (x_val < bins(0)) {
      x_bin = 0;
    } else if (x_val >= bins(nbin)) {
      x_bin = nbin + 1;
    } else if (!logscale) {
      x_bin = static_cast<int>((x_val - bins(0)) / step_size) + 1;
    } else {
      x_bin = static_cast<int>(std::log10(x_val / bins(0)) / step_size) + 1;
    }

    int y_bin = 0;
    if (dimension == 2) {
      const Real y_val = outvars(1, m, k, j, i);
      if (y_val < bins2(0)) {
        y_bin = 0;
      } else if (y_val >= bins2(nbin2)) {
        y_bin = nbin2 + 1;
      } else if (!logscale2) {
        y_bin = static_cast<int>((y_val - bins2(0)) / step_size2) + 1;
      } else {
        y_bin = static_cast<int>(std::log10(y_val / bins2(0)) / step_size2) + 1;
      }
    }

    Real weight = cell_measure(m, i);
    if (weight == 0.0) return;
    if (mass_weighted) weight *= state(m, IDN, k, j, i);
    scatter.access()(y_bin, x_bin) += weight;
  });
}

}  // namespace pdf

#endif  // OUTPUTS_PDF_ACCUMULATE_HPP_
