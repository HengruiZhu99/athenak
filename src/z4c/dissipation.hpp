#ifndef Z4C_DISSIPATION_HPP_
#define Z4C_DISSIPATION_HPP_
#include "z4c/z4c.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
namespace z4c {
// Caller checks the continuum RHS axis subspace before this operator. The axis
// projection below applies to the KO-completed RHS, matching the leaf path.
template<typename Centering,typename Symmetry,int NGHOST>
void AddZ4cDissipation(const Z4cGridLayout &layout,const DualArray1D<RegionSize> &size,
    const DualArray2D<BoundaryFlag> &mb_bcs,const subcycling::BlockBatches &rhs_batches,
    const DvceArray5D<Real> &u0,const DvceArray5D<Real> &u_rhs,Real diss,
    bool collect_chi_provenance,const DvceArray5D<Real> &chi_provenance_terms) {
  const int is=layout.is,ie=layout.ie,js=layout.js,je=layout.je;
  const int ks=layout.ks,ke=layout.ke,nx1=layout.nx1,nx3=layout.nx3,nz4c=Z4c::nz4c;
  if constexpr(std::is_same_v<Centering,VertexCenteredZ4c> &&
               std::is_same_v<Symmetry,CartoonSO2>) {
    rhs_batches.For4("SO2-invariant vertex K-O dissipation",
            ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real values[Z4c::nz4c];
          Real idx[] = {1 / size.d_view(m).dx1, 1 / size.d_view(m).dx2,
                        1 / size.d_view(m).dx3};
          auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
              idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
          for (int n = 0; n < Z4c::nz4c; ++n) {
            values[n] = u_rhs(m, n, k, j, i);
            for (int direction = 0; direction < 3; ++direction) {
              const Real before = values[n];
              values[n] += derivatives.DirectionalComponentDissipation(
                               direction, n, u0) * diss;
              if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
                const int term = direction == 0 ? chi_ko_rho
                                 : (direction == 1 ? chi_ko_z : chi_ko_y);
                const int cumulative = direction == 0 ? chi_rhs_after_ko_rho
                                       : (direction == 1 ? chi_rhs_after_ko_z
                                                         : chi_rhs_after_ko_y);
                chi_provenance_terms(m, term, k, j, i) = values[n] - before;
                chi_provenance_terms(m, cumulative, k, j, i) = values[n];
              }
            }
          }
          if (i == is &&
              mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis) {
            ProjectVertexAxisZ4cValues(values);
          }
          for (int n = 0; n < Z4c::nz4c; ++n) {
            u_rhs(m, n, k, j, i) = values[n];
          }
          if (collect_chi_provenance) {
            chi_provenance_terms(m, chi_rhs_after_ko, k, j, i) =
                values[Z4c::I_Z4C_CHI];
          }
        });
  } else {
    rhs_batches.For5("K-O Dissipation",
    0,nz4c-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
      auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
          idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
      // Keep the established multiply-then-accumulate order for Cartesian roundoff.
      for (int direction = 0; direction < 3; ++direction) {
        if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
          const Real rhs_before = u_rhs(m,n,k,j,i);
          u_rhs(m,n,k,j,i) +=
              derivatives.DirectionalComponentDissipation(direction, n, u0) * diss;
          const Real contribution = u_rhs(m,n,k,j,i) - rhs_before;
          const int term = direction == 0 ? chi_ko_rho
                           : (direction == 1 ? chi_ko_z : chi_ko_y);
          const int cumulative = direction == 0 ? chi_rhs_after_ko_rho
                                 : (direction == 1 ? chi_rhs_after_ko_z
                                                   : chi_rhs_after_ko_y);
          chi_provenance_terms(m, term, k, j, i) = contribution;
          chi_provenance_terms(m, cumulative, k, j, i) = u_rhs(m,n,k,j,i);
        } else {
          u_rhs(m,n,k,j,i) +=
              derivatives.DirectionalComponentDissipation(direction, n, u0) * diss;
        }
      }
      if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
        chi_provenance_terms(m, chi_rhs_after_ko, k, j, i) = u_rhs(m,n,k,j,i);
      }
    });
  }
}
}  // namespace z4c
#endif  // Z4C_DISSIPATION_HPP_
