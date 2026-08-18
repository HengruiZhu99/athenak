//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_constraints.cpp
//! \brief Formulation-independent vacuum ADM Hamiltonian and momentum diagnostics.

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/finite_diff.hpp"

namespace adm {

// This diagnostic deliberately operates only on the reconstructed ADM gamma_ij and K_ij.
// Both FO-GH and Z4c call this same kernel so their direct comparison uses an identical
// finite-difference operator and contraction convention.  It does not feed either system.
template <int NGHOST>
void ADM::ComputeVacuumConstraints(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const auto vars = adm;
  const auto common = u_common;

  Kokkos::deep_copy(common, 0.0);
  par_for("common vacuum ADM constraints", DevExeSpace(),
  0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> M_u;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;
    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};

    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          dg_ddd(c, a, b) = Dx<NGHOST>(c, idx, vars.g_dd, m, a, b, k, j, i);
          dK_ddd(c, a, b) = Dx<NGHOST>(c, idx, vars.vK_dd, m, a, b, k, j, i);
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = c; d < 3; ++d) {
            ddg_dddd(a, b, c, d) = (a == b)
                ? Dxx<NGHOST>(a, idx, vars.g_dd, m, c, d, k, j, i)
                : Dxy<NGHOST>(a, b, idx, vars.g_dd, m, c, d, k, j, i);
          }
        }
      }
    }

    const Real detg = SpatialDet(vars.g_dd(m, 0, 0, k, j, i),
                                 vars.g_dd(m, 0, 1, k, j, i),
                                 vars.g_dd(m, 0, 2, k, j, i),
                                 vars.g_dd(m, 1, 1, k, j, i),
                                 vars.g_dd(m, 1, 2, k, j, i),
                                 vars.g_dd(m, 2, 2, k, j, i));
    SpatialInv(1.0/detg,
               vars.g_dd(m, 0, 0, k, j, i), vars.g_dd(m, 0, 1, k, j, i),
               vars.g_dd(m, 0, 2, k, j, i), vars.g_dd(m, 1, 1, k, j, i),
               vars.g_dd(m, 1, 2, k, j, i), vars.g_dd(m, 2, 2, k, j, i),
               &g_uu(0, 0), &g_uu(0, 1), &g_uu(0, 2),
               &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          Gamma_ddd(c, a, b) = 0.5*(dg_ddd(a, b, c) + dg_ddd(b, a, c)
                                      - dg_ddd(c, a, b));
          Gamma_udd(c, a, b) = 0.0;
          for (int d = 0; d < 3; ++d) {
            Gamma_udd(c, a, b) += g_uu(c, d)*Gamma_ddd(d, a, b);
          }
        }
      }
    }

    Real ricci = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        R_dd(a, b) = 0.0;
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            for (int e = 0; e < 3; ++e) {
              R_dd(a, b) += g_uu(c, d)*Gamma_udd(e, a, c)*Gamma_ddd(e, b, d);
              R_dd(a, b) -= g_uu(c, d)*Gamma_udd(e, a, b)*Gamma_ddd(e, c, d);
            }
            R_dd(a, b) += 0.5*g_uu(c, d)*(
                -ddg_dddd(c, d, a, b) - ddg_dddd(a, b, c, d)
                +ddg_dddd(a, c, b, d) + ddg_dddd(b, c, a, d));
          }
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) ricci += g_uu(a, b)*R_dd(a, b);
    }

    Real trace_k = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        K_ud(a, b) = 0.0;
        for (int c = 0; c < 3; ++c) {
          K_ud(a, b) += g_uu(a, c)*vars.vK_dd(m, c, b, k, j, i);
        }
      }
      trace_k += K_ud(a, a);
    }
    Real kk = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) kk += K_ud(a, b)*K_ud(b, a);
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = b; c < 3; ++c) {
          DK_ddd(a, b, c) = dK_ddd(a, b, c);
          for (int d = 0; d < 3; ++d) {
            DK_ddd(a, b, c) -= Gamma_udd(d, a, b)*vars.vK_dd(m, d, c, k, j, i);
            DK_ddd(a, b, c) -= Gamma_udd(d, a, c)*vars.vK_dd(m, b, d, k, j, i);
          }
          DK_udd(a, b, c) = 0.0;
          for (int d = 0; d < 3; ++d) DK_udd(a, b, c) += g_uu(a, d)*DK_ddd(d, b, c);
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      M_u(a) = 0.0;
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          M_u(a) += g_uu(a, b)*DK_udd(c, b, c)
                    - g_uu(b, c)*DK_udd(a, b, c);
        }
      }
    }
    Real momentum2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        momentum2 += vars.g_dd(m, a, b, k, j, i)*M_u(a)*M_u(b);
      }
    }
    common(m, I_COMMON_H, k, j, i) = ricci + trace_k*trace_k - kk;
    common(m, I_COMMON_M2, k, j, i) = momentum2;
  });
}

template void ADM::ComputeVacuumConstraints<2>(MeshBlockPack *pmbp);
template void ADM::ComputeVacuumConstraints<3>(MeshBlockPack *pmbp);
template void ADM::ComputeVacuumConstraints<4>(MeshBlockPack *pmbp);

} // namespace adm
