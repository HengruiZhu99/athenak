//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_field_oscillator.cpp
//! \brief Homogeneous real, complex, and quartic scalar-field oscillator tests.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "scalar_field/scalar_field.hpp"
#include "z4c/z4c.hpp"

namespace {

Real oscillator_lapse = 1.0;
Real oscillator_initial_pi = 0.0;
Real oscillator_expansion_rate = 0.0;

enum class OscillatorCase {
  real_free,
  complex_free,
  real_quartic,
  dynamic_flrw,
  smooth_excision
};

struct OscillatorState {
  Real phi[2];
  Real pi[2];
};

[[noreturn]] void OscillatorFatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

OscillatorCase ReadOscillatorCase(ParameterInput *pin) {
  const std::string test_case =
      pin->GetOrAddString("problem", "test_case", "real_free");
  if (test_case == "real_free") {
    return OscillatorCase::real_free;
  }
  if (test_case == "complex_free") {
    return OscillatorCase::complex_free;
  }
  if (test_case == "real_quartic") {
    return OscillatorCase::real_quartic;
  }
  if (test_case == "dynamic_flrw") {
    return OscillatorCase::dynamic_flrw;
  }
  if (test_case == "smooth_excision") {
    return OscillatorCase::smooth_excision;
  }
  OscillatorFatal("Unknown scalar oscillator test_case: " + test_case);
}

OscillatorState InitialState(const OscillatorCase test_case,
                             const Real amplitude, const Real mass) {
  OscillatorState state;
  state.phi[0] = amplitude;
  state.phi[1] = 0.0;
  state.pi[0] = 0.0;
  state.pi[1] = 0.0;
  if (test_case == OscillatorCase::complex_free) {
    state.pi[1] = -mass*amplitude;
  } else if (test_case == OscillatorCase::dynamic_flrw) {
    state.pi[0] = oscillator_initial_pi;
  }
  return state;
}

void QuarticRhs(const Real phi, const Real pi, const Real mass,
                const Real quartic, Real *dphi, Real *dpi) {
  *dphi = -pi;
  *dpi = (mass*mass + quartic*phi*phi)*phi;
}

OscillatorState QuarticReference(const Real time, const Real amplitude,
                                  const Real mass, const Real quartic) {
  OscillatorState state;
  state.phi[0] = amplitude;
  state.phi[1] = 0.0;
  state.pi[0] = 0.0;
  state.pi[1] = 0.0;
  const int nsteps = std::max(1, static_cast<int>(std::ceil(time/1.0e-4)));
  const Real dt = time/nsteps;

  for (int step = 0; step < nsteps; ++step) {
    Real k1_phi;
    Real k1_pi;
    Real k2_phi;
    Real k2_pi;
    Real k3_phi;
    Real k3_pi;
    Real k4_phi;
    Real k4_pi;
    QuarticRhs(state.phi[0], state.pi[0], mass, quartic, &k1_phi, &k1_pi);
    QuarticRhs(state.phi[0] + 0.5*dt*k1_phi,
               state.pi[0] + 0.5*dt*k1_pi, mass, quartic,
               &k2_phi, &k2_pi);
    QuarticRhs(state.phi[0] + 0.5*dt*k2_phi,
               state.pi[0] + 0.5*dt*k2_pi, mass, quartic,
               &k3_phi, &k3_pi);
    QuarticRhs(state.phi[0] + dt*k3_phi, state.pi[0] + dt*k3_pi,
               mass, quartic, &k4_phi, &k4_pi);
    state.phi[0] +=
        (dt/6.0)*(k1_phi + 2.0*k2_phi + 2.0*k3_phi + k4_phi);
    state.pi[0] +=
        (dt/6.0)*(k1_pi + 2.0*k2_pi + 2.0*k3_pi + k4_pi);
  }
  return state;
}

OscillatorState ReferenceState(const OscillatorCase test_case, const Real time,
                               const Real amplitude, const Real mass,
                               const Real quartic) {
  if (test_case == OscillatorCase::real_quartic) {
    return QuarticReference(time, amplitude, mass, quartic);
  }
  if (test_case == OscillatorCase::dynamic_flrw) {
    const Real scale = 1.0 + oscillator_expansion_rate*time;
    OscillatorState state = {
      {Real(amplitude - oscillator_initial_pi/(2.0*oscillator_expansion_rate) *
                        (1.0 - 1.0/(scale*scale))),
       0.0},
      {oscillator_initial_pi/(scale*scale*scale), 0.0}
    };
    return state;
  }
  if (test_case == OscillatorCase::smooth_excision) {
    return InitialState(test_case, amplitude, mass);
  }

  const Real cosine = std::cos(mass*time);
  const Real sine = std::sin(mass*time);
  OscillatorState state = {
    {amplitude*cosine, 0.0},
    {amplitude*mass*sine, 0.0}
  };
  if (test_case == OscillatorCase::complex_free) {
    state.phi[1] = amplitude*sine;
    state.pi[1] = -amplitude*mass*cosine;
  }
  return state;
}

KOKKOS_INLINE_FUNCTION
void HomogeneousMatter(const OscillatorState &state, const int ncomponents,
                       const Real mass, const Real quartic, Real *energy,
                       Real *pressure, Real *charge) {
  Real phi_squared = 0.0;
  Real pi_squared = 0.0;
  for (int component = 0; component < ncomponents; ++component) {
    phi_squared += state.phi[component]*state.phi[component];
    pi_squared += state.pi[component]*state.pi[component];
  }
  const Real q = 0.5*phi_squared;
  const Real potential = mass*mass*q + quartic*q*q;
  *energy = 0.5*pi_squared + potential;
  *pressure = 0.5*pi_squared - potential;
  *charge = (ncomponents == 2)
      ? state.phi[1]*state.pi[0] - state.phi[0]*state.pi[1] : 0.0;
}

KOKKOS_INLINE_FUNCTION
Real WrappedPhaseError(const Real numerical, const Real exact) {
  const Real difference = numerical - exact;
  return fabs(atan2(sin(difference), cos(difference)));
}

void SetScalarOscillatorADM(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = pmbp->nmb_thispack;
  const Real lapse = oscillator_lapse;
  const Real scale =
      1.0 + oscillator_expansion_rate*pmbp->padm->metric_time;
  const Real metric_diagonal = scale*scale;
  const Real curvature_diagonal = -scale*oscillator_expansion_rate;
  auto &adm = pmbp->padm->adm;

  par_for("pgen_scalar_oscillator_adm", DevExeSpace(), 0, nmb - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm.alpha(m, k, j, i) = lapse;
    for (int direction = 0; direction < 3; ++direction) {
      adm.beta_u(m, direction, k, j, i) = 0.0;
    }
    adm.psi4(m, k, j, i) = 1.0;

    adm.g_dd(m, 0, 0, k, j, i) = metric_diagonal;
    adm.g_dd(m, 0, 1, k, j, i) = 0.0;
    adm.g_dd(m, 0, 2, k, j, i) = 0.0;
    adm.g_dd(m, 1, 1, k, j, i) = metric_diagonal;
    adm.g_dd(m, 1, 2, k, j, i) = 0.0;
    adm.g_dd(m, 2, 2, k, j, i) = metric_diagonal;

    adm.vK_dd(m, 0, 0, k, j, i) = curvature_diagonal;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 1, k, j, i) = curvature_diagonal;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 2, 2, k, j, i) = curvature_diagonal;
  });
}

void ScalarOscillatorErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const OscillatorCase test_case = ReadOscillatorCase(pin);
  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 0.2);
  const Real mass = pin->GetReal("scalar_field", "mass");
  const Real quartic = pin->GetReal("scalar_field", "lambda");
  const int ncomponents = pmbp->pscalar->ncomponents;
  const OscillatorState initial =
      InitialState(test_case, amplitude, mass);
  const OscillatorState exact =
      ReferenceState(test_case, pm->time, amplitude, mass, quartic);

  Real initial_energy;
  Real initial_pressure;
  Real initial_charge;
  HomogeneousMatter(initial, ncomponents, mass, quartic,
                    &initial_energy, &initial_pressure, &initial_charge);
  Real exact_energy;
  Real exact_pressure;
  Real exact_charge;
  HomogeneousMatter(exact, ncomponents, mass, quartic,
                    &exact_energy, &exact_pressure, &exact_charge);
  (void)initial_pressure;
  (void)exact_charge;

  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nkji = nx1*nx2*nx3;
  const int ncell = pmbp->nmb_thispack*nkji;
  const Real phase_scale =
      sqrt(mass*mass + quartic*amplitude*amplitude);
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->pscalar->u0;

  array_sum::GlobalSum local_sum;
  Kokkos::parallel_reduce(
      "scalar oscillator errors",
      Kokkos::RangePolicy<DevExeSpace>(0, ncell),
      KOKKOS_LAMBDA(const int index, array_sum::GlobalSum &sum) {
        const int m = index/nkji;
        const int cell = index - m*nkji;
        const int k = ks + cell/(nx1*nx2);
        const int row = cell - (k - ks)*nx1*nx2;
        const int j = js + row/nx1;
        const int i = is + row - (j - js)*nx1;
        const Real volume =
            size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

        OscillatorState numerical = {
          {u0(m, scalar_field::ScalarField::I_SF_PHI0, k, j, i), 0.0},
          {u0(m, scalar_field::ScalarField::I_SF_PI0, k, j, i), 0.0}
        };
        if (ncomponents == 2) {
          numerical.phi[1] =
              u0(m, scalar_field::ScalarField::I_SF_PHI1, k, j, i);
          numerical.pi[1] =
              u0(m, scalar_field::ScalarField::I_SF_PI1, k, j, i);
        }

        Real energy;
        Real pressure;
        Real charge;
        HomogeneousMatter(numerical, ncomponents, mass, quartic,
                          &energy, &pressure, &charge);
        Real numerical_phase;
        Real exact_phase;
        if (ncomponents == 2) {
          numerical_phase = atan2(numerical.phi[1], numerical.phi[0]);
          exact_phase = atan2(exact.phi[1], exact.phi[0]);
        } else if (phase_scale > 0.0) {
          numerical_phase =
              atan2(numerical.pi[0]/phase_scale, numerical.phi[0]);
          exact_phase = atan2(exact.pi[0]/phase_scale, exact.phi[0]);
        } else {
          numerical_phase = 0.0;
          exact_phase = 0.0;
        }

        array_sum::GlobalSum point;
        point.the_array[0] =
            volume*fabs(numerical.phi[0] - exact.phi[0]);
        point.the_array[1] =
            volume*fabs(numerical.pi[0] - exact.pi[0]);
        point.the_array[2] =
            volume*fabs(numerical.phi[1] - exact.phi[1]);
        point.the_array[3] =
            volume*fabs(numerical.pi[1] - exact.pi[1]);
        point.the_array[4] =
            volume*WrappedPhaseError(numerical_phase, exact_phase);
        point.the_array[5] = volume*fabs(energy - initial_energy);
        point.the_array[6] = volume*fabs(charge - initial_charge);
        // With zero gradients, S_i and off-diagonal S_ij vanish identically.
        point.the_array[7] =
            volume*(fabs(energy - exact_energy) +
                    3.0*fabs(pressure - exact_pressure));
        point.the_array[8] = volume;
        for (int n = 9; n < NREDUCTION_VARIABLES; ++n) {
          point.the_array[n] = 0.0;
        }
        sum += point;
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum));

  Real reduced[9];
  for (int n = 0; n < 9; ++n) {
    reduced[n] = local_sum.the_array[n];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, reduced, 9, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int n = 0; n < 8; ++n) {
    reduced[n] /= reduced[8];
  }

  if (global_variable::my_rank == 0) {
    std::string filename = pin->GetString("job", "basename");
    filename.append("-errs.dat");
    FILE *file = std::fopen(filename.c_str(), "r");
    if (file != nullptr) {
      file = std::freopen(filename.c_str(), "a", file);
    } else {
      file = std::fopen(filename.c_str(), "w");
      if (file != nullptr) {
        std::fprintf(file, "# Nx1  Ncycle  time  phi0_L1  Pi0_L1  ");
        std::fprintf(file, "phi1_L1  Pi1_L1  phase_L1  energy_drift  ");
        std::fprintf(file, "charge_drift  Tmunu_L1\n");
      }
    }
    if (file == nullptr) {
      OscillatorFatal("Scalar oscillator error file could not be opened.");
    }
    std::fprintf(file, "%04d  %05d  %.17e", pm->mesh_indcs.nx1,
                 pm->ncycle, pm->time);
    for (int n = 0; n < 8; ++n) {
      std::fprintf(file, "  %.17e", reduced[n]);
    }
    std::fprintf(file, "\n");
    std::fclose(file);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::ScalarFieldOscillator()
//! \brief Initialize homogeneous canonical scalar oscillator regression cases.

void ProblemGenerator::ScalarFieldOscillator(ParameterInput *pin,
                                             const bool restart) {
  pgen_final_func = ScalarOscillatorErrors;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->padm == nullptr || pmbp->pscalar == nullptr) {
    OscillatorFatal(
        "Scalar oscillator requires <adm> and <scalar_field> blocks.");
  }

  const OscillatorCase test_case = ReadOscillatorCase(pin);
  const bool is_complex = test_case == OscillatorCase::complex_free;
  const bool is_quartic = test_case == OscillatorCase::real_quartic;
  const bool is_dynamic = test_case == OscillatorCase::dynamic_flrw;
  const bool is_excision = test_case == OscillatorCase::smooth_excision;
  if (pmbp->padm->is_dynamic != is_dynamic) {
    OscillatorFatal(
        "Scalar oscillator adm/dynamic setting does not match test_case.");
  }
  if (pmbp->pscalar->excision != is_excision) {
    OscillatorFatal(
        "Scalar oscillator excision setting does not match test_case.");
  }
  if ((pmbp->pscalar->ncomponents == 2) != is_complex) {
    OscillatorFatal("Scalar oscillator field_type does not match test_case.");
  }
  const std::string potential =
      pin->GetOrAddString("scalar_field", "potential", "free");
  if ((potential == "mass_quartic") != is_quartic) {
    OscillatorFatal("Scalar oscillator potential does not match test_case.");
  }

  oscillator_lapse = pin->GetOrAddReal("problem", "lapse", 1.0);
  if (oscillator_lapse <= 0.0) {
    OscillatorFatal("Scalar oscillator lapse must be positive.");
  }
  oscillator_initial_pi =
      pin->GetOrAddReal("problem", "initial_pi", 0.0);
  oscillator_expansion_rate =
      pin->GetOrAddReal("problem", "expansion_rate", 0.0);
  if (is_dynamic && oscillator_expansion_rate <= 0.0) {
    OscillatorFatal("Dynamic FLRW expansion_rate must be positive.");
  }
  if (!is_excision) {
    if (pmbp->pz4c == nullptr) {
      pmbp->padm->SetADMVariables = SetScalarOscillatorADM;
    }
  }
  if (pmbp->pz4c == nullptr) {
    pmbp->padm->SetADMVariablesAtTime(pmbp, pmy_mesh_->time);
  }
  if (restart) {
    return;
  }

  const Real amplitude = pin->GetOrAddReal("problem", "amplitude", 0.2);
  const Real mass = pin->GetReal("scalar_field", "mass");
  const OscillatorState initial =
      InitialState(test_case, amplitude, mass);
  auto &indcs = pmy_mesh_->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = pmbp->nmb_thispack;
  const int ncomponents = pmbp->pscalar->ncomponents;
  auto &u0 = pmbp->pscalar->u0;

  if (pmbp->pz4c != nullptr) {
    auto &pz4c = pmbp->pz4c;
    auto &z4c = pz4c->z4c;
    Kokkos::deep_copy(pz4c->u0, 0.0);
    Kokkos::deep_copy(pz4c->u1, 0.0);
    Kokkos::deep_copy(pz4c->u_rhs, 0.0);
    par_for("pgen_scalar_oscillator_z4c", DevExeSpace(), 0, nmb - 1,
    0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      z4c.chi(m, k, j, i) = 1.0;
      z4c.alpha(m, k, j, i) = 1.0;
      for (int direction = 0; direction < 3; ++direction) {
        z4c.g_dd(m, direction, direction, k, j, i) = 1.0;
      }
    });
    pz4c->Z4cToADM(pmbp);
  }

  par_for("pgen_scalar_oscillator", DevExeSpace(), 0, nmb - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, scalar_field::ScalarField::I_SF_PHI0, k, j, i) = initial.phi[0];
    u0(m, scalar_field::ScalarField::I_SF_PI0, k, j, i) = initial.pi[0];
    if (ncomponents == 2) {
      u0(m, scalar_field::ScalarField::I_SF_PHI1, k, j, i) =
          initial.phi[1];
      u0(m, scalar_field::ScalarField::I_SF_PI1, k, j, i) =
          initial.pi[1];
    }
  });
}
