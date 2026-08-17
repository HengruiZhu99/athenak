//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tensor_unit.cpp
//! \brief CPU/GPU unit tests for spacetime and mixed-dimension tensor storage.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

static_assert(TensorDOF<TensorSymm::SYM2, 4, 2> == 10,
              "A symmetric spacetime rank-2 tensor must have 10 components.");
static_assert(sizeof(AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>) == 4*sizeof(Real),
              "A spacetime vector must store exactly 4 components.");
static_assert(MixedTensor<Real, 3, 4>::ndof == 30,
              "Phi_iab storage must have 30 components.");
static_assert(sizeof(MixedTensor<Real, 3, 4>) == 30*sizeof(Real),
              "MixedTensor must store exactly its independent components.");

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::FoGhTensorUnit()
//! \brief Exercise 4D and mixed tensor access in the configured Kokkos execution space.

void ProblemGenerator::FoGhTensorUnit(ParameterInput *pin, const bool restart) {
  (void)pin;
  (void)restart;
  int errors = 0;
  Kokkos::parallel_reduce(
      "FO-GH tensor unit", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, int &local_errors) {
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> vector;
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> symmetric;
        MixedTensor<Real, 3, 4> phi;

        vector.ZeroClear();
        symmetric.ZeroClear();
        phi.ZeroClear();

        for (int a = 0; a < 4; ++a) {
          vector(a) = static_cast<Real>(a + 1);
          for (int b = a; b < 4; ++b) {
            symmetric(a, b) = static_cast<Real>(10*a + b + 1);
          }
        }
        for (int i = 0; i < 3; ++i) {
          for (int a = 0; a < 4; ++a) {
            for (int b = a; b < 4; ++b) {
              phi(i, a, b) = static_cast<Real>(100*i + 10*a + b + 1);
            }
          }
        }

        for (int a = 0; a < 4; ++a) {
          if (vector(a) != static_cast<Real>(a + 1)) {
            ++local_errors;
          }
          for (int b = 0; b < 4; ++b) {
            const int lo = (a < b ? a : b);
            const int hi = (a < b ? b : a);
            if (symmetric(a, b) != static_cast<Real>(10*lo + hi + 1)) {
              ++local_errors;
            }
          }
        }
        for (int i = 0; i < 3; ++i) {
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              const int lo = (a < b ? a : b);
              const int hi = (a < b ? b : a);
              if (phi(i, a, b) != static_cast<Real>(100*i + 10*lo + hi + 1)) {
                ++local_errors;
              }
            }
          }
        }
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH tensor unit test failed with " << errors
              << " indexing errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH tensor unit test passed." << std::endl;
}
