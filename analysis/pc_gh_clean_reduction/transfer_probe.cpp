#include "pc_gh/commuting_transfer.hpp"

// CPU ABI probe of the source primitive. This is not an AthenaK mesh evolution.
extern "C" void transfer_probe(int rows, int cols, const double *weights,
    const double *auxiliary, const double *target, const double *destination,
    double *result) {
  for (int i = 0; i < rows; ++i) {
    result[i] = pc_gh::TransferReductionResidual(cols, weights+i*cols,
                                                auxiliary, target, destination[i]);
  }
}
