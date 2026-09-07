// Residual-aware auxiliary transfer algebra. Halo construction belongs to callers.
#ifndef PC_GH_COMMUTING_TRANSFER_HPP_
#define PC_GH_COMMUTING_TRANSFER_HPP_

namespace pc_gh {

// Applicable to prolongation, restriction, or same-level exchange. source_target
// and destination_target must use the actual synchronized primaries on their
// respective meshes. For alpha, form rho*w before applying the derivative.
// This function neither obtains missing stencil support nor synchronizes ghosts.
// Keeping targets explicit prevents pretending that interpolation commutes with D.
template <typename Scalar>
inline Scalar TransferReductionResidual(int count, const Scalar *weight,
    const Scalar *source_auxiliary, const Scalar *source_target,
    Scalar destination_target) {
  Scalar value = destination_target;
  for (int q = 0; q < count; ++q) {
    value += weight[q]*(source_auxiliary[q] - source_target[q]);
  }
  return value;
}

}  // namespace pc_gh
#endif  // PC_GH_COMMUTING_TRANSFER_HPP_
