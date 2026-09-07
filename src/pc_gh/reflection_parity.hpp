// Tensor and derivative parity shared by PC-GH boundary fill and reconstruction.
#ifndef PC_GH_REFLECTION_PARITY_HPP_
#define PC_GH_REFLECTION_PARITY_HPP_
#include "athena.hpp"
#include "pc_gh/pc_gh.hpp"
namespace pc_gh {
KOKKOS_INLINE_FUNCTION
constexpr bool ReflectOdd(int n, int axis) {
  if ((n >= pc_gh::PcGh::I_ZX && n <= pc_gh::PcGh::I_ZZ)
      || (n >= pc_gh::PcGh::I_BETAX && n <= pc_gh::PcGh::I_BETAZ)
      || (n >= pc_gh::PcGh::I_P1 && n <= pc_gh::PcGh::I_P3)
      || (n >= pc_gh::PcGh::I_L1 && n <= pc_gh::PcGh::I_L3)) {
    int const first = (n <= pc_gh::PcGh::I_ZZ) ? pc_gh::PcGh::I_ZX
        : ((n <= pc_gh::PcGh::I_BETAZ) ? pc_gh::PcGh::I_BETAX
        : ((n <= pc_gh::PcGh::I_P3) ? pc_gh::PcGh::I_P1 : pc_gh::PcGh::I_L1));
    return n - first == axis;
  }

  int tensor_component = -1;
  int derivative_axis = -1;
  if (n >= pc_gh::PcGh::I_GTXX && n <= pc_gh::PcGh::I_GTZZ) {
    tensor_component = n - pc_gh::PcGh::I_GTXX;
  } else if (n >= pc_gh::PcGh::I_ATXX && n <= pc_gh::PcGh::I_ATZZ) {
    tensor_component = n - pc_gh::PcGh::I_ATXX;
  } else if (n >= pc_gh::PcGh::I_Q1XX && n <= pc_gh::PcGh::I_Q3ZZ) {
    int const offset = n - pc_gh::PcGh::I_Q1XX;
    derivative_axis = offset/6;
    tensor_component = offset % 6;
  }
  if (tensor_component >= 0) {
    int first_index = 0;
    int second_index = 0;
    if (tensor_component == 1) {
      second_index = 1;
    } else if (tensor_component == 2) {
      second_index = 2;
    } else if (tensor_component == 3) {
      first_index = second_index = 1;
    } else if (tensor_component == 4) {
      first_index = 1;
      second_index = 2;
    } else if (tensor_component == 5) {
      first_index = second_index = 2;
    }
    int reflected_indices = (first_index == axis) + (second_index == axis)
        + (derivative_axis == axis);
    return reflected_indices % 2 == 1;
  }

  if (n >= pc_gh::PcGh::I_B11 && n <= pc_gh::PcGh::I_B33) {
    int const offset = n - pc_gh::PcGh::I_B11;
    int const derivative = offset/3;
    int const vector_component = offset % 3;
    return ((derivative == axis) + (vector_component == axis)) == 1;
  }
  return false;
}
}  // namespace pc_gh
#endif
