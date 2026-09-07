#include <fstream>
#include <iomanip>
#include <stdexcept>
#include "pc_gh/intrinsic_finite_difference.hpp"

struct PlaneWave {
  int column;
  double amplitude, phase, theta[3];
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int n, int k, int j, int i) const {
    return (n < 2 ? 1.0 : 0.0) + (n == column ? amplitude
           *cos(phase+theta[0]*i+theta[1]*j+theta[2]*k) : 0.0);
  }
};
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  std::ifstream file(argv[1]); int count; file >> count;
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double**> input("input",count,15), output("output",count,50);
    auto host=Kokkos::create_mirror_view(input);
    for (int r=0;r<count;++r) for (int n=0;n<15;++n) file >> host(r,n);
    if (!file) throw std::runtime_error("incomplete stencil input");
    Kokkos::deep_copy(input,host);
    Kokkos::parallel_for("intrinsic FD plane wave",count,KOKKOS_LAMBDA(int r) {
      PlaneWave field{static_cast<int>(input(r,2)),input(r,3),input(r,4),
                      {input(r,5),input(r,6),input(r,7)}};
      Real idx[3]={input(r,8),input(r,9),input(r,10)};double rhs[50];
      int dimensions=static_cast<int>(input(r,1));
      if (input(r,0)==2) pc_gh::intrinsic::FiniteDifferenceRHS<2>(field,0,0,0,0,idx,dimensions,input(r,11),input(r,12),input(r,13),input(r,14),rhs);
      else if (input(r,0)==4) pc_gh::intrinsic::FiniteDifferenceRHS<3>(field,0,0,0,0,idx,dimensions,input(r,11),input(r,12),input(r,13),input(r,14),rhs);
      else pc_gh::intrinsic::FiniteDifferenceRHS<4>(field,0,0,0,0,idx,dimensions,input(r,11),input(r,12),input(r,13),input(r,14),rhs);
      for (int n=0;n<50;++n) output(r,n)=rhs[n];
    });
    Kokkos::fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
    std::ofstream out(argv[2]);out << std::setprecision(17);
    for (int r=0;r<count;++r) {for (int n=0;n<50;++n) out << (n?" ":"") << result(r,n);out << '\n';}
    std::ofstream config(argv[3]);Kokkos::print_configuration(config,true);
  }
  Kokkos::finalize();
}
