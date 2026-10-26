// Materialize the actual compiled radial scalar operators, without evolution.
// The first six rows use zero ghost extension; the right boundary is the
// primary solver's actual closure. This is not a multipatch energy proof.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
int main(int argc,char **argv) {
  if(argc!=2)return 1;
  Kokkos::initialize(argc,argv);
  {
    constexpr int n=65;const double h=1./(n-1);
    Kokkos::View<double*[5],Kokkos::LayoutRight,Kokkos::HostSpace> u("operator basis",n+2*NG);
    std::ofstream out(argv[1]);out<<std::setprecision(17)<<"row,col,Dc,Dplus,Dminus,Q\n";
    for(int j=0;j<n;++j) {
      Kokkos::deep_copy(u,0.);u(NG+j,0)=1;
      for(int i=0;i<n;++i) {
        KernelCell c;c.stride[0]=1;c.idx[0]=1/h;c.normal_upwind=c.normal_sbp=c.radial_sbp63=1;c.radial_end=n-1-i;
        if(c.radial_end<6){c.outer_distance=c.radial_end;c.outer_norm=radial_sbp::H[c.radial_end];}
        int l=NG+i;out<<i<<","<<j<<","<<deriv(u,l,0,0,c)<<","<<upwind(u,l,0,0,1,c)<<","<<upwind(u,l,0,0,-1,c)<<","<<radial_sbp_dissipation(u,l,0,c)<<"\n";
      }
    }
  }
  Kokkos::finalize();return 0;
}
