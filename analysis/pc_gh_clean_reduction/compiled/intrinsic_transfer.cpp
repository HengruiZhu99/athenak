#include <fstream>
#include <iomanip>
#include <limits>
#include "pc_gh/intrinsic_transfer_target.hpp"

struct PrimaryPolynomial {
  int order,dimensions;
  KOKKOS_INLINE_FUNCTION
  int extent_int(int d) const {return (d==2 && dimensions==2)?1:16;}
  KOKKOS_INLINE_FUNCTION
  Real operator()(int,int n,int k,int j,int i) const {
    if (n<0 || n>=10 || i<0 || i>=16 || j<0 || j>=16 || k<0 || k>=extent_int(2))
      return std::numeric_limits<double>::quiet_NaN();
    double x=(i-3.5)*.125,y=(j-3.5)*.1625,z=dimensions==3?(k-3.5)*.2125:0;
    if (n==0) return 1+.01*(x+2*y+3*z);
    if (n==1) return 1.2+.02*(2*x-y+z);
    double px=1,py=1,pz=1;
    for (int r=0;r<order;++r) {px*=x;py*=y;pz*=z;}
    return .001*((n+1)*px+.5*(n+2)*py+.25*(n+3)*pz+.1*(n+1)*x*y);
  }
};
int main(int argc,char **argv) {
  if (argc!=4) return 2;
  std::ifstream input(argv[1]);int count;input>>count;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<int**> points("points",count,5);Kokkos::View<double**> results("targets",count,30);
    auto host=Kokkos::create_mirror_view(points);
    for (int r=0;r<count;++r) for (int n=0;n<5;++n) input>>host(r,n);
    if (!input) return 3;
    Kokkos::deep_copy(points,host);
    Kokkos::parallel_for("intrinsic polynomial halo targets",count,KOKKOS_LAMBDA(int r) {
      PrimaryPolynomial field{points(r,0),points(r,1)};Real idx[3]={8,1/.1625,1/.2125};
      for (int n=20;n<50;++n) {
        if (field.order==2) results(r,n-20)=pc_gh::intrinsic::IntrinsicTransferTarget<2>(field,0,n,points(r,2),points(r,3),points(r,4),idx);
        else if (field.order==4) results(r,n-20)=pc_gh::intrinsic::IntrinsicTransferTarget<4>(field,0,n,points(r,2),points(r,3),points(r,4),idx);
        else results(r,n-20)=pc_gh::intrinsic::IntrinsicTransferTarget<6>(field,0,n,points(r,2),points(r,3),points(r,4),idx);
      }
    });
    Kokkos::fence();auto values=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),results);
    std::ofstream out(argv[2]);out<<std::setprecision(17);
    for (int r=0;r<count;++r) {for (int n=0;n<30;++n) out<<(n?" ":"")<<values(r,n);out<<'\n';}
    std::ofstream config(argv[3]);Kokkos::print_configuration(config,true);
  }
  Kokkos::finalize();
}
