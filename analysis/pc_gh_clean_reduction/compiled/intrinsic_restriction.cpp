#include <iostream>
#include <iomanip>
#include <limits>
#include "pc_gh/intrinsic_restriction.hpp"
struct Polynomial {
  int nx,px,py;
  KOKKOS_INLINE_FUNCTION Real operator()(int,int,int,int j,int i) const {
    if (i<4 || i>=4+nx || j<4 || j>=4+nx)
      return std::numeric_limits<double>::quiet_NaN();
    Real value=1;
    for(int k=0;k<px;++k) value*=Real(i-4)/(nx-1);
    for(int k=0;k<py;++k) value*=Real(j-4)/(nx-1);
    return value;
  }
};
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);double worst=0;
  for(int nx : {6,8,16,32}) {
    double error=0;int half=nx/2;
    Kokkos::parallel_reduce("point restriction polynomial",36*half*half,
      KOKKOS_LAMBDA(int r,double &maximum) {
        int i=r%half,j=(r/half)%half,p=r/(half*half),px=p%6,py=p/6;
        Polynomial u{nx,px,py};int fi=4+2*i,fj=4+2*j;
        double value=pc_gh::intrinsic::PointRestrict2D(u,0,0,0,fj,fi,4,3+nx,4,3+nx);
        double exact=1;
        for(int k=0;k<px;++k) exact*=(fi+.5-4)/(nx-1);
        for(int k=0;k<py;++k) exact*=(fj+.5-4)/(nx-1);
        double diff=Kokkos::abs(value-exact);
        if (!(diff<=2e-12)) diff=1;
        if(diff>maximum) maximum=diff;
      },Kokkos::Max<double>(error));
    std::cout<<nx<<" "<<std::setprecision(17)<<error<<"\n";
    if(error>worst) worst=error;
  }
  Kokkos::finalize();return worst<=2e-12?0:1;
}
