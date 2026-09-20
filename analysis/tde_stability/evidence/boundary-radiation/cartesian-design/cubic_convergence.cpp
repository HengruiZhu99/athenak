#include <cmath>
#include <iostream>
#include <limits>
#ifndef HEADER_UNDER_TEST
#define HEADER_UNDER_TEST "z4c/z4c_constraint_radiation.hpp"
#endif
#include HEADER_UNDER_TEST
using z4c::Z4c;
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    std::cout.precision(16);std::cout<<"[";
    for(int n:{9,17,33,65}) {
      RegionIndcs in{};in.ng=2;in.nx1=n;in.nx2=in.nx3=5;
      in.is=in.js=in.ks=2;in.ie=n+1;in.je=in.ke=6;
      DvceArray5D<Real> bg("bg",1,25,9,9,n+4),u("u",1,25,9,9,n+4),rhs("rhs",1,25,9,9,n+4);
      const Real nan=std::numeric_limits<Real>::quiet_NaN();
      Kokkos::deep_copy(bg,nan);Kokkos::deep_copy(u,nan);Kokkos::deep_copy(rhs,nan);
      const Real h=1./(n-1),idx[3]={1/h,1/h,1/h};
      for(int k=2;k<=6;++k)for(int j=2;j<=6;++j)for(int i=2;i<=n+1;++i) {
        for(int f=0;f<25;++f)bg(0,f,k,j,i)=u(0,f,k,j,i)=rhs(0,f,k,j,i)=0;
        for(int f:{Z4c::I_Z4C_CHI,Z4c::I_Z4C_ALPHA,Z4c::I_Z4C_GXX,Z4c::I_Z4C_GYY,Z4c::I_Z4C_GZZ})
          bg(0,f,k,j,i)=u(0,f,k,j,i)=1;
        const Real x=1+(i-2)*h;u(0,Z4c::I_Z4C_GXX,k,j,i)=1+.05*x*x*x;
      }
      const Real x=2,a=1+.05*x*x*x,ap=.15*x*x,app=.3*x;
      const Real z=-ap/(4*a),zp=-(app*a-ap*ap)/(4*a*a);
      const Real exact=2/(a*std::sqrt(a))*(zp+z/(x+1));
      const Real xyz[3]={x,0,0},nd[3]={std::sqrt(a),0,0},nu[3]={1/std::sqrt(a),0,0};
      const int side[3]={1,0,0};Z4c::Options opt{};opt.characteristic_radiation_areal_shift=1;
      Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,rhs,0,4,4,n+1,in,side,idx,nd,nu,xyz,opt,ft,fq);
      if(n!=9)std::cout<<",";
      std::cout<<"{\"nx\":"<<n<<",\"h\":"<<h<<",\"status\":"<<status<<",\"FQ_x\":"<<fq[0]<<",\"exact\":"<<exact<<",\"error\":"<<std::abs(fq[0]-exact)<<"}";
    }
    std::cout<<"]\n";
  }
  Kokkos::finalize();return 0;
}
