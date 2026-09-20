#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "z4c/z4c_boundary_stencil.hpp"

// Original helpers retained as a regression witness from production8b694211.
// Only names are changed. Its physical-side selection reads unfilled RHS
// ghosts at an internal tangential block edge when the raised normal is oblique.
KOKKOS_INLINE_FUNCTION
Real OriginalCoordinateDerivative(const DvceArray5D<Real> &u, int m, int n,
                          int k, int j, int i, int dir, int side,
                          const Real idx[3]) {
  if (side != 0) {
    const int inward = -side;
    Real f0 = u(m,n,k,j,i);
    Real f1;
    Real f2;
    if (dir == 0) {
      f1 = u(m,n,k,j,i + inward);
      f2 = u(m,n,k,j,i + 2*inward);
    } else if (dir == 1) {
      f1 = u(m,n,k,j + inward,i);
      f2 = u(m,n,k,j + 2*inward,i);
    } else {
      f1 = u(m,n,k + inward,j,i);
      f2 = u(m,n,k + 2*inward,j,i);
    }
    return 0.5*side*idx[dir]*(3.0*f0 - 4.0*f1 + f2);
  }

  if (dir == 0) {
    return 0.5*idx[0]*(u(m,n,k,j,i+1) - u(m,n,k,j,i-1));
  } else if (dir == 1) {
    return 0.5*idx[1]*(u(m,n,k,j+1,i) - u(m,n,k,j-1,i));
  }
  return 0.5*idx[2]*(u(m,n,k+1,j,i) - u(m,n,k-1,j,i));
}

KOKKOS_INLINE_FUNCTION
Real OriginalNormalDerivative(const DvceArray5D<Real> &u, int m, int n,
                      int k, int j, int i, const Real normal_u[3],
                      const int side[3], const Real idx[3]) {
  Real derivative = 0.0;
  for (int a = 0; a < 3; ++a) {
    if (fabs(normal_u[a]) > 1.0e-12) {
      derivative += normal_u[a] *
          OriginalCoordinateDerivative(u, m, n, k, j, i, a, side[a], idx);
    }
  }
  return derivative;
}


Real FixedNormal(const DvceArray5D<Real>& u,int k,int j,int i,
                 const Real n[3],const RegionIndcs& in,const Real idx[3]) {
  Real d=0;for(int a=0;a<3;++a)if(std::abs(n[a])>1e-12)
    d+=n[a]*z4c::BoundaryCoordinateDerivative2(u,0,0,k,j,i,a,in,idx);
  return d;
}
void Normal(const int side[3],Real n[3]) {
  // SPD determinant-one full conformal metric with three off-diagonal entries.
  const Real g[3][3]={{1,.1,.04},{.1,1,-.02},{.04,-.02,1}};
  const Real det=.98784,scale=std::pow(det,-1./3.);
  Real inv[3][3];
  for(int a=0;a<3;++a)for(int b=0;b<3;++b) {
    const int c=(a+1)%3,d=(a+2)%3,e=(b+1)%3,f=(b+2)%3;
    inv[b][a]=(g[c][e]*g[d][f]-g[c][f]*g[d][e])/(det*scale);
  }
  Real norm2=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)
    norm2+=side[a]*inv[a][b]*side[b];
  for(int a=0;a<3;++a){n[a]=0;for(int b=0;b<3;++b)n[a]+=inv[a][b]*side[b]/std::sqrt(norm2);}
}
void Coordinates(int k,int j,int i,Real x[3]) {
  x[0]=(i-3.5)/4;x[1]=(j-3.5)/8;x[2]=(k-3.5)/16;
}
Real Polynomial(const Real x[3]) {
  return x[0]*x[0]+2*x[1]*x[1]-3*x[2]*x[2]+.5*x[0]*x[1]-.25*x[0]*x[2]+.125*x[1]*x[2]+.2*x[0]-.3*x[1]+.4*x[2];
}
void Gradient(const Real x[3],Real d[3]) {
 d[0]=2*x[0]+.5*x[1]-.25*x[2]+.2;
 d[1]=4*x[1]+.5*x[0]+.125*x[2]-.3;
 d[2]=-6*x[2]-.25*x[0]+.125*x[1]+.4;
}
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);bool passed=true;
 {
  RegionIndcs in{};in.ng=4;in.nx1=in.nx2=in.nx3=8;
  in.is=in.js=in.ks=4;in.ie=in.je=in.ke=11;
  const Real idx[3]={4,8,16},nan=std::numeric_limits<Real>::quiet_NaN();
  DvceArray5D<Real> state("state",1,1,16,16,16),rhs("rhs",1,1,16,16,16),sum("sum",1,1,16,16,16),zero("zero",1,1,16,16,16);
  for(auto u:{state,rhs,sum,zero})Kokkos::deep_copy(u,nan);
  for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
   Real x[3];Coordinates(k,j,i,x);state(0,0,k,j,i)=Polynomial(x);
   rhs(0,0,k,j,i)=.3*x[0]*x[0]-.2*x[1]*x[2]+.5*x[2];
   sum(0,0,k,j,i)=state(0,0,k,j,i)+.25*rhs(0,0,k,j,i);zero(0,0,k,j,i)=0;
  }
  Real coordinate_error=0,normal_error=0,linearity_error=0,zero_error=0;
  int coordinate_tests=0,normal_tests=0,old_nonfinite=0,new_nonfinite=0,nonzero_response=0;
  for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
   Real x[3],g[3];Coordinates(k,j,i,x);Gradient(x,g);
   for(int a=0;a<3;++a) {
    auto got=z4c::BoundaryCoordinateDerivative2(state,0,0,k,j,i,a,in,idx);
    passed&=std::isfinite(got);coordinate_error=std::max(coordinate_error,std::abs(got-g[a]));++coordinate_tests;
   }
  }
  // All six faces, twelve physical edges, eight corners; also probe both
  // INTERNAL tangential block edges on every physically unconstrained axis.
  for(int sz=-1;sz<=1;++sz)for(int sy=-1;sy<=1;++sy)for(int sx=-1;sx<=1;++sx) {
   if(!(sx||sy||sz))continue;
   const int side[3]={sx,sy,sz};Real n[3];Normal(side,n);
   for(int kk:{4,8,11})for(int jj:{4,8,11})for(int ii:{4,8,11}) {
    const int pos[3]={ii,jj,kk};bool use=true;
    for(int a=0;a<3;++a)if(side[a])use&=pos[a]==(side[a]<0?4:11);
    if(!use)continue;
    Real x[3],g[3];Coordinates(kk,jj,ii,x);Gradient(x,g);
    Real exact=0;for(int a=0;a<3;++a)exact+=n[a]*g[a];
    const Real got=FixedNormal(state,kk,jj,ii,n,in,idx);
    const Real old=OriginalNormalDerivative(state,0,0,kk,jj,ii,n,side,idx);
    old_nonfinite+=!std::isfinite(old);new_nonfinite+=!std::isfinite(got);
    normal_error=std::max(normal_error,std::abs(got-exact));nonzero_response+=std::abs(got)>1e-10;
    zero_error=std::max(zero_error,std::abs(FixedNormal(zero,kk,jj,ii,n,in,idx)));
    const Real rate=FixedNormal(rhs,kk,jj,ii,n,in,idx);
    const Real combined=FixedNormal(sum,kk,jj,ii,n,in,idx);
    linearity_error=std::max(linearity_error,std::abs(combined-got-.25*rate));
    ++normal_tests;
   }
  }
  // An inactive direction must not inspect even its center or adjacent ghosts.
  DvceArray5D<Real> poison("poison",1,1,16,16,16);Kokkos::deep_copy(poison,nan);
  int inactive_passed=0;
  for(int a=0;a<3;++a) {
   RegionIndcs inactive=in;
   if(a==0)inactive.ie=inactive.is;else if(a==1)inactive.je=inactive.js;else inactive.ke=inactive.ks;
   const Real d=z4c::BoundaryCoordinateDerivative2(poison,0,0,4,4,4,a,inactive,idx);
   inactive_passed+=d==0 && std::isfinite(d);
  }
  // Exactly flat normal hides the old bug despite the same poisoned ghosts.
  const int side[3]={1,0,0};const Real flat[3]={1,0,0};
  const bool flat_masks_bug=std::isfinite(OriginalNormalDerivative(state,0,0,8,4,11,flat,side,idx));
  passed &= inactive_passed==3 && coordinate_tests==1536 && normal_tests==98 && old_nonfinite==72 && new_nonfinite==0 && flat_masks_bug && nonzero_response==98 && zero_error==0 && coordinate_error<1e-11 && normal_error<1e-11 && linearity_error<1e-11;
  std::cout.precision(17);
  std::cout<<"{\"passed\":"<<(passed?"true":"false")<<",\"inactive_axes_passed\":"<<inactive_passed<<",\"coordinate_tests\":"<<coordinate_tests<<",\"normal_tests\":"<<normal_tests<<",\"original_nonfinite\":"<<old_nonfinite<<",\"fixed_nonfinite\":"<<new_nonfinite<<",\"flat_masks_bug\":"<<(flat_masks_bug?"true":"false")<<",\"nonzero_responses\":"<<nonzero_response<<",\"coordinate_error\":"<<coordinate_error<<",\"normal_error\":"<<normal_error<<",\"state_rhs_linearity_error\":"<<linearity_error<<",\"zero_error\":"<<zero_error<<"}\n";
 }
 Kokkos::finalize();return passed?0:1;
}
