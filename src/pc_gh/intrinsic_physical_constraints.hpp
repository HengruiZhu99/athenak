// AthenaK astrophysical plasma code, 3-clause BSD License (LICENSE).
#ifndef PC_GH_INTRINSIC_PHYSICAL_CONSTRAINTS_HPP_
#define PC_GH_INTRINSIC_PHYSICAL_CONSTRAINTS_HPP_
#include "athena.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh::intrinsic {
// Diagnostic materialization: w,rho,K,g[9],A[9]. There are no independent
// reductions or GH fields in this input. Output: H,M[3],alpha*M[3].
// Direct Dxx and tensor-product Dxy require radius Stencil-1 per direction;
// unlike repeated derivatives of Christoffels, FD6 fits four valid ghosts.
template<int Stencil, typename State>
KOKKOS_INLINE_FUNCTION
void PhysicalConstraints(State field, int m, int k, int j, int i,
                         const Real idx[3], int dimensions, double out[7]) {
  double g[3][3],gu[3][3],a[3][3],dg[3][3][3]={},ddg[3][3][3][3]={};
  double da[3][3][3]={},dw[3]={},ddw[3][3]={},dk[3]={};
  double w=field(m,0,k,j,i),rho=field(m,1,k,j,i),kval=field(m,2,k,j,i);
  for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
    int n=3+3*b+c;g[b][c]=field(m,n,k,j,i);a[b][c]=field(m,n+9,k,j,i);
    for (int d=0;d<dimensions;++d) {
      dg[d][b][c]=Dx<Stencil>(d,idx,field,m,n,k,j,i);
      da[d][b][c]=Dx<Stencil>(d,idx,field,m,n+9,k,j,i);
      for (int e=0;e<dimensions;++e)
        ddg[d][e][b][c]=d==e ? Dxx<Stencil>(d,idx,field,m,n,k,j,i)
          : Dxy<Stencil>(d,e,idx,field,m,n,k,j,i);
    }
  }
  double det=g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
    -g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
    +g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
  for (int b=0;b<3;++b) for (int c=0;c<3;++c)
    gu[b][c]=(g[(c+1)%3][(b+1)%3]*g[(c+2)%3][(b+2)%3]
             -g[(c+1)%3][(b+2)%3]*g[(c+2)%3][(b+1)%3])/det;
  for (int d=0;d<dimensions;++d) {
    dw[d]=Dx<Stencil>(d,idx,field,m,0,k,j,i);
    dk[d]=Dx<Stencil>(d,idx,field,m,2,k,j,i);
    for (int e=0;e<dimensions;++e)
      ddw[d][e]=d==e ? Dxx<Stencil>(d,idx,field,m,0,k,j,i)
        : Dxy<Stencil>(d,e,idx,field,m,0,k,j,i);
  }
  double dgu[3][3][3]={},gam[3][3][3]={},dgam[3][3][3][3]={},mix[3][3]={};
  for (int d=0;d<3;++d) for (int b=0;b<3;++b) for (int c=0;c<3;++c)
    for (int r=0;r<3;++r) for (int s=0;s<3;++s)
      dgu[d][b][c]-=gu[b][r]*dg[d][r][s]*gu[s][c];
  for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
    for (int r=0;r<3;++r) mix[b][c]+=gu[b][r]*a[r][c];
    for (int d=0;d<3;++d) for (int r=0;r<3;++r) {
      double low=0.5*(dg[c][r][d]+dg[d][r][c]-dg[r][c][d]);
      gam[b][c][d]+=gu[b][r]*low;
      for (int e=0;e<3;++e)
        dgam[e][b][c][d]+=dgu[e][b][r]*low+0.5*gu[b][r]
          *(ddg[e][c][r][d]+ddg[e][d][r][c]-ddg[e][r][c][d]);
    }
  }
  out[0]=2.0/3.0*kval*kval;
  for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
    double ricci=0,hessian=ddw[b][c];
    for (int d=0;d<3;++d) {
      ricci+=dgam[d][d][b][c]-dgam[c][d][b][d];
      hessian-=gam[d][b][c]*dw[d];
      for (int e=0;e<3;++e)
        ricci+=gam[d][d][e]*gam[e][b][c]-gam[d][c][e]*gam[e][b][d];
    }
    out[0]+=-mix[b][c]*mix[c][b]+gu[b][c]*(w*w*ricci+4*w*hessian-6*dw[b]*dw[c]);
  }
  for (int b=0;b<3;++b) {
    double momentum=-2.0/3.0*dk[b],contraction=0;
    for (int c=0;c<3;++c) {
      contraction+=mix[c][b]*dw[c];
      for (int d=0;d<3;++d) {
        momentum+=dgu[c][c][d]*a[d][b]+gu[c][d]*da[c][d][b];
        momentum+=gam[c][c][d]*mix[d][b]-gam[d][c][b]*mix[c][d];
      }
    }
    out[1+b]=momentum-3*contraction/w;
    out[4+b]=rho*w*momentum-3*rho*contraction;
  }
}
}  // namespace pc_gh::intrinsic
#endif  // PC_GH_INTRINSIC_PHYSICAL_CONSTRAINTS_HPP_
