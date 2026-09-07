// Complete 50-field intrinsic candidate point kernel; no mesh/RK integration.
#ifndef PC_GH_INTRINSIC_RHS_HPP_
#define PC_GH_INTRINSIC_RHS_HPP_
#include "pc_gh/intrinsic_sources.hpp"
namespace pc_gh::intrinsic {
KOKKOS_INLINE_FUNCTION
void PointRHS(const double u[nvar], const double du[3][nvar], double lambda,
              double eta, double kappa, double rhs[nvar]) {
  Geometry<double> geo; BuildGeometry(u,geo);
  auto &g=geo.metric; auto &gu=geo.inverse_metric; auto &a=geo.curvature;
  auto &q=geo.gradient;
  const double w=u[W], rho=u[RHO], alpha=w*rho, kval=u[K], c=u[C];
  const double theta=u[B]+u[B+4]+u[B+8];
  double dg[3][3][3], dgu[3][3][3], da[3][3][3], dq[3][3][3][3];
  double df[3][10], f[10], ell[3][3];
  ConfigurationSources(u,eta,f,ell);
  for (int k=0;k<3;++k) {
    Jet state[nvar];
    for (int n=0;n<nvar;++n) state[n]={u[n],du[k][n]};
    Geometry<Jet> jet; BuildGeometry(state,jet);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      dg[k][i][j]=jet.metric[i][j].derivative;
      dgu[k][i][j]=jet.inverse_metric[i][j].derivative;
      da[k][i][j]=jet.curvature[i][j].derivative;
      for (int r=0;r<3;++r) dq[k][r][i][j]=jet.gradient[r][i][j].derivative;
    }
    double dell[3][3]; DifferentiateConfiguration(u,du[k],eta,df[k],dell);
  }
  double low[3][3][3], gam[3][3][3]={}, dlambda[3][3];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
    low[i][j][k]=(q[j][i][k]+q[k][i][j]-q[i][j][k])/2;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) for (int k=0;k<3;++k)
    for (int r=0;r<3;++r) gam[i][j][k]+=gu[i][r]*low[r][j][k];
  for (int d=0;d<3;++d) for (int i=0;i<3;++i) {
    dlambda[d][i]=-du[d][Z+i];
    for (int j=0;j<3;++j) for (int k=0;k<3;++k) {
      dlambda[d][i]+=dgu[d][j][k]*gam[i][j][k];
      for (int r=0;r<3;++r)
        dlambda[d][i]+=gu[j][k]*(dgu[d][i][r]*low[r][j][k]
            +gu[i][r]*(dq[d][j][r][k]+dq[d][k][r][j]-dq[d][r][j][k])/2);
    }
  }
  double ricci[3][3]={}, cp[3][3], cl[3][3], amix[3][3]={};
  double aup[3][3]={}, zcov[3]={}, ap[3]={}, aM[3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    cp[i][j]=du[i][P+j]; cl[i][j]=du[i][LAPSE_GRADIENT+j];
    zcov[i]+=g[i][j]*u[Z+j]; ap[i]+=gu[i][j]*u[P+j];
    for (int k=0;k<3;++k) {
      cp[i][j]-=gam[k][i][j]*u[P+k]; cl[i][j]-=gam[k][i][j]*u[LAPSE_GRADIENT+k];
      amix[i][j]+=gu[i][k]*a[k][j];
      ricci[i][j]+=(g[k][i]*dlambda[j][k]+g[k][j]*dlambda[i][k])/2;
      for (int l=0;l<3;++l) {
        aup[i][j]+=gu[i][k]*a[k][l]*gu[j][l];
        ricci[i][j]-=gu[k][l]*dq[k][l][i][j]/2;
        for (int m=0;m<3;++m)
          ricci[i][j]+=gu[k][l]*(gam[m][k][l]*(low[i][j][m]+low[j][i][m])/2
              +gam[m][k][i]*low[j][m][l]+gam[m][k][j]*low[i][m][l]
              +gam[m][i][k]*low[m][j][l]);
      }
    }
  }
  double a2=0, h=2*kval*kval/3, ntrace=0, pdotl=0, pdotz=0, zdamping=0;
  for (int i=0;i<3;++i) {
    h-=6*u[P+i]*ap[i]; pdotl+=ap[i]*u[LAPSE_GRADIENT+i];
    pdotz+=u[P+i]*u[Z+i]; zdamping+=(rho*u[P+i]+u[LAPSE_GRADIENT+i])*u[Z+i];
    for (int j=0;j<3;++j) {
      a2+=a[i][j]*aup[i][j]; h+=w*w*gu[i][j]*ricci[i][j]+4*w*gu[i][j]*cp[i][j];
      ntrace+=gu[i][j]*cl[i][j];
    }
  }
  h-=a2;
  for (int i=0;i<3;++i) {
    double momentum=-2*du[i][K]/3, contraction=0;
    for (int j=0;j<3;++j) {
      contraction+=amix[j][i]*u[P+j];
      for (int k=0;k<3;++k)
        momentum+=dgu[j][j][k]*a[k][i]+gu[j][k]*da[j][k][i]
            +gam[j][j][k]*amix[k][i]-gam[k][j][i]*amix[j][k];
    }
    aM[i]=alpha*momentum-3*rho*contraction;
  }
  double st[3][3], tt[3][3], trst=0, trtt=0, fa[3][3], ahdot[3][3]={};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    st[i][j]=alpha*w*w*ricci[i][j]+alpha*w*(cp[i][j]+cp[j][i])/2
        -w*w*(cl[i][j]+cl[j][i])/2
        -w*(u[LAPSE_GRADIENT+i]*u[P+j]+u[LAPSE_GRADIENT+j]*u[P+i]);
    tt[i][j]=-w*(zcov[i]*u[P+j]+zcov[j]*u[P+i]);
    for (int k=0;k<3;++k) tt[i][j]-=w*w*u[Z+k]*q[k][i][j]/2;
    trst+=gu[i][j]*st[i][j]; trtt+=gu[i][j]*tt[i][j];
  }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    fa[i][j]=st[i][j]-g[i][j]*trst/3+alpha*(tt[i][j]-g[i][j]*trtt/3)
        -(2.0/3.0)*theta*a[i][j]+alpha*(kval-c)*a[i][j];
    for (int k=0;k<3;++k)
      fa[i][j]+=u[B+3*i+k]*a[k][j]+a[i][k]*u[B+3*j+k]-2*alpha*a[i][k]*amix[k][j];
  }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    for (int r=0;r<3;++r) {
      ahdot[i][j]-=ell[i][r]*geo.ahat[r][j]+geo.ahat[i][r]*ell[j][r];
      for (int s=0;s<3;++s) ahdot[i][j]+=geo.inverse_tri[i][r]*fa[r][s]*geo.inverse_tri[j][s];
    }
  }
  for (int n=0;n<nvar;++n) {
    rhs[n]=0; for (int k=0;k<3;++k) rhs[n]+=u[BETA+k]*du[k][n];
  }
  rhs[W]+=f[0]; rhs[RHO]+=rho*(-2*kval-(alpha*kval-theta)/3);
  for (int n=0;n<5;++n) rhs[CHART+n]+=f[2+n];
  for (int n=0;n<3;++n) rhs[BETA+n]+=f[7+n];
  rhs[K]+=alpha*a2+alpha*kval*kval/3-w*w*ntrace+w*pdotl
      +alpha*(h-kval*c)+alpha*w*pdotz-1.5*kappa*alpha*c;
  rhs[C]+=alpha*(h-kval*c)+w*w*zdamping-2*kappa*alpha*c;
  rhs[AHAT]+=ahdot[0][0]; rhs[AHAT+1]+=ahdot[0][1]; rhs[AHAT+2]+=ahdot[0][2];
  rhs[AHAT+3]+=ahdot[1][1]; rhs[AHAT+4]+=ahdot[1][2];
  for (int i=0;i<3;++i) {
    rhs[Z+i]+=(2*theta/3-(2*alpha*kval/3+kappa*alpha))*u[Z+i];
    for (int j=0;j<3;++j) {
      double curl=0;
      for (int d=0;d<3;++d) curl+=du[d][B+3*j+d]-du[j][B+3*d+d];
      rhs[Z+i]+=gu[i][j]*(-2*aM[j]-alpha*du[j][C]+c*u[LAPSE_GRADIENT+j]+curl)
          -u[Z+j]*u[B+3*j+i];
    }
  }
  for (int i=0;i<3;++i) for (int family=0;family<10;++family) {
    int offset=family==0?P+i : family==1?LAPSE_GRADIENT+i
        : family<7?S+5*i+family-2 : B+3*i+family-7;
    double dpot=family==0?du[i][W] : family==1?rho*du[i][W]+w*du[i][RHO]
        : family<7?du[i][CHART+family-2] : du[i][BETA+family-7];
    rhs[offset]+=df[i][family]-lambda*(u[offset]-dpot);
    for (int j=0;j<3;++j) {
      int other=family==0?P+j : family==1?LAPSE_GRADIENT+j
          : family<7?S+5*j+family-2 : B+3*j+family-7;
      rhs[offset]+=du[i][BETA+j]*u[other];
    }
  }
}
}  // namespace pc_gh::intrinsic
#endif
