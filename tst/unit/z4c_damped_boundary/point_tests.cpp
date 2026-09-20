#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "z4c/z4c_constraint_radiation.hpp"

using z4c::Z4c;
int main(int argc, char **argv) {
  Kokkos::initialize(argc,argv);
  int failures = 0;
  {
    RegionIndcs in{}; in.ng=4; in.nx1=in.nx2=in.nx3=8;
    in.is=in.js=in.ks=4; in.ie=in.je=in.ke=11;
    constexpr int n=16;
    DvceArray5D<Real> bg("bg",1,25,n,n,n), u("u",1,25,n,n,n);
    DvceArray5D<Real> rhs("rhs",1,25,n,n,n), zero("zero",1,25,n,n,n);
    DvceArray5D<Real> plus("plus",1,25,n,n,n), minus("minus",1,25,n,n,n);
    const Real nan=std::numeric_limits<Real>::quiet_NaN();
    Kokkos::deep_copy(bg,nan);Kokkos::deep_copy(u,nan);
    Kokkos::deep_copy(rhs,nan);Kokkos::deep_copy(zero,nan);
    const Real idx[3]={8,8,8};
    Z4c::Options opt{};opt.characteristic_radiation_areal_shift=1;
    opt.characteristic_radiation_areal_falloff=true; opt.damp_kappa1=0;
    Real zero_error=0,manufactured_error=0,time_error=0,matched_z_error=0;
    int calls=0,badstatus=0;
    auto xyz_at=[](int k,int j,int i,Real x[3]) {
      x[0]=1.5+(i-3.5)*.125;x[1]=-.5+(j-3.5)*.125;x[2]=-.5+(k-3.5)*.125;
    };
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
      for(int f=0;f<25;++f) bg(0,f,k,j,i)=u(0,f,k,j,i)=rhs(0,f,k,j,i)=zero(0,f,k,j,i)=0;
      for(int f:{Z4c::I_Z4C_CHI,Z4c::I_Z4C_ALPHA,Z4c::I_Z4C_GXX,Z4c::I_Z4C_GYY,Z4c::I_Z4C_GZZ})
        bg(0,f,k,j,i)=u(0,f,k,j,i)=1;
      Real x[3];xyz_at(k,j,i,x);
      for(int a=0;a<3;++a) bg(0,Z4c::I_Z4C_BETAX+a,k,j,i)=u(0,Z4c::I_Z4C_BETAX+a,k,j,i)=.03*(a+1);
    }
    const int points[3][3]={{4,4,4},{11,8,11},{11,11,11}};
    for(auto &p:points) {
      const int k=p[0],j=p[1],i=p[2];Real x[3];xyz_at(k,j,i,x);
      const int side[3]={1,1,1};const Real normal[3]={1/std::sqrt(3.),1/std::sqrt(3.),1/std::sqrt(3.)};
      Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,rhs,0,k,j,i,in,side,idx,normal,normal,x,opt,ft,fq);
      ++calls;badstatus+=status!=0;zero_error=std::max(zero_error,std::abs(ft));for(auto q:fq)zero_error=std::max(zero_error,std::abs(q));
    }
    // Flat-space polynomial physical constraints: all active derivative stencils
    // differentiate these quadratics exactly, including edges/corners.
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
      Real x[3];xyz_at(k,j,i,x);
      u(0,Z4c::I_Z4C_THETA,k,j,i)=.02*(x[0]*x[0]+x[1]+2*x[2]);
      rhs(0,Z4c::I_Z4C_THETA,k,j,i)=.004;
      for(int a=0;a<3;++a) {
        u(0,Z4c::I_Z4C_GAMX+a,k,j,i)=.01*(a+1)*(x[0]+x[1]*x[1]+3*x[2]);
        rhs(0,Z4c::I_Z4C_GAMX+a,k,j,i)=.003*(a+1);
      }
    }
    for(auto &p:points) {
      const int k=p[0],j=p[1],i=p[2];Real x[3];xyz_at(k,j,i,x);
      const int side[3]={1,1,1};const Real normal[3]={1/std::sqrt(3.),1/std::sqrt(3.),1/std::sqrt(3.)};
      Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,rhs,0,k,j,i,in,side,idx,normal,normal,x,opt,ft,fq);
      ++calls;badstatus+=status!=0;
      Real v[3],r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),omega=0;
      for(int a=0;a<3;++a){v[a]=normal[a]-.03*(a+1);omega+=v[a]*x[a]/(r*(r+1));}
      const Real exact=.004+.02*(2*x[0]*v[0]+v[1]+2*v[2])+omega*u(0,Z4c::I_Z4C_THETA,k,j,i);
      manufactured_error=std::max(manufactured_error,std::abs(ft-exact));
      for(int a=0;a<3;++a) {
        const Real eq=.003*(a+1)+.01*(a+1)*(v[0]+2*x[1]*v[1]+3*v[2])+omega*u(0,Z4c::I_Z4C_GAMX+a,k,j,i);
        manufactured_error=std::max(manufactured_error,std::abs(fq[a]-eq));
      }
    }
    // Nonconstant, non-unit-determinant metric and nontrivial metric RHS.
    // Finite-difference the physical Z functional itself to audit Gamma_t,
    // inverse-metric time terms and the lowering-metric product rule.
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
      Real x[3];xyz_at(k,j,i,x);
      for(int a=0;a<3;++a)for(int b=a;b<3;++b) {
        const int f=Z4c::I_Z4C_GXX+z4c::RadiationSymmetricOffset(a,b);
        u(0,f,k,j,i)=(a==b?1.:0.)+.005*(a+b+1)*(x[0]*x[1]+x[2]*x[2]);
        rhs(0,f,k,j,i)=.004*(a+1)*(b+1)*(x[0]*x[0]+x[1]*x[2]);
      }
    }
    const Real eps=1e-4;
    Kokkos::deep_copy(plus,u);Kokkos::deep_copy(minus,u);
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i)
      for(int f=0;f<25;++f){plus(0,f,k,j,i)+=eps*rhs(0,f,k,j,i);minus(0,f,k,j,i)-=eps*rhs(0,f,k,j,i);}
    for(auto &p:points) {
      const int k=p[0],j=p[1],i=p[2];Real x[3];xyz_at(k,j,i,x);
      const int side[3]={1,1,1};const Real normal[3]={1/std::sqrt(3.),1/std::sqrt(3.),1/std::sqrt(3.)};
      Real ft,fq[3],ft0,fq0[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,rhs,0,k,j,i,in,side,idx,normal,normal,x,opt,ft,fq);
      status+=z4c::ComputeConstraintRadiationResidual(u,bg,zero,0,k,j,i,in,side,idx,normal,normal,x,opt,ft0,fq0);
      Real zp[3],zm[3],g[3][3],gi[3][3],dg[3][3][3],gm[3];
      status+=!z4c::RadiationZ(plus,0,k,j,i,in,idx,zp);status+=!z4c::RadiationZ(minus,0,k,j,i,in,idx,zm);
      status+=!z4c::RadiationGeometry(u,0,k,j,i,in,idx,g,gi,dg,gm);
      ++calls;badstatus+=status!=0;
      for(int a=0;a<3;++a) {
        Real exact=0;for(int b=0;b<3;++b)exact+=gi[a][b]*(zp[b]-zm[b])/eps;
        time_error=std::max(time_error,std::abs(fq[a]-fq0[a]-exact));
      }
    }
    // A nontrivial metric with its evolved Gamma equal to the metric-defined
    // Gamma must not be mistaken for a Z perturbation. This checks Z/Theta
    // radiation only, not Hamiltonian/momentum or full evolution constraints.
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
      Real g[3][3],gi[3][3],dg[3][3][3],gm[3];
      z4c::RadiationGeometry(u,0,k,j,i,in,idx,g,gi,dg,gm);
      for(int a=0;a<3;++a)u(0,Z4c::I_Z4C_GAMX+a,k,j,i)=gm[a];
      z4c::RadiationGeometry(bg,0,k,j,i,in,idx,g,gi,dg,gm);
      for(int a=0;a<3;++a)bg(0,Z4c::I_Z4C_GAMX+a,k,j,i)=gm[a];
      u(0,Z4c::I_Z4C_THETA,k,j,i)=bg(0,Z4c::I_Z4C_THETA,k,j,i)=0;
    }
    for(auto &p:points) {
      const int k=p[0],j=p[1],i=p[2];Real x[3];xyz_at(k,j,i,x);
      const int side[3]={1,1,1};const Real normal[3]={1/std::sqrt(3.),1/std::sqrt(3.),1/std::sqrt(3.)};
      Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,zero,0,k,j,i,in,side,idx,normal,normal,x,opt,ft,fq);
      ++calls;badstatus+=status!=0;matched_z_error=std::max(matched_z_error,std::abs(ft));
      for(auto q:fq)matched_z_error=std::max(matched_z_error,std::abs(q));
      status=z4c::ComputeConstraintRadiationResidual(u,u,zero,0,k,j,i,in,side,idx,normal,normal,x,opt,ft,fq);
      ++calls;badstatus+=status!=0;zero_error=std::max(zero_error,std::abs(ft));
      for(auto q:fq)zero_error=std::max(zero_error,std::abs(q));
    }
    // Independently prescribed physical Q and Theta polynomials on a constant,
    // non-Euclidean conformal metric. Cover every signed face/edge/corner normal.
    // Q_n is contracted with the covector normal here; this independently tests
    // the helper's conversion through Z_i and its sqrt(chi) factor.
    const Real diagonal[3]={1.2,.8,1.0/.96};
    const Real alpha=.8,chi=.64,kappa=.1,sigma=alpha*kappa;
    opt.characteristic_radiation_areal_falloff=false;
    opt.damp_kappa1=kappa;
    for(int k=4;k<=11;++k)for(int j=4;j<=11;++j)for(int i=4;i<=11;++i) {
      Real x[3];xyz_at(k,j,i,x);
      for(int f=0;f<25;++f)u(0,f,k,j,i)=bg(0,f,k,j,i)=rhs(0,f,k,j,i)=0;
      u(0,Z4c::I_Z4C_ALPHA,k,j,i)=bg(0,Z4c::I_Z4C_ALPHA,k,j,i)=alpha;
      u(0,Z4c::I_Z4C_CHI,k,j,i)=bg(0,Z4c::I_Z4C_CHI,k,j,i)=chi;
      for(int a=0;a<3;++a) {
        const int f=Z4c::I_Z4C_GXX+z4c::RadiationSymmetricOffset(a,a);
        u(0,f,k,j,i)=bg(0,f,k,j,i)=diagonal[a];
        u(0,Z4c::I_Z4C_BETAX+a,k,j,i)=bg(0,Z4c::I_Z4C_BETAX+a,k,j,i)=.03*(a+1);
        u(0,Z4c::I_Z4C_GAMX+a,k,j,i)=.01*(a+1)*(x[0]+x[1]*x[1]+3*x[2]);
        rhs(0,Z4c::I_Z4C_GAMX+a,k,j,i)=.003*(a+1);
      }
      u(0,Z4c::I_Z4C_THETA,k,j,i)=.02*(x[0]*x[0]+x[1]+2*x[2]);
      rhs(0,Z4c::I_Z4C_THETA,k,j,i)=.004;
    }
    Real damped_error=0,nonzero_response=0;int orientation_count=0;
    for(int sx=-1;sx<=1;++sx)for(int sy=-1;sy<=1;++sy)for(int sz=-1;sz<=1;++sz) {
      if(sx==0&&sy==0&&sz==0)continue;
      const int side[3]={sx,sy,sz};
      const int i=sx<0?4:sx>0?11:8,j=sy<0?4:sy>0?11:8,k=sz<0?4:sz>0?11:8;
      Real x[3];xyz_at(k,j,i,x);Real nd[3],nu[3],norm=0;
      for(int a=0;a<3;++a)norm+=side[a]*side[a]/diagonal[a];
      for(int a=0;a<3;++a){nd[a]=side[a]/std::sqrt(norm);nu[a]=nd[a]/diagonal[a];}
      Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,rhs,0,k,j,i,in,side,idx,nd,nu,x,opt,ft,fq);
      ++calls;++orientation_count;badstatus+=status!=0;
      Real vel[3],qn=0;for(int a=0;a<3;++a){vel[a]=alpha*std::sqrt(chi)*nu[a]-.03*(a+1);qn+=nd[a]*u(0,Z4c::I_Z4C_GAMX+a,k,j,i);}
      const Real exact=.004+.02*(2*x[0]*vel[0]+vel[1]+2*vel[2])+sigma*(u(0,Z4c::I_Z4C_THETA,k,j,i)-.5*std::sqrt(chi)*qn);
      damped_error=std::max(damped_error,std::abs(ft-exact));nonzero_response=std::max(nonzero_response,std::abs(ft));
      for(int a=0;a<3;++a) {
        const Real eq=.003*(a+1)+.01*(a+1)*(vel[0]+2*x[1]*vel[1]+3*vel[2])+sigma*u(0,Z4c::I_Z4C_GAMX+a,k,j,i);
        damped_error=std::max(damped_error,std::abs(fq[a]-eq));nonzero_response=std::max(nonzero_response,std::abs(fq[a]));
      }
      // Equal, nonzero full/background data must subtract exactly at every
      // orientation. NaNs in all state/RHS ghosts remain throughout this test.
      status=z4c::ComputeConstraintRadiationResidual(u,u,zero,0,k,j,i,in,side,idx,nd,nu,x,opt,ft,fq);
      ++calls;badstatus+=status!=0;zero_error=std::max(zero_error,std::abs(ft));
      for(auto q:fq)zero_error=std::max(zero_error,std::abs(q));
    }
    failures=badstatus!=0||zero_error!=0||matched_z_error!=0||manufactured_error>1e-12||time_error>1e-8||damped_error>1e-12||nonzero_response<1e-3;
    std::cout<<"{\"calls\":"<<calls<<",\"status_failures\":"<<badstatus
             <<",\"zero_error\":"<<zero_error<<",\"manufactured_error\":"<<manufactured_error
             <<",\"metric_time_derivative_error\":"<<time_error<<",\"matched_metric_Gamma_Z_error\":"<<matched_z_error
             <<",\"damped_orientation_count\":"<<orientation_count<<",\"damped_manufactured_error\":"<<damped_error
             <<",\"nonzero_response\":"<<nonzero_response
             <<",\"rhs_ghosts_poisoned\":true,\"state_ghosts_poisoned\":true,\"passed\":"<<(failures?"false":"true")<<"}\n";
  }
  Kokkos::finalize();return failures;
}
