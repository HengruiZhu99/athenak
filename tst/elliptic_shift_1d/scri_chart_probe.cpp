// Independent formal-series substitution into the actual nonlinear tensor kernel.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "taylor.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
using T=scri_series::Taylor;
using scri_series::Dx;using scri_series::DivX;using scri_series::Integral;
namespace h=z4c::hyperboloidal;
struct Result {long double residue[3],H4,M3,T4,Z3,low_poles,clock_rate,chi_rate;};
Result Evaluate(long double pf,long double af1,long double ell0,long double gauge_scale,long double restoring=0) {
  T x;x.c[1]=1;T r=Kokkos::sqrt(1-2*x),r2=1-2*x;
  T C;C.c[0]=.02L*gauge_scale;C.c[1]=-.01L*gauge_scale;
  C.c[2]=.003L*gauge_scale;C.c[3]=.001L*gauge_scale;C.c[4]=.0002L*gauge_scale;
  T tau;tau.c[0]=.007L;tau.c[1]=.003L;
  T ell(ell0);ell.c[1]=.002L;
  T L;L.c[0]=.03L*gauge_scale;L.c[1]=.002L*gauge_scale;
  T Pf(pf);Pf.c[1]=.001L;T Af(.004L);Af.c[1]=af1;Af.c[2]=.0003L;
  T P0=Dx(C)/2-C/2-C*C/8;
  T E0=4*Dx(C)/3-5*C*C/6;
  T P1=(9*C*C*C+18*C*C-18*C*Dx(C)-20*Dx(C)+4*Dx(Dx(C)))/8;
  T chi=1+x*C,K=-3-2*x*C+x*x*P0+x*x*x*P1+x*x*x*x*Pf;
  T B=2*C/3+x*E0+x*x*Af,Arr=r2*B;
  T alpha=1-x+x*L,theta=x*x*x*x*tau,lambda=x*x*x*ell;
  // Local Taylor expansion of the exact nonlinear elliptic shift at scri:
  // (beta/r)_x=-3 alpha B/2, beta(0)=-1.
  T beta=r*Integral(-3*alpha*B/2,-1);
  T q[h::NFIELDS]{},d[h::NFIELDS]{},dd[h::NFIELDS]{};
  q[h::DCHI]=chi-1;q[h::ARR]=Arr;q[h::DK]=K-2*theta+3;
  q[h::THETA]=theta;q[h::LAMBDA]=lambda;q[h::DALPHA]=x*L;
  for(int f=0;f<h::NFIELDS;++f){d[f]=-r*Dx(q[f]);dd[f]=-Dx(q[f])+r2*Dx(Dx(q[f]));}
  h::CMCReference<T> ref{1,1};auto u=h::SphericalJet(r,q,d,dd,ref);
  u.beta=h::RadialVector(beta,-r*Dx(beta),-Dx(beta)+r2*Dx(Dx(beta)),r);
  h::OmegaJet<T> o{};o.omega=x;o.gradient[0]=-r;for(int j=0;j<3;++j)o.hessian[j][j]=-1;
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  auto parts=h::ConformalRHS(u,o,T(4),T(-.99L),true);if(!parts.valid)throw std::runtime_error("Invalid series kernel");
  const auto con=h::EvolvedConstraints(u,o);if(!con.valid)throw std::runtime_error("Invalid series constraints");
  auto assemble=[](const T&R,const T&S){return R+DivX(S);};
  T chi_t=assemble(parts.regular.chi,parts.pole.chi);
  T A_t=assemble(parts.regular.a[0][0],parts.pole.a[0][0]);
  T p_t=assemble(parts.regular.trace,parts.pole.trace);
  T theta_t=assemble(parts.regular.theta,parts.pole.theta);
  T lambda_t=assemble(parts.regular.lambda[0],parts.pole.lambda[0]);
  T Ct=DivX(chi_t);
  T P0t=Dx(Ct)/2-Ct/2-C*Ct/4;
  T P1t=((27*C*C+36*C-18*Dx(C))*Ct-(18*C+20)*Dx(Ct)+4*Dx(Dx(Ct)))/8;
  T Pnumer=p_t+2*theta_t+2*x*Ct-x*x*P0t-x*x*x*P1t;
  const auto gp=h::UnfactoredReferenceGauge(ref,ref.At(r,T(0),T(0)),u,
      h::GaugeParameters<T>{T(1),T(0),T(restoring),T(0)},false,true);
  T alpha_t=assemble(gp.regular.alpha,gp.pole.alpha);
  Result out{};out.clock_rate=alpha_t.c[0];out.chi_rate=chi_t.c[0];out.residue[0]=theta_t.c[3];out.residue[1]=lambda_t.c[2];out.residue[2]=Pnumer.c[3];
  out.H4=con.hamiltonian.c[4];out.M3=con.momentum[0].c[3];out.T4=theta.c[4];out.Z3=con.z4.z_covector[0].c[3];
  for(int i=0;i<4;++i)out.low_poles=std::max(out.low_poles,std::abs(con.hamiltonian.c[i]));
  for(int i=0;i<3;++i)out.low_poles=std::max(out.low_poles,std::abs(con.momentum[0].c[i]));
  out.low_poles=std::max(out.low_poles,std::abs(chi_t.c[0]));
  out.low_poles=std::max(out.low_poles,std::abs(parts.pole.a[0][0].c[0]));
  return out;
}
int main(int argc,char**argv){
  if(argc!=5 && argc!=6)return 2;
  auto z=Evaluate(std::stold(argv[1]),std::stold(argv[2]),std::stold(argv[3]),std::stold(argv[4]),argc==6?std::stold(argv[5]):0);
  std::cout<<std::setprecision(19)<<"{\"residue\":["<<z.residue[0]<<","<<z.residue[1]<<","<<z.residue[2]<<"],\"H4\":"<<z.H4<<",\"M3\":"<<z.M3<<",\"T4\":"<<z.T4<<",\"Z3\":"<<z.Z3<<",\"lower_weight_defect\":"<<z.low_poles<<",\"raw_clock_rate\":"<<z.clock_rate<<",\"raw_chi_rate\":"<<z.chi_rate<<"}\n";
}
