#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
namespace h=z4c::hyperboloidal;
using Z=z4c::Z4c;
void require(bool v,const char *why) {if(!v)throw std::runtime_error(why);}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    std::cout<<std::setprecision(17);
    const cubed_sphere::Map map{.15,1.,0.};
    h::HyperbolicShiftParameters<Real> parameters;
    parameters.bulk_mu=4;parameters.Validate();
    Real previous=0;
    // Use AthenaK's native component ordering/storage and complete J/H operator.
    // This is an operator test; it does not run Extension's evolution task graph.
    for(int resolution:{12,24,48}) {
      const Real dx=1./resolution;
      DvceArray5D<Real> state("gauge modes",6,Z::nz4c,7,7,7);
      DvceArray2D<Real> errors("gauge errors",6,3);
      par_for("Cartesian modes on patches",DevExeSpace(),0,5,0,6,0,6,0,6,
          KOKKOS_LAMBDA(int patch,int k,int j,int i) {
        Real q[3]={.45+(i-3)*dx,.13+(j-3)*dx,-.21+(k-3)*dx},x[3];
        map.Forward(patch,q,x);
        for(int n=0;n<Z::nz4c;++n)state(patch,n,k,j,i)=0;
        for(int a=0;a<3;++a) {
          // Constant plus linear modes about an OFF-CENTER origin.
          state(patch,Z::I_Z4C_BETAX+a,k,j,i)=.17*(a+1);
          for(int b=0;b<3;++b)
            state(patch,Z::I_Z4C_BETAX+a,k,j,i)+=(.1*(a+1)-.07*b)*(x[b]-.037*(b+1));
        }
        state(patch,Z::I_Z4C_ALPHA,k,j,i)=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
        state(patch,Z::I_Z4C_CHI,k,j,i)=1;
      });
      Kokkos::parallel_for("mapped Cartesian mode derivatives",6,KOKKOS_LAMBDA(int p) {
        cubed_sphere::Stencil s{};s.mapped=true;
        Real q[3]={.45,.13,-.21};s.p=map.Geometry(p,q);
        for(int a=0;a<3;++a)s.idx[a]=1/dx;
        Real error=0,constant=0;
        for(int a=0;a<3;++a) {
          constant=fmax(constant,fabs(Dx<3>(a,s,state,p,Z::I_Z4C_CHI,3,3,3)));
          error=fmax(error,fabs(Dx<3>(a,s,state,p,Z::I_Z4C_ALPHA,3,3,3)-s.p.x[a]));
          for(int b=0;b<3;++b) {
            error=fmax(error,fabs(Dx<3>(b,s,state,p,Z::I_Z4C_BETAX+a,3,3,3)-(.1*(a+1)-.07*b)));
            error=fmax(error,fabs(Dxy<3>(a,b,s,state,p,Z::I_Z4C_ALPHA,3,3,3)-(a==b ? 1:0)));
          }
        }
        // Analytic fixed Minkowski background: no measured residual subtraction.
        h::Z4cJet<Real> u{};u.chi.value=1;u.trace.value=-3;
        Real r2=0;for(int a=0;a<3;++a)r2+=s.p.x[a]*s.p.x[a];
        u.alpha.value=(1+r2)/2;
        for(int a=0;a<3;++a) {
          u.metric.g[a][a]=1;u.alpha.d[a]=s.p.x[a];u.alpha.dd[a][a]=1;
          u.beta.value[a]=-s.p.x[a];u.beta.d[a][a]=-1;
        }
        h::VectorJet<Real> B{};Real D[3]{};
        const auto gauge=h::HyperbolicGauge(s.p.x,u,B,D);
        Real equilibrium=fmax(fabs(gauge.lapse_regular),fabs(gauge.lapse_pole));
        for(int a=0;a<3;++a) equilibrium=fmax(equilibrium,fabs(gauge.shift_regular[a]));
        errors(p,0)=error;errors(p,1)=constant;errors(p,2)=equilibrium;
      });
      const auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),errors);
      Real error=0;
      for(int p=0;p<6;++p) {
        error=fmax(error,e(p,0));require(e(p,1)<1.e-12,"constant Cartesian field");
        require(e(p,2)<1.e-13,"Minkowski gauge source balance");
      }
      std::cout<<"resolution="<<resolution<<" derivative_error="<<error<<"\n";
      if(previous)require(previous/error>12,"mapped mode fourth-order convergence");
      previous=error;
    }
    // Actual Omega=0 traces on every face. No Omega floor or finite-radius BC.
    for(int p=0;p<6;++p)for(Real u:{-1.,-.3,1.})for(Real v:{-1.,.2,1.}) {
      Real q[3]={1,u,v},x[3];map.Forward(p,q,x);
      Real metric[3][3]={{1.2,.1,0},{.1,.9,.05},{0,.05,1.1}};
      Real trial[3]={.2,-.1,.3},metric_t[3][3]={{.2,.03,0},{.03,-.1,0},{0,0,.07}};
      auto b=h::ReconstructScriShift(Real(0),Real(1),metric,x,trial);
      Real source[3]={.1,.2,.3},D[3]={.3,-.2,.4};
      require(h::ReconstructScriAuxiliary(b,Real(.03),metric_t,source,Real(1.5),D),"scri validity");
      Real normal=0,rate=0;
      for(int a=0;a<3;++a) {
        normal+=b.beta[a]*b.covariant[a];
        rate+=b.covariant[a]*(source[a]+1.5*D[a]);
      }
      require(fabs(normal+1)<2.e-14,"null scri trace");
      require(fabs(rate-b.normal_rate)<2.e-14,"dependent auxiliary trace");
    }
    std::cout<<"54 face/edge/corner closure identities passed; no evolution claim\n";
    // Translation at the coordinate origin is allowed and is not silently set
    // to zero by a spherical parity rule. Test independent connection transport.
    h::Z4cJet<Real> u{};u.alpha.value=.5;u.chi.value=1;u.trace.value=-3;
    h::VectorJet<Real> B{},D{};Real x[3]{},lambda_rhs[3]={.7,-.3,.2},rate[3]{};
    for(int i=0;i<3;++i) {
      u.metric.g[i][i]=1;u.beta.value[i]=.1*(i+1);u.beta.d[i][i]=-1;
      D.value[i]=.03*(i+1);u.lambda.value[i]=9*(i+1);
      for(int j=0;j<3;++j){D.d[j][i]=.02*(j+1);u.lambda.d[j][i]=.04*(i+1);}
    }
    const auto gauge=h::HyperbolicGauge(x,u,B,D.value,parameters);
    require(gauge.valid,"translated origin gauge validity");
    require(h::HyperbolicAuxiliaryRate(u,D,lambda_rhs,rate),"independent auxiliary rate");
    for(int i=0;i<3;++i) {
      require(fabs(gauge.shift_regular[i]+u.beta.value[i]-4*D.value[i])<1.e-14,"constant shift mode retained");
      Real expected=lambda_rhs[i]-D.value[i];
      for(int j=0;j<3;++j)expected+=u.beta.value[j]*(D.d[j][i]-u.lambda.d[j][i]);
      require(fabs(rate[i]-expected)<1.e-14,"independent evolved connection RHS");
      require(u.lambda.value[i]==9*(i+1),"connection overwritten");
    }
    std::cout<<"constant translation at origin and independent connection transport passed\n";
  }
  Kokkos::finalize();
}
