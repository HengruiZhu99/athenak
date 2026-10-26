#ifndef FROZEN_TEST_SCALAR_GEOMETRY_HPP_
#define FROZEN_TEST_SCALAR_GEOMETRY_HPP_
#include <array>
#include <cmath>
#include <stdexcept>
#include "mesh/cubed_sphere/map.hpp"
namespace test_scalar {
// First spatial jets, evaluated only during host-side geometry construction.
// No finite-difference derivatives of background coefficients enter evolution.
struct AD {
  long double f=0,d[3]{};
  AD()=default; AD(long double x):f(x) {}
  static AD var(long double x,int a) {AD z(x);z.d[a]=1;return z;}
};
inline AD operator+(AD a,AD b) {AD z(a.f+b.f);for(int i=0;i<3;++i)z.d[i]=a.d[i]+b.d[i];return z;}
inline AD operator-(AD a,AD b) {AD z(a.f-b.f);for(int i=0;i<3;++i)z.d[i]=a.d[i]-b.d[i];return z;}
inline AD operator-(AD a) {return AD(0)-a;}
inline AD operator*(AD a,AD b) {AD z(a.f*b.f);for(int i=0;i<3;++i)z.d[i]=a.d[i]*b.f+a.f*b.d[i];return z;}
inline AD power(AD a,long double p) {AD z(std::pow(a.f,p));for(int i=0;i<3;++i)z.d[i]=p*std::pow(a.f,p-1)*a.d[i];return z;}
inline AD operator/(AD a,AD b) {return a*power(b,-1);}
inline AD exp(AD a) {AD z(std::exp(a.f));for(int i=0;i<3;++i)z.d[i]=z.f*a.d[i];return z;}
inline AD smooth(AD z) {if(z.f<=0)return 0;if(z.f>=1)return 1;return z*z*z*(AD(10)+z*(AD(-15)+AD(6)*z));}
struct Parameters {
  double mass=0,spin=0,L=3,S=24,C=10,cap=1,excision=0,cfl=.18,ko=.1,reduction_damping=0,radial_stretch=0;
  int n=12; bool characteristic_transfer=false,smooth_layer=false,compatible_reduction=false,normal_upwind=false;int outer_sbp=0; double tlim=12,start_time=0,wall_seconds=0;std::string restart_file,restart_physical_hash,restart_reference_hash; std::string boundary="exact",data="angular";
};
inline AD layer_smooth(AD z,const Parameters &p) {
  if(!p.smooth_layer)return smooth(z);
  if(z.f<=0)return 0;if(z.f>=1)return 1;
  AD a=exp(AD(-1)/z),b=exp(AD(-1)/(AD(1)-z));
  // A reciprocal product can perturb an exactly saturated value by one ulp.
  // Its complement is subsequently multiplied by singular endpoint factors.
  // The quotient jet below retains the exponentially small numerator instead.
  const long double sum=a.f+b.f;AD w(a.f/sum);
  for(int i=0;i<3;++i)w.d[i]=(a.d[i]*b.f-a.f*b.d[i])/(sum*sum);
  return w;
}
inline AD layer_weight_product(AD z) {
  // w(1-w)=ab/(a+b)^2, without subtracting a rounded w from one.
  AD a=exp(AD(-1)/z),b=exp(AD(-1)/(AD(1)-z)),sum=a+b;
  return a*b/(sum*sum);
}
struct Coefficients {
  double g[4][4]{},b[4]{},beta_gradient[3][3]{},alpha_gradient[3]{},potential=0,density=1,omega=1,x[3]{};
  cubed_sphere::Point map;
  int active=1;
};
// Full coefficient differences are diagnostics, never perturbative RHS inputs.
struct CoefficientDifference {double g[4][4]{},b[4]{},potential=0,density=0;};
inline CoefficientDifference difference(const Coefficients &actual,const Coefficients &reference) {
  CoefficientDifference d;for(int i=0;i<4;++i){d.b[i]=actual.b[i]-reference.b[i];for(int j=0;j<4;++j)d.g[i][j]=actual.g[i][j]-reference.g[i][j];}
  d.potential=actual.potential-reference.potential;d.density=actual.density-reference.density;return d;
}
inline AD omega(AD rho,const Parameters &p) {
  return AD(1)-layer_smooth((rho-AD(8))/AD(8),p)*rho/AD(p.S);
}
// Exact R0=M Dennison--Baumgarte--Montero trumpet in Cartesian
// coordinates r=R_BL-M (upstream branch provenance saved with the evidence).
// All expressions below are full metric expressions, not perturbation series.
inline Coefficients coefficients(const double y[3],const Parameters &p) {
  Coefficients c;for(int i=0;i<3;++i)c.x[i]=y[i];
  AD x[3];for(int i=0;i<3;++i)x[i]=AD::var(y[i],i);
  AD rho=power(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],.5L);
  if(rho.f==0)throw std::runtime_error("the trumpet puncture is excluded");
  AD n[3];for(int i=0;i<3;++i)n[i]=x[i]/rho;
  AD az[3]={-n[1],n[0],0};
  AD o=omega(rho,p);
  if(o.f < -1e-12L)throw std::runtime_error("geometry beyond scri");
  AD z=(rho-AD(8))/AD(8),w=layer_smooth(z,p),wp=0;
  if(z.f>0 && z.f<1) {
    if(p.smooth_layer)wp=layer_weight_product(z)*(AD(1)/(z*z)+AD(1)/((AD(1)-z)*(AD(1)-z)))/AD(8);
    else wp=AD(30)*z*z*(AD(1)-z)*(AD(1)-z)/AD(8);
  }
  AD op=-(w+rho*wp)/AD(p.S),lambda=o-rho*op;
  const long double M=p.mass,a=p.spin*M,c0=std::sqrt(M*M-a*a),A=2*M+c0,D=M*M+a*a;
  AD Rb=rho+AD(M)*o,Sigma=Rb*Rb+AD(a*a)*o*o*n[2]*n[2];
  AD Ab=Rb*Rb+AD(a*a)*o*o,Xb=Ab*Ab-AD(a*a)*rho*rho*o*o*(AD(1)-n[2]*n[2]);
  AD root=power(rho*rho+AD(p.C*p.C)*o*o,.5L);
  AD delta_over_o2=AD(-p.C*p.C)/(root*(root+rho));
  AD hp=rho/root+AD(A)*o/rho,wh=1;
  if(M!=0 && o.f>1e-12L)wh=layer_smooth((rho/o-AD(8))/AD(8),p);
  hp=wh*hp;
  AD g[4][4];
  if(wh.f==1) {
    // Exact algebraic factorization by Omega^2, including every higher term.
    // It avoids cancellation and evaluates the same metric at exact scri.
    AD n2=AD(-4*M*(M+c0))+AD(a*a)*(AD(1)-n[2]*n[2]);
    AD n3=AD(-4*M*D-4*M*c0*A-2*c0*D-2*A*c0*c0)/rho;
    AD n4=AD(-(D+c0*A)*(D+c0*A))/(rho*rho);
    AD base=AD(1)+AD(A)*o/rho,rad=rho*rho-AD(c0*c0)*o*o;
    g[0][0]=(n2+n3*o+n4*o*o+delta_over_o2*(AD(-2*c0)*Ab*o/rho+AD(2)*base*rad)
                +delta_over_o2*delta_over_o2*o*o*rad)/Sigma;
  } else {
    g[0][0]=(-Xb/(rho*rho)-AD(2*c0)*hp*Ab*o/rho+hp*hp*(rho*rho-AD(c0*c0)*o*o))/(Sigma*o*o);
  }
  AD radial=(AD(c0)*Ab*o/rho-hp*(rho*rho-AD(c0*c0)*o*o))/(lambda*Sigma);
  AD tangent=(AD(-a)*(AD(2*M)*rho+AD(D)*o)*o/rho-hp*AD(a*c0)*o)/Sigma;
  for(int i=0;i<3;++i) {
    g[0][i+1]=g[i+1][0]=radial*n[i]+tangent*az[i];
    for(int j=0;j<3;++j)g[i+1][j+1]=
        (rho*rho*(AD(i==j)-n[i]*n[j])-AD(a*a)*o*o*az[i]*az[j]
         +AD(a*c0)*o*o*o/lambda*(n[i]*az[j]+az[i]*n[j])
         +o*o/(lambda*lambda)*(rho*rho-AD(c0*c0)*o*o)*n[i]*n[j])/Sigma;
  }
  AD density=lambda*Sigma/(rho*rho);
  // sqrt(-g_phys)=Sigma_phys/r^2. The azimuthal radial flux has zero
  // divergence; hence Box_g Omega = 1/Sigma_phys d_r(Delta Omega_r).
  AD flux=(rho*rho-AD(c0*c0)*o*o)*op/lambda;
  long double fluxprime=0;for(int i=0;i<3;++i)fluxprime+=n[i].f*flux.d[i];
  c.potential=-o.f*fluxprime/(lambda.f*Sigma.f);
  c.density=density.f;c.omega=o.f;
  for(int mu=0;mu<4;++mu) {
    for(int nu=0;nu<4;++nu)c.g[mu][nu]=g[mu][nu].f;
    for(int i=0;i<3;++i)c.b[mu]+=(density*g[i+1][mu]).d[i]/density.f;
  }
  for(int i=0;i<3;++i) {c.alpha_gradient[i]=power(-g[0][0],-.5L).d[i];for(int j=0;j<3;++j)c.beta_gradient[i][j]=(-g[0][j+1]/g[0][0]).d[i];}
  if(!(c.g[0][0]<0) || !std::isfinite(c.potential))throw std::runtime_error("non-spacelike or nonfinite geometry");
  return c;
}
inline std::array<double,2> speeds(const Coefficients &c,const double normal[3]) {
  double gn=0,nn=0;for(int i=0;i<3;++i) {
    gn+=c.g[0][i+1]*normal[i];
    for(int j=0;j<3;++j)nn+=c.g[i+1][j+1]*normal[i]*normal[j];
  }
  double b=gn/c.g[0][0],d=std::sqrt(b*b-nn/c.g[0][0]);return {b-d,b+d};
}
// Gaussian derivative using the probabilists' recurrence in its scaled argument.
inline AD gaussian(AD u,int order) {
  AD z=(u+AD(4))/AD(1.2),h0=1,h1=AD(-2)*z;
  if(order==0)return exp(-z*z);
  for(int k=1;k<order;++k) {AD next=AD(-2)*z*h1-AD(2*k)*h0;h0=h1;h1=next;}
  return exp(-z*z)*h1/AD(std::pow(1.2,order));
}
inline std::array<double,5> exact(const double y[3],double t,const Parameters &p) {
  AD x[3];for(int i=0;i<3;++i)x[i]=AD::var(y[i],i);
  AD rho=power(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],.5L),o=omega(rho,p);
  AD root=power(rho*rho+AD(p.C*p.C)*o*o,.5L);
  AD um=AD(t-p.C)+AD(p.C*p.C)*o/(root+rho);
  AD up=0; bool ingoing=o.f>1e-12L;
  if(ingoing)up=AD(t-p.C)+(root+rho)/o;
  AD value[2];
  for(int k=0;k<2;++k) {
    AD f0=gaussian(um,k),f1=gaussian(um,k+1),f2=gaussian(um,k+2);
    AD e0=ingoing?gaussian(up,k):AD(0),e1=ingoing?gaussian(up,k+1):AD(0),e2=ingoing?gaussian(up,k+2):AD(0);
    value[k]=x[0]*x[1]/(rho*rho)*((f2-e2)/rho+AD(3)*(f1+e1)*o/(rho*rho)+AD(3)*(f0-e0)*o*o/(rho*rho*rho));
  }
  return {double(value[0].f),double(value[1].f),double(value[0].d[0]),double(value[0].d[1]),double(value[0].d[2])};
}
}
#endif
