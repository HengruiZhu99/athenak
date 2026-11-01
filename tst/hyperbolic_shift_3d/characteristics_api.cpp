// Read-only frozen-principal-symbol audit. Never linked into an evolution target.
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
namespace h=z4c::hyperboloidal;
extern "C" int hyperbolic_symbol(double,double,const double*,const double*,double,double,double,double,double,double,double*);
namespace {
// Proposed single transition. Retain the COMPLETE existing outer gauge, but
// extend its outer mu choice inward and suppress its old Euclidean B coupling.
// This candidate is an audit only, not an adopted evolution equation.
h::HyperbolicGaugeParts<double> Candidate(double r,const h::Z4cJet<double>&u,
 const h::VectorJet<double>&B,const double*D){
 double x[3]={r,0,0};const double o=(1-r*r)/2;
 auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value);
 h::HyperbolicShiftParameters<double> p;p.bulk_mu=1.5*u.chi.value*c.F;
 h::VectorJet<double> zero{};
 auto out=h::DonorHyperbolicGauge(x,u,zero,D,p);
 for(int i=0;i<3;++i){out.flux_rate[i]=-B.value[i]/p.tau;for(int j=0;j<3;++j)out.flux_rate[i]+=u.beta.value[j]*B.d[j][i];}
 const double s=h::Smoothstep9((.9-r)/.7);
 if(s==0)return out;
 const auto g=h::Geometry(u.metric);const double a=u.alpha.value/o;
 const double chi=fmax(o*o*u.chi.value,1e-5);
 double lapse=-2*u.alpha.value*u.trace.value;
 for(int i=0;i<3;++i){
  lapse+=u.beta.value[i]*(u.alpha.d[i]+a*x[i]);
  for(int j=0;j<3;++j)lapse+=o*chi*g.inverse[i][j]*B.d[i][j];
  double flux=((.2/3)*(u.alpha.d[i]+a*x[i])/o-B.value[i])/(.1/3);
  double shift=u.lambda.value[i]-6*u.beta.value[i];
  for(int j=0;j<3;++j){flux+=u.beta.value[j]*B.d[j][i];shift+=u.beta.value[j]*u.beta.d[j][i];}
  out.flux_rate[i]=(1-s)*out.flux_rate[i]+s*flux;
  out.shift_regular[i]=(1-s)*out.shift_regular[i]+s*shift;out.shift_pole[i]*=1-s;
 }
 out.lapse_regular=(1-s)*out.lapse_regular+s*lapse;out.lapse_pole*=1-s;
 return out;
}
}
// Variables: dchi, 5 dg, 5 A, p/Omega, Theta/Omega, 3 Lambda,
// dalpha_bar, 3 dbeta, 3 dB, 3 D. Freeze in the comoving frame.
// The geometric symbol in these normalized variables equals ConformalRHS at
// Omega=1. Omega derivatives, damping and background forcing are lower order.
extern "C" int inner_gauge_symbol(int mode,double r,double alpha,double chi,
 const double*metric,const double*n,double*P){
 if(mode<0||mode>2||r<0||r>1)return 1;
 int code=hyperbolic_symbol(alpha,chi,metric,n,1.,1.,0.,.005,.02,4.,P);if(code)return code;
 for(int row=16;row<23;++row)for(int col=0;col<26;++col)P[row*26+col]=0;
 h::Z4cJet<double>base{};base.alpha.value=alpha;base.chi.value=chi;base.trace.value=-3;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)base.metric.g[i][j]=metric[3*i+j];
 const double o=(1-r*r)/2,eps=1e-4;double x[3]={r,0,0};
 for(int col=0;col<26;++col){
  h::HyperbolicGaugeParts<double> v[2];
  for(int k=0;k<2;++k){
   const double e=(k==0?-eps:eps);auto u=base;h::VectorJet<double>B{};double D[3]{};
   if(col==11)u.trace.value+=e;
   if(col>=13&&col<16)u.lambda.value[col-13]=e;
   if(col==16)for(int i=0;i<3;++i)u.alpha.d[i]=e*n[i];
   if(col>=20&&col<23)for(int i=0;i<3;++i)B.d[i][col-20]=e*n[i];
   if(col>=23)D[col-23]=e;
   v[k]=mode==0?h::DonorHyperbolicGauge(x,u,B,D):mode==1?h::HyperbolicGauge(x,u,B,D):Candidate(r,u,B,D);
   if(!v[k].valid)return 4;
  }
  // Differencing pole numerators before division avoids fake loss of rank
  // near scri. The p/Omega column multiplies the regular contribution by Omega.
  if(col==11)P[16*26+col]=(o*(v[1].lapse_regular-v[0].lapse_regular)+v[1].lapse_pole-v[0].lapse_pole)/(2*eps);
  else if(col>=20&&col<23)P[16*26+col]=(v[1].lapse_regular-v[0].lapse_regular)/(2*eps);
  for(int i=0;i<3;++i){
   if((col>=13&&col<16)||col>=23)P[(17+i)*26+col]=(v[1].shift_regular[i]-v[0].shift_regular[i])/(2*eps);
   if(col==16)P[(20+i)*26+col]=(v[1].flux_rate[i]-v[0].flux_rate[i])/(2*eps);
  }
 }
 return 0;
}

extern "C" double candidate_outer_rhs_error(){
 double worst=0;
 for(double r:{.9,.95,.999,1.}) {
  double x[3]={r,0,0},D[3]={.2,-.1,.3};h::Z4cJet<double>u{};h::VectorJet<double>B{};
  u.alpha.value=.83;u.chi.value=1.12;u.trace.value=-2.7;
  for(int i=0;i<3;++i){u.metric.g[i][i]=1.;u.beta.value[i]=.3*(i-1);u.alpha.d[i]=.11*(i+1);B.value[i]=.14*(i-1);
   for(int j=0;j<3;++j){u.beta.d[j][i]=.02*(1+i-j);B.d[j][i]=.03*(i+2*j);}}
  auto a=h::DonorHyperbolicGauge(x,u,B,D),b=Candidate(r,u,B,D);
  worst=fmax(worst,fabs(a.lapse_regular-b.lapse_regular));worst=fmax(worst,fabs(a.lapse_pole-b.lapse_pole));
  for(int i=0;i<3;++i){worst=fmax(worst,fabs(a.shift_regular[i]-b.shift_regular[i]));worst=fmax(worst,fabs(a.shift_pole[i]-b.shift_pole[i]));worst=fmax(worst,fabs(a.flux_rate[i]-b.flux_rate[i]));}
 }
 return worst;
}
