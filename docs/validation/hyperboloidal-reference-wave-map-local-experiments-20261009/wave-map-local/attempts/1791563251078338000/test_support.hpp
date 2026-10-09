#ifndef RWM_TEST_SUPPORT_HPP_
#define RWM_TEST_SUPPORT_HPP_
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "reference_wave_map.hpp"
namespace audit {
namespace hyp=z4c::hyperboloidal;
using Jet=hyp::Z4cJet<double>;
struct Four {double g[4][4]{},inv[4][4]{},d[4][4][4]{},Gamma[4][4][4]{};};
inline void Inverse(const double a[4][4],double inv[4][4]){
 double b[4][8]{};for(int i=0;i<4;++i)for(int j=0;j<4;++j){b[i][j]=a[i][j];b[i][4+j]=i==j;}
 for(int k=0;k<4;++k){int p=k;for(int i=k+1;i<4;++i)if(std::abs(b[i][k])>std::abs(b[p][k]))p=i;
  if(b[p][k]==0)throw std::runtime_error("singular four-matrix");for(int j=0;j<8;++j)std::swap(b[k][j],b[p][j]);
  const double v=b[k][k];for(int j=0;j<8;++j)b[k][j]/=v;
  for(int i=0;i<4;++i)if(i!=k){const double c=b[i][k];for(int j=0;j<8;++j)b[i][j]-=c*b[k][j];}}
 for(int i=0;i<4;++i)for(int j=0;j<4;++j)inv[i][j]=b[i][4+j];
}
inline Four Metric4(const Jet&u,const hyp::LayerPoint<double>&p,
    const hyp::Z4cRHS<double>*geom=nullptr,const hyp::GaugeRHS<double>*gauge=nullptr,bool physical=false){
 Four f{};const auto b=hyp::PenroseMetric(u.metric,u.chi);double bd[4][3][3]{};
 const double a=u.alpha.value;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){f.g[i+1][j+1]=b.g[i][j];
  if(geom)bd[0][i][j]=geom->metric[i][j]/u.chi.value-u.metric.g[i][j]*geom->chi/(u.chi.value*u.chi.value);
  for(int d=0;d<3;++d)bd[d+1][i][j]=b.dg[d][i][j];}
 f.g[0][0]=-a*a;for(int i=0;i<3;++i){for(int j=0;j<3;++j){f.g[0][i+1]+=b.g[i][j]*u.beta.value[j];f.g[0][0]+=b.g[i][j]*u.beta.value[i]*u.beta.value[j];}f.g[i+1][0]=f.g[0][i+1];}
 for(int d=0;d<4;++d){const double da=d?u.alpha.d[d-1]:(gauge?gauge->alpha:0);
  double db[3]{};for(int i=0;i<3;++i)db[i]=d?u.beta.d[d-1][i]:(gauge?gauge->beta[i]:0);
  f.d[d][0][0]=-2*a*da;
  for(int i=0;i<3;++i){for(int j=0;j<3;++j){f.d[d][i+1][j+1]=bd[d][i][j];f.d[d][0][i+1]+=bd[d][i][j]*u.beta.value[j]+b.g[i][j]*db[j];f.d[d][0][0]+=bd[d][i][j]*u.beta.value[i]*u.beta.value[j]+2*b.g[i][j]*u.beta.value[i]*db[j];}f.d[d][i+1][0]=f.d[d][0][i+1];}}
 if(physical){const double o=p.omega;for(int a=0;a<4;++a)for(int b=0;b<4;++b){for(int d=0;d<4;++d)f.d[d][a][b]=f.d[d][a][b]/(o*o)-(d?2*p.domega[d-1]*f.g[a][b]/(o*o*o):0);f.g[a][b]/=o*o;}}
 Inverse(f.g,f.inv);
 for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int c=0;c<4;++c)for(int l=0;l<4;++l)f.Gamma[a][b][c]+=.5*f.inv[a][l]*(f.d[b][l][c]+f.d[c][l][b]-f.d[l][b][c]);
 return f;
}
inline std::array<double,4> Contract(const Four&f){std::array<double,4>v{};for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int c=0;c<4;++c)v[a]+=f.inv[b][c]*f.Gamma[a][b][c];return v;}
// Independent full Jacobian/Hessian embedding, using direct radial height jets.
// This intentionally does not use the simplified scaled connection formulas.
inline Four Embedding(const hyp::LayerPoint<double>&p,const double x[3],double S,double a,const hyp::LayerParameters&geo){
 Four f{};const double r=p.radius;if(geo.enabled&&r<=geo.r0)return f;
 if(r==0){for(int i=0;i<3;++i)f.Gamma[0][i+1][i+1]=1/(a*p.omega*p.omega);return f;}
 const auto w=hyp::SmoothCutoff(r,geo.r0,geo.r1);const double out=(S*S-r*r)/(2*a*S),op0=-r/(a*S),opp0=-1/(a*S);
 const double o=geo.enabled?(1-w.value)+w.value*out:out;
 const double op=geo.enabled?w.d*(out-1)+w.value*op0:op0;
 const double opp=geo.enabled?w.dd*(out-1)+2*w.d*op0+w.value*opp0:opp0;
 const double b=geo.enabled?r*w.value/a:r/a,bp=geo.enabled?(w.value+r*w.d)/a:1/a;
 const double L=o-r*op,Lp=-r*opp,ah=std::hypot(o,b),ahp=(o*op+b*bp)/ah;
 const double hp=b*L/(ah*o*o),hpp=(bp*L+b*Lp)/(ah*o*o)-b*L*ahp/(ah*ah*o*o)-2*b*L*op/(ah*o*o*o);
 double jac[4][4]{},invjac[4][4]{},H[4][3][3]{};jac[0][0]=1;
 double n[3];for(int i=0;i<3;++i)n[i]=x[i]/r;
 for(int i=0;i<3;++i){jac[0][i+1]=hp*n[i];for(int I=0;I<3;++I)jac[I+1][i+1]=(I==i?1.:0.)/o-x[I]*op*n[i]/(o*o);}
 Inverse(jac,invjac);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){const double oi=op*n[i],oj=op*n[j],oij=opp*n[i]*n[j]+op*((i==j?1.:0.)-n[i]*n[j])/r;
  H[0][i][j]=hpp*n[i]*n[j]+hp*((i==j?1.:0.)-n[i]*n[j])/r;
  for(int I=0;I<3;++I)H[I+1][i][j]=-((I==i?oj:0)+(I==j?oi:0)+x[I]*oij)/(o*o)+2*x[I]*oi*oj/(o*o*o);
  for(int aa=0;aa<4;++aa)for(int A=0;A<4;++A)f.Gamma[aa][i+1][j+1]+=invjac[aa][A]*H[A][i][j];}
 return f;
}
inline hyp::OmegaJet<double> Omega(const Jet&u,const hyp::LayerPoint<double>&p){hyp::OmegaJet<double>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;}
// Direct, unfactored independent evaluation of the equivalent retained-P rows.
inline hyp::GaugeRHS<double> RawGauge(const hyp::LayerPoint<double>&p,const Jet&u,const rwm::Connection<double>&c){
 hyp::GaugeRHS<double>f{};const auto gi=hyp::Geometry(u.metric);const double a=u.alpha.value;
 double B=0,L[3][3]{};for(int i=0;i<3;++i){B+=u.beta.value[i]*p.domega[i];for(int j=0;j<3;++j)L[i][j]=a*a*u.chi.value*gi.inverse[i][j]-u.beta.value[i]*u.beta.value[j];}
 double Sa=-a*a*u.trace.value-a*B;for(int i=0;i<3;++i)for(int j=0;j<3;++j)Sa-=a*L[i][j]*c.scaled[0][i][j];
 f.alpha=Sa/p.omega;for(int i=0;i<3;++i)f.alpha+=u.beta.value[i]*u.alpha.d[i];
 for(int i=0;i<3;++i){f.beta[i]=a*a*u.chi.value*u.lambda.value[i];double pole=0;
  for(int j=0;j<3;++j){f.beta[i]+=u.beta.value[j]*u.beta.d[j][i]+.5*a*a*gi.inverse[i][j]*u.chi.d[j]-a*u.chi.value*gi.inverse[i][j]*u.alpha.d[j];pole+=2*a*a*u.chi.value*gi.inverse[i][j]*p.domega[j];for(int k=0;k<3;++k)pole-=L[j][k]*(c.scaled[i+1][j][k]+u.beta.value[i]*c.scaled[0][j][k]);}f.beta[i]+=pole/p.omega;}
 return f;
}
inline double ScaleError(double a,double b){return std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)});}
}
#endif
