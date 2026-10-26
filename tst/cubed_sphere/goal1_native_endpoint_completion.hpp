// Actual native second-order endpoint compatibility; jet coefficients are
// analysis inputs only, never additional independently evolved fields.
#pragma once
#include "goal1_native_endpoint.hpp"
#include <vector>
namespace cubed_sphere {namespace native_endpoint_completion {
namespace h=z4c::hyperboloidal;namespace ne=native_endpoint;
struct Result {std::array<Real,22>rhs{};Real first[13][3]{};std::vector<Real>residual;};
inline Result Evaluate(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&p,const Real x[3],bool omit_metric_gradient_time=false,Real unused=0){
 Result out;auto ref=ne::OuterReference(p,x);ref.omega.f=0;auto c=ne::OuterConnection(x);
 auto f=ne::NativeParts(ne::Spatial(u,unused),ref,c);if(!f.valid)throw std::runtime_error("invalid endpoint completion source");
 Real norm=0;for(int i=0;i<3;++i)norm+=ref.omega.d[i]*ref.omega.d[i];
 for(int n=0;n<22;++n){Real q=0;for(int i=0;i<3;++i)q+=ref.omega.d[i]*f.pole[n].d[i]/norm;out.rhs[n]=f.regular[n].f+q;out.residual.push_back(f.pole[n].f);}
 for(int n=0;n<22;++n)for(int i=0;i<3;++i)out.residual.push_back(f.pole[n].d[i]-ref.omega.d[i]*(out.rhs[n]-f.regular[n].f));
 for(int a=0;a<13;++a){int n=ne::first_rows[a];Real q=out.rhs[n]-f.regular[n].f,R[3][3],nn=0,dq[3]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){R[i][j]=f.pole[n].h[i][j]-ref.omega.h[i][j]*q;nn+=ref.omega.d[i]*ref.omega.d[j]*R[i][j];}
  for(int j=0;j<3;++j){for(int i=0;i<3;++i)dq[j]+=ref.omega.d[i]*R[i][j]/norm;dq[j]-=ref.omega.d[j]*nn/(2*norm*norm);out.first[a][j]=f.regular[n].d[j]+dq[j];}
  for(int i=0;i<3;++i)for(int j=i;j<3;++j)out.residual.push_back(R[i][j]-ref.omega.d[i]*dq[j]-ref.omega.d[j]*dq[i]);
 }
 auto time=typed_support::Map<Jet>(u,[](int,Real a){return Jet(a);});
 time.chi.value.d[0]=out.rhs[0];time.alpha.value.d[0]=out.rhs[18];time.trace.value.d[0]=out.rhs[7];time.theta.value.d[0]=out.rhs[17];
 auto first=[&](int n,int d){for(int a=0;a<13;++a)if(ne::first_rows[a]==n)return out.first[a][d];throw std::runtime_error("missing endpoint RHS gradient");};
 for(int d=0;d<3;++d){time.chi.d[d].d[0]=first(0,d);time.alpha.d[d].d[0]=first(18,d);time.trace.d[d].d[0]=first(7,d);time.theta.d[d].d[0]=first(17,d);for(int i=0;i<3;++i)time.beta.d[d][i].d[0]=first(19+i,d);}
 for(int n=0;n<6;++n){int i=ne::ma[n],j=ne::mb[n];time.metric.g[i][j].d[0]=time.metric.g[j][i].d[0]=out.rhs[1+n];time.a.k[i][j].d[0]=time.a.k[j][i].d[0]=out.rhs[8+n];if(!omit_metric_gradient_time)for(int d=0;d<3;++d)time.metric.dg[d][i][j].d[0]=time.metric.dg[d][j][i].d[0]=first(1+n,d);}
 for(int i=0;i<3;++i){time.beta.value[i].d[0]=out.rhs[19+i];time.lambda.value[i].d[0]=out.rhs[14+i];}
 auto fixed=typed_support::Constant<Jet>(p);fixed.omega=Jet(0);auto connection=c;for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)connection.scaled[k][i][j]=Jet(c.scaled[k][i][j].f);
 auto ft=ne::NativeParts(time,fixed,connection);if(!ft.valid)throw std::runtime_error("invalid time compatibility source");for(int n=0;n<22;++n)out.residual.push_back(ft.pole[n].d[0]);
 auto geo=h::Geometry(u.metric);Real up[3][3]{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)up[i][j]+=geo.inverse[i][k]*u.a.k[k][l]*geo.inverse[j][l];
 for(int d=0;d<3;++d)for(int e=d;e<3;++e){Real v=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){v+=geo.inverse[i][j]*u.metric.ddg[d][e][i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)v-=geo.inverse[i][j]*u.metric.dg[d][j][k]*geo.inverse[k][l]*u.metric.dg[e][l][i];}out.residual.push_back(v);}
 for(int d=0;d<3;++d){Real v=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)v+=geo.inverse[i][j]*u.a.dk[d][i][j]-up[i][j]*u.metric.dg[d][i][j];out.residual.push_back(v);}
 if(out.residual.size()!=197)throw std::runtime_error("endpoint completion residual dimension");return out;
}
inline std::vector<int> Unknowns(){std::vector<int>v;for(int b:{0,10,20,30,40,60,80})for(int j=4;j<10;++j)v.push_back(b+j);for(int b:{50,70,90})for(int j=1;j<=3;++j)v.push_back(b+j);for(int n=0;n<6;++n){for(int j=4;j<10;++j)v.push_back(100+14*n+j);for(int j=11;j<14;++j)v.push_back(100+14*n+j);}return v;}
}}
