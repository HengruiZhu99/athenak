// Native second-spatial-order endpoint jet analysis only. No evolved jet fields.
#pragma once
#include "goal1_native_endpoint_completion.hpp"
namespace cubed_sphere {namespace native_endpoint_invariant {
namespace h=z4c::hyperboloidal;namespace ec=native_endpoint_completion;
inline ec::Result Evaluate(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&p,const Real x[3],Real unused=0){
 auto out=ec::Evaluate(u,p,x,false,unused);auto t=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});
 auto first=[&](int row,int d){for(int j=0;j<13;++j)if(native_endpoint::first_rows[j]==row)return out.first[j][d];throw std::runtime_error("missing native endpoint first time gradient");};
 t.chi.value.d[0]=out.rhs[0];t.alpha.value.d[0]=out.rhs[18];t.trace.value.d[0]=out.rhs[7];t.theta.value.d[0]=out.rhs[17];
 for(int d=0;d<3;++d){t.chi.d[d].d[0]=first(0,d);t.alpha.d[d].d[0]=first(18,d);t.trace.d[d].d[0]=first(7,d);t.theta.d[d].d[0]=first(17,d);for(int i=0;i<3;++i)t.beta.d[d][i].d[0]=first(19+i,d);}
 for(int j=0;j<6;++j){int a=native_endpoint::ma[j],b=native_endpoint::mb[j];t.metric.g[a][b].d[0]=t.metric.g[b][a].d[0]=out.rhs[1+j];t.a.k[a][b].d[0]=t.a.k[b][a].d[0]=out.rhs[8+j];for(int d=0;d<3;++d)t.metric.dg[d][a][b].d[0]=t.metric.dg[d][b][a].d[0]=first(1+j,d);}
 for(int i=0;i<3;++i){t.beta.value[i].d[0]=out.rhs[19+i];t.lambda.value[i].d[0]=out.rhs[14+i];}
 h::OmegaJet<Jet>o{};o.omega=Jet(0);Real norm=0;for(int i=0;i<3;++i){o.gradient[i]=Jet(p.domega[i]);norm+=p.domega[i]*p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=Jet(p.omega_hessian[i][j]);}
 h::SetStationaryOmegaNormal(t.alpha.value,t.beta.value,t.alpha.d,t.beta.d,o);auto g=h::Geometry(t.metric);Jet tau,N1;
 for(int d=0;d<3;++d){tau+=Jet(p.domega[d]/norm)*t.theta.d[d];Jet dG;for(int i=0;i<3;++i)for(int j=0;j<3;++j){Jet dC=t.chi.d[d]*g.inverse[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)dC-=t.chi.value*g.inverse[i][k]*t.metric.dg[d][k][l]*g.inverse[l][j];dG+=dC*Jet(p.domega[i]*p.domega[j])+Jet(2)*t.chi.value*g.inverse[i][j]*Jet(p.omega_hessian[d][i]*p.domega[j]);}N1+=Jet(p.domega[d]/norm)*(dG-Jet(2)*o.normal*o.dnormal[d]);}
 out.residual.push_back(tau.d[0]);for(int i=0;i<3;++i)out.residual.push_back((t.lambda.value[i]-g.contracted[i]).d[0]/2);out.residual.push_back(N1.d[0]);if(out.residual.size()!=202)throw std::runtime_error("native invariant endpoint dimension");return out;
}
}}
