// Exact CMC-scri point quotient on compatible jets, not a boundary closure.
#pragma once
#include "goal1_native_source_ad.hpp"
namespace cubed_sphere {namespace endpoint_point {
namespace h=z4c::hyperboloidal;
inline rwm::Connection<Jet> OuterConnection(const Real x[3]){
 Jet r2;Jet y[3];for(int i=0;i<3;++i){y[i]=Jet(x[i]);y[i].d[i]=1;r2+=y[i]*y[i];}
 Jet omega=Jet(1)-r2,L=Jet(1)+r2;rwm::Connection<Jet>c{};c.valid=true;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){c.scaled[0][i][j]=(i==j?Jet(2)/L:Jet(0));for(int k=0;k<3;++k)c.scaled[k+1][i][j]=Jet(2)*((k==i?y[j]:Jet(0))+(k==j?y[i]:Jet(0)))+(i==j?Jet(2)*y[k]*omega/L:Jet(0));}
 return c;
}
inline h::Z4cJet<Jet> First(const h::Z4cJet<Real>&u,Real unused=0){
 auto v=typed_support::Map<Jet>(u,[](int,Real a){return Jet(a);});
 auto scalar=[](const h::ScalarJet<Real>&s,h::ScalarJet<Jet>&t){for(int d=0;d<3;++d){t.value.d[d]=s.d[d];for(int i=0;i<3;++i)t.d[i].d[d]=s.dd[d][i];}};
 scalar(u.chi,v.chi);scalar(u.alpha,v.alpha);scalar(u.trace,v.trace);scalar(u.theta,v.theta);
 for(int d=0;d<3;++d)for(int i=0;i<3;++i){v.beta.value[i].d[d]=u.beta.d[d][i];v.lambda.value[i].d[d]=u.lambda.d[d][i];for(int j=0;j<3;++j){v.beta.d[j][i].d[d]=u.beta.dd[d][j][i];v.lambda.d[j][i].d[d]=u.lambda.dd[d][j][i];v.metric.g[i][j].d[d]=u.metric.dg[d][i][j];v.a.k[i][j].d[d]=u.a.dk[d][i][j];for(int k=0;k<3;++k){v.metric.dg[k][i][j].d[d]=u.metric.ddg[d][k][i][j];v.metric.ddg[k][d][i][j].d[d]=unused;v.a.dk[k][i][j].d[d]=unused;}}v.chi.dd[d][i].d[d]=unused;v.alpha.dd[d][i].d[d]=unused;v.trace.dd[d][i].d[d]=unused;v.theta.dd[d][i].d[d]=unused;for(int j=0;j<3;++j)v.beta.dd[d][i][j].d[d]=unused;}
 return v;
}
template<class T>std::array<T,22> Rows(const h::Z4cRHS<T>&f,const h::GaugeRHS<T>&g){
 std::array<T,22>r{};r[0]=f.chi;r[7]=f.trace;r[17]=f.theta;r[18]=g.alpha;int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};for(int n=0;n<6;++n){r[1+n]=f.metric[a[n]][b[n]];r[8+n]=f.a[a[n]][b[n]];}for(int i=0;i<3;++i){r[14+i]=f.lambda[i];r[19+i]=g.beta[i];}return r;
}
struct Result{bool valid=false;std::array<Real,22>rhs{},pole{},quotient{};Real pole_max=0,tangent_max=0;};
inline Result Evaluate(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&p,const Real x[3],Real unused=0){
 Result result;Real rr=0;for(int i=0;i<3;++i)rr+=x[i]*x[i];if(std::abs(rr-1)>1e-14||std::abs(p.omega)>1e-14)return result;
 auto state=First(u,unused);Real third[3][3][3]{};auto ref=native_source_ad::ReferenceFirst(p,third);auto connection=OuterConnection(x);h::OmegaJet<Jet>o{};o.omega=Jet(0);for(int i=0;i<3;++i){o.omega.d[i]=p.domega[i];o.gradient[i]=ref.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=ref.omega_hessian[i][j];}h::SetStationaryOmegaNormal(state.alpha.value,state.beta.value,state.alpha.d,state.beta.d,o);
 auto f=h::ConformalRHS(state,o,Jet(10)/state.alpha.value,Jet(0));auto g=rwm::Gauge(ref,state,connection);if(!f.valid||!g.valid)return result;auto regular=Rows(f.regular,g.regular),pole=Rows(f.pole,g.pole);Real norm=0;for(int i=0;i<3;++i)norm+=p.domega[i]*p.domega[i];
 for(int n=0;n<22;++n){Real quotient=0;for(int i=0;i<3;++i)quotient+=p.domega[i]*pole[n].d[i]/norm;result.pole[n]=pole[n].f;result.quotient[n]=quotient;result.rhs[n]=regular[n].f+quotient;result.pole_max=std::max(result.pole_max,std::abs(pole[n].f));for(int i=0;i<3;++i)result.tangent_max=std::max(result.tangent_max,std::abs(pole[n].d[i]-quotient*p.domega[i]));if(!std::isfinite(result.rhs[n]))return result;}
 result.valid=result.pole_max<=1e-10&&result.tangent_max<=1e-10;return result;
}
}}
