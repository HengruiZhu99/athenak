// Native22 second-order Z4c endpoint equations on compatible analytic jets.
// AD jets are equation-analysis inputs, never additional evolved fields.
#pragma once
#include "goal1_typed_support.hpp"
namespace cubed_sphere {namespace native_endpoint {
namespace h=z4c::hyperboloidal;
constexpr int first_rows[13]={0,1,2,3,4,5,6,7,17,18,19,20,21};
constexpr int ma[6]={0,0,0,1,1,2},mb[6]={0,1,2,1,2,2};
inline h::Z4cJet<Jet> Spatial(const h::Z4cJet<Real>&u,Real unused=0){
 auto v=typed_support::Map<Jet>(u,[](int,Real a){return Jet(a);});
 auto scalar=[](const h::ScalarJet<Real>&s,h::ScalarJet<Jet>&t){for(int d=0;d<3;++d){t.value.d[d]=s.d[d];for(int i=0;i<3;++i){t.value.h[d][i]=s.dd[d][i];t.d[i].d[d]=s.dd[d][i];}}};scalar(u.chi,v.chi);scalar(u.alpha,v.alpha);scalar(u.trace,v.trace);scalar(u.theta,v.theta);
 for(int d=0;d<3;++d)for(int i=0;i<3;++i){v.beta.value[i].d[d]=u.beta.d[d][i];v.lambda.value[i].d[d]=u.lambda.d[d][i];for(int j=0;j<3;++j){v.beta.value[i].h[d][j]=u.beta.dd[d][j][i];v.lambda.value[i].h[d][j]=u.lambda.dd[d][j][i];v.beta.d[j][i].d[d]=u.beta.dd[d][j][i];v.lambda.d[j][i].d[d]=unused;v.metric.g[i][j].d[d]=u.metric.dg[d][i][j];v.a.k[i][j].d[d]=u.a.dk[d][i][j];for(int k=0;k<3;++k){v.metric.g[i][j].h[d][k]=u.metric.ddg[d][k][i][j];v.metric.dg[k][i][j].d[d]=u.metric.ddg[d][k][i][j];v.metric.ddg[k][d][i][j].d[d]=unused*(i+j+k+1);v.a.dk[k][i][j].d[d]=unused;}}v.chi.dd[d][i].d[d]=unused;v.alpha.dd[d][i].d[d]=unused;for(int j=0;j<3;++j)v.beta.dd[d][i][j].d[d]=unused;}
 return v;
}
inline h::LayerPoint<Jet> OuterReference(const h::LayerPoint<Real>&p,const Real x[3]){
 auto r=typed_support::Constant<Jet>(p);Jet y[3],radius2;for(int i=0;i<3;++i){y[i]=Jet(x[i]);y[i].d[i]=1;radius2+=y[i]*y[i];}r.omega=Jet(1)-radius2;r.L=r.alpha=r.state.alpha.value=Jet(1)+radius2;
 for(int i=0;i<3;++i){r.domega[i]=-Jet(2)*y[i];r.beta[i]=r.state.beta.value[i]=-Jet(2)*y[i];r.dalpha[i]=r.state.alpha.d[i]=Jet(2)*y[i];for(int j=0;j<3;++j){r.omega_hessian[i][j]=Jet(i==j?-2:0);r.state.alpha.dd[i][j]=Jet(i==j?2:0);r.state.beta.d[i][j]=Jet(i==j?-2:0);}}return r;
}
inline rwm::Connection<Jet> OuterConnection(const Real x[3]){
 Jet y[3],r2;for(int i=0;i<3;++i){y[i]=Jet(x[i]);y[i].d[i]=1;r2+=y[i]*y[i];}Jet omega=Jet(1)-r2,L=Jet(1)+r2;rwm::Connection<Jet>c{};c.valid=true;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){c.scaled[0][i][j]=i==j?Jet(2)/L:Jet(0);for(int k=0;k<3;++k)c.scaled[k+1][i][j]=Jet(2)*((k==i?y[j]:Jet(0))+(k==j?y[i]:Jet(0)))+(i==j?Jet(2)*y[k]*omega/L:Jet(0));}return c;
}
template<class T>std::array<T,22>Rows(const h::Z4cRHS<T>&f,const h::GaugeRHS<T>&g){std::array<T,22>r{};r[0]=f.chi;r[7]=f.trace;r[17]=f.theta;r[18]=g.alpha;for(int n=0;n<6;++n){r[1+n]=f.metric[ma[n]][mb[n]];r[8+n]=f.a[ma[n]][mb[n]];}for(int i=0;i<3;++i){r[14+i]=f.lambda[i];r[19+i]=g.beta[i];}return r;}
struct Parts{bool valid=false;std::array<Jet,22>regular,pole;};
inline Parts NativeParts(const h::Z4cJet<Jet>&u,const h::LayerPoint<Jet>&p,const rwm::Connection<Jet>&c){
 h::OmegaJet<Jet>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);auto f=h::ConformalRHS(u,o,Jet(10)/u.alpha.value,Jet(0));auto g=rwm::Gauge(p,u,c);Parts out;if(!f.valid||!g.valid)return out;out.regular=Rows(f.regular,g.regular);out.pole=Rows(f.pole,g.pole);out.valid=true;return out;
}
struct Result{
 bool valid=false;std::array<Real,22>rhs{};std::array<std::array<Real,3>,13>first{};
 Real pole=0,tangent=0,second=0,pole_time=0,nullness_time=0,z_time=0;
 Real First(int row,int d)const{for(int n=0;n<13;++n)if(first_rows[n]==row)return first[n][d];throw std::runtime_error("unsupported native endpoint RHS derivative");}
};
inline Result Evaluate(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&p,const Real x[3],Real unused=0,bool omit_metric_gradient_time=false){
 Result out;Real radius2=0;for(int d=0;d<3;++d)radius2+=x[d]*x[d];if(std::abs(radius2-1)>1e-14||std::abs(p.omega)>1e-14)return out;auto ref=OuterReference(p,x);
 // This function is defined on the known r=1 face: Omega is exactly zero.
 // Its analytic spatial jets are retained. This is not an Omega floor.
 ref.omega.f=0;auto c=OuterConnection(x);auto state=Spatial(u,unused);auto f=NativeParts(state,ref,c);if(!f.valid)return out;Real norm=0;for(int d=0;d<3;++d)norm+=ref.omega.d[d]*ref.omega.d[d];
 for(int n=0;n<22;++n){Real quotient=0;for(int d=0;d<3;++d)quotient+=ref.omega.d[d]*f.pole[n].d[d]/norm;out.pole=std::max(out.pole,std::abs(f.pole[n].f));for(int d=0;d<3;++d)out.tangent=std::max(out.tangent,std::abs(f.pole[n].d[d]-ref.omega.d[d]*quotient));out.rhs[n]=f.regular[n].f+quotient;}
 for(int a=0;a<13;++a){int n=first_rows[a];Real q=out.rhs[n]-f.regular[n].f,R[3][3],nn=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){R[i][j]=f.pole[n].h[i][j]-ref.omega.h[i][j]*q;nn+=ref.omega.d[i]*ref.omega.d[j]*R[i][j];}Real dq[3]{};for(int j=0;j<3;++j){for(int i=0;i<3;++i)dq[j]+=ref.omega.d[i]*R[i][j]/norm;dq[j]-=ref.omega.d[j]*nn/(2*norm*norm);out.first[a][j]=f.regular[n].d[j]+dq[j];}for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.second=std::max(out.second,std::abs(R[i][j]-ref.omega.d[i]*dq[j]-ref.omega.d[j]*dq[i]));}
 if(std::max({out.pole,out.tangent,out.second})>1e-10)return out;
 // Time differentiation of the full actual native pole system. Only native
 // configuration/P/Theta RHS gradients are needed; A/Lambda gradients are not
 // supplied or claimed. No extra evolution variable is introduced.
 auto time=typed_support::Map<Jet>(u,[](int,Real a){return Jet(a);});time.chi.value.d[0]=out.rhs[0];time.alpha.value.d[0]=out.rhs[18];time.trace.value.d[0]=out.rhs[7];time.theta.value.d[0]=out.rhs[17];
 for(int d=0;d<3;++d){time.chi.d[d].d[0]=out.First(0,d);time.alpha.d[d].d[0]=out.First(18,d);time.trace.d[d].d[0]=out.First(7,d);time.theta.d[d].d[0]=out.First(17,d);for(int i=0;i<3;++i){time.beta.d[d][i].d[0]=out.First(19+i,d);}}
 for(int n=0;n<6;++n){time.metric.g[ma[n]][mb[n]].d[0]=time.metric.g[mb[n]][ma[n]].d[0]=out.rhs[1+n];time.a.k[ma[n]][mb[n]].d[0]=time.a.k[mb[n]][ma[n]].d[0]=out.rhs[8+n];if(!omit_metric_gradient_time)for(int d=0;d<3;++d)time.metric.dg[d][ma[n]][mb[n]].d[0]=time.metric.dg[d][mb[n]][ma[n]].d[0]=out.First(1+n,d);}
 for(int i=0;i<3;++i){time.beta.value[i].d[0]=out.rhs[19+i];time.lambda.value[i].d[0]=out.rhs[14+i];}
 auto fixed=typed_support::Constant<Jet>(p);fixed.omega=Jet(0);auto connection=c;for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)connection.scaled[k][i][j]=Jet(c.scaled[k][i][j].f);auto ft=NativeParts(time,fixed,connection);if(!ft.valid)return out;for(int n=0;n<22;++n)out.pole_time=std::max(out.pole_time,std::abs(ft.pole[n].d[0]));
 auto geometry=h::Geometry(time.metric);Jet normal,gradient2;for(int i=0;i<3;++i){normal+=time.beta.value[i]*Jet(p.domega[i]);for(int j=0;j<3;++j)gradient2+=time.chi.value*geometry.inverse[i][j]*Jet(p.domega[i]*p.domega[j]);Jet z=(time.lambda.value[i]-geometry.contracted[i])/Jet(2);out.z_time=std::max(out.z_time,std::abs(z.d[0]));}Jet nullness=gradient2-normal*normal/(time.alpha.value*time.alpha.value);out.nullness_time=std::abs(nullness.d[0]);
 out.valid=std::max({out.pole,out.tangent,out.second,out.pole_time})<=1e-10;for(Real v:out.rhs)out.valid=out.valid&&std::isfinite(v);return out;
}
}}
