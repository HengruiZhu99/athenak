// CMC endpoint first-jet velocity map. No endpoint evolution closure.
#pragma once
#include "goal1_velocity_momentum.hpp"
#include "goal1_endpoint_point.hpp"
namespace cubed_sphere {namespace velocity_endpoint {
namespace h=z4c::hyperboloidal;
inline h::LayerPoint<Jet> OuterReference(const h::LayerPoint<Real>&p,const Real x[3]){
 auto ref=typed_support::Constant<Jet>(p);Jet y[3],r2;for(int i=0;i<3;++i){y[i]=Jet(x[i]);y[i].d[i]=1;r2+=y[i]*y[i];}
 ref.omega=Jet(1)-r2;ref.L=ref.alpha=ref.state.alpha.value=Jet(1)+r2;ref.radius=Root(r2);ref.b=Jet(2)*ref.radius;
 for(int i=0;i<3;++i){ref.domega[i]=-Jet(2)*y[i];ref.beta[i]=ref.state.beta.value[i]=-Jet(2)*y[i];ref.dalpha[i]=ref.state.alpha.d[i]=Jet(2)*y[i];for(int j=0;j<3;++j){ref.omega_hessian[i][j]=Jet(i==j?-2:0);ref.state.alpha.dd[i][j]=Jet(i==j?2:0);ref.state.beta.d[i][j]=Jet(i==j?-2:0);}}return ref;
}
inline h::Z4cJet<Jet> NativeSecondValues(const h::Z4cJet<Real>&u){
 auto state=endpoint_point::First(u);auto scalar=[](const h::ScalarJet<Real>&a,h::ScalarJet<Jet>&b){for(int i=0;i<3;++i)for(int j=0;j<3;++j)b.value.h[i][j]=a.dd[i][j];};scalar(u.chi,state.chi);scalar(u.alpha,state.alpha);scalar(u.trace,state.trace);scalar(u.theta,state.theta);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k){state.beta.value[k].h[i][j]=u.beta.dd[i][j][k];state.lambda.value[k].h[i][j]=u.lambda.dd[i][j][k];for(int l=0;l<3;++l)state.metric.g[k][l].h[i][j]=u.metric.ddg[i][j][k][l];}return state;
}
struct Quotient{bool valid=false;Jet first;Real pole=0,tangent=0,second=0;};
// F=Omega*Q. Value and gradient of Q are uniquely determined by F's first
// and second jets if the value/tangential and second compatibility hold.
inline Quotient DivideSecond(const Jet&F,const Jet&omega){
 Quotient out;Real norm=0;for(int i=0;i<3;++i)norm+=omega.d[i]*omega.d[i];if(!(norm>0)||std::abs(omega.f)>1e-14)return out;
 for(int i=0;i<3;++i)out.first.f+=omega.d[i]*F.d[i]/norm;Real R[3][3],nn=0;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){R[i][j]=F.h[i][j]-omega.h[i][j]*out.first.f;nn+=omega.d[i]*omega.d[j]*R[i][j];}
 for(int j=0;j<3;++j){for(int i=0;i<3;++i)out.first.d[j]+=omega.d[i]*R[i][j]/norm;out.first.d[j]-=omega.d[j]*nn/(2*norm*norm);}
 out.pole=std::abs(F.f);for(int i=0;i<3;++i){out.tangent=std::max(out.tangent,std::abs(F.d[i]-omega.d[i]*out.first.f));for(int j=0;j<3;++j)out.second=std::max(out.second,std::abs(R[i][j]-omega.d[i]*out.first.d[j]-omega.d[j]*out.first.d[i]));}
 out.valid=std::max({out.pole,out.tangent,out.second})<=1e-10;return out;
}
struct ValueFirst{bool valid=false;std::array<Jet,10>value;Real compatibility=0;};
inline ValueFirst Assemble(const velocity_momentum::Parts<Jet>&parts,const Jet&omega){
 ValueFirst out;if(!parts.valid)return out;for(int n=0;n<10;++n){auto quotient=DivideSecond(parts.pole[n],omega);out.compatibility=std::max({out.compatibility,quotient.pole,quotient.tangent,quotient.second});if(!quotient.valid)return out;out.value[n].f=parts.regular[n].f+quotient.first.f;for(int d=0;d<3;++d)out.value[n].d[d]=parts.regular[n].d[d]+quotient.first.d[d];}out.valid=true;return out;
}
}}
