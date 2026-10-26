// Private first spatial reference-connection jet; reference files unchanged.
#pragma once
#include "goal1_native_ad.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace cubed_sphere {namespace reference_connection_ad {
namespace h=z4c::hyperboloidal;
struct Result {rwm::Connection<Jet>connection;Real omega_third[3][3][3]{};};
inline Result Build(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&p,const Real x[3]){
 Result out;const auto original=rwm::ReferenceConnection(p,x);out.connection.valid=original.valid;if(!original.valid)return out;
 const Real r=p.radius;
 if(r<=descriptor.layer.r0){for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.connection.scaled[k][i][j]=Jet(original.scaled[k][i][j]);return out;}
 const Real S=descriptor.scri_radius,a=descriptor.curvature_radius;const auto w=h::SmoothCutoff(r,Real(descriptor.layer.r0),Real(descriptor.layer.r1));Real n[3];for(int i=0;i<3;++i)n[i]=x[i]/r;
 Real op=0,opp=0;for(int i=0;i<3;++i){op+=n[i]*p.domega[i];for(int j=0;j<3;++j)opp+=n[i]*n[j]*p.omega_hessian[i][j];}
 const Real outer=(S-r)*(S+r)/(2*a*S),o3=w.ddd*(outer-1)+3*w.dd*(-r/(a*S))+3*w.d*(-1/(a*S));
 if(r<descriptor.layer.r1)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)out.omega_third[i][j][k]=(o3-3*opp/r+3*op/(r*r))*n[i]*n[j]*n[k]+(opp/r-op/(r*r))*((i==j?n[k]:0)+(i==k?n[j]:0)+(j==k?n[i]:0));
 auto radial=[&](Real value,Real derivative){Jet q(value);for(int d=0;d<3;++d)q.d[d]=derivative*n[d];return q;};
 Jet radius=radial(r,1),omega=radial(p.omega,op),oprime=radial(op,opp),b=radial(p.b,(w.value+r*w.d)/a),bp=radial((w.value+r*w.d)/a,(2*w.d+r*w.dd)/a),L=radial(p.L,-r*opp),alpha(p.alpha),normal[3],coordinate[3],ograd[3],ohess[3][3];
 for(int i=0;i<3;++i){alpha.d[i]=p.dalpha[i];normal[i]=Jet(n[i]);coordinate[i]=Jet(x[i]);ograd[i]=Jet(p.domega[i]);for(int d=0;d<3;++d){normal[i].d[d]=((i==d?1:0)-n[i]*n[d])/r;coordinate[i].d[d]=i==d?1:0;ograd[i].d[d]=p.omega_hessian[d][i];}for(int j=0;j<3;++j){ohess[i][j]=Jet(p.omega_hessian[i][j]);for(int d=0;d<3;++d)ohess[i][j].d[d]=out.omega_third[d][i][j];}}
 const Jet ar=L*(omega*bp-b*oprime)/(alpha*alpha*alpha),at=b/(alpha*radius);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){out.connection.scaled[0][i][j]=ar*normal[i]*normal[j]+at*(Jet(i==j?1:0)-normal[i]*normal[j]);for(int k=0;k<3;++k)out.connection.scaled[k+1][i][j]=-(k==i?ograd[j]:Jet(0))-(k==j?ograd[i]:Jet(0))-coordinate[k]*omega*ohess[i][j]/L;}
 for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j){auto&q=out.connection.scaled[k][i][j];q.f=original.scaled[k][i][j];for(auto&row:q.h)for(auto&v:row)v=0;out.connection.valid=out.connection.valid&&JetFinite(q);}
 return out;
}
}}
