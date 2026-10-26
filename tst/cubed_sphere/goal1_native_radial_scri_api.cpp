// Local spherical-sector caller of the complete native Cartesian tensor source.
// No independently evolved gradient, momentum, or first-order spatial reduction.
#include "goal1_native_endpoint.hpp"
#include "goal1_radial_scri_clock.hpp"
#include "goal1_native_algebraic_rhs.hpp"
#include "goal1_operator_oracle.hpp"
#include <string>
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace ne=cs::native_endpoint;
namespace {std::string error;const int rows[10]={0,18,7,17,19,14,1,4,8,11};
h::LayerReference<Real> reference(1.,.5,{true,.05,.95});
cs::radial_scri_clock::Height&Height(){static cs::radial_scri_clock::Height value(48);return value;}
h::Z4cJet<Real>Jet(const Real*v,const Real*d,const Real*dd,Real r){h::Z4cJet<Real>a{};const int ids[4]={0,1,2,3};h::ScalarJet<Real>*q[4]={&a.chi,&a.alpha,&a.trace,&a.theta};for(int i=0;i<4;++i){q[i]->value=v[ids[i]];q[i]->d[0]=d[ids[i]];q[i]->dd[0][0]=dd[ids[i]];}a.beta.value[0]=v[4];a.beta.d[0][0]=d[4];a.beta.dd[0][0][0]=dd[4];a.lambda.value[0]=v[5];a.lambda.d[0][0]=d[5];a.lambda.dd[0][0][0]=dd[5];for(int i=0;i<3;++i){int g=i==0?6:7,k=i==0?8:9;a.metric.g[i][i]=v[g];a.metric.dg[0][i][i]=d[g];a.metric.ddg[0][0][i][i]=dd[g];a.a.k[i][i]=v[k];a.a.dk[0][i][i]=d[k];}Real n[3]={1,0,0};return h::CartesianRadialJet(a,r,n);}
void TensorRates(const std::array<Real,22>&f,h::Z4cRHS<Real>&v){v.chi=f[0];v.trace=f[7];v.theta=f[17];for(int i=0;i<6;++i)v.metric[ne::ma[i]][ne::mb[i]]=v.metric[ne::mb[i]][ne::ma[i]]=f[1+i];for(int i=0;i<6;++i)v.a[ne::ma[i]][ne::mb[i]]=v.a[ne::mb[i]][ne::ma[i]]=f[8+i];for(int i=0;i<3;++i)v.lambda[i]=f[14+i];}
}
extern "C" {
const char* radial_error(){return error.c_str();}
int radial_init(){try{Kokkos::initialize();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
void radial_finalize(){Kokkos::finalize();}
Real radial_height_constant(){return Height().constant;}
int radial_reference(int n,const Real*r,Real*out){try{for(int i=0;i<n;++i){auto p=reference.At(r[i],0.,0.);for(int k=0;k<10;++k){auto s=cs::operator_oracle::Component(p.state,rows[k]);out[i*30+k]=s.value;out[i*30+10+k]=s.d[0];out[i*30+20+k]=s.dd[0][0];}}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int radial_oracle(int n,const Real*r,Real time,Real epsilon,Real centre,Real*out,Real*rate){try{for(int i=0;i<n;++i){auto q=cs::radial_scri_clock::Evaluate(r[i],time,epsilon,Height(),.2,centre);for(int k=0;k<10;++k){out[i*30+k]=q.f[k].a[0][0];out[i*30+10+k]=q.f[k].a[1][0];out[i*30+20+k]=2*q.f[k].a[2][0];rate[i*10+k]=q.f[k].a[0][1];}}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int radial_jets(int n,const Real*r,const Real*v,const Real*d,const Real*dd,Real*out){try{for(int i=0;i<n;++i){auto u=Jet(v+10*i,d+10*i,dd+10*i,r[i]);for(int k=0;k<22;++k){auto q=cs::operator_oracle::Component(u,k);Real*o=out+(i*22+k)*13;o[0]=q.value;for(int j=0;j<3;++j){o[1+j]=q.d[j];for(int e=0;e<3;++e)o[4+j*3+e]=q.dd[j][e];}}}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
int radial_source(int n,const Real*r,const Real*v,const Real*d,const Real*dd,const Real*extra,Real*out,Real*meta,Real*endpoint_full){try{for(int i=0;i<n;++i){auto u=Jet(v+10*i,d+10*i,dd+10*i,r[i]);auto p=reference.At(r[i],0.,0.);std::array<Real,22>f{};Real pole=0,ptangent=0;
 if(r[i]==1){Real x[3]={1,0,0};auto pp=ne::OuterReference(p,x);pp.omega.f=0;auto parts=ne::NativeParts(ne::Spatial(u),pp,ne::OuterConnection(x));if(!parts.valid)throw std::runtime_error("invalid known-scri native parts");for(int k=0;k<22;++k){f[k]=parts.regular[k].f-parts.pole[k].d[0]/2;pole=std::max(pole,std::abs(parts.pole[k].f));ptangent=std::max({ptangent,std::abs(parts.pole[k].d[1]),std::abs(parts.pole[k].d[2])});endpoint_full[k]=f[k];}}
 else{if(!(p.omega>0))throw std::runtime_error("nonpositive interior Omega");Real x[3]={r[i],0,0};f=cs::typed_support::Core(u,p,rwm::ReferenceConnection(p,x));}
 Real raw=0;for(Real q:f){if(!std::isfinite(q))throw std::runtime_error("nonfinite native source");raw=std::max(raw,std::abs(q));}h::Z4cRHS<Real>z{};TensorRates(f,z);z.chi+=extra[10*i];z.trace+=extra[10*i+2];z.theta+=extra[10*i+3];z.lambda[0]+=extra[10*i+5];z.metric[0][0]+=extra[10*i+6];z.a[0][0]+=extra[10*i+8];for(int k=1;k<3;++k){z.metric[k][k]+=extra[10*i+7];z.a[k][k]+=extra[10*i+9];}auto tangent=cs::native_algebraic_rhs::Tangent(u.metric,u.a,z);if(!tangent.valid)throw std::runtime_error("native algebraic tangent invalid");Real rates[10]={z.chi,f[18]+extra[10*i+1],z.trace,z.theta,f[19]+extra[10*i+4],z.lambda[0],z.metric[0][0],z.metric[1][1],z.a[0][0],z.a[1][1]};for(int k=0;k<10;++k)out[10*i+k]=rates[k];
 auto omega=h::CartesianOmega(u,p);auto c=h::EvolvedConstraints(u,omega);if(!c.valid)throw std::runtime_error("native constraint invalid");auto g=h::Geometry(u.metric);Real trace=0,dg=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)trace+=g.inverse[a][b]*u.a.k[a][b];for(int a=0;a<3;++a){Real t=0;for(int b=0;b<3;++b)for(int k=0;k<3;++k)t+=g.inverse[b][k]*u.metric.dg[a][b][k];dg+=t*t;}Real light=u.alpha.value*sqrt(u.chi.value/u.metric.g[0][0]);Real values[18]={p.omega,c.hamiltonian,sqrt(c.momentum_conformal_norm2),sqrt(c.momentum_physical_norm2),sqrt(c.z4.z_conformal_norm2),sqrt(c.z4.z_physical_norm2),u.theta.value,g.determinant-1,trace,sqrt(dg),c.null_residual,pole,ptangent,raw,-u.beta.value[0]-light,-u.beta.value[0]+light,c.momentum[0],c.z4.z_covector[0]};for(int k=0;k<18;++k)meta[18*i+k]=values[k];
 }return 0;}catch(const std::exception&e){error=e.what();return 1;}}
}
