// Independent binding of saved exact-flat jets to the actual C0 and RWM rows.
// No evolution, projections, reference RHS counterterm, or absent jet padding.
#include "reference_wave_map.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace hyp=z4c::hyperboloidal;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
using Raw=std::array<double,22>;
bool Configuration(int col){return col<=6||col>=18;}
void Read(double&v){if(!(std::cin>>v)||!std::isfinite(v))throw std::runtime_error("missing/nonfinite input");}
hyp::ScalarJet<double> ReadJet(bool second){
 hyp::ScalarJet<double>s{};Read(s.value);for(double&v:s.d)Read(v);
 for(auto&row:s.dd)for(double&v:row){if(second)Read(v);else v=std::numeric_limits<double>::quiet_NaN();}
 return s;
}
void Put(hyp::Z4cJet<double>&u,int col,const hyp::ScalarJet<double>&s){
 if(col==0){u.chi=s;return;}if(col==7){u.trace=s;return;}if(col==17){u.theta=s;return;}if(col==18){u.alpha=s;return;}
 if(col>=1&&col<=6){int i=ti[col-1],j=tj[col-1];u.metric.g[i][j]=u.metric.g[j][i]=s.value;
  for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=s.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=s.dd[d][e];}return;}
 if(col>=8&&col<=13){int i=ti[col-8],j=tj[col-8];u.a.k[i][j]=u.a.k[j][i]=s.value;for(int d=0;d<3;++d)u.a.dk[d][i][j]=u.a.dk[d][j][i]=s.d[d];return;}
 const bool shift=col>=19;int i=col-(shift?19:14);auto&v=shift?u.beta:u.lambda;v.value[i]=s.value;
 for(int d=0;d<3;++d){v.d[d][i]=s.d[d];for(int e=0;e<3;++e)v.dd[d][e][i]=s.dd[d][e];}
}
hyp::OmegaJet<double> Omega(const hyp::LayerPoint<double>&p,const hyp::Z4cJet<double>&u){
 hyp::OmegaJet<double>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
 hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 if(!std::isfinite(o.normal))throw std::runtime_error("invalid stationary Omega normal");for(double v:o.dnormal)if(!std::isfinite(v))throw std::runtime_error("invalid stationary Omega normal derivative");return o;
}
Raw Pack(const hyp::Z4cRHS<double>&f,const hyp::GaugeRHS<double>&g){
 Raw r{};r[0]=f.chi;r[7]=f.trace;r[17]=f.theta;r[18]=g.alpha;
 for(int t=0;t<6;++t){r[1+t]=f.metric[ti[t]][tj[t]];r[8+t]=f.a[ti[t]][tj[t]];}
 for(int i=0;i<3;++i){r[14+i]=f.lambda[i];r[19+i]=g.beta[i];}return r;
}
template<std::size_t N>void Print(const std::array<double,N>&a){std::cout<<'[';for(std::size_t i=0;i<N;++i){if(i)std::cout<<',';if(!std::isfinite(a[i]))throw std::runtime_error("nonfinite output");std::cout<<a[i];}std::cout<<']';}
void Batch(){
 hyp::LayerReference<double>reference(1.,.5,{true,.05,.95});reference.Validate();double time;
 while(std::cin>>time){if(!std::isfinite(time))throw std::runtime_error("nonfinite time");std::array<double,3>x{};for(double&v:x)Read(v);
  hyp::Z4cJet<double>u{};for(int col=0;col<22;++col)Put(u,col,ReadJet(Configuration(col)));const auto submitted_omega=ReadJet(true);
  const auto p=reference.At(x[0],x[1],x[2]);const auto o=Omega(p,u);if(!(p.omega>0))throw std::runtime_error("point outside finite Omega domain");
  hyp::Z4cRHS<double>f{};if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),o.omega,f))throw std::runtime_error("invalid actual C0 source");
  const auto c=rwm::ReferenceConnection(p,x.data());hyp::GaugeRHS<double>g{};if(!rwm::Assemble(rwm::Gauge(p,u,c),p.omega,g))throw std::runtime_error("invalid actual RWM source");
  const auto q=hyp::EvolvedConstraints(u,o);if(!q.valid)throw std::runtime_error("invalid actual constraints");const auto geom=hyp::Geometry(u.metric);if(!geom.valid)throw std::runtime_error("invalid actual spatial metric");
  std::array<double,8>constraints{q.hamiltonian,q.momentum[0],q.momentum[1],q.momentum[2],q.z4.z_covector[0],q.z4.z_covector[1],q.z4.z_covector[2],q.z4.theta_physical};
  std::array<double,2>normals{geom.determinant-1,0},rate_normals{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){normals[1]+=geom.inverse[i][j]*u.a.k[i][j];rate_normals[0]+=geom.inverse[i][j]*f.metric[i][j];rate_normals[1]+=geom.inverse[i][j]*f.a[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)rate_normals[1]-=geom.inverse[i][k]*geom.inverse[j][l]*u.a.k[k][l]*f.metric[i][j];}
  std::array<double,13>omega_difference{};int n=0;omega_difference[n++]=submitted_omega.value-p.omega;for(int i=0;i<3;++i)omega_difference[n++]=submitted_omega.d[i]-p.domega[i];for(int i=0;i<3;++i)for(int j=0;j<3;++j)omega_difference[n++]=submitted_omega.dd[i][j]-p.omega_hessian[i][j];
  std::array<double,4>source{};rwm::ScaledSource(p,u,c,source.data());std::array<double,36>connection{};n=0;for(int a=0;a<4;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)connection[n++]=c.scaled[a][i][j];
  std::cout<<"{\"point\":";Print(std::array<double,4>{time,x[0],x[1],x[2]});std::cout<<",\"actual_rhs22\":";Print(Pack(f,g));std::cout<<",\"physical_constraints8\":";Print(constraints);std::cout<<",\"input_normals2\":";Print(normals);std::cout<<",\"rate_normals2\":";Print(rate_normals);std::cout<<",\"submitted_minus_native_omega13\":";Print(omega_difference);std::cout<<",\"scaled_source4\":";Print(source);std::cout<<",\"scaled_reference_connection36\":";Print(connection);std::cout<<"}\n";
 }
 if(!std::cin.eof())throw std::runtime_error("malformed batch tail");
}
int main(){try{std::cout<<std::setprecision(17);Batch();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
