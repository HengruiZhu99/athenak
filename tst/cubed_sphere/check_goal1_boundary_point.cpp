// Reuse full native jets from saved v2 checkpoints; no new native trajectories.
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
#include "mesh/cubed_sphere/research_boundary_hierarchy.hpp"
namespace cubed_sphere {
namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace bh=cs::boundary_hierarchy;namespace ck=cs::canonical_ko;
using Jet=h::Z4cJet<Real>;
void NativeU(const Jet&u,const h::LayerPoint<Real>&p,Real U[4]) {
 auto result=h::EvolvedConstraints(u,h::CartesianOmega(u,p));if(!result.valid)throw std::runtime_error("invalid native constraint fixture");
 auto g=h::Geometry(u.metric);U[0]=u.theta.value/p.omega;
 for(int i=0;i<3;++i){U[i+1]=0;for(int j=0;j<3;++j)U[i+1]+=2*g.inverse[i][j]*result.z4.z_covector[j];}
}
void ShiftJet(Jet&u,const Jet&ref,const bh::Hierarchy&b,const Real dz[10],const Real dpi[10]) {
 Real pi[10],after[10],m[3][3]{},desired[10];ck::Momentum(u,ref,b.n,b.omega,pi);ck::InverseMetricChart(ck::MetricFrame(u),dz+1,m);
 for(int a=0;a<3;++a){u.chi.d[a]+=b.n[a]*dz[0];u.alpha.d[a]+=b.n[a]*dz[6];
  for(int i=0;i<3;++i){u.beta.d[a][i]+=b.n[a]*dz[7+i];for(int j=0;j<3;++j)u.metric.dg[a][i][j]+=b.n[a]*m[i][j];}}
 ck::Momentum(u,ref,b.n,b.omega,after);for(int j=0;j<10;++j)desired[j]=pi[j]+dpi[j]-after[j];h::Z4cRHS<Real> p{};ck::Lift(u,b.omega,desired,p);
 u.trace.value+=p.trace;u.theta.value+=p.theta;for(int i=0;i<3;++i){u.lambda.value[i]+=p.lambda[i];for(int j=0;j<3;++j)u.a.k[i][j]+=p.a[i][j];}
}
struct TimeResult {Real H=0,M=0,theta=0,z=0,theta_identity=0,z_identity=0,regular_q_forcing=0,gamma_variation=0;};
TimeResult ActualTime(const Jet&u,const h::LayerPoint<Real>&p,const Real x[3],const bh::Hierarchy&b) {
 auto con=h::EvolvedConstraints(u,h::CartesianOmega(u,p));h::Z4cRHS<Real> f{};h::GaugeRHS<Real> gauge{};
 if(!h::AssembleInterior(h::ConformalRHS(u,h::CartesianOmega(u,p),10/u.alpha.value,0.),p.omega,f) ||
    !rwm::Assemble(rwm::Gauge(p,u,rwm::ReferenceConnection(p,x)),p.omega,gauge))throw std::runtime_error("invalid full native RHS fixture");
 auto g=h::Geometry(u.metric);Real df[3][3][3]{},div=0,ddiv[3]{},invdot[3][3]{},connDot[3][3][3]{},gammaDot[3]{};
 for(int a=0;a<3;++a){div+=u.beta.d[a][a];for(int d=0;d<3;++d)ddiv[d]+=u.beta.dd[d][a][a];}
 for(int d=0;d<3;++d)for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  df[d][i][j]=-2*(u.alpha.d[d]*u.a.k[i][j]+u.alpha.value*u.a.dk[d][i][j])-Real(2./3.)*(u.metric.dg[d][i][j]*div+u.metric.g[i][j]*ddiv[d]);
  for(int a=0;a<3;++a)df[d][i][j]+=u.beta.d[d][a]*u.metric.dg[a][i][j]+u.beta.value[a]*u.metric.ddg[d][a][i][j]+u.metric.dg[d][a][j]*u.beta.d[i][a]+u.metric.g[a][j]*u.beta.dd[d][i][a]+u.metric.dg[d][i][a]*u.beta.d[j][a]+u.metric.g[i][a]*u.beta.dd[d][j][a];
 }
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int a=0;a<3;++a)for(int d=0;d<3;++d)invdot[i][j]-=g.inverse[i][a]*f.metric[a][d]*g.inverse[d][j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k){
  for(int a=0;a<3;++a)connDot[i][j][k]+=.5*(invdot[i][a]*(u.metric.dg[j][a][k]+u.metric.dg[k][a][j]-u.metric.dg[a][j][k])+g.inverse[i][a]*(df[j][a][k]+df[k][a][j]-df[a][j][k]));
  gammaDot[i]+=invdot[j][k]*g.connection[i][j][k]+g.inverse[j][k]*connDot[i][j][k];
 }
 TimeResult t;t.H=con.hamiltonian;t.M=sqrt(con.momentum_conformal_norm2);t.theta=f.theta;
 for(Real step:{Real(1e-5),Real(5e-6)}){auto plus=u.metric,minus=u.metric;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){plus.g[i][j]+=step*f.metric[i][j];minus.g[i][j]-=step*f.metric[i][j];for(int d=0;d<3;++d){plus.dg[d][i][j]+=step*df[d][i][j];minus.dg[d][i][j]-=step*df[d][i][j];}}
  auto gp=h::Geometry(plus),gm=h::Geometry(minus);for(int i=0;i<3;++i)t.gamma_variation=std::max(t.gamma_variation,fabs((gp.contracted[i]-gm.contracted[i])/(2*step)-gammaDot[i])/std::max(Real(1),fabs(gammaDot[i])));
 }
 t.theta_identity=fabs(f.theta-u.alpha.value*con.hamiltonian/(2*p.omega));
 for(int i=0;i<3;++i){Real zdot=0;for(int j=0;j<3;++j)zdot+=.5*(f.metric[i][j]*(u.lambda.value[j]-g.contracted[j])+u.metric.g[i][j]*(f.lambda[j]-gammaDot[j]));
  t.z=std::max(t.z,fabs(zdot));t.z_identity=std::max(t.z_identity,fabs(zdot-u.alpha.value*con.momentum[i]/p.omega));}
 Real z[10]{},pi[10],qdot[10]{};b.ConfigurationNormal(u,p.state,z);ck::Momentum(u,p.state,b.n,p.omega,pi);
 qdot[0]=f.chi;ck::MetricChart(ck::MetricFrame(u),f.metric,qdot+1);qdot[6]=gauge.alpha;for(int i=0;i<3;++i)qdot[7+i]=gauge.beta[i];
 for(int j=0;j<10;++j)t.regular_q_forcing=std::max(t.regular_q_forcing,fabs(qdot[j]-b.beta*z[j]-pi[j]));return t;
}
void Audit(const Jet&u,const h::LayerPoint<Real>&p,const Real x[3],const Real n[3],const std::string&kind,int index) {
 bh::Hierarchy b;if(!b.Build(u,n,p.omega))throw std::runtime_error("singular full native boundary hierarchy");
 Real data[6]{},incoming[10],outgoing[10],target[10],forcing[4];b.Target(u,p.state,data,incoming,outgoing,target,forcing);
 Real rowerr=0;
 for(int sign:{1,-1})for(int col=0;col<10;++col){Jet plus=u,minus=u;Real dq[10]{},dp[10]{};dq[col]=1;dp[col]=sign*b.c;ShiftJet(plus,p.state,b,dq,dp);dq[col]=-1;dp[col]=-sign*b.c;ShiftJet(minus,p.state,b,dq,dp);
  Real up[4],um[4];NativeU(plus,p,up);NativeU(minus,p,um);for(int i=0;i<4;++i){Real expected=(sign==1?b.Jin:b.Jout)[i][col];rowerr=std::max(rowerr,fabs((up[i]-um[i])/2-expected)/std::max(Real(1),fabs(expected)));}}
 auto corrected=u;bh::IncomingFixtureCorrection(corrected,p.state,b,incoming,target);
 Real U[4],ui[10],uo[10],unused[10],hh[4];NativeU(corrected,p,U);b.Target(corrected,p.state,data,ui,uo,unused,hh);
 Real constraint=0,free=0,outchange=0,corr=0,omitted=0,trace=0,atrace=0,config=0;
 for(int i=0;i<4;++i){constraint=std::max(constraint,fabs(U[i]));omitted=std::max(omitted,fabs(forcing[i]));}
 for(int i=0;i<6;++i){Real f=0;for(int j=0;j<10;++j)f+=b.F[i][j]*ui[j];free=std::max(free,fabs(f));}
 for(int j=0;j<10;++j){outchange=std::max(outchange,fabs(uo[j]-outgoing[j]));corr=std::max(corr,fabs(target[j]-incoming[j]));}
 auto g=h::Geometry(u.metric);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){atrace+=g.inverse[i][j]*(corrected.a.k[i][j]-u.a.k[i][j]);config=std::max(config,fabs(corrected.metric.g[i][j]-u.metric.g[i][j]));}
 for(int a=0;a<3;++a){Real v=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)v+=g.inverse[i][j]*(corrected.metric.dg[a][i][j]-u.metric.dg[a][i][j]);trace=std::max(trace,fabs(v));}
 auto before=ActualTime(u,p,x,b),after=ActualTime(corrected,p,x,b);
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"Omega\":"<<p.omega<<",\"radius\":"<<p.radius<<",\"beta\":"<<b.beta<<",\"c\":"<<b.c
  <<",\"normal_row_error\":"<<rowerr<<",\"corrected_constraint_error\":"<<constraint<<",\"free_condition_error\":"<<free<<",\"outgoing_change\":"<<outchange
  <<",\"incoming_correction\":"<<corr<<",\"omitted_full_forcing\":"<<omitted<<",\"metric_derivative_trace_change\":"<<trace<<",\"A_trace_change\":"<<fabs(atrace)<<",\"configuration_value_change\":"<<config
  <<",\"H_after\":"<<after.H<<",\"M_after\":"<<after.M<<",\"Theta_time_after\":"<<after.theta<<",\"Zcov_time_after\":"<<after.z
  <<",\"naive_value_only_Theta_time_defect\":"<<after.theta_identity<<",\"naive_value_only_Zcov_time_defect\":"<<after.z_identity<<",\"Gamma_time_variation_error\":"<<after.gamma_variation
  <<",\"full_q_rhs_remainder_before\":"<<before.regular_q_forcing<<",\"full_q_rhs_remainder_after\":"<<after.regular_q_forcing<<",\"K\":[";
 for(int i=0;i<10;++i){std::cout<<(i?",":"")<<"[";for(int j=0;j<10;++j)std::cout<<(j?",":"")<<b.K[i][j];std::cout<<"]";}std::cout<<"],\"R\":[";
 for(int i=0;i<10;++i){std::cout<<(i?",":"")<<"[";for(int j=0;j<10;++j)std::cout<<(j?",":"")<<b.R[i][j];std::cout<<"]";}std::cout<<"]}\n";
}
template<class T>void Binary(std::istream&in,T&x){in.read(reinterpret_cast<char*>(&x),sizeof(x));if(!in)throw std::runtime_error("truncated fixture");}
void NativeCase(const char*checkpoint,const char*reference,const std::string&label) {
 std::ifstream ckfile(checkpoint,std::ios::binary),ref(reference,std::ios::binary);if(!ckfile||!ref)throw std::runtime_error("missing fixture");
 char magic[16],tag[64];ckfile.read(magic,16);if(std::string(magic,16)!="ATHENA_CSP_CK02!")throw std::runtime_error("wrong checkpoint version");ckfile.read(tag,64);
 Real signature[17];ckfile.read(reinterpret_cast<char*>(signature),sizeof(signature));std::uint64_t fingerprint;Real time;std::int64_t cycle;Binary(ckfile,fingerprint);Binary(ckfile,time);Binary(ckfile,cycle);
 std::string line;std::getline(ref,line);if(line!="CUBED_SPHERE_REFERENCE_V1")throw std::runtime_error("wrong immutable reference");
 for(int region=0;region<2;++region){std::uint64_t shape[5],n,rs[5];for(auto &v:shape)Binary(ckfile,v);Binary(ref,n);for(auto &v:rs)Binary(ref,v);for(int i=0;i<5;++i)if(shape[i]!=rs[i])throw std::runtime_error("shape mismatch");
  std::vector<cs::Cell> cells(n);ref.read(reinterpret_cast<char*>(cells.data()),n*sizeof(cs::Cell));
  DvceArray5D<Real> dev("saved full native deviation",shape[0],shape[1],shape[2],shape[3],shape[4]);auto host=Kokkos::create_mirror_view(dev);std::vector<Real> values(host.size()),background(host.size());
  ckfile.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(Real));ref.read(reinterpret_cast<char*>(background.data()),background.size()*sizeof(Real));if(!ref||!ckfile)throw std::runtime_error("truncated native payload");
  if(region==0)continue;for(size_t i=0;i<values.size();++i)host.data()[i]=values[i]-background[i];Kokkos::deep_copy(dev,host);
  auto fields=h::BindCartesianFields(dev);auto op=cs::MakeResearchRadialSBP(shape[4]-8,4,1./(shape[4]-8));int count=0;
  for(auto &cell:cells)if(cell.i==static_cast<int>(shape[4])-5){Real n[3],norm=0;for(int i=0;i<3;++i)norm+=cell.geometry.J[0][i]*cell.geometry.J[0][i];norm=sqrt(norm);for(int i=0;i<3;++i)n[i]=cell.geometry.J[0][i]/norm;
   const auto u=cs::LoadProbeJet(fields,dev,cell,cell.reference.state,op);Audit(u,cell.reference,cell.geometry.x,n,label,count++);}
  if(count!=static_cast<int>(shape[0]*(shape[2]-8)*(shape[3]-8)))throw std::runtime_error("incomplete native outer coverage");
 }
 char tail;ref.read(&tail,1);if(!ref.eof())throw std::runtime_error("reference Cell ABI mismatch");
}
void Synthetic() {
 h::LayerReference<Real> reference(1.,.5,{true,.05,.95});Real normals[3][3]={{1,0,0},{.8,-.48,.36},{1/sqrt(Real(3)),1/sqrt(Real(3)),1/sqrt(Real(3))}};int index=0;
 for(Real radius:{.25,.7,.98})for(auto &n:normals){Real x[3];for(int i=0;i<3;++i)x[i]=radius*n[i];auto p=reference.At(x[0],x[1],x[2]);Audit(p.state,p,x,n,"synthetic-reference",index);
  auto u=p.state;u.chi.value*=1.01;u.alpha.value*=1.02;u.trace.value+=.02;u.theta.value+=.001;
  auto frame=ck::MetricFrame(u);Real v[5]={.01,-.007,.003,.006,-.002},m[3][3]{};ck::InverseMetricChart(frame,v,m);
  for(int a=0;a<3;++a){u.chi.d[a]+=.01*(a+1);u.alpha.d[a]-=.02*(a+1);for(int i=0;i<3;++i){u.lambda.value[i]+=(a==0?.01*(i+1):0);u.beta.d[a][i]+=.007*(a+1)*(i+1);for(int j=0;j<3;++j){u.metric.dg[a][i][j]+=(a+1)*(m[i][j]+.003*u.metric.g[i][j]);u.a.k[i][j]+=(a==0?.3*m[i][j]:0);}}}
  Audit(u,p,x,n,"synthetic-off-constraint",index++);
 }
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 {if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else if(argc==1)Synthetic();else throw std::runtime_error("use checkpoint reference label, or noargs synthetic");}
 Kokkos::finalize();return 0;
}
