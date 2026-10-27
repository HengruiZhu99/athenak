// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <fstream>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/spherical_tensor.hpp"
#include "z4c/hyperboloidal/spherical_matched_evolution.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 if(argc!=2)throw std::invalid_argument("radial fixture path required");
 std::ifstream in(argv[1]);int count;in>>count;if(!in||count!=196)throw std::runtime_error("fixture count");
 double errors[8]{},unattenuated=0,trace_error=0;
 for(int trial=0;trial<count;++trial){
  double r,chi,alpha,q[8],d[8],dd[8],beta,db,ddb,B,Bp,expected[8];
  in>>r>>chi>>alpha;for(auto*data:{q,d,dd})for(int j=0;j<8;++j)in>>data[j];
  in>>beta>>db>>ddb>>B>>Bp;for(auto&v:expected)in>>v;
  if(!in)throw std::runtime_error("truncated radial fixture");
  q[7]=beta+r;d[7]=db+1;dd[7]=ddb;
  auto u=h::SphericalJet(r,q,d,dd,h::CMCReference<double>{1,1});
  u.chi.value=chi;u.alpha.value=alpha;
  const double x[3]={r,0,0};const auto omega=h::CompositeOmegaAt(x,u);
  const auto parts=h::ConformalRHS(u,omega,4.,-.99,true);h::Z4cRHS<double> rate{};
  if(!h::AssembleInterior(parts,omega.omega,rate))throw std::runtime_error("bad radial tensor rate");
  const auto w=h::ReferenceTelegraphSupport(r);
  const double A=(1+r*r)/2,gradA[3]={r,0,0},gradw[3]={w.radial_derivative,0,0};
  const auto flux=h::RadialVector(B,Bp,0.,r);
  const auto old=h::ReferenceTelegraph(u,flux,gradA,w.value,gradw,.02,.005);
  const auto addition=h::ReferenceTelegraph(u,flux,gradA,w.value,gradw,.02,.005,
      h::ReferenceTelegraphAttenuation(alpha,A,1e-4));
  const double actual[8]={rate.chi,rate.metric[0][0],rate.a[0][0],rate.trace,rate.theta,
                          rate.lambda[0],h::WormholeLapseRate(x,u)+addition.lapse,addition.flux[0]};
  for(int f=0;f<8;++f)errors[f]=std::max(errors[f],std::abs(actual[f]-expected[f])/(1+std::abs(expected[f])));
  unattenuated=std::max(unattenuated,std::abs(old.lapse-addition.lapse));
  // Independent endpoint trace formula, allowing nonzero constraint traces.
  q[6]=0;auto edge=h::SphericalJet(1.,q,d,dd,h::CMCReference<double>{1,1});
  const double end[3]={1,0,0};const auto C=h::ScriRegularityTraces(edge,h::CompositeOmegaAt(end,edge,true));
  trace_error=std::max({trace_error,std::abs(C.a[0][0]-(q[2]+2*d[0]/3)),
                      std::abs(C.lambda[0]-q[5]),std::abs(C.a[1][1]+C.a[0][0]/2)});
 }
 std::string extra;if(in>>extra)throw std::runtime_error("trailing fixture data");
 // General non-diagonal, variable-metric boundary control: zero regularity
 // traces cancel the curvature pole in the chosen normalization. This fixture
 // construction is not a projection in the evolution code.
 double general_pole=0;
 for(int trial=0;trial<32;++trial){
  h::Z4cJet<double> u{};const double x[3]={.6,0,.8};
  for(int i=0;i<3;++i){u.metric.g[i][i]=1+.2*i;
   for(int j=0;j<i;++j)u.metric.g[i][j]=u.metric.g[j][i]=.03*std::sin(trial+i+j);
   for(int d=0;d<3;++d)for(int j=0;j<=i;++j)
     u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=.02*std::sin(trial+i+2*j+3*d);}
  auto g=h::Geometry(u.metric);double normal=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)normal+=g.inverse[i][j]*x[i]*x[j];
  u.chi.value=1/normal;u.alpha.value=1;u.trace.value=-3;
  for(int i=0;i<3;++i){u.chi.d[i]=.04*std::cos(trial+i);u.lambda.value[i]=g.contracted[i];}
  if(!h::FixedAngleScriShift(g,1.,u.chi.value,x,u.beta.value))throw std::runtime_error("manufactured scri shift");
  auto omega=h::CompositeOmegaAt(x,u,true);const auto before=h::ScriRegularityTraces(u,omega);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=-before.a[i][j];
  const auto parts=h::ConformalRHS(u,omega,4.,-.99,true);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)general_pole=std::max(general_pole,std::abs(parts.pole.a[i][j]));
 }

 std::cout<<std::setprecision(17)<<"{\"cases\":"<<count<<",\"scaled_errors_chi_g_A_Khat_Theta_Lambda_alpha_B\":[";
 for(int f=0;f<8;++f)std::cout<<(f?",":"")<<errors[f];
 std::cout<<"],\"scri_trace_error\":"<<trace_error<<",\"general_metric_curvature_pole\":"<<general_pole<<",\"missing_attenuation_falsifier\":"<<unattenuated<<"}\n";
 if(*std::max_element(errors,errors+8)>2e-11||trace_error>2e-13||general_pole>2e-13||unattenuated<1e-3)
   throw std::runtime_error("matched radial kernel gate failed");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
