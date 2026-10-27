// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "z4c/hyperboloidal/puncture_position.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
using J=cubed_sphere::Jet;
h::Z4cJet<Real> State(const Real x[3],const h::PuncturePosition &point,Real exponent,bool curved){
 auto u=manufactured::State(x,curved);J y[3],r2(0);for(int i=0;i<3;++i){y[i]=J::Variable(x[i]-point.position[i],i);r2=r2+y[i]*y[i];}
 const Real B[3][3]={{.12,.03,-.02},{-.04,.08,.01},{.02,-.03,-.05}},b[3]={.03,-.02,.01};
 J bump(0);if(r2.f<.18*.18)bump=cubed_sphere::Power(J(1)-r2/J(.18*.18),8);
 for(int i=0;i<3;++i){J linear(b[i]);for(int j=0;j<3;++j)linear=linear+J(B[i][j])*y[j];
  J beta;
  if(exponent==-2)beta=J(b[i]);else if(exponent==-1)beta=linear;
  else{J fractional(0);if(r2.f==0){if(exponent<=1)throw std::domain_error("point-collocated fractional source needs a weak/endpoint discretization");}
    else fractional=cubed_sphere::Power(r2/J(.18*.18),exponent/2)*y[i]*J(.25);
    beta=J(-x[i])+bump*(linear+fractional);beta.d[i]-=1;}
  u.beta.value[i]=beta.f;for(int a=0;a<3;++a){u.beta.d[a][i]=beta.d[a];for(int c=0;c<3;++c)u.beta.dd[a][c][i]=beta.h[a][c];}
 }
 // Exact full-variable-metric source, independently cross-checked below via
 // the metric-rate variation. This is not Einstein black-hole initial data.
 const auto rate=h::ShiftMetricRate(u.metric,u.beta);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.a.k[i][j]=rate.h[i][j]/2;for(int a=0;a<3;++a)u.a.dk[a][i][j]=rate.dh[a][i][j]/2;}
 return u;
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 const int n=argc>1?std::stoi(argv[1]):8,location=argc>4?std::stoi(argv[4]):0;
 const Real exponent=argc>2?std::stod(argv[2]):1.09,power=argc>3?std::stod(argv[3]):2;
 const bool curved=argc>5?std::stoi(argv[5]):true,scaled=argc>6?std::stoi(argv[6]):false;
 const Real mix=argc>7?std::stod(argv[7]):.5;
 h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25,{.65,1},1,n,2,2),4,true);
 auto &g=stage.grid;h::PuncturePosition point;
 if(location==1)point.position={.037,.013,-.021};
 if(location==2||location==3){size_t nearest=0;Real r=1;for(size_t p=0;p<g.active.size();++p){Real q=0;for(Real x:g.geometry[p].x)q+=x*x;if(q<r){r=q;nearest=p;}}
  for(int i=0;i<3;++i)point.position[i]=g.geometry[nearest].x[i];if(location==3)point.position[0]+=1e-6;}
 multigrid::Vector source,exact,solution(3*g.active.size(),0);std::vector<h::MetricJet<Real>> metric;std::vector<Real> chi;
 h::ShiftSolveControl control;control.conformal_norm_fraction=mix;control.maximum_residual_tolerance=1e-7;
 if(scaled)control.column_scale.resize(g.active.size());
 Real minimum_radius=1,source_check=0,min_chi=1;
 for(size_t p=0;p<g.active.size();++p){const auto &x=g.geometry[p].x;const auto u=State(x,point,exponent,curved);metric.push_back(u.metric);
  const Real r2=point.RadiusSquared(x);minimum_radius=std::min(minimum_radius,std::sqrt(r2));
  Real c=1;if(power>0){if(r2==0)throw std::domain_error("point-collocated infinite physical-volume norm needs explicit endpoint measure");c=std::pow(r2/(r2+.04),power/2);}
  chi.push_back(c);min_chi=std::min(min_chi,c);
  if(scaled){const Real t=std::max(Real(0),1-r2/.09);control.column_scale[p]=1-.5*t*t*t*t;}
  Real s[3],independent[3];h::EllipticShiftSource(u,h::Geometry(u.metric),0.,s);h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),u.beta,independent);
  for(int i=0;i<3;++i){source_check=std::max(source_check,std::abs(s[i]-independent[i]));source.push_back(s[i]);exact.push_back(u.beta.value[i]);}
 }
 stage.SetGeometry(metric,chi);
 const auto boundary=[&](const std::array<Real,3>&x){const auto u=State(x.data(),point,exponent,curved);return std::array<Real,3>{u.beta.value[0],u.beta.value[1],u.beta.value[2]};};
 const auto start=std::chrono::steady_clock::now();const auto report=stage.Solve(source,boundary,solution,1e-10,0.,160,control);
 multigrid::Vector error(exact.size());Real max_error=0,core_error=0;
 for(size_t p=0;p<g.active.size();++p)for(int i=0;i<3;++i){error[3*p+i]=solution[3*p+i]-exact[3*p+i];max_error=std::max(max_error,std::abs(error[3*p+i]));if(point.InNeighborhood(g.geometry[p].x,.1))core_error=std::max(core_error,std::abs(error[3*p+i]));}
 std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"exponent\":"<<exponent<<",\"chi_power\":"<<power<<",\"location\":"<<location
  <<",\"curved\":"<<curved<<",\"bounded_scaling\":"<<scaled<<",\"conformal_norm_fraction\":"<<mix<<",\"converged\":"<<report.converged
  <<",\"iterations\":"<<report.iterations<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
  <<",\"minimum_radius\":"<<minimum_radius<<",\"minimum_chi\":"<<min_chi<<",\"source_identity_error\":"<<source_check
  <<",\"physical_residual_rms\":"<<report.physical_residual_rms<<",\"physical_residual_l2\":"<<report.physical_residual_l2
  <<",\"physical_volume_quadrature\":"<<report.physical_volume<<",\"conformal_residual_rms\":"<<report.conformal_residual_rms<<",\"residual_max\":"<<report.residual_max
  <<",\"physical_error_rms\":"<<std::sqrt(stage.Dot(error,error))<<",\"error_max\":"<<max_error<<",\"puncture_error_max\":"<<core_error<<"}\n";
 if(!report.converged||source_check>1e-11)status=1;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=2;}Kokkos::finalize();return status;}
