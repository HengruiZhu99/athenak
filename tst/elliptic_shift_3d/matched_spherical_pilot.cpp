// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Genuine Cartesian/shell host evolution, no radial update or projections.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/spherical_matched_evolution.hpp"
#include "composed_metric_operator_probe.hpp"
namespace h=z4c::hyperboloidal;namespace c=cubed_sphere;using V=multigrid::Vector;
c::CompositeMesh Mesh(int n){if(n!=6&&n!=8&&n!=10)throw std::invalid_argument("N6/8/10 budget");
 return c::CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,n,2,2);}
void Record(h::SphericalMatchedEvolution&e,const std::string&prefix,double time,double dt){
 const auto rhs=e.RHS(e.state,dt);const auto&g=e.shift.grid;const size_t N=g.active.size();
 const auto padded=e.Padded(e.state);std::vector<h::Z4cJet<Real>>edge;
 for(size_t p=0;p<e.scri->rows.size();++p){const auto&x=e.scri->rows[p].geometry.x;
  const auto s=e.scri->Sample(p,padded,e.scri->values,22);
  edge.push_back(h::UnpackComposite(x,s.value,s.d,s.dd,true,0,2));}
 composed_metric_probe::Operator actual(e.shift,*e.scri,e.jets,edge);
 V rates(6*N),boundary(6*edge.size());
 for(size_t p=0;p<N;++p)for(int t=0;t<6;++t)rates[6*p+t]=rhs[22*p+1+t];
 for(size_t p=0;p<edge.size();++p)for(int t=0;t<6;++t)boundary[6*p+t]=e.boundary_rhs[22*p+1+t];
 const auto freeze=actual.Variation(rates,boundary);
 std::ofstream csv(prefix+".csv");csv<<std::setprecision(17)
 <<"x,y,z,r,weight,alpha,chi,g_rr,A_rr,Khat,Theta,Lambda_r,beta_r,B,H,M_norm,freeze_x,freeze_y,freeze_z\n";
 long double H2=0,M2=0,V2=0,volume=0;Real vmax=0,minr=1;
 for(size_t p=0;p<N;++p){const auto&x=g.geometry[p].x;const auto&u=e.jets[p];
  const auto constraint=h::EvolvedConstraints(u,h::CompositeOmegaAt(x,u));
  if(!constraint.valid)throw std::runtime_error("invalid all-cell constraints");
  const Real weight=g.volumes[p]*std::sqrt(e.shift.geometry[p].determinant)/std::pow(u.chi.value,1.5);
  const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);minr=std::min(minr,r);
  Real grr=0,arr=0,lambda=0,beta=0,B=0,f2=0;
  for(int i=0;i<3;++i){lambda+=u.lambda.value[i]*x[i]/r;beta+=u.beta.value[i]*x[i]/r;B+=e.state[22*p+19+i]*x[i]/r;
   f2+=freeze[3*p+i]*freeze[3*p+i];vmax=std::max(vmax,std::abs(freeze[3*p+i]));
   for(int j=0;j<3;++j){grr+=u.metric.g[i][j]*x[i]*x[j]/(r*r);arr+=u.a.k[i][j]*x[i]*x[j]/(r*r);}}
  volume+=weight;H2+=weight*constraint.hamiltonian*constraint.hamiltonian;M2+=weight*constraint.momentum_conformal_norm2;V2+=weight*f2;
  csv<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<','<<weight<<','<<u.alpha.value<<','<<u.chi.value<<','<<grr<<','<<arr<<','
   <<u.trace.value<<','<<u.theta.value<<','<<lambda<<','<<beta<<','<<B<<','<<constraint.hamiltonian<<','<<std::sqrt(constraint.momentum_conformal_norm2);
  for(int i=0;i<3;++i)csv<<','<<freeze[3*p+i];csv<<'\n';}
 if(!csv)throw std::runtime_error("profile write failure");
 std::ofstream report(prefix+".json");report<<std::setprecision(17)
 <<"{\"time\":"<<time<<",\"cells\":"<<N<<",\"minimum_radius\":"<<minr<<",\"H_all_rms\":"<<std::sqrt(H2/volume)
 <<",\"M_all_rms\":"<<std::sqrt(M2/volume)<<",\"actual_metric_Vt_rms\":"<<std::sqrt(V2/volume)<<",\"actual_metric_Vt_max\":"<<vmax
 <<",\"expanded_residual_max_over_solves\":"<<e.max_shift_residual<<",\"scri_pole\":"<<e.boundary_pole<<",\"scri_trace\":"<<e.boundary_regularity
 <<",\"initial_floor_clamps\":"<<e.floor.initial_clamps<<",\"evolution_floor_clamps\":"<<e.floor.evolution_clamps
 <<",\"nonpositive_clamps\":"<<e.floor.nonpositive_clamps<<",\"maximum_floor_radius\":"<<e.floor.maximum_activation_radius
 <<",\"lapse_attenuation\":"<<e.controls.lapse_attenuation<<",\"physical_lapse_floor\":"<<e.controls.physical_lapse_floor
 <<",\"total_lapse_update\":"<<e.controls.total_lapse_update<<",\"additional_scri_traces\":"<<e.controls.scri_regularity_traces
 <<",\"all_active_cells_retained\":true,\"scientific_validation\":false,\"native_Driver_integration\":false}\n";
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 if(argc<4)throw std::invalid_argument("coordinates N path | evolve N initial output steps dt [epsilon floor]");
 const int n=std::stoi(argv[2]);auto mesh=Mesh(n);
 if(std::string(argv[1])=="coordinates"){
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";size_t id=0;
  for(size_t m=0;m<mesh.blocks.size();++m)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
   const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(m,q,x);
   out<<0<<','<<id++<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<'\n';}
  if(!out)throw std::runtime_error("coordinate output");
 }else{
  if(std::string(argv[1])!="evolve"||argc<7)throw std::invalid_argument("missing evolution controls");
  const int steps=std::stoi(argv[5]);const Real dt=std::stod(argv[6]);
  if(steps<0||steps>32||!(dt>0)||steps*dt>.01)throw std::invalid_argument("bounded matched pilot duration");
  const std::string out=argv[4];if(std::filesystem::exists(out))throw std::runtime_error("output exists");std::filesystem::create_directory(out);
  h::SphericalMatchedControls options;if(argc>7)options.lapse_attenuation=std::stod(argv[7]);if(argc>8)options.physical_lapse_floor=std::stod(argv[8]);
  h::SphericalMatchedEvolution e(std::move(mesh),1e-10,options);e.Initialize(0,false);e.InitializeWormhole(argv[3]);
  Record(e,out+"/initial",0,dt);
  for(int step=0;step<steps;++step){e.Step(dt);std::cerr<<"accepted "<<step+1<<'/'<<steps<<'\n';}
  Record(e,out+"/final",steps*dt,dt);
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
