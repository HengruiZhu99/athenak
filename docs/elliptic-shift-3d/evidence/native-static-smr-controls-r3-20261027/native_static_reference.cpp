// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Matched stage/control only. Actual native Driver exports both leaf bounds and
// live RK coefficients; this harness does not independently validate the PDE.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "z4c/hyperboloidal/native_core_map.hpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "z4c/hyperboloidal/native_composite_stage.hpp"
namespace h=z4c::hyperboloidal;
using Engine=h::CompositeTensorEvolution;using State=h::NativeCompositeStageState;
cubed_sphere::CompositeMesh ReadMesh(const char *path){std::ifstream file(path);std::string magic;int n,ng,bulk;Real half;size_t count,layers,coarse;
 file>>magic>>n>>ng>>half>>bulk>>count>>layers;
 if(!file||magic!="NATIVE_STATIC_COMPOSITE_MESH_V1"||n<6||n>16||ng!=4||count>232||layers<1||layers>16)throw std::runtime_error("native static mesh receipt layout");
 std::vector<Real> radii(layers);for(Real &r:radii)file>>r;std::vector<h::NativeCartesianBlock> records(count);
 for(auto &b:records){file>>b.native>>b.refinement;for(int a=0;a<3;++a)file>>b.lo[a]>>b.hi[a];}file>>coarse;
 if(!file)throw std::runtime_error("native static receipt truncated");std::string extra;if(file>>extra)throw std::runtime_error("native static receipt trailing payload");
 auto mesh=h::NativeCartesianCoreMap::FromNativeLeaves(half,radii,1,n,ng,bulk,records);const auto plan=multigrid::CompositeCoarseningPlan::Build(mesh);
 if(!plan.dense_supported||plan.TerminalCells()!=coarse)throw std::runtime_error("native static receipt hierarchy mismatch");return mesh;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int code=0;try{
 if(argc!=9&&argc!=10)throw std::invalid_argument("reference mesh rk dt prefix wormhole amplitude angular tolerance [initial]");
 auto mesh=ReadMesh(argv[1]);const bool wormhole=std::stoi(argv[5])!=0;const Real amplitude=std::stod(argv[6]);const bool angular=std::stoi(argv[7])!=0;const Real tolerance=std::stod(argv[8]);
 if(wormhole!=(argc==10)||!(tolerance>0&&tolerance<=1e-11)||std::abs(amplitude)>1e-3)throw std::invalid_argument("native static reference controls");
 std::ifstream receipt(argv[2]);std::string magic;h::NativeCompositeRK rk;receipt>>magic>>rk.integrator>>rk.stages;
 if(magic!="NATIVE_COMPOSITE_RK_V1")throw std::runtime_error("native RK receipt magic");for(auto *a:{&rk.gam0,&rk.gam1,&rk.beta,&rk.delta})for(Real &x:*a)receipt>>x;
 if(!receipt)throw std::runtime_error("native RK receipt truncated");rk.Validate();
 std::ifstream clock(argv[3]);std::vector<Real>dts;Real dt;while(clock>>dt){if(!(dt>0&&dt<=.0001))throw std::runtime_error("reference dt bound");dts.push_back(dt);}if(dts.empty()||dts.size()>4)throw std::runtime_error("reference cycle bound");
 Engine engine(std::move(mesh),true,tolerance,4,true,wormhole,0,6,true,wormhole?2:0,.24,wormhole,true,true);
 engine.Initialize(amplitude,angular);if(wormhole)engine.InitializeWormhole(argv[9],amplitude,angular);
 State stage({engine.state,engine.scri->values,{}},engine.beta_delta.size());
 auto load=[&]{engine.state=stage.Coefficients().volume;engine.scri->values=stage.Coefficients().scri;engine.beta_delta=stage.ShiftWarmStart();};
 auto solve=[&]{load();engine.RHS(engine.state,0);stage.InstallShift(stage.CurrentEpoch(),engine.beta_delta);};
 auto write=[&](size_t cycle){std::ostringstream number;number<<std::setw(5)<<std::setfill('0')<<cycle;std::ofstream out(std::string(argv[4])+"."+number.str()+".csv");out<<std::setprecision(17)<<"x,y,z";for(int f=0;f<22;++f)out<<",state_"<<f;for(int a=0;a<3;++a)out<<",delta_beta_"<<a;out<<'\n';
  for(size_t p=0;p<engine.shift.grid.active.size();++p){const auto &x=engine.shift.grid.geometry[p].x;out<<x[0]<<','<<x[1]<<','<<x[2];for(int f=0;f<22;++f)out<<','<<engine.state[22*p+f];for(int a=0;a<3;++a)out<<','<<engine.beta_delta[3*p+a];out<<'\n';}
  std::ofstream surface(std::string(argv[4])+"."+number.str()+".scri.csv");surface<<std::setprecision(17)<<"x,y,z";for(int f=0;f<22;++f)surface<<",state_"<<f;for(int a=0;a<3;++a)surface<<",delta_beta_"<<a;surface<<'\n';
  for(size_t p=0;p<engine.scri->rows.size();++p){const auto &x=engine.scri->rows[p].geometry.x;surface<<x[0]<<','<<x[1]<<','<<x[2];for(int f=0;f<22;++f)surface<<','<<engine.scri->values[22*p+f];for(int a=0;a<3;++a)surface<<','<<engine.scri->beta[3*p+a];surface<<'\n';}
  std::ofstream shift(std::string(argv[4])+"."+number.str()+".shift.bin",std::ios::binary);const uint64_t count=engine.beta_delta.size();shift.write(reinterpret_cast<const char*>(&count),sizeof(count));shift.write(reinterpret_cast<const char*>(engine.beta_delta.data()),count*sizeof(Real));
  if(!out||!surface||!shift)throw std::runtime_error("native static reference output");};
 solve();write(0);Real time=0;for(size_t cycle=0;cycle<dts.size();++cycle){stage.BeginCycle(rk,dts[cycle],time);
  for(int s=1;s<=rk.stages;++s){const auto epoch=stage.BeginStage(s);load();State::Values rate{engine.RHS(engine.state,dts[cycle]),engine.boundary_rhs,{}};stage.InstallShift(epoch,engine.beta_delta);stage.AdvanceStage(epoch,rate);load();if(!stage.Active())solve();}
  time+=dts[cycle];write(cycle+1);}
 std::cout<<"{\"cycles\":"<<dts.size()<<",\"cells\":"<<engine.shift.grid.active.size()<<",\"ng\":4,\"actual_native_topology_and_RK\":true,\"independent_PDE_validation\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
