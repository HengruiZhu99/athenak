// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Matched host-engine control using ACTUAL Driver RK coefficients exported by
// the native run. No independent native-task or scientific validation claim.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "z4c/hyperboloidal/native_composite_stage.hpp"
namespace h=z4c::hyperboloidal;
using Engine=h::CompositeTensorEvolution;using Stage=h::NativeCompositeStageState;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 if(argc==4&&std::string(argv[1])=="--coordinates"){
  const int n=std::stoi(argv[2]);if(n<6||n>10)throw std::invalid_argument("coordinate control n6..10");
  cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,n,3,2),4,true);
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"x,y,z,r\n";
  for(const auto &g:grid.geometry){const auto &x=g.x;out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<'\n';}
  if(!out)throw std::runtime_error("coordinate export failed");Kokkos::finalize();return 0;
 }
 if(argc<5||argc>8)throw std::invalid_argument("native reference: initial.dat rk.dat dt.dat output-prefix [core-bulk [n [tolerance]]]");
 const int bulk=argc>=6?std::stoi(argv[5]):1;if(bulk!=1&&bulk!=2)throw std::invalid_argument("native reference static core bulk1/2");
 std::ifstream receipt(argv[2]);std::string magic;h::NativeCompositeRK rk;
 receipt>>magic>>rk.integrator>>rk.stages;
 if(magic!="NATIVE_COMPOSITE_RK_V1")throw std::runtime_error("native RK receipt magic");
 for(auto *a:{&rk.gam0,&rk.gam1,&rk.beta,&rk.delta})for(Real &x:*a)receipt>>x;
 if(!receipt)throw std::runtime_error("native RK receipt truncated");std::string extra;if(receipt>>extra)throw std::runtime_error("native RK receipt trailing payload");rk.Validate();
 std::ifstream timestep(argv[3]);std::vector<Real> dts;Real dt;
 while(timestep>>dt){if(!(dt>0&&dt<=1.e-4))throw std::runtime_error("native reference dt bound");dts.push_back(dt);}
 if(dts.empty()||dts.size()>4)throw std::runtime_error("native reference only one to four cycles");
 const int n=argc>=7?std::stoi(argv[6]):6;const Real tolerance=argc>=8?std::stod(argv[7]):1.e-11;
 if(n<6||n>10||!(tolerance>0&&tolerance<=1.e-11))throw std::invalid_argument("native reference n/tolerance bounds");
 Engine engine(cubed_sphere::CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,n,3,bulk),true,tolerance,4,true,true,0,6,true,2,.24,true,true,true);
 engine.Initialize(0,false);engine.InitializeWormhole(argv[1],0,false);
 Stage stage({engine.state,engine.scri->values,{}},engine.beta_delta.size());
 auto load=[&]{engine.state=stage.Coefficients().volume;engine.scri->values=stage.Coefficients().scri;engine.beta_delta=stage.ShiftWarmStart();};
 auto solve=[&]{load();engine.RHS(engine.state,0);stage.InstallShift(stage.CurrentEpoch(),engine.beta_delta);};
 auto write=[&](size_t cycle){std::ostringstream number;number<<std::setw(5)<<std::setfill('0')<<cycle;
   std::ofstream out(std::string(argv[4])+"."+number.str()+".csv");out<<std::setprecision(17)<<"x,y,z";
   for(int f=0;f<22;++f)out<<",state_"<<f;for(int a=0;a<3;++a)out<<",delta_beta_"<<a;out<<'\n';
   for(size_t p=0;p<engine.shift.grid.active.size();++p){const auto &x=engine.shift.grid.geometry[p].x;out<<x[0]<<','<<x[1]<<','<<x[2];
     for(int f=0;f<22;++f)out<<','<<engine.state[22*p+f];for(int a=0;a<3;++a)out<<','<<engine.beta_delta[3*p+a];out<<'\n';}
   if(!out)throw std::runtime_error("native reference output write failed");};
 solve();write(0);Real time=0;
 for(size_t cycle=0;cycle<dts.size();++cycle){stage.BeginCycle(rk,dts[cycle],time);
   for(int s=1;s<=rk.stages;++s){const auto epoch=stage.BeginStage(s);load();
     Stage::Values rate{engine.RHS(engine.state,dts[cycle]),engine.boundary_rhs,{}};
     stage.InstallShift(epoch,engine.beta_delta);stage.AdvanceStage(epoch,rate);load();
     if(!stage.Active())solve();}
   time+=dts[cycle];write(cycle+1);
 }
 std::cout<<std::setprecision(17)<<"{\"cycles\":"<<dts.size()<<",\"time\":"<<time<<",\"cells\":"<<engine.shift.grid.active.size()
   <<",\"ng\":3,\"core_bulk\":"<<bulk<<",\"actual_native_RK_receipt\":true,\"qualification\":\"same-engine stage recurrence control; no independent equation or scientific pass\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
