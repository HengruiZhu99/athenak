// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define main native_stage_reference_main_unused
#include "test_native_stage.cpp"
#undef main
void Cycle(Stage &s,const RK &rk,Real dt){const Real time=s.Time();s.BeginCycle(rk,dt,time);
 for(int stage=1;stage<=rk.stages;++stage){const auto e=s.BeginStage(stage);s.InstallShift(e,Beta(stage));s.AdvanceStage(e,Split(ODE(Flatten(s.Coefficients()),s.Time())));} }
int main(int argc,char **argv){try{const auto source=argc>1?std::filesystem::path(argv[1]):std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()/"src/driver/driver.cpp";int stage_checks=0;
 for(const auto &name:{"rk3","rk4"}){const auto rk=ReadNative(source,name);auto untouched=Initial();Cycle(untouched,rk,.025);auto reset=untouched;const auto before=reset.Save();
  Check(!Same(before.registers,before.values),"test did not create nontrivial old scratch registers");reset.ReplaceAcceptedState(reset.Coefficients(),reset.ShiftWarmStart(),reset.GeometryMetadata());const auto after=reset.Save();
  Check(after.time==before.time&&after.cycle==before.cycle&&after.completed_cycles==before.completed_cycles&&after.has_rk==before.has_rk&&after.rk==before.rk,"reset changed accepted time/cycle/RK");Check(after.shift_warm_start==before.shift_warm_start,"reset changed separate algebraic beta");
  untouched.BeginCycle(rk,.017,untouched.Time());reset.BeginCycle(rk,.017,reset.Time());
  for(int stage=1;stage<=rk.stages;++stage){const auto ea=untouched.BeginStage(stage),eb=reset.BeginStage(stage);Check(Same(untouched.Save().registers,reset.Save().registers),"newstage1 did not overwrite every volume/scri/endpoint register");Check(Same(untouched.Coefficients(),reset.Coefficients())&&untouched.Time()==reset.Time(),"stage state/time changed after accepted reset");
   untouched.InstallShift(ea,Beta(stage));reset.InstallShift(eb,Beta(stage));auto rate=Split(ODE(Flatten(untouched.Coefficients()),untouched.Time()));untouched.AdvanceStage(ea,rate);reset.AdvanceStage(eb,rate);Check(Same(untouched.Coefficients(),reset.Coefficients()),"accepted reset changed nontrivial next-cycle ODE result");++stage_checks;}
  Check(untouched.Time()==reset.Time()&&untouched.CompletedCycles()==reset.CompletedCycles(),"next cycle metadata changed");auto mid=Initial();mid.BeginCycle(rk,.01,0);Reject([&]{mid.ReplaceAcceptedState(mid.Coefficients(),mid.ShiftWarmStart(),mid.GeometryMetadata());},"midstage register reset accepted");
 }
 std::cout<<"{\"methods\":[\"rk3\",\"rk4\"],\"next_stage_bitwise_checks\":"<<stage_checks<<",\"all_volume_scri_endpoint_registers\":true,\"time_cycle_RK_beta_preserved\":true,\"midcycle_rejected\":true,\"status\":\"passed\"}\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
