// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "driver/classical_rk4.hpp"
#include "z4c_hyperboloidal/tasks.hpp"

namespace hyp = z4c_hyperboloidal;
namespace nr = numrel;

namespace {
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// Bind the production queue function directly to native TaskList. Physics
// discovery is separate from this test: the delayed module below deliberately
// has no Mesh or numerical PDE. No second task scheduler is used here.
class NativeLists {
 public:
  std::array<TaskList,3> lists;
  std::array<std::map<nr::TaskName,TaskID>,3> ids;

  template<class F>
  void QueueTask(F function, nr::TaskName name, const std::string &,
                 nr::TaskLocation location,
                 std::vector<nr::TaskName> dependencies = {}) {
    TaskID deps(0);
    for (auto dep : dependencies) {
      Require(ids[location].count(dep)==1, "missing or cross-list dependency");
      deps=deps | ids[location].at(dep);
    }
    Require(ids[location].count(name)==0, "duplicate named task");
    ids[location][name]=lists[location].AddTask(function,deps);
  }

  template<class F, class Module>
  void QueueTask(F function, Module *module, nr::TaskName name,
                 const std::string &label, nr::TaskLocation location,
                 std::vector<nr::TaskName> dependencies = {}) {
    QueueTask([=](Driver *driver, int stage) {
      return (module->*function)(driver,stage);
    },name,label,location,dependencies);
  }

  void Reset() {for (auto &list : lists) list.Reset();}
  TaskListStatus Poll(nr::TaskLocation location, int stage) {
    return lists[location].DoAvailable(nullptr,stage);
  }
};

// Inject independently delayed volume/surface traffic and send completion. The
// NaN-filled dependencies make accidental early computation fail, while unique
// stage tags expose using a prior stage's halos or normal correction.
struct DelayedModule {
  bool posted=false, copied=false, prepared=false, bulk_sent=false, surface_sent=false;
  bool allow_bulk=false, allow_surface=false, allow_send_clear=false;
  bool bulk_received=false, surface_received=false, local_filled=false;
  bool remote_filled=false, normal_ready=false, surface_rhs_ready=false;
  bool updated=false, projected=false, sends_cleared=false, recvs_cleared=false;
  int stage_tag=0, copy_calls=0, base_copies=0, update_calls=0, projection_calls=0;
  int bulk_polls=0, surface_polls=0, send_polls=0;
  int dependent_calls=0;
  std::array<int,4> rhs_calls{};
  std::array<bool,4> empty{};
  std::array<Real,4> rhs{}, state{1,2,3,4}, base{}, sum{};
  Real local_halo=0, remote_halo=0, surface_sample=0, normal=0;
  Real surface_rhs=0, surface_state=5, surface_base=0, surface_sum=0;

  void BeginStage() {
    Require(stage_tag==0 || (projected && sends_cleared && recvs_cleared),
            "stage recycled live buffers or unprojected state");
    posted=copied=prepared=bulk_sent=surface_sent=false;
    bulk_received=surface_received=local_filled=remote_filled=false;
    normal_ready=surface_rhs_ready=updated=projected=false;
    sends_cleared=recvs_cleared=false;
    allow_bulk=allow_surface=allow_send_clear=false;
    rhs_calls.fill(0);
    dependent_calls=0;
    const Real poison=std::numeric_limits<Real>::quiet_NaN();
    rhs.fill(poison);local_halo=remote_halo=surface_sample=normal=poison;
    surface_rhs=poison;
  }

  TaskStatus InitRecv(Driver *, int stage) {
    Require(!posted,"receive posting repeated");
    stage_tag=stage;posted=true;return TaskStatus::complete;
  }
  TaskStatus CopyU(Driver *, int stage) {
    Require(posted && stage_tag==stage,"copy precedes stage receive posting");
    ++copy_calls;copied=true;
    if (stage==1) {base=state;sum.fill(0);surface_base=surface_state;surface_sum=0;++base_copies;}
    return TaskStatus::complete;
  }
  TaskStatus PrepareStage(Driver *, int stage) {
    Require(copied && stage==stage_tag,"prepare precedes immutable base copy");
    prepared=true;return TaskStatus::complete;
  }
  TaskStatus SendBulk(Driver *, int) {
    Require(prepared && posted,"bulk pack precedes projection or receive posting");
    bulk_sent=true;return TaskStatus::complete;
  }
  TaskStatus SendSurface(Driver *, int) {
    Require(prepared && posted,"surface pack precedes projection or receive posting");
    surface_sent=true;return TaskStatus::complete;
  }
  TaskStatus FillLocalHalos(Driver *, int stage) {
    Require(bulk_sent,"local fill reads unpacked donor values");
    local_halo=100+stage;local_filled=true;return TaskStatus::complete;
  }
  TaskStatus RecvBulk(Driver *, int) {
    Require(bulk_sent,"bulk receive precedes pack");++bulk_polls;
    if (!allow_bulk) return TaskStatus::incomplete;
    bulk_received=true;return TaskStatus::complete;
  }
  TaskStatus FillRemoteHalos(Driver *, int stage) {
    Require(bulk_received && local_filled,"remote fill precedes receive completion");
    remote_halo=200+stage;remote_filled=true;return TaskStatus::complete;
  }
  TaskStatus RecvSurface(Driver *, int stage) {
    Require(surface_sent,"surface receive precedes pack");++surface_polls;
    if (!allow_surface) return TaskStatus::incomplete;
    surface_sample=300+stage;surface_received=true;return TaskStatus::complete;
  }
  TaskStatus SolveNormalCompatibility(Driver *, int stage) {
    Require(surface_received && surface_sample==300+stage,
            "normal solve uses absent or stale surface samples");
    normal=surface_sample+stage;normal_ready=true;return TaskStatus::complete;
  }
  TaskStatus CalcSurfaceRHS(Driver *, int stage) {
    Require(normal_ready && normal==300+2*stage,"surface RHS lacks current normal lift");
    surface_rhs=normal;surface_rhs_ready=true;return TaskStatus::complete;
  }
  TaskStatus CalcRHS(Driver *, int stage, hyp::RHSSubset subset) {
    if (subset==hyp::RHSSubset::Dependent) {
      Require(local_filled && remote_filled && normal_ready && dependent_calls++==0,
              "joined RHS precedes dependencies or ran twice");
      for (int batch=1;batch<4;++batch)
        CalcRHS(nullptr,stage,static_cast<hyp::RHSSubset>(batch));
      return TaskStatus::complete;
    }
    const int index=static_cast<int>(subset);
    Require(prepared && rhs_calls[index]++==0,"duplicate or unprepared volume RHS");
    Real value=10+stage;
    if (subset==hyp::RHSSubset::LocalHalo) {
      Require(local_filled && local_halo==100+stage,"local RHS lacks current local halo");
      value+=local_halo;
    } else if (subset==hyp::RHSSubset::RemoteHalo) {
      Require(remote_filled && remote_halo==200+stage,
              "remote RHS lacks current remote halo");
      value+=remote_halo;
    } else if (subset==hyp::RHSSubset::Surface) {
      Require(remote_filled && normal_ready && normal==300+2*stage,
              "outer RHS precedes bulk halos or current normal lift");
      value+=remote_halo+normal;
    }
    // Empty batches legitimately complete, but must not skip any nonempty batch.
    rhs[index]=empty[index] ? 0 : value;
    return TaskStatus::complete;
  }
  TaskStatus ExpRKUpdate(Driver *, int stage) {
    Require(surface_rhs_ready,"update lacks the surface RHS");
    Require(dependent_calls==1,"update lacks the joined volume RHS");
    for (int p=0;p<4;++p) {
      Require(rhs_calls[p]==1 && std::isfinite(rhs[p]),"update lacks a volume batch");
      state[p]=time_integration::ClassicalRK4::Advance(stage,.01,base[p],rhs[p],sum[p]);
    }
    surface_state=time_integration::ClassicalRK4::Advance(
        stage,.01,surface_base,surface_rhs,surface_sum);
    updated=true;++update_calls;return TaskStatus::complete;
  }
  TaskStatus ProjectSurface(Driver *, int) {
    Require(updated,"surface projection precedes update");
    projected=true;++projection_calls;return TaskStatus::complete;
  }
  TaskStatus ClearSend(Driver *, int) {
    Require(projected,"end tasks precede stage completion");++send_polls;
    if (!allow_send_clear) return TaskStatus::incomplete;
    sends_cleared=true;return TaskStatus::complete;
  }
  TaskStatus ClearRecv(Driver *, int) {
    Require(projected && bulk_received && surface_received,
            "receive clear precedes unpack or completed stage");
    recvs_cleared=true;return TaskStatus::complete;
  }
};

void CheckGraph(int first_available, unsigned int empty_mask) {
  DelayedModule module;
  for (int p=0;p<4;++p) module.empty[p]=(empty_mask & (1U<<p))!=0;
  NativeLists queue;
  hyp::QueueZ4cHyperboloidalStageTasks(&queue,&module);
  Require(queue.lists[nr::Task_Start].Size()==1,"wrong stage-start task count");
  Require(queue.lists[nr::Task_Run].Size()==14,"wrong stage task count");
  Require(queue.lists[nr::Task_End].Size()==2,"wrong stage-end task count");
  for (int stage=1;stage<=4;++stage) {
    module.BeginStage();queue.Reset();
    Require(queue.Poll(nr::Task_Start,stage)==TaskListStatus::complete,
            "receive posting stalled");
    Require(queue.Poll(nr::Task_Run,stage)==TaskListStatus::running,
            "unavailable communication did not stall dependents");
    Require(module.rhs_calls[0]==1 && module.local_filled,
            "interior work or local halo fill waited for remote communication");
    Require(module.rhs_calls[1]==0 && module.rhs_calls[2]==0 && module.rhs_calls[3]==0 &&
            !module.surface_rhs_ready && !module.updated,
            "dependent work ran before remote communication");
    for (int poll=0;poll<3;++poll) queue.Poll(nr::Task_Run,stage);
    Require(module.rhs_calls[0]==1 && module.dependent_calls==0,
            "completed independent work reran while polling");

    if (first_available==0) module.allow_bulk=true;
    if (first_available==1) module.allow_surface=true;
    if (first_available<2) {
      queue.Poll(nr::Task_Run,stage);
      Require(!module.updated && module.rhs_calls[3]==0,
              "one communication stream released a coupled update");
      if (first_available==0) {
        Require(module.remote_filled && module.rhs_calls[1]==0 &&
                module.rhs_calls[2]==0 && !module.normal_ready,
                "halo fill stalled or joined RHS ran before scri");
      } else {
        Require(module.surface_rhs_ready && module.rhs_calls[2]==0,
                "surface solve waited for unrelated bulk halos");
      }
    }
    module.allow_bulk=module.allow_surface=true;
    Require(queue.Poll(nr::Task_Run,stage)==TaskListStatus::complete,
            "completed receives did not release full stage");
    Require(module.updated && module.projected,"stage finished before projection");
    Require(queue.Poll(nr::Task_End,stage)==TaskListStatus::running,
            "pending sends did not retain buffers");
    Require(module.recvs_cleared && !module.sends_cleared,
            "receive retirement waited for unrelated sends");
    module.allow_send_clear=true;
    Require(queue.Poll(nr::Task_End,stage)==TaskListStatus::complete,
            "completed sends did not release next stage");
  }
  Require(module.base_copies==1 && module.copy_calls==4 && module.update_calls==4 &&
          module.projection_calls==4,"classical RK4 lifecycle call count");
  // Stage-labelled payloads have exact weighted mean 2.5. Delays and ordering
  // must not change either accepted volume or accepted surface state.
  const Real expected_rhs[4]={12.5,115,215,520};
  for (int p=0;p<4;++p) {
    const Real expected=p+1+.01*(module.empty[p] ? 0 : expected_rhs[p]);
    Require(std::abs(module.state[p]-expected)<4e-15,"delays changed accepted bulk state");
  }
  Require(std::abs(module.surface_state-8.05)<4e-15,"delays changed accepted scri state");
}
}  // namespace

int main() {
  try {
    for (int order=0;order<3;++order)
      for (unsigned int empty=0;empty<16;++empty) CheckGraph(order,empty);
    std::cout<<"PASS native hyperboloidal task graph: 48 delayed/empty-batch cases, "
                "192 RK stages; independent volume/surface progress, current-stage "
                "dependencies and buffer retirement\n";
  } catch (const std::exception &error) {
    std::cerr<<error.what()<<"\n";
    return 1;
  }
  return 0;
}
