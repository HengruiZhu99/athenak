#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/classical_rk4_update.hpp"
#include "driver/subcycle_schedule.hpp"
void Check(bool ok) {if(!ok) throw std::runtime_error("level RK regression");}
struct Evolution {
  z4c::Z4cGridLayout l;
  DualArray1D<int> levels;
  DvceArray5D<Real> state,initial,rhs,sum;
  subcycling::BlockBatches batches;
  int updates[4]{};
  Evolution():levels("levels",5),state("state",5,2,1,5,5),
      initial("initial",5,2,1,5,5),rhs("rhs",5,2,1,5,5) {
    l.is=l.js=1;l.ie=l.je=3;l.ks=l.ke=0;
    int lev[5]={3,1,2,3,0};
    for(int m=0;m<5;++m) levels.h_view(m)=lev[m];
    levels.modify_host();levels.sync_device();
    Kokkos::deep_copy(state,1.0);Kokkos::deep_copy(initial,-7.0);Kokkos::deep_copy(rhs,-9.0);
  }
  void Advance(const subcycling::StepContext &step) {
    ++updates[step.maximum_level];
    batches.Update(true,levels,5,step.minimum_level,step.maximum_level);
    auto old=Kokkos::create_mirror(state);Kokkos::deep_copy(old,state);
    const auto u=state,uinit=initial,f=rhs;
    batches.For5("copy selected initial state",0,1,0,0,1,3,1,3,
        KOKKOS_LAMBDA(int m,int n,int k,int j,int i){uinit(m,n,k,j,i)=u(m,n,k,j,i);});
    for(int stage=1;stage<=4;++stage) {
      const double time=step.StageTime(classical_rk4::StageTime(stage));
      batches.For5("nonautonomous ODE RHS",0,1,0,0,1,3,1,3,
          KOKKOS_LAMBDA(int m,int n,int k,int j,int i){f(m,n,k,j,i)=time*u(m,n,k,j,i);});
      classical_rk4::Update(batches,l,step.Dt(),stage,state,initial,rhs,sum,1,1);
    }
    auto now=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state);
    for(int m=0;m<5;++m) for(int n=0;n<2;++n) for(int j=0;j<5;++j) for(int i=0;i<5;++i) {
      const bool selected=levels.h_view(m)>=step.minimum_level && levels.h_view(m)<=step.maximum_level;
      if(!selected || j==0 || j==4 || i==0 || i==4) Check(now(m,n,0,j,i)==old(m,n,0,j,i));
      else if(n==1) Check(now(m,n,0,j,i)==0);
    }
  }
  void Synchronize(const subcycling::StepContext &) {}
};
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    double previous=0;
    for(int steps : {4,8,16,32}) {
      Evolution e;
      subcycling::Schedule scheduler(0,3,4);
      for(int n=0;n<steps;++n) scheduler.Run(double(n)/steps,1./steps,e);
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),e.state);
      double error=0;
      for(int m=0;m<5;++m) error=std::max(error,std::abs(h(m,0,0,2,2)-std::exp(.5)));
      if(previous>0) {double ratio=previous/error;std::cout << "recursive RK ratio " << ratio << '\n';Check(ratio>12 && ratio<20);}
      previous=error;
      Check(e.updates[1]==steps && e.updates[2]==2*steps && e.updates[3]==4*steps);
      // Updating only a level absent from this pack must leave everything alone.
      auto saved=Kokkos::create_mirror(e.state);Kokkos::deep_copy(saved,e.state);
      e.batches.Update(true,e.levels,5,7,7);
      classical_rk4::Update(e.batches,e.l,.01,1,e.state,e.initial,e.rhs,e.sum);
      auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),e.state);
      for(int m=0;m<5;++m) Check(after(m,0,0,2,2)==saved(m,0,0,2,2));
    }
    std::cout << "PASS: recursive selected-level RK updates, prescribed components, untouched levels/ghosts\n";
  }
  Kokkos::finalize();
}
