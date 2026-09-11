#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/rk4_predictor_states.hpp"
#include "driver/rk4_physical_maximum.hpp"
void Check(bool ok) { if (!ok) throw std::runtime_error("predictor state regression"); }
template<class F> void Reject(F f) {
  bool rejected=false;
  try { f(); } catch(const std::exception &) { rejected=true; }
  Check(rejected);
}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    z4c::Z4cGridLayout l;
    l.is=l.js=2; l.ie=6; l.je=8; l.ks=l.ke=0;
    DvceArray5D<Real> state("state",7,2,1,11,9), rhs("rhs",7,2,1,11,9), out;
    auto h=Kokkos::create_mirror(state);
    for(int m=0;m<7;++m) for(int v=0;v<2;++v) for(int j=0;j<11;++j) for(int i=0;i<9;++i)
      h(m,v,0,j,i)=m+v*.1+j*.01+i*.001;
    Kokkos::deep_copy(state,h);
    subcycling::RK4PredictorStates p;
    Reject([&]{p.Capture(rhs,1);});
    Reject([&]{p.Begin(state,l,{1,1},3,.25);});
    Reject([&]{p.Begin(state,l,{7},3,.25);});
    Reject([&]{p.Begin(state,l,{1},3,0);});
    p.Begin(state,l,{5,1},3,.25);
    Check(p.StartTime()==3 && p.Bytes()==5*2*2*1*7*5*sizeof(Real));
    Reject([&]{p.EvaluateStage(0,.125,1,out);});
    Reject([&]{p.Capture(rhs,2);});
    // Overwrite the source after Begin: retained beginning state must be owned.
    Kokkos::deep_copy(state,-77.0);
    for(int stage=1;stage<=4;++stage) {
      auto hr=Kokkos::create_mirror_view(rhs);
      for(int m=0;m<7;++m) for(int v=0;v<2;++v) for(int j=0;j<11;++j) for(int i=0;i<9;++i)
        hr(m,v,0,j,i)=stage*stage+h(m,v,0,j,i);
      Kokkos::deep_copy(rhs,hr);p.Capture(rhs,stage);
    }
    Kokkos::deep_copy(rhs,-99.0);
    Check(p.CompletedStages()==4);
    for(double fraction : {0.,.5}) for(int stage=1;stage<=4;++stage) {
      p.EvaluateStage(fraction,.125,stage,out);
      auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
      for(int b=0;b<2;++b) for(int v=0;v<2;++v) for(int j=0;j<7;++j) for(int i=0;i<5;++i) {
        const double y=h(b==0?5:1,v,0,j+2,i+2);
        subcycling::RK4DenseBoundary expected{y,.25,1+y,4+y,9+y,16+y};
        Check(std::abs(ho(b,v,0,j,i)-expected.StageBoundary(fraction,.125,stage))<1e-14);
      }
    }
    // A prescribed cubic-in-time solution has quadratic RHS. Classical RK
    // stage sampling and dense physical reconstruction must reproduce it at
    // off-stage times; this does not compare the implementation to itself.
    p.Begin(state,l,{5,1},3,.25);
    for(int stage=1;stage<=4;++stage) {
      const double t=3+(stage==1?0:stage==4?.25:.125);
      Kokkos::deep_copy(rhs,3*t*t-4*t+1);p.Capture(rhs,stage);
    }
    auto primitive=[](double t){return t*t*t-2*t*t+t;};
    for(double f:{0.,.17,.5,.83,1.}) {
      p.EvaluateValue(f,out);
      auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
      const double expected=-77+primitive(3+.25*f)-primitive(3);
      for(int b=0;b<2;++b) for(int v=0;v<2;++v) for(int j=0;j<7;++j) for(int i=0;i<5;++i)
        Check(std::abs(ho(b,v,0,j,i)-expected)<3e-14);
    }
    Reject([&]{p.EvaluateValue(-.01,out);});
    Reject([&]{p.EvaluateValue(1.01,out);});
    Reject([&]{p.EvaluateValue(std::numeric_limits<double>::quiet_NaN(),out);});
    Reject([&]{p.EvaluateStage(.75,.125,1,out);});
    Reject([&]{p.EvaluateStage(0,.125,5,out);});
    Reject([&]{p.Capture(rhs,4);});
    p.Begin(state,l,{2},4,.125);
    Check(p.CompletedStages()==0);
    Reject([&]{p.EvaluateValue(.5,out);});
    Reject([&]{p.EvaluateStage(0,.0625,1,out);});
    p.Begin(state,l,{},4,.125);
    for(int stage=1;stage<=4;++stage) p.Capture(rhs,stage);
    p.EvaluateStage(0,.0625,1,out);Check(out.size()==0);
    p.EvaluateValue(.5,out);Check(out.size()==0);
    // Covered source block 5 is deliberately enormous. Select only leaf1;
    // include Theta independently so reducing Khat alone cannot pass.
    Kokkos::deep_copy(state,1000000.0);
    auto physical=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state);
    for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i) {
      physical(1,0,0,j,i)=-3;physical(1,1,0,j,i)=.5;
    }
    Kokkos::deep_copy(state,physical);p.Begin(state,l,{5,1},3,.25);
    Kokkos::deep_copy(rhs,2.0);
    for(int stage=1;stage<=4;++stage) p.Capture(rhs,stage);
    subcycling::RK4PhysicalMaximum maximum;
    Reject([&]{maximum.Build({5,1},{2},0,1);});
    Reject([&]{maximum.Build({5,1},{1,1},0,1);});
    maximum.Build({5,1},{1},0,1);
    for(double f:{0.,.17,.5,1.})
      Check(std::abs(maximum.Evaluate(p,f)-std::abs(-2+1.5*f))<1e-13);
    maximum.Build({5,1},{},0,1);Check(maximum.Evaluate(p,.5)==0);
    // The maximizing leaf changes inside the interval. Reduce interpolated
    // fields, rather than interpolating a maximum measured at endpoints.
    for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i) {
      physical(5,0,0,j,i)=1;physical(5,1,0,j,i)=0;
    }
    Kokkos::deep_copy(state,physical);p.Begin(state,l,{5,1},3,.25);
    for(int stage=1;stage<=4;++stage) p.Capture(rhs,stage);
    maximum.Build({5,1},{5,1},0,1);
    for(double f:{0.,.17,1./3,.5,1.})
      Check(std::abs(maximum.Evaluate(p,f)-std::max(1+1.5*f,2-1.5*f))<1e-13);
    physical(1,0,0,l.js,l.is)=std::numeric_limits<Real>::quiet_NaN();
    Kokkos::deep_copy(state,physical);p.Begin(state,l,{5,1},3,.25);
    for(int stage=1;stage<=4;++stage) p.Capture(rhs,stage);
    Reject([&]{maximum.Evaluate(p,.5);});
    std::cout << "PASS: RK stage/physical histories, cubic reconstruction, leaf maxima, switching, nonfinite rejection\n";
  }
  Kokkos::finalize();
}
