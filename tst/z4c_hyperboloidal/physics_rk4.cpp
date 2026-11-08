// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include "driver/classical_rk4.hpp"
#include "z4c_hyperboloidal/fields.hpp"
#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"

using RK=time_integration::ClassicalRK4;
namespace nh=z4c_hyperboloidal;
void Require(bool yes,const char *message) {if(!yes)throw std::runtime_error(message);}
bool SameBits(Real a,Real b) {return std::memcmp(&a,&b,sizeof(Real))==0;}

template<class RHS>
Real Integrate(Real initial,int steps,RHS rhs) {
  const Real dt=Real(1)/steps;Real state=initial;
  for(int n=0;n<steps;++n) {
    const Real base=state,time=n*dt;Real sum=0;
    for(int stage=1;stage<=4;++stage)
      state=RK::Advance(stage,dt,base,rhs(RK::RHSTime(time,dt,stage),state),sum);
  }
  return state;
}

void TestOrder() {
  const auto autonomous=[](Real,Real y){return y*(1-y);};
  const auto nonautonomous=[](Real t,Real y){return t*y;};
  Real previous_a=0,previous_n=0;
  std::cout<<"fourth-order errors (logistic, time-dependent):";
  for(int steps:{4,8,16,32}) {
    const Real a=std::abs(Integrate(.2,steps,autonomous)-1/(1+4*std::exp(-1.)));
    const Real n=std::abs(Integrate(1,steps,nonautonomous)-std::exp(.5));
    if(previous_a>0) {
      Require(previous_a/a>13 && previous_a/a<19,"autonomous fourth-order convergence");
      Require(previous_n/n>12 && previous_n/n<21,"time-dependent fourth-order convergence");
    }
    previous_a=a;previous_n=n;
    std::cout<<" "<<steps<<":"<<a<<","<<n;
  }
  std::cout<<"\n";
}

void TestLegacyArithmetic() {
  std::mt19937_64 rng(0x4c1a551ca1ULL);
  std::uniform_real_distribution<Real> random(-2,2);
  for(int sample=0;sample<4096;++sample) {
    const Real base=random(rng),time=random(rng),dt=.01+.01*std::abs(random(rng));
    Real k[4];for(auto &value:k)value=random(rng);
    Real sum=0;
    for(int stage=1;stage<=4;++stage) {
      const Real result=RK::Advance(stage,dt,base,k[stage-1],sum);
      const Real expected=stage==4 ? base+dt*(k[0]+2*k[1]+2*k[2]+k[3])/6
                                  : base+(stage<3?dt/2:dt)*k[stage-1];
      Require(SameBits(result,expected),"legacy classical stage arithmetic changed");
      const Real expected_rhs=stage==1?time:time+(stage==4?dt:dt/2);
      const Real expected_state=time+(stage<3?dt/2:dt);
      Require(SameBits(RK::RHSTime(time,dt,stage),expected_rhs),"RHS stage time");
      Require(SameBits(RK::StateTime(time,dt,stage),expected_state),"updated stage time");
    }
  }
  Require(std::abs(RK::Weight(1)+RK::Weight(2)+RK::Weight(3)+RK::Weight(4)-1)<2e-16,
          "RK weights do not sum to one");
}

void TestConservationOnDevice() {
  // A conservative exchange pair: x'=-k*x, y'=k*x. Different scales and 400
  // cycles exercise independent register accumulation and linear conservation.
  DvceArray2D<Real> states("RK4 conservation",128,2);
  Kokkos::parallel_for("classical RK4 conservative pair",128,KOKKOS_LAMBDA(int i) {
    Real x=.2+.001*i,y=.8-.001*i;
    for(int n=0;n<400;++n) {
      const Real base_x=x,base_y=y;Real sum_x=0,sum_y=0;
      for(int stage=1;stage<=4;++stage) {
        const Real flux=.6*x;
        x=RK::Advance(stage,.05,base_x,-flux,sum_x);
        y=RK::Advance(stage,.05,base_y,flux,sum_y);
      }
    }
    states(i,0)=x;states(i,1)=y;
  });
  const auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),states);
  for(int i=0;i<128;++i) {
    Require(std::abs(h(i,0)+h(i,1)-1)<8e-15,"RK4 linear conservation");
    Require(std::abs(h(i,0)-(.2+.001*i)*std::exp(-12))<3e-12,
            "RK4 conservative exchange trajectory");
  }
}

void TestIndependentFieldProjection() {
  nh::CartesianChartJets current{};
  current[0].f=1;current[1].f=.1;current[2].f=.02;current[3].f=-.03;
  current[4].f=.07;current[5].f=.01;
  current[7].f=.03;current[8].f=.02;current[9].f=-.01;
  current[10].f=-.04;current[11].f=.015;
  for(int f=28;f<31;++f)current[f].f=100+f;
  nh::ExpandCartesianChart(current);
  const auto base=current;Real sum[31]{};int updates=0;
  for(int stage=1;stage<=4;++stage) {
    for(int f=0;f<31;++f)if(nh::Fields::Independent(f)) {
      const Real rhs=.01*std::sin(f+RK::RHSFraction(stage)*.1);
      current[f].f=RK::Advance(stage,.1,base[f].f,rhs,sum[f]);++updates;
    }
    nh::ExpandCartesianChart(current);
    const Real a=1+current[1].f,b=current[2].f,c=current[3].f,
               d=1+current[4].f,e=current[5].f,f=1+current[6].f;
    const Real det=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
    const Real trace=(d*f-e*e)*current[7].f+2*(c*e-b*f)*current[8].f
        +2*(b*e-c*d)*current[9].f+(a*f-c*c)*current[10].f
        +2*(b*c-a*e)*current[11].f+(a*d-b*b)*current[12].f;
    Require(std::abs(det-1)<8e-16,"determinant stage projection");
    Require(std::abs(trace)<8e-17,"tracefree stage projection");
    for(int field=28;field<31;++field)
      Require(SameBits(current[field].f,base[field].f),"dormant RK variable changed");
  }
  Require(updates==4*26,"classical RK4 must advance 26 independent fields");
}

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);int result=0;
  try {
    std::cout<<std::setprecision(9);
    TestOrder();TestLegacyArithmetic();TestConservationOnDevice();TestIndependentFieldProjection();
    std::cout<<"PASS classical RK4: fourth order, 4096 exact legacy stages/clocks, "
                "128 conservative device trajectories, 26-field projection\n";
  }catch(const std::exception &error){std::cerr<<"FAIL: "<<error.what()<<"\n";result=1;}
  Kokkos::finalize();return result;
}
