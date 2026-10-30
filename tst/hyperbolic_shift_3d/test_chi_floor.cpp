#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/positive_variables.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool condition,const char*message){if(!condition)throw std::runtime_error(message);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;
try{
  const Real floor=1e-8,radius=.03,A=.50005;bool clipped=false;
  Require(h::MaterializePositive(-1e6,A,0,1e-10,clipped,floor,radius)==floor&&clipped,"finite underflow was not floored");
  const Real logarithm=std::log(1e-5);
  Require(h::MaterializePositive(logarithm,A,0,1e-10,clipped,floor,radius)==std::exp(logarithm)&&!clipped,"unfloored arithmetic changed");
  Require(h::MaterializePositive(-1e6,.505,0,1e-10,clipped,floor,radius)==0&&!clipped,"floor escaped puncture region");
  Require(h::MaterializePositive(-1e6,A,0,1e-10,clipped)==0&&!clipped,"default scheme changed");
  Require(h::MaterializePositive(-std::numeric_limits<Real>::infinity(),A,0,1e-10,clipped,floor,radius)==0&&!clipped,"nonfinite log was rescued");
  Require(std::isnan(h::MaterializePositive(std::numeric_limits<Real>::quiet_NaN(),A,0,1e-10,clipped,floor,radius))&&!clipped,"NaN was rescued");
  const Real old_lapse=h::MaterializePositive(-1e6,A,18,1e-10,clipped);Require(clipped,"lapse floor missing");
  Require(h::MaterializePositive(-1e6,A,18,1e-10,clipped,floor,radius)==old_lapse&&clipped,"lapse policy changed");
  const Real lower=1e-10*(1-A)-A;
  Require(h::FloorDirectLapse(-A-1,A,1e-10,clipped)==lower&&clipped,"direct lapse undershoot not floored");
  Require(h::FloorDirectLapse(.2-A,A,1e-10,clipped)==.2-A&&!clipped,"inactive direct lapse changed");
  Require(h::FloorDirectLapse(-2,1,1e-10,clipped)==-2&&!clipped,"direct lapse floor imposed at scri");
  Require(std::isinf(h::FloorDirectLapse(std::numeric_limits<Real>::infinity(),A,1e-10,clipped))&&!clipped,"direct overflow was rescued");
  Require(std::isnan(h::FloorDirectLapse(std::numeric_limits<Real>::quiet_NaN(),A,1e-10,clipped))&&!clipped,"direct NaN was rescued");
  std::cout<<"chi floor: underflow, inactive parity, spatial support, nonfinite rejection and log/direct lapse policies passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
Kokkos::finalize();return code;}
