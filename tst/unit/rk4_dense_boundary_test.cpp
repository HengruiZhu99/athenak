#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/rk4_dense_boundary.hpp"

void Check(bool ok) { if (!ok) throw std::runtime_error("RK4 dense boundary regression"); }
double RHS(double t, double y) { return y*y-std::exp(2*t)+std::exp(t); }
std::array<double,4> Stages(double t, double h, double y) {
  double y2=y+0.5*h*RHS(t,y);
  double y3=y+0.5*h*RHS(t+0.5*h,y2);
  double y4=y+h*RHS(t+0.5*h,y3);
  return {y,y2,y3,y4};
}
double Error(double H) {
  auto coarse=Stages(0,H,1);
  subcycling::RK4DenseBoundary p{1,H,RHS(0,coarse[0]),
      RHS(H/2,coarse[1]),RHS(H/2,coarse[2]),RHS(H,coarse[3])};
  Check(p.Value(0)==1);
  const double endpoint=1+H*(p.k1+2*p.k2+2*p.k3+p.k4)/6;
  Check(std::abs(p.Value(1)-endpoint)<1e-14);
  double error=0;
  for (double q : {0.0,0.5}) {
    auto expected=Stages(q*H,H/2,std::exp(q*H));
    for (int stage=1; stage<=4; ++stage) {
      error=std::max(error,std::abs(p.StageBoundary(q,H/2,stage)-expected[stage-1]));
    }
  }
  return error;
}
int main() {
  double previous=Error(0.2);
  for (double H : {0.1,0.05,0.025}) {
    double error=Error(H), ratio=previous/error;
    std::cout << H << " error=" << error << " ratio=" << ratio << '\n';
    Check(ratio>12 && ratio<24); previous=error;
  }
  // Stage2 at a half-step is an Euler stage, not the physical dense state.
  auto coarse=Stages(0,0.1,1);
  subcycling::RK4DenseBoundary p{1,0.1,RHS(0,coarse[0]),
      RHS(.05,coarse[1]),RHS(.05,coarse[2]),RHS(.1,coarse[3])};
  Check(std::abs(p.StageBoundary(0,.05,2)-p.Value(.25))>1e-5);
  // Prescribed boundary RHS must generate the same stage vectors AND endpoint
  // through the actual classical RK algebra, including a shifted child step.
  for(double q:{0.,.25,.5}) for(double h:{.01,.025}) {
    const double y=p.Value(q),k1=p.StageRHS(q,h,1),k2=p.StageRHS(q,h,2),
                 k3=p.StageRHS(q,h,3),k4=p.StageRHS(q,h,4);
    Check(std::abs(y+h*k1/2-p.StageBoundary(q,h,2))<1e-14);
    Check(std::abs(y+h*k2/2-p.StageBoundary(q,h,3))<1e-14);
    Check(std::abs(y+h*k3-p.StageBoundary(q,h,4))<1e-14);
    Check(std::abs(y+h*(k1+2*k2+2*k3+k4)/6-p.Value(q+h/p.H))<1e-14);
  }
  std::cout << "PASS: child stage consistency and fourth-order local boundary error\n";
}
