#include "candidate_injection.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unused_versioned_gauge_main
#include "/Users/hz0693/research/hyperboloidal/tst/hyperboloidal/test_physical_gauge.cpp"
#undef main
#pragma clang diagnostic pop
int main(int argc, char**) {
  if (argc==2) {
    std::cout<<std::setprecision(17)<<'[';
    bool first=true;
    for(double a:{.5,.75,1.,2.}) {
      if(!first)std::cout<<',';first=false;
      PrintPole(1.5,5.,true,a);
    }
    std::cout<<"]\n";
  } else { ReferenceAudit(); }
}
