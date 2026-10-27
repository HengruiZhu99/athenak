// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"
int main(){using P=z4c::hyperboloidal::EnrichedLogDerivative;std::vector<P::Point>points;
  const Real h=.25/6;for(int k=0;k<6;++k)for(int j=0;j<6;++j)for(int i=0;i<6;++i)points.push_back({-.25+(i+.5)*h,-.25+(j+.5)*h,-.25+(k+.5)*h});
  const P::Point query={-.25+.25*h,-.25+.25*h,-.25+.25*h};
  for(Real width:{2,4,8,16}){P plan;P::Options options;options.gaussian_width=width;
    const bool built=plan.Build(query,points,h,{},options);const auto&d=plan.diagnostics();Real l1=0;
    if(built){std::vector<Real> weights(points.size());Real sum=0;
      for(size_t i=0;i<points.size();++i){std::vector<Real> unit(points.size(),0);unit[i]=1;P::Result value;
        if(!plan.Evaluate(unit,{},value))return 2;weights[i]=value.fitted_value;sum+=weights[i];}
      weights[0]+=1-sum;for(Real value:weights)l1+=std::abs(value);}
    const bool ready=built&&l1<=8;const auto reason=built?(ready?std::string("ready"):std::string("value_weight_amplification")):"basis_"+d.status;
    std::cout<<"{\"width\":"<<width<<",\"ready\":"<<ready<<",\"rank\":"<<d.rank<<",\"condition\":"<<d.condition<<",\"value_L1\":"<<l1<<",\"status\":\""<<reason<<"\"}\n";
    if(ready)return 3; // retained counterexample: broad weighting is no cure
  }
}
