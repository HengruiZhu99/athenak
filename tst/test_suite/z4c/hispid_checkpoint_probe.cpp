// Standalone real checkpoint-reader/sampler check, without starting evolution.
#include "hispid_checkpoint.hpp"
#include <iomanip>
#include <iostream>
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  try {
    auto c=hispid_import::Read(argv[1],argv[2]);
    hispid_import::Context data(HiSpID_create_with_seed_family(&c.config,c.seed_family,0,0,1));
    if(!data || HiSpID_seed_family(data.get())!=c.seed_family ||
       HiSpID_set_unknowns(data.get(),c.unknowns.data(),c.unknowns.size()))
      throw std::runtime_error(HiSpID_last_error());
    const double xyz[]={1.2,.7,-.4,2.,-1.,.8};
    HiSpID_Point points[2];double dg[54];
    if(HiSpID_sample_with_derivatives(data.get(),2,xyz,points,dg))
      throw std::runtime_error(HiSpID_last_error());
    std::cout<<std::setprecision(17)<<"{\"seed_family\":"<<c.seed_family<<",\"values\":[";
    bool first=true;
    auto emit=[&](double v){if(!first)std::cout<<",";first=false;std::cout<<v;};
    for(auto &p:points){for(double v:p.gamma)emit(v);for(double v:p.Kij)emit(v);}
    for(double v:dg)emit(v);
    std::cout<<"]}\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
