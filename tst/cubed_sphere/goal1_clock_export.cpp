// Passive oracle evaluation at saved native coordinates. No evolution.
#include "goal1_clock_runtime.hpp"
#include "goal1_operator_oracle.hpp"
#include <fstream>
#include <cstdint>
#include <iostream>
int main(int argc,char**argv){if(argc!=3)return 2;std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);double time,epsilon;std::uint64_t count,cols=286;in.read(reinterpret_cast<char*>(&time),8);in.read(reinterpret_cast<char*>(&epsilon),8);in.read(reinterpret_cast<char*>(&count),8);out.write(reinterpret_cast<const char*>(&count),8);out.write(reinterpret_cast<const char*>(&cols),8);cubed_sphere::clock_oracle::Height height(48);
for(std::uint64_t k=0;k<count;++k){Real x[3],n[3]{};in.read(reinterpret_cast<char*>(x),24);Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);if(r)for(int i=0;i<3;++i)n[i]=x[i]/r;auto radial=cubed_sphere::clock_oracle::Evaluate(r,time,epsilon,height);auto target=cubed_sphere::clock_oracle::NativeJet(radial,r,n);for(int field=0;field<22;++field){auto q=cubed_sphere::operator_oracle::Component(target,field);out.write(reinterpret_cast<const char*>(&q.value),8);out.write(reinterpret_cast<const char*>(q.d),24);out.write(reinterpret_cast<const char*>(q.dd),72);}}
if(!in||!out)return 3;return 0;}
