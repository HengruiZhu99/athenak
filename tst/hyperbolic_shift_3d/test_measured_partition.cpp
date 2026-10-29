#include <iostream>
#include <sstream>
#include "z4c/hyperboloidal/measured_partition.hpp"
int main(){
  using z4c::hyperboloidal::ReadMeasuredPartition;
  const std::string header="HYP3D_COST_PARTITION_V1\n16 100 2\n"+std::string(64,'a')+"\n";
  std::istringstream valid(header+"0 67 100\n");
  auto plan=ReadMeasuredPartition(valid,16,100,2);
  if(plan.cuts!=std::vector<int>({0,67,100})||plan.source_report_sha256!=std::string(64,'a'))return 1;
  auto reject=[](const std::string&text,int n,int cells,int ranks){
    try{std::istringstream in(text);ReadMeasuredPartition(in,n,cells,ranks);return false;}
    catch(const std::invalid_argument&){return true;}
  };
  for(const std::string cuts:{"0 0 100","0 100 100","0 101 100","1 67 100","0 67 99","0 67","0 67 100 extra","0 99999999999999999999 100"})
    if(!reject(header+cuts,16,100,2))return 2;
  if(!reject(header+"0 67 100",12,100,2)||!reject(header+"0 67 100",16,101,2)||!reject(header+"0 67 100",16,100,3))return 3;
  if(!reject("HYP3D_COST_PARTITION_V1 16 100 2 "+std::string(64,'z')+" 0 67 100",16,100,2))return 4;
  std::cout<<"PASS: fixed cost-partition descriptor, exact grid/rank identity, coverage, trailing/truncated/overflow rejection\n";
}
