// Exercise callback attribution without requiring a profiling-enabled backend.
#include "amr_allocation_tool.cpp"
#include <iostream>
int main(){
  SpaceHandle space{};std::string("SYCLDeviceUSM").copy(space.name,13);
  int first=0,second=0,unknown=0;
  kokkosp_allocate_data(space,"scratch",&first,128);
  kokkosp_allocate_data(space,"scratch",&second,256);
  kokkosp_deallocate_data(space,"[unlabeled]",&first,128);
  kokkosp_deallocate_data(space,"scratch",&second,256);
  kokkosp_deallocate_data(space,"unknown",&unknown,1024);
  const auto s=spaces.at("SYCLDeviceUSM"),l=labels.at("SYCLDeviceUSM:scratch");
  if(s.live||l.live||s.peak!=384||l.peak!=384||unmatched_frees!=1||mismatched_labels!=1||mismatched_sizes)return 1;
  std::cout<<"PASS: pointer-based free attribution, mismatched label and unmatched free accounting\n";
}
