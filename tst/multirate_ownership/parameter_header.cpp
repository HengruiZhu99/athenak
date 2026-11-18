#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "globals.hpp"
#include "parameter_input.hpp"

int main(int argc,char **argv) {
  global_variable::my_rank=0;global_variable::nranks=1;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc,&argv);
#endif
  Kokkos::initialize(argc,argv);
  int result=0;
  char path[]="/tmp/athena-owner-header-XXXXXX";
  const int fd=mkstemp(path);
  if(fd>=0)close(fd);
  try {
    if(fd<0)throw std::runtime_error("Cannot create temporary checkpoint header");
    std::string owners;
    for(int gid=0;gid<30000;++gid){if(gid)owners+=',';owners+=std::to_string(gid%3);}
    ParameterInput original;
    original.SetString("multirate","block_owners",owners);
    for(int setting=0;setting<20;++setting)
      original.SetInteger("multirate","setting"+std::to_string(setting),setting);
    std::ostringstream out;original.ParameterDump(out);
    auto bytes=out.str();
    if(bytes.size()>owners.size()+4000)
      throw std::runtime_error("Owner map multiplied checkpoint padding");
    // Force <par_end> across a read-buffer boundary to test incremental search.
    const auto marker=bytes.find("<par_end>");
    auto padding=(4092+4096-marker%4096)%4096;
    if(padding<2)padding+=4096;
    bytes.insert(marker,"#"+std::string(padding-2,' ')+"\n");
    {std::ofstream file(path,std::ios::binary);file<<bytes<<"binary-payload";}
    IOWrapper file;file.Open(path,IOWrapper::FileMode::read,true);
    ParameterInput restored;restored.LoadFromFile(file,true);restored.CheckBlockNames();
    if(file.GetPosition(true)!=bytes.size() ||
        restored.GetString("multirate","block_owners")!=owners ||
        restored.GetInteger("multirate","setting19")!=19)
      throw std::runtime_error("Large owner header did not round-trip at the payload offset");
    file.Close(true);
  }catch(const std::exception &error){std::cerr<<error.what()<<'\n';result=1;}
  if(fd>=0)std::remove(path);
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return result;
}
