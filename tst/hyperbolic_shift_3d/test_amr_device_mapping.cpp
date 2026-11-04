#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#ifdef KOKKOS_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  Kokkos::initialize(argc,argv);int bad=0;
  {
    Kokkos::View<double*> send("mapping send",size),recv("mapping receive",size);
    Kokkos::parallel_for("mapping initialize",size,KOKKOS_LAMBDA(int i){send(i)=10000*rank+i;});Kokkos::fence();
    MPI_Alltoall(send.data(),1,MPI_DOUBLE,recv.data(),1,MPI_DOUBLE,MPI_COMM_WORLD);
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),recv);
    for(int i=0;i<size;++i)bad|=host(i)!=10000*i+rank;
    char name[MPI_MAX_PROCESSOR_NAME];int len;MPI_Get_processor_name(name,&len);
    std::ostringstream info;Kokkos::print_configuration(info,true);
    const char*prefix=std::getenv("HYPERBOLIC_MAPPING_PREFIX");
    if(prefix){std::ofstream out(std::string(prefix)+".rank"+std::to_string(rank)+".txt");out<<"rank="<<rank<<" host="<<name<<"\n";
      for(const char*key:{"ZE_AFFINITY_MASK","PALS_LOCAL_RANKID","MPIR_CVAR_ENABLE_GPU","EnableImplicitScaling"})out<<key<<'='<<(std::getenv(key)?std::getenv(key):"unset")<<'\n';
      out<<info.str();
#ifdef KOKKOS_ENABLE_SYCL
      const auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);out<<"visible_gpu_devices="<<devices.size()<<'\n';bad|=devices.size()!=1;
      for(const auto&d:devices)out<<"gpu="<<d.get_info<sycl::info::device::name>()<<" driver="<<d.get_info<sycl::info::device::driver_version>()<<'\n';
#endif
      std::ifstream affinity("/proc/self/status");std::string line;while(std::getline(affinity,line))if(line.find("Cpus_allowed_list")==0)out<<line<<'\n';
      out<<"device_pointer_alltoall_passed="<<!bad<<'\n';
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return bad;
}
