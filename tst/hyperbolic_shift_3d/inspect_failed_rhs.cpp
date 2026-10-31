// Read-only diagnostic of a rejected snapshot. No stepping or checkpoint writes.
#define main hyperbolic_unused_evolution_main
#include "coupled_mpi.cpp"
#undef main

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);int code=0;
  try {
    if(argc!=3||global_variable::nranks!=1)throw std::invalid_argument("usage: inspect_failed_rhs n capture (one rank)");
    const int n=std::stoi(argv[1]);Evolution e(n,2,2,"");
    V values(31*e.total),edge(31*e.surface_total);
    std::ifstream input(argv[2],std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(Real));
    input.read(reinterpret_cast<char*>(edge.data()),edge.size()*sizeof(Real));
    if(!input||input.peek()!=EOF)throw std::runtime_error("capture length mismatch");
    std::cout<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"modes\":[";
    for(int mode=0;mode<4;++mode) {
      V v=values;int clipped=0;
      if(mode==3)for(int p=0;p<e.total;++p)if(v[31*p]<=0){v[31*p]=1e-8;++clipped;}
      e.Upload(v);Evolution::CopyFromHost(edge,e.surface);
      const auto halo=e.exchange()->Fill(e.state);
      const auto stats=e.kernel()->Evaluate(e.exchange()->padded,.0005*16/n,false,
          mode==1?0:.5,e.surface,mode==2?Buffer{}:e.exchange()->positive_ko,false);
      const auto rhs=e.Download(e.kernel()->LocalRHS());
      std::vector<int> bad,counts(31,0);
      for(int p=0;p<e.total;++p){bool invalid=false;for(int f=0;f<31;++f)
        if(!std::isfinite(rhs[31*p+f])){++counts[f];invalid=true;}
        if(invalid)bad.push_back(p);}
      if(mode)std::cout<<',';
      std::cout<<"{\"mode\":"<<JsonString(mode==0?"original_log_KO":mode==1?"KO_disabled":mode==2?"physical_variable_KO":"snapshot_zero_chi_floor_log_KO")
        <<",\"halo_invalid\":"<<halo.invalid<<",\"bad_cells\":"<<stats.bad_cells
        <<",\"first_bad\":"<<stats.first_bad<<",\"reason\":"<<stats.reason
        <<",\"snapshot_clipped_cells\":"<<clipped<<",\"nonfinite_rhs_cells\":[";
      for(size_t i=0;i<bad.size();++i){if(i)std::cout<<',';std::cout<<bad[i];}
      std::cout<<"],\"nonfinite_field_counts\":[";
      for(int f=0;f<31;++f){if(f)std::cout<<',';std::cout<<counts[f];}
      std::cout<<"],\"cell3071_chi_rhs\":"<<JsonNumber(rhs[31*3071])<<'}';
    }
    std::cout<<"],\"zero_chi_donors_for_cell3071\":[";bool comma=false;
    auto &grid=e.Model().grid;
    for(int A=0;A<3;++A)for(int a=0;a<5;++a){
      const int start=grid.ShiftActive(3071,A,-a);
      if(start<0||grid.ShiftActive(start,A,4)<0)continue;
      for(int k=0;k<5;++k){const int donor=grid.ShiftActive(start,A,k);
        if(values[31*donor]>0)continue;
        if(comma)std::cout<<',';comma=true;
        std::cout<<"{\"axis\":"<<A<<",\"start\":"<<start<<",\"target_slot\":"<<a
          <<",\"donor_slot\":"<<k<<",\"donor\":"<<donor<<'}';}}
    std::cout<<"]}\n";
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';code=1;}
  Kokkos::finalize();MPI_Finalize();return code;
}
