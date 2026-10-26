// Read-only full scalar RHS benchmark. No time steps or background updates.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
#include <chrono>
int main(int argc,char **argv) {
  if(argc!=5)return 1;Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;p.n=std::stoi(argv[1]);p.mass=1;p.spin=0;p.excision=.25;
    p.normal_upwind=true;p.outer_sbp=2;p.ko=0;p.reduction_damping=0;
    p.smooth_layer=true;p.radial_stretch=-2;p.boundary="radiation";
    p.data="ringdown";p.tlim=100;const int repeats=std::stoi(argv[4]);
    const std::string input=argv[2],out=argv[3];
    if(repeats<1 || std::filesystem::exists(out))throw std::runtime_error("Invalid or existing benchmark");
    Solver s(p,out);auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.active);
    auto state=Kokkos::create_mirror_view(s.u);
    std::ifstream f(input,std::ios::binary);f.read(reinterpret_cast<char*>(state.data()),state.size()*sizeof(double));
    if(!f || f.peek()!=EOF)throw std::runtime_error("Native checkpoint shape mismatch");
    Kokkos::deep_copy(s.u,state);
    auto field_hash=[&]() {
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.u);
      uint64_t value=14695981039346656037ULL;
      for(size_t i=0;i<ids.extent(0);++i)for(int v=0;v<5;++v) {
        uint64_t bits;double x=h(ids(i),v);std::memcpy(&bits,&x,8);
        for(int j=0;j<8;++j){value^=(bits>>(8*j))&255;value*=1099511628211ULL;}
      }return value;
    };
    const auto before=field_hash();
    for(int i=0;i<3;++i){s.fill(100);s.evaluate();Kokkos::fence();}
    double fill_seconds=0,evaluate_seconds=0;
    for(int i=0;i<repeats;++i) {
      auto a=std::chrono::steady_clock::now();s.fill(100);Kokkos::fence();
      auto b=std::chrono::steady_clock::now();s.evaluate();Kokkos::fence();
      auto c=std::chrono::steady_clock::now();
      fill_seconds+=std::chrono::duration<double>(b-a).count();
      evaluate_seconds+=std::chrono::duration<double>(c-b).count();
    }
    std::ifstream numa("/proc/self/numa_maps");std::ofstream maps(out+"/numa_maps.txt");maps<<numa.rdbuf();
    const auto after=field_hash();
    auto rate=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.rhs);
    std::ofstream output(out+"/active-rhs.f64",std::ios::binary);
    for(size_t i=0;i<ids.extent(0);++i)for(int v=0;v<5;++v) {
      const double x=rate(ids(i),v);if(!std::isfinite(x))throw std::runtime_error("Nonfinite scalar RHS");
      output.write(reinterpret_cast<const char*>(&x),sizeof(double));
    }
    bool unchanged=s.fingerprint(false)==s.initial_physical && s.fingerprint(true)==s.initial_reference
      && s.fingerprint(false,true)==s.initial_physical_derivatives && s.fingerprint(true,true)==s.initial_reference_derivatives;
    std::ofstream meta(out+"/benchmark.json");meta<<std::setprecision(17)
      <<"{\"n\":"<<p.n<<",\"active_cells\":"<<ids.extent(0)<<",\"repeats\":"<<repeats
      <<",\"fill_seconds_per_call\":"<<fill_seconds/repeats<<",\"evaluate_seconds_per_call\":"<<evaluate_seconds/repeats
      <<",\"active_state_unchanged\":"<<(before==after?"true":"false")<<",\"frozen_unchanged\":"<<(unchanged?"true":"false")
      <<",\"active_state_before\":\""<<std::hex<<before<<"\",\"active_state_after\":\""<<after
      <<"\",\"physical_before\":\""<<s.initial_physical<<"\",\"reference_before\":\""<<s.initial_reference
      <<"\",\"physical_derivatives_before\":\""<<s.initial_physical_derivatives<<"\",\"reference_derivatives_before\":\""<<s.initial_reference_derivatives<<"\"}\n";
    if(!unchanged || before!=after || !output || !meta)throw std::runtime_error("Read-only benchmark integrity failed");
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
