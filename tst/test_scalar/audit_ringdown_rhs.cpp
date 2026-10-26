// Evaluate the actual compiled scalar RHS on saved black-hole states.
// No time stepping and no geometry, gauge or constraint evolution.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
int main(int argc,char **argv) {
  if(argc!=6)return 1;Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;p.n=std::stoi(argv[1]);p.mass=1;p.spin=std::stod(argv[2]);p.excision=std::stod(argv[3]);
    p.normal_upwind=true;p.outer_sbp=2;p.ko=0;p.reduction_damping=0;p.smooth_layer=true;p.radial_stretch=-2;
    p.boundary="radiation";p.data="ringdown";p.tlim=100;
    const std::string run=argv[4],out=argv[5];
    if(std::filesystem::exists(out))throw std::runtime_error("Existing RHS audit is immutable");
    Solver s(p,out);auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.active);
    auto state=Kokkos::create_mirror_view(s.u);double restore=0;
    for(double t:{2.,6.,12.,100.}) {
      std::ostringstream label;label<<std::fixed<<std::setprecision(3)<<t;
      std::ifstream input(run+"/fields-t"+label.str()+".f64",std::ios::binary);
      std::vector<std::array<double,12>> rows(ids.extent(0));input.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(rows[0]));
      if(!input || input.peek()!=EOF)throw std::runtime_error("Snapshot size mismatch");
      if(t==100) {
        std::ifstream checkpoint(run+"/checkpoint.f64",std::ios::binary);checkpoint.read(reinterpret_cast<char*>(state.data()),state.size()*sizeof(double));
        if(!checkpoint || checkpoint.peek()!=EOF)throw std::runtime_error("Checkpoint size mismatch");
      }
      for(size_t i=0;i<ids.extent(0);++i) {
        int l=ids(i);const auto &c=s.host[l].physical;const auto &row=rows[i];double betaD=0;
        for(int a=0;a<3;++a) {
          if(row[a]!=c.x[a])throw std::runtime_error("Saved state grid differs from actual compiled operator");
          betaD+=-c.g[0][a+1]/c.g[0][0]*row[9+a];
        }
        double Pi=(betaD-row[5])*std::sqrt(-c.g[0][0]);
        if(t==100) {
          restore=std::max(restore,std::abs(Pi-state(l,1)));
          if(row[4]!=state(l,0))throw std::runtime_error("Checkpoint scalar differs from snapshot");
          for(int a=0;a<3;++a)if(row[6+a]!=state(l,2+a))throw std::runtime_error("Checkpoint scalar gradient differs");
        }else {state(l,0)=row[4];state(l,1)=Pi;for(int a=0;a<3;++a)state(l,2+a)=row[6+a];}
      }
      Kokkos::deep_copy(s.u,state);s.fill(t);s.evaluate();Kokkos::fence();
      auto rate=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.rhs);
      std::ofstream output(out+"/actual-rhs-t"+label.str()+".f64",std::ios::binary);
      for(size_t i=0;i<ids.extent(0);++i)output.write(reinterpret_cast<const char*>(&rate(ids(i),0)),sizeof(double));
      if(!output)throw std::runtime_error("Actual RHS write failed");
    }
    bool unchanged=s.fingerprint(false)==s.initial_physical && s.fingerprint(true)==s.initial_reference
      && s.fingerprint(false,true)==s.initial_physical_derivatives && s.fingerprint(true,true)==s.initial_reference_derivatives;
    std::ofstream meta(out+"/actual-rhs.json");meta<<std::setprecision(17)<<"{\"n\":"<<p.n<<",\"spin\":"<<p.spin
      <<",\"excision\":"<<p.excision<<",\"Pi_restore_max\":"<<restore<<",\"frozen_unchanged\":"<<(unchanged?"true":"false")
      <<",\"physical_before\":\""<<std::hex<<s.initial_physical<<"\",\"reference_before\":\""<<s.initial_reference
      <<"\",\"physical_derivatives_before\":\""<<s.initial_physical_derivatives<<"\",\"reference_derivatives_before\":\""<<s.initial_reference_derivatives<<"\"}\n";
    if(!unchanged || !meta)throw std::runtime_error("Actual RHS frozen geometry check failed");
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
