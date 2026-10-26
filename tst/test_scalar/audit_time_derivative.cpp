// Evaluate the actual compiled scalar RHS on saved states; never time step.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
int main(int argc,char **argv) {
  if(argc!=4)return 1;
  Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;p.n=std::stoi(argv[1]);p.normal_upwind=true;p.outer_sbp=2;p.ko=0;p.reduction_damping=0;p.smooth_layer=true;p.radial_stretch=-2;
    const std::string run=argv[2],out=argv[3];
    if(std::filesystem::exists(out))throw std::runtime_error("existing derivative audit is immutable");
    Solver s(p,out);auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.active);auto state=Kokkos::create_mirror_view(s.u);
    std::ofstream csv(out+"/actual-time-derivative.csv");csv<<std::setprecision(17)<<"time,region,RHS_L2,RHS_Linf,reconstruction_L2,reconstruction_Linf,Pi_restore_max\n";
    for(double t:{2.,6.,12.}) {
      std::ostringstream name;name<<run<<"/fields-t"<<std::fixed<<std::setprecision(3)<<t<<".f64";std::ifstream input(name.str(),std::ios::binary);
      std::vector<std::array<double,12>> rows(ids.extent(0));input.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(rows[0]));
      if(!input || input.peek()!=EOF)throw std::runtime_error("snapshot size mismatch");
      double restore=0;
      if(t==12) {
        std::ifstream checkpoint(run+"/checkpoint.f64",std::ios::binary);checkpoint.read(reinterpret_cast<char*>(state.data()),state.size()*sizeof(double));
        if(!checkpoint || checkpoint.peek()!=EOF)throw std::runtime_error("checkpoint size mismatch");
      }
      for(size_t z=0;z<ids.extent(0);++z) {
        int l=ids(z);const auto &c=s.host[l].physical;const auto &row=rows[z];double betaD=0;
        for(int i=0;i<3;++i){if(row[i]!=c.x[i])throw std::runtime_error("snapshot grid differs");betaD+=-c.g[0][i+1]/c.g[0][0]*row[9+i];}
        double Pi=(betaD-row[5])*std::sqrt(-c.g[0][0]);
        if(t==12) {restore=std::max(restore,std::abs(Pi-state(l,1)));if(row[4]!=state(l,0))throw std::runtime_error("checkpoint scalar differs");for(int i=0;i<3;++i)if(row[6+i]!=state(l,2+i))throw std::runtime_error("checkpoint gradient differs");}
        else {state(l,0)=row[4];state(l,1)=Pi;for(int i=0;i<3;++i)state(l,2+i)=row[6+i];}
      }
      Kokkos::deep_copy(s.u,state);s.fill(t);s.evaluate();Kokkos::fence();auto rate=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.rhs);double sum[3][2]{},maximum[3][2]{},volume[3]{};
      for(size_t z=0;z<ids.extent(0);++z) {
        int l=ids(z);const auto &c=s.host[l];double exact=ts::exact(c.physical.x,t,p)[1],error[2]={rate(l,0)-exact,rows[z][5]-exact};
        for(int region:{0,c.block?2:1}){volume[region]+=c.weight;for(int k=0;k<2;++k){sum[region][k]+=c.weight*error[k]*error[k];maximum[region][k]=std::max(maximum[region][k],std::abs(error[k]));}}
      }
      for(int region=0;region<3;++region)csv<<t<<","<<(region==0?"all":region==1?"bulk":"shell")<<","<<std::sqrt(sum[region][0]/volume[region])<<","<<maximum[region][0]<<","<<std::sqrt(sum[region][1]/volume[region])<<","<<maximum[region][1]<<","<<restore<<"\n";
    }
    if(s.fingerprint(false)!=s.initial_physical || s.fingerprint(true)!=s.initial_reference || s.fingerprint(false,true)!=s.initial_physical_derivatives || s.fingerprint(true,true)!=s.initial_reference_derivatives)throw std::runtime_error("frozen audit coefficients changed");
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
