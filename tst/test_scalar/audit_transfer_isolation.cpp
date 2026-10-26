// Replace only frozen scalar transfer rows, and evaluate the initial RHS.
// This is a diagnostic, not an admitted time evolution or a metric model.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
#include <cstdint>
static void save_rate(Solver &s,const std::string &file) {
  s.fill(0);s.evaluate();Kokkos::fence();
  auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.active);
  auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.rhs);
  std::ofstream f(file,std::ios::binary);
  for(size_t j=0;j<ids.extent(0);++j)
    f.write(reinterpret_cast<const char*>(&r(ids(j),0)),5*sizeof(double));
  if(!f)throw std::runtime_error("RHS export failed");
}
int main(int argc,char **argv) {
  if(argc!=4)return 1;Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;p.n=std::stoi(argv[1]);p.normal_upwind=true;p.outer_sbp=2;
    p.ko=0;p.reduction_damping=0;p.smooth_layer=true;p.radial_stretch=-2;
    std::string candidate=argv[2],out=argv[3];
    if(std::filesystem::exists(out))throw std::runtime_error("Existing audit is immutable");
    Solver s(p,out);save_rate(s,out+"/original-rhs.f64");
    auto rows=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.ghosts);
    std::ifstream integers(candidate+"/transfer.i32",std::ios::binary|std::ios::ate);
    std::ifstream weights(candidate+"/transfer.f64",std::ios::binary|std::ios::ate);
    if(!integers || !weights || integers.tellg()!=std::streamoff(rows.extent(0)*16*4)
       || weights.tellg()!=std::streamoff(rows.extent(0)*50*8))
      throw std::runtime_error("Transfer schema or row count differs");
    integers.seekg(0);weights.seekg(0);
    size_t changed_indices=0;
    for(size_t j=0;j<rows.extent(0);++j) {
      int32_t a[16];double w[50];integers.read(reinterpret_cast<char*>(a),sizeof(a));
      weights.read(reinterpret_cast<char*>(w),sizeof(w));auto &r=rows(j);
      if(a[0]!=r.target || a[15]!=0)throw std::runtime_error("Target/order or transfer type differs");
      changed_indices+=a[1]!=r.start;r.start=a[1];
      for(int A=0;A<3;++A) {
        if(a[2+A]!=r.stride[A] || (a[5+A]!=1 && a[5+A]!=6))
          throw std::runtime_error("Transfer stride/count differs");
        r.stride[A]=a[2+A];r.count[A]=a[5+A];
      }
      for(int k=0;k<r.count[2];++k)for(int y=0;y<r.count[1];++y)for(int x=0;x<r.count[0];++x) {
        int l=r.start+x*r.stride[0]+y*r.stride[1]+k*r.stride[2];
        if(l<0 || size_t(l)>=s.host.size() || !s.host[l].physical.active)
          throw std::runtime_error("Transfer donor is not active");
      }
      for(int A=0;A<3;++A)for(int i=0;i<6;++i) {
        if(!std::isfinite(w[6*A+i]))throw std::runtime_error("Nonfinite weight");
        r.w[A][i]=w[6*A+i];
      }
    }
    if(!integers || !weights)throw std::runtime_error("Transfer read failed");
    Kokkos::deep_copy(s.ghosts,rows);save_rate(s,out+"/imported-rhs.f64");
    bool unchanged=s.fingerprint(false)==s.initial_physical && s.fingerprint(true)==s.initial_reference
      && s.fingerprint(false,true)==s.initial_physical_derivatives
      && s.fingerprint(true,true)==s.initial_reference_derivatives;
    std::ofstream meta(out+"/isolation.json");meta<<"{\"changed_start_indices\":"<<changed_indices
      <<",\"geometry_unchanged\":"<<(unchanged?"true":"false")
      <<",\"only_transfer_rows_replaced\":true,\"time_evolution\":false}\n";
    if(!unchanged || !meta)throw std::runtime_error("Frozen geometry changed");
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
