// Export the actual frozen interpolation/metric operator and initial RHS.
// No time stepping and no modification of a production executable.
#define main frozen_scalar_program_main
#include "../../src/test_scalar/main.cpp"
#undef main
int main(int argc,char **argv) {
  if(argc!=3)return 1;Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;p.n=std::stoi(argv[1]);p.normal_upwind=true;p.outer_sbp=2;p.ko=0;p.reduction_damping=0;p.smooth_layer=true;p.radial_stretch=-2;
    const std::string out=argv[2];if(std::filesystem::exists(out))throw std::runtime_error("existing operator export is immutable");
    Solver s(p,out);auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.active);auto rows=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.ghosts);auto kernel=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.kernel);auto initial=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.u);
    std::ofstream mi(out+"/transfer.i32",std::ios::binary),mw(out+"/transfer.f64",std::ios::binary),mx(out+"/transfer-positions.f64",std::ios::binary),ki(out+"/kernel.i32",std::ios::binary),kw(out+"/kernel.f64",std::ios::binary),ui(out+"/initial.f64",std::ios::binary);
    for(size_t i=0;i<rows.extent(0);++i) {
      const auto &r=rows(i);int a[16]={r.target,r.start};for(int j=0;j<3;++j){a[2+j]=r.stride[j];a[5+j]=r.count[j];a[9+j]=r.own_stride[j];a[12+j]=r.own_count[j];}a[8]=r.own_start;a[15]=r.characteristic;mi.write(reinterpret_cast<const char*>(a),sizeof(a));
      double w[50];int z=0;for(auto &v:r.w)for(double x:v)w[z++]=x;for(auto &v:r.own_w)for(double x:v)w[z++]=x;for(double x:r.normal)w[z++]=x;for(double x:r.ginvn)w[z++]=x;for(double x:r.beta)w[z++]=x;w[z++]=r.alpha;w[z++]=r.root;for(double x:r.speeds)w[z++]=x;if(z!=50)throw std::runtime_error("row schema error");mw.write(reinterpret_cast<const char*>(w),sizeof(w));mx.write(reinterpret_cast<const char*>(s.host[r.target].physical.x),3*sizeof(double));
    }
    for(size_t j=0;j<ids.extent(0);++j) {
      int l=ids(j);const auto &c=kernel(l);const auto &f=c.physical;int a[15]={l};for(int i=0;i<3;++i){a[1+i]=c.stride[i];a[4+i]=c.one[i];a[7+i]=c.trim_ko[i];}a[10]=c.outer_distance;a[11]=c.radial_sbp63;a[12]=c.radial_end;a[13]=c.normal_upwind;a[14]=c.normal_sbp;ki.write(reinterpret_cast<const char*>(a),sizeof(a));
      double w[56];int z=0;for(auto &v:f.g)for(double x:v)w[z++]=x;for(double x:f.b)w[z++]=x;for(auto &v:f.beta_gradient)for(double x:v)w[z++]=x;for(double x:f.alpha_gradient)w[z++]=x;w[z++]=f.potential;for(double x:f.x)w[z++]=x;for(auto &v:f.map.J)for(double x:v)w[z++]=x;for(double x:c.idx)w[z++]=x;for(double x:c.ko)w[z++]=x;w[z++]=c.boundary_rate;for(double x:c.radial_speeds)w[z++]=x;w[z++]=c.outer_norm;w[z++]=c.boundary_amplitude_gradient;if(z!=56)throw std::runtime_error("kernel schema error");kw.write(reinterpret_cast<const char*>(w),sizeof(w));ui.write(reinterpret_cast<const char*>(&initial(l,0)),5*sizeof(double));
    }
    s.fill(0);s.evaluate();Kokkos::fence();auto rate=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),s.rhs);std::ofstream rr(out+"/initial-rhs.f64",std::ios::binary);for(size_t j=0;j<ids.extent(0);++j)rr.write(reinterpret_cast<const char*>(&rate(ids(j),0)),5*sizeof(double));
    bool unchanged=s.fingerprint(false)==s.initial_physical && s.fingerprint(true)==s.initial_reference && s.fingerprint(false,true)==s.initial_physical_derivatives && s.fingerprint(true,true)==s.initial_reference_derivatives;
    std::ofstream meta(out+"/operator.json");meta<<std::setprecision(17)<<"{\"n\":"<<p.n<<",\"active\":"<<ids.extent(0)<<",\"rows\":"<<rows.extent(0)<<",\"dt\":"<<s.dt<<",\"physical\":\""<<std::hex<<s.initial_physical<<"\",\"reference\":\""<<s.initial_reference<<"\",\"physical_derivatives\":\""<<s.initial_physical_derivatives<<"\",\"reference_derivatives\":\""<<s.initial_reference_derivatives<<"\",\"frozen_unchanged\":"<<(unchanged?"true":"false")<<"}\n";
    if(!unchanged || !mi || !mw || !mx || !ki || !kw || !ui || !rr || !meta)throw std::runtime_error("operator export failed");
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
