//========================================================================================
// HiSpID physical initial-data import and optional initial-time FastFlow check.
//========================================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/fastflow.hpp"
#include "hispid_checkpoint.hpp"
#if HISPID_DYNAMIC_IMAGE
#include <dlfcn.h>
#endif
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
constexpr int full[6]={0,1,2,4,5,8};
void Refine(MeshBlockPack *pack) { pack->pz4c->pamr->Refine(pack); }
void PositiveMetric(const double *g) {
  double det=g[0]*(g[4]*g[8]-g[5]*g[7])-g[1]*(g[3]*g[8]-g[5]*g[6])+g[2]*(g[3]*g[7]-g[4]*g[6]);
  if (!(g[0]>0 && g[0]*g[4]-g[1]*g[3]>0 && det>0) || !std::isfinite(det))
    throw std::runtime_error("HiSpID metric is not finite positive definite");
}
void Flat(HiSpID_Point &p, double *dg=nullptr) {
  p={};p.gamma[0]=p.gamma[4]=p.gamma[8]=1;p.psi=1;p.attenuation=1;
  if (dg) std::fill(dg,dg+27,0.0);
}
struct ResetSource {
  FastFlow &finder;
  ~ResetSource() { finder.geometry_source={};finder.initial_shape={}; }
};
void VerifyConsumerImage() {
#if HISPID_DYNAMIC_IMAGE
  // A compiled SHA describes the linked file. Also check the actual images
  // supplying these entry points, so loader interposition cannot silently
  // replace the sampler used by the separately proved migration.
  const auto expected=std::filesystem::canonical(HISPID_LIBRARY_PATH);
  auto address=[](const char *name,bool required) {
    dlerror();void *value=dlsym(RTLD_DEFAULT,name);const char *error=dlerror();
    if (required && (!value || error))
      throw std::runtime_error(std::string("HiSpID consumer symbol unavailable: ")+name);
    return error?nullptr:value;
  };
  for (const auto name:{"HiSpID_create_sampler","HiSpID_sample","HiSpID_sample_with_derivatives"}) {
    Dl_info info{};
    if (!dladdr(address(name,true),&info) || !info.dli_fname ||
        std::filesystem::canonical(info.dli_fname)!=expected)
      throw std::runtime_error("HiSpID loaded consumer image differs from the configured library");
  }
  for (const auto name:{"AB_To_XR","C_To_c","PK_solve","Puncture_execution_initialize"}) {
    const auto symbol=address(name,std::string(name)=="AB_To_XR" || std::string(name)=="C_To_c");
    if (!symbol) continue;
    Dl_info info{};
    if (!dladdr(symbol,&info) || !info.dli_fname)
      throw std::runtime_error("HiSpID puncture dependency image unavailable");
    const auto image=std::filesystem::canonical(info.dli_fname);
    if (image!=expected && (image.parent_path()!=expected.parent_path() ||
        (image.filename()!="libTwoPunctures.so" && image.filename()!="libTwoPunctures.dylib")))
      throw std::runtime_error("HiSpID loaded puncture dependency differs from the configured directory");
    if (global_variable::my_rank==0)
      std::cout << "HiSpID consumer_dependency symbol=" << std::quoted(name)
                << " image=" << std::quoted(image.string()) << std::endl;
  }
  if (global_variable::my_rank==0)
    std::cout << "HiSpID consumer_image=" << std::quoted(expected.string()) << std::endl;
#endif
}
void Initialize(MeshBlockPack *pack, ParameterInput *pin) {
  VerifyConsumerImage();
  const auto checkpoint=hispid_import::Read(pin->GetString("problem","hispid_filename"),
                                           pin->GetString("problem","hispid_source_sha256"));
  if (checkpoint.library_sha!=HISPID_LIBRARY_SHA256 &&
      !pin->GetOrAddBoolean("problem","hispid_allow_library_migration",false))
    throw std::runtime_error("Checkpoint/consumer-library SHA differs; validate the new build before explicit migration");
  if (checkpoint.acceptance=="diagnostic" && !pin->GetOrAddBoolean("problem","hispid_allow_diagnostic",false))
    throw std::runtime_error("Diagnostic checkpoint requires hispid_allow_diagnostic=true");
  hispid_import::Context data(HiSpID_create_sampler(&checkpoint.config));
  if (!data || HiSpID_set_unknowns(data.get(),checkpoint.unknowns.data(),checkpoint.unknowns.size()))
    throw std::runtime_error(std::string("HiSpID checkpoint: ")+HiSpID_last_error());
  const bool flat=pin->GetOrAddBoolean("problem","hispid_flat_control",false);
  auto ind=pack->pmesh->mb_indcs;auto &sizes=pack->pmb->mb_size;sizes.sync<HostMemSpace>();
  // create_mirror allocates independent storage even for a Serial host build;
  // create_mirror_view may alias u_adm and would invalidate the round-trip test.
  auto initial=Kokkos::create_mirror(pack->padm->u_adm);Kokkos::deep_copy(initial,0.0);
  std::vector<double> positions(3*(ind.nx1+2*ind.ng));
  std::vector<HiSpID_Point> samples(ind.nx1+2*ind.ng);
  const int nx=ind.nx1+2*ind.ng,ny=ind.nx2+2*ind.ng,nz=ind.nx3+2*ind.ng;
  std::vector<double> attenuation(static_cast<std::size_t>(pack->nmb_thispack)*nx*ny*nz);
  for (int m=0;m<pack->nmb_thispack;++m) {
    auto size=sizes.h_view(m);
    for (int k=0;k<ind.nx3+2*ind.ng;++k) for (int j=0;j<ind.nx2+2*ind.ng;++j) {
      for (int i=0;i<ind.nx1+2*ind.ng;++i) {
        positions[3*i]=CellCenterX(i-ind.is,ind.nx1,size.x1min,size.x1max);
        positions[3*i+1]=CellCenterX(j-ind.js,ind.nx2,size.x2min,size.x2max);
        positions[3*i+2]=CellCenterX(k-ind.ks,ind.nx3,size.x3min,size.x3max);
      }
      if (!flat && HiSpID_sample(data.get(),samples.size(),positions.data(),samples.data()))
        throw std::runtime_error(std::string("HiSpID mesh sampling: ")+HiSpID_last_error());
      for (int i=0;i<ind.nx1+2*ind.ng;++i) {
        if (flat) Flat(samples[i]);PositiveMetric(samples[i].gamma);
        attenuation[((static_cast<std::size_t>(m)*nz+k)*ny+j)*nx+i]=samples[i].attenuation;
        for (int c=0;c<6;++c) {
          if (!std::isfinite(samples[i].Kij[full[c]])) throw std::runtime_error("Nonfinite imported Kij");
          // Native gamma and Kij are already PHYSICAL tensors.
          initial(m,adm::ADM::I_ADM_GXX+c,k,j,i)=samples[i].gamma[full[c]];
          initial(m,adm::ADM::I_ADM_KXX+c,k,j,i)=samples[i].Kij[full[c]];
        }
      }
    }
  }
  Kokkos::deep_copy(pack->padm->u_adm,initial);Kokkos::deep_copy(pack->pz4c->u0,0.0);
  auto state=pack->pz4c->z4c;
  par_for("HiSpID initial lapse",DevExeSpace(),0,pack->nmb_thispack-1,
      0,ind.nx3+2*ind.ng-1,0,ind.nx2+2*ind.ng-1,0,ind.nx1+2*ind.ng-1,
      KOKKOS_LAMBDA(int m,int k,int j,int i) { state.alpha(m,k,j,i)=1.0; });
  switch (ind.ng) {
    case 2: pack->pz4c->ADMToZ4c<2>(pack,pin);break;
    case 3: pack->pz4c->ADMToZ4c<3>(pack,pin);break;
    case 4: pack->pz4c->ADMToZ4c<4>(pack,pin);break;
    default: throw std::runtime_error("HiSpID requires 2,3 or4 ghosts");
  }
  pack->pz4c->Z4cToADM(pack);pack->pz4c->GaugePreCollapsedLapse(pack,pin);
  auto roundtrip=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->padm->u_adm);
  Real error=0;
  for (int m=0;m<pack->nmb_thispack;++m)
  for (int k=0;k<ind.nx3+2*ind.ng;++k) for (int j=0;j<ind.nx2+2*ind.ng;++j)
  for (int i=0;i<ind.nx1+2*ind.ng;++i) for (int c=0;c<12;++c) {
    const int v=c<6?adm::ADM::I_ADM_GXX+c:adm::ADM::I_ADM_KXX+c-6;
    if (!std::isfinite(roundtrip(m,v,k,j,i))) throw std::runtime_error("Nonfinite ADM/Z4c round-trip field");
    error=std::max(error,std::abs(roundtrip(m,v,k,j,i)-initial(m,v,k,j,i))/
                              std::max(Real(1),std::abs(initial(m,v,k,j,i))));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (!std::isfinite(error) || error>1e-11) throw std::runtime_error("HiSpID ADM/Z4c round trip failed");
  if (global_variable::my_rank==0) {
    std::cout << "HiSpID import source=" << checkpoint.library_sha << " acceptance=" << checkpoint.acceptance
              << " consumer=" << HISPID_LIBRARY_SHA256
              << " ADM/Z4c relative error=" << std::setprecision(17) << error << std::endl;
  }
  if (pin->GetOrAddBoolean("problem","hispid_mesh_constraints",false)) {
    switch (ind.ng) {
      case 2: pack->pz4c->ADMConstraints<2>(pack);break;
      case 3: pack->pz4c->ADMConstraints<3>(pack);break;
      case 4: pack->pz4c->ADMConstraints<4>(pack);break;
    }
    auto con=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pz4c->u_con);
    const double min_radius=pin->GetOrAddReal("problem","hispid_mesh_constraint_min_radius",0.0);
    if (!std::isfinite(min_radius) || min_radius<0) throw std::runtime_error("Invalid mesh constraint exclusion radius");
    // Volume-weighted RMS and maxima in two explicit regions: g=1, g<1.
    // A stencil-safe outer mask is a third region, excluding each active
    // puncture by max(min_radius,g_max+derivative-stencil halfwidth).
    // Set min_radius above all grid guards for a common refinement region.
    double sums[9]={},maxima[6]={};
    for (int m=0;m<pack->nmb_thispack;++m) {
      auto size=sizes.h_view(m);const double volume=size.dx1*size.dx2*size.dx3;
      const double guard=ind.ng*std::sqrt(size.dx1*size.dx1+size.dx2*size.dx2+size.dx3*size.dx3);
      for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j) for (int i=ind.is;i<=ind.ie;++i) {
        const double H=con(m,z4c::Z4c::I_CON_H,k,j,i),M2=con(m,z4c::Z4c::I_CON_M,k,j,i);
        if (!std::isfinite(H) || !std::isfinite(M2) || M2<0) throw std::runtime_error("Nonfinite mesh constraints");
        const bool vacuum=attenuation[((static_cast<std::size_t>(m)*nz+k)*ny+j)*nx+i]==1.0;
        double pos[3]={CellCenterX(i-ind.is,ind.nx1,size.x1min,size.x1max),
                       CellCenterX(j-ind.js,ind.nx2,size.x2min,size.x2max),
                       CellCenterX(k-ind.ks,ind.nx3,size.x3min,size.x3max)};
        bool outer=vacuum;
        for (int h=0;h<2;++h) if (checkpoint.config.hole[h].mass>0) {
          const auto hole=checkpoint.config.hole[h];double r2=0;
          for (int d=0;d<3;++d) r2+=(pos[d]-hole.center[d])*(pos[d]-hole.center[d]);
          outer &= std::sqrt(r2)>std::max(min_radius,checkpoint.config.inner_max[h]+guard);
        }
        for (int region=0;region<3;++region) if ((region==0&&vacuum)||(region==1&&!vacuum)||(region==2&&outer)) {
          sums[3*region]+=volume;sums[3*region+1]+=volume*H*H;sums[3*region+2]+=volume*M2;
          maxima[2*region]=std::max(maxima[2*region],std::abs(H));maxima[2*region+1]=std::max(maxima[2*region+1],std::sqrt(M2));
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,sums,9,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,maxima,6,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank==0) {
      auto basename=pin->GetString("job","basename");std::ofstream out(basename+".hispid_mesh_constraints.json");
      out<<std::setprecision(17)<<"{\"source_library_sha256\":\""<<checkpoint.library_sha
         <<"\",\"consumer_library_sha256\":\""<<HISPID_LIBRARY_SHA256
         <<"\",\"rms_measure\":\"coordinate_cell_volume\",\"fd_stencil\":"<<pack->pz4c->opt.fd_stencil
         <<",\"roundtrip_relative_error\":"<<error<<",\"minimum_radius\":"<<min_radius<<",\"regions\":[";
      for (int region=0;region<3;++region) {
        if (region) out<<',';
        out<<"{\"region\":\""<<(region==0?"g_equals_one":region==1?"g_less_than_one":"stencil_safe_outer")
           <<"\",\"volume\":"<<sums[3*region];
        if (sums[3*region]>0) out<<",\"H_rms\":"<<std::sqrt(sums[3*region+1]/sums[3*region])
                              <<",\"M_rms\":"<<std::sqrt(sums[3*region+2]/sums[3*region])
                              <<",\"H_max\":"<<maxima[2*region]<<",\"M_max\":"<<maxima[2*region+1];
        out<<'}';
      }
      out<<"]}\n";if (!out) throw std::runtime_error("Cannot write mesh constraint diagnostics");
    }
  }
  if (!pin->GetOrAddBoolean("problem","hispid_initial_horizons",false)) return;
  if (pack->pz4c->pfastflow.empty()) throw std::runtime_error("Initial horizon check requires a configured FastFlow surface");
  std::vector<int> active;
  for (int a=0;a<2;++a) if (checkpoint.config.hole[a].mass>0) active.push_back(a);
  const bool common=pin->GetOrAddBoolean("problem","hispid_common_horizon",false);
  if (common && active.size()!=2)
    throw std::runtime_error("Initial common horizon search requires two active holes");
  if (common && pack->pz4c->pfastflow.size()!=1)
    throw std::runtime_error("Initial common horizon search requires exactly one configured finder");
  if (!common && pack->pz4c->pfastflow.size()!=active.size())
    throw std::runtime_error("Initial component horizon verification needs one finder for each active hole");
  const bool direct=pin->GetOrAddBoolean("problem","hispid_direct_horizon_geometry",true);
  if (!direct) {
    for (auto &finder:pack->pz4c->pfastflow) switch (ind.ng) {
      case 2: finder->MetricDerivatives<2>(0.0);break;
      case 3: finder->MetricDerivatives<3>(0.0);break;
      case 4: finder->MetricDerivatives<4>(0.0);break;
    }
  }
  const bool seed_guess=pin->GetOrAddBoolean("problem","hispid_seed_horizon_guess",true);
  const Real scale=pin->GetOrAddReal("problem","hispid_horizon_guess_scale",1.05);
  if (!std::isfinite(scale) || scale<=0) throw std::runtime_error("Invalid horizon guess scale");
  int h=0;
  for (auto &owned:pack->pz4c->pfastflow) {
    auto &finder=*owned;ResetSource reset{finder};
    if (!common && h>=static_cast<int>(active.size())) throw std::runtime_error("Need one configured active hole per horizon");
    const auto hole=checkpoint.config.hole[active[h]];
    if (!(finder.start_time<=0 && finder.stop_time>=0)) throw std::runtime_error("Initial finder time window must include time zero");
    if (finder.lmax<1 || finder.ntheta<=finder.lmax)
      throw std::runtime_error("Initial finder needs lmax>=1 and ntheta>lmax");
    if (!common && !pin->GetOrAddBoolean("problem","hispid_preserve_horizon_centers",false))
      for (int d=0;d<3;++d) finder.center[d]=hole.center[d];
    finder.require_complete_surface=true;
    if (finder.expansion_rms_tol<=0)
      finder.expansion_rms_tol=pin->GetOrAddReal("problem","hispid_expansion_rms_tol",1e-6);
    if (!std::isfinite(finder.expansion_rms_tol) || finder.expansion_rms_tol<=0)
      throw std::runtime_error("Initial horizon checks require a positive expansion RMS tolerance");
    if (direct) finder.geometry_source=[&](const Real *x,Real *g,Real *K,Real *dg) {
      const auto domain=pack->pmesh->mesh_size;
      if (x[0]<domain.x1min || x[0]>domain.x1max || x[1]<domain.x2min || x[1]>domain.x2max ||
          x[2]<domain.x3min || x[2]>domain.x3max) throw std::runtime_error("Horizon point outside mesh domain");
      double pos[3]={x[0],x[1],x[2]},gradient[27];HiSpID_Point p;
      if (flat) Flat(p,gradient);
      else if (HiSpID_sample_with_derivatives(data.get(),1,pos,&p,gradient))
        throw std::runtime_error(HiSpID_last_error());
      PositiveMetric(p.gamma);
      for (int c=0;c<6;++c) { g[c]=p.gamma[full[c]];K[c]=p.Kij[full[c]]; }
      for (int d=0;d<3;++d) for (int c=0;c<6;++c) dg[6*d+c]=gradient[9*d+full[c]];
    };
    // A common search uses the configured center/radius. A single-hole seed
    // ellipsoid is not a common-horizon initial guess.
    if (seed_guess && !common && !flat) {
      const double s2=hole.spin[0]*hole.spin[0]+hole.spin[1]*hole.spin[1]+hole.spin[2]*hole.spin[2];
      const double rh=.5*hole.mass*std::sqrt(1-s2/std::pow(hole.mass,4));
      const double v2=hole.velocity[0]*hole.velocity[0]+hole.velocity[1]*hole.velocity[1]+hole.velocity[2]*hole.velocity[2];
      finder.initial_shape=[=](Real th,Real ph) {
        double vn=hole.velocity[0]*std::sin(th)*std::cos(ph)+hole.velocity[1]*std::sin(th)*std::sin(ph)+hole.velocity[2]*std::cos(th);
        return scale*rh/std::sqrt(1+vn*vn/(1-v2));
      };
    }
    finder.Find(0,0.0);finder.Write(0,0.0);
    if (global_variable::my_rank==0)
      std::cout << std::setprecision(17) << "HiSpID horizon " << h << " geometry=" << (direct?"native":"mesh")
                << " found=" << finder.ah_found << " area=" << finder.Area()
                << " expansion_rms=" << finder.ExpansionRMS() << " min_radius=" << finder.rr_min
                << " attempt_area=" << finder.last_area << " attempt_expansion_rms=" << finder.last_expansion_rms
                << " center_x=" << finder.center[0] << " center_y=" << finder.center[1]
                << " center_z=" << finder.center[2] << " kind=" << (common?"common":"component") << std::endl;
    if (!finder.ah_found || !std::isfinite(finder.Area()) || finder.Area()<=0 ||
        !std::isfinite(finder.ExpansionRMS()) || finder.ExpansionRMS()>finder.expansion_rms_tol ||
        !std::isfinite(finder.rr_min) || finder.rr_min<=0)
      throw std::runtime_error("Initial HiSpID horizon search failed its geometry/expansion checks");
    ++h;
  }
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin,const bool restart) {
  user_ref_func=Refine;if (restart) return;
  auto *pack=pmy_mesh_->pmb_pack;
  try {
    if (!pack->pz4c || !pmy_mesh_->three_d) throw std::runtime_error("HiSpID requires 3D and <z4c>");
    Initialize(pack,pin);
  } catch (const std::exception &e) {
    std::cerr << "HiSpID pgen: " << e.what() << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }
}
