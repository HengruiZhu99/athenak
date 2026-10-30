#include <mpi.h>
#include <cstring>
#include <iostream>
#include "z4c/hyperboloidal/hyperbolic_mesh_plans.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"
#include "fixed_composite_mesh.hpp"
namespace h=z4c::hyperboloidal;
using Bundle=h::HyperbolicMeshPlans;
using Buffer=Bundle::Buffer;
using V=std::vector<Real>;
static void Require(bool okay,const char*reason){
  int bad=!okay;MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(bad)throw std::runtime_error(reason);
}
template<class F>static void Reject(F f,const char*reason=""){bool caught=false;try{f();}catch(const std::invalid_argument&e){caught=std::string(e.what()).find(reason)!=std::string::npos;}
  Require(caught,"expected collective mesh-plan rejection");}
static Buffer Upload(const V&v){Buffer b("bundle test upload",v.size());auto host=Kokkos::create_mirror_view(b);
  for(size_t i=0;i<v.size();++i)host(i)=v[i];Kokkos::deep_copy(b,host);return b;}
static V Download(Buffer b){auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),b);V v(b.extent(0));
  for(size_t i=0;i<v.size();++i)v[i]=host(i);return v;}
static bool Bits(const V&a,const V&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0);}
static void Append(V&out,Buffer b){const auto v=Download(b);out.insert(out.end(),v.begin(),v.end());}
// Evaluate actual native operators, with fresh stage fields each time. This
// fixture has no field transfer, time integration or horizon reconstruction.
static V Evaluate(Bundle&bundle,h::MeshPlanEpoch epoch,int stage){
  Buffer global,owned,surface,local_surface;int first=0,local=0,total=0,rank=0;V full;
  bundle.Use(epoch,[&](auto&g){const auto&m=g.Host();const auto&l=g.layout;
    first=l.first;local=l.local;total=l.total;rank=l.rank;full.resize(31*total);V edge(31*l.surface_total);
    auto seed=[&](const Real*x,Real*v){
      for(int f=0;f<31;++f)v[f]=(f==0?1.:0.)+.00001*(stage+1)*(1+f%4)*std::sin((1+.11*f)*x[0]+.7*x[1]-.9*x[2]);
      if(!h::ProjectAlgebraicNativeState(v).valid)throw std::runtime_error("invalid analytic bundle fixture");};
    for(int p=0;p<total;++p)seed(m.grid.geometry[p].x,&full[31*p]);
    for(int p=0;p<l.surface_total;++p)seed(m.scri.rows[p].geometry.x,&edge[31*p]);
    owned=Upload(V(full.begin()+31*first,full.begin()+31*(first+local)));
    global=Buffer("bundle partial global",full.size());surface=Upload(edge);
  });
  bundle.Exchange(epoch,owned,global,true);V out;
  if(stage==0){
    Buffer complete("bundle complete snapshot",full.size());bundle.Gather(epoch,owned,complete);
    Require(Bits(Download(complete),full),"cached snapshot gather differs from full analytic fields");
    Buffer wrong=rank==0?Buffer{}:owned;
    Reject([&]{bundle.Gather(epoch,wrong,complete);});
    Buffer alias=owned;if(rank==0)alias=Kokkos::subview(complete,std::make_pair(size_t(31*first),size_t(31*(first+local))));
    Reject([&]{bundle.Gather(epoch,alias,complete);});
  }
  bundle.Use(epoch,[&](auto&g){
    const auto&l=g.layout;
    const auto projected=g.global_projector->Project(surface);
    if(projected.bad_rows)throw std::runtime_error("bundle projection rejected");
    Kokkos::deep_copy(surface,g.global_projector->LocalSurface());
    g.normal_halo->surface=surface;g.halo->surface=surface;
    const auto nh=g.normal_halo->Fill(global);
    const auto n=g.normal->Evaluate(g.normal_halo->padded,surface);
    if(nh.invalid||n.bad_rows)throw std::runtime_error("bundle normal sampling rejected");
    const auto residual=g.angular->Residual(surface);
    const auto angular=g.angular->Apply(g.angular->LocalResidual(),g.Host().angular_epsilon);
    if(residual.bad_values||angular.bad_values)throw std::runtime_error("bundle angular operator rejected");
    const auto additive=Kokkos::subview(g.angular->LocalRHS(),std::make_pair(size_t(31*l.surface_first),size_t(31*(l.surface_first+l.surface_local))));
    const auto scri=g.boundary->Evaluate(global,surface,.0001,additive,g.normal->LocalPoleSamples());
    if(scri.bad_rows)throw std::runtime_error("bundle scri operator rejected");
    Append(out,g.normal->LocalPoleSamples());Append(out,g.angular->LocalRHS());
    Append(out,g.boundary->LocalRHS());local_surface=g.boundary->LocalSurface();
  });
  bundle.Gather(epoch,local_surface,surface,true);
  bundle.Use(epoch,[&](auto&g){
    const auto halo=g.halo->Fill(global);
    const auto bulk=g.bulk->Evaluate(g.halo->padded,.0001,false,.5,surface,g.halo->positive_ko);
    const auto diagnostic=g.bulk->Diagnostics(g.halo->padded,0,surface);
    if(halo.invalid||bulk.bad_cells||diagnostic.invalid_cells)throw std::runtime_error("bundle native operator rejected");
    Append(out,surface);Append(out,g.bulk->LocalRHS());Append(out,g.reference_lapse);
    for(Real x:{diagnostic.total.volume,diagnostic.total.h2,diagnostic.total.m2,diagnostic.total.theta2,diagnostic.total.z2,
                diagnostic.min_chi,diagnostic.min_lapse,diagnostic.max_determinant_residual,diagnostic.max_tracefree_residual})out.push_back(x);
    for(Real x:out)if(!std::isfinite(x))throw std::runtime_error("nonfinite bundle result");
  });
  return out;
}
static std::vector<int> Cuts(const cubed_sphere::CompositeMesh&mesh,int size,bool repartition=false){
  // Only the descriptor's host difference is needed to determine ownership.
  cubed_sphere::CompositeDifference grid(mesh,4,true);const int n=grid.active.size();std::vector<int> cuts(size+1);
  for(int r=0;r<=size;++r)cuts[r]=n*r/size;
  if(repartition&&size>1)++cuts[1];return cuts;
}
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int rank,size,code=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  try{
    Bundle bundle;Bundle::Options options;h::MeshPlanEpoch epoch{};uint64_t expected_builds=0;int comparisons=0;
    Reject([&]{bundle.Borrow(epoch);});
    auto check=[&](const cubed_sphere::CompositeMesh&mesh,bool repartition=false){
      const auto cuts=Cuts(mesh,size,repartition);
      if(rank==0)std::cout<<"CHECK generation "<<epoch.topology<<","<<epoch.geometry<<","<<epoch.partition<<" blocks="<<mesh.blocks.size()<<" cells="<<mesh.cells<<std::endl;
      bundle.Rebuild(mesh,options,cuts,epoch);++expected_builds;
      // Exercise automatic ownership without constructing a second host grid
      // merely to determine cuts. Explicit repartitioning still uses its cuts.
      Bundle fresh;fresh.Rebuild(mesh,options,repartition?cuts:std::vector<int>{},epoch);
      for(int stage=0;stage<2;++stage){Require(Bits(Evaluate(bundle,epoch,stage),Evaluate(fresh,epoch,stage)),"rebuilt/fresh native fields differ");++comparisons;}
      bundle.Rebuild(mesh,options,cuts,epoch);Require(bundle.Builds()==expected_builds,"unchanged plan was rebuilt");
      if(rank==0)bundle.Invalidate();
      Reject([&]{bundle.Use(epoch,[](auto&){});});
      bundle.Rebuild(mesh,options,cuts,epoch);++expected_builds;
      auto stale=epoch;if(rank==0)++stale.geometry;Reject([&]{bundle.Use(stale,[](auto&){});});
      auto wrong=epoch;++wrong.topology;Reject([&]{bundle.Borrow(wrong);});
      // Changing coordinates without advancing the epoch fails even at identical shape.
      auto changed=mesh;changed.radii[0]+=.001;
      Reject([&]{bundle.Rebuild(changed,options,cuts,epoch);});Reject([&]{bundle.Use(epoch,[](auto&){});});
      Reject([&]{bundle.Borrow(epoch);});
      Reject([&]{bundle.Exchange(epoch,Buffer{},Buffer{});});Reject([&]{bundle.Gather(epoch,Buffer{},Buffer{},true);});
      bundle.Rebuild(mesh,options,cuts,epoch);++expected_builds;
      // Inject a local state-staging rejection after device construction, before MPI-plan construction.
      bundle.Invalidate();
      Reject([&]{bundle.Rebuild(mesh,options,cuts,epoch,[&](const auto&){if(rank==0)throw std::runtime_error("local staging failure");});});
      Reject([&]{bundle.Use(epoch,[](auto&){});});
      bundle.Rebuild(mesh,options,cuts,epoch);++expected_builds;
      Require(Bits(Evaluate(bundle,epoch,2),Evaluate(fresh,epoch,2)),"recovered generation differs");++comparisons;
      auto inconsistent=mesh;if(rank==0)inconsistent.half_width+=.001;
      if(size>1){Reject([&]{bundle.Rebuild(inconsistent,options,cuts,epoch);});
        bundle.Rebuild(mesh,options,cuts,epoch);++expected_builds;}
      Require(bundle.Builds()==expected_builds,"wrong bundle build count");
    };
    auto mesh=hyperbolic_reproduction::FixedMesh(6);check(mesh);
    const auto parent=mesh.blocks[0];mesh.RefineCartesian(0);++epoch.topology;check(mesh);
    mesh.CoarsenCartesian(parent);++epoch.topology;check(mesh);
    mesh.radii[0]=.46;++epoch.geometry;check(mesh);
    ++epoch.partition;check(mesh,true);
    // Optimized production geometry: same count but distinct maps and references.
    auto continuous=hyperbolic_reproduction::ContinuousCoreMesh(12);++epoch.topology;++epoch.geometry;check(continuous);
    // The current continuous derivative policy cannot represent mixed refinement levels.
    const auto oldcuts=Cuts(continuous,size);continuous.RefineCartesian(0);++epoch.topology;
    Reject([&]{bundle.Rebuild(continuous,options,oldcuts,epoch);},"continuous core requires a uniform Cartesian level");Reject([&]{bundle.Use(epoch,[](auto&){});});
    Require(comparisons==18,"missing mesh-generation comparisons");
    if(rank==0)std::cout<<"PASS: full host/device bundle; 18 fresh/rebuilt native comparisons; refine/coarsen, geometry, repartition, unchanged reuse, one-rank stale key, descriptor mismatch, local staging failure/recovery and unsupported continuous-core refinement rejection; ranks="<<size<<" builds="<<bundle.Builds()<<" backend="<<Kokkos::DefaultExecutionSpace::name()<<std::endl;
  }catch(const std::exception&e){std::cerr<<rank<<": "<<e.what()<<std::endl;code=1;}
  MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return code;
}
