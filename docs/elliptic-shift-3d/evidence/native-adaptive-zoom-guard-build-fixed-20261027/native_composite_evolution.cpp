// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "z4c/hyperboloidal/native_composite_evolution.hpp"
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "z4c/hyperboloidal/composite_regrid.hpp"
#include "z4c/hyperboloidal/native_composite_fields.hpp"
#include "z4c/hyperboloidal/native_core_map.hpp"
#include "z4c/hyperboloidal/native_composite_stage.hpp"
#include "z4c/hyperboloidal/athenak_bridge.hpp"
namespace z4c {namespace hyperboloidal {
namespace {
using State=NativeCompositeStageState;
using V=State::Vector;
struct Buffer {
 std::vector<char> bytes;size_t cursor=0;
 template<class T>void Put(const T &v){static_assert(std::is_trivially_copyable<T>::value,"POD checkpoint");const char *p=reinterpret_cast<const char*>(&v);bytes.insert(bytes.end(),p,p+sizeof(T));}
 template<class T>T Get(){if(cursor+sizeof(T)>bytes.size())throw std::runtime_error("truncated native composite checkpoint");T v;std::memcpy(&v,bytes.data()+cursor,sizeof(T));cursor+=sizeof(T);return v;}
 void Vector(const V &v){Put(uint64_t(v.size()));for(Real x:v)Put(x);}
 V Vector(){const auto size=Get<uint64_t>();if(size>2200000)throw std::runtime_error("native composite checkpoint vector bound");V v(size);for(auto &x:v)x=Get<Real>();return v;}
 void String(const std::string &s){Put(uint64_t(s.size()));bytes.insert(bytes.end(),s.begin(),s.end());}
 std::string String(){auto n=Get<uint64_t>();if(n>4096||cursor+n>bytes.size())throw std::runtime_error("native composite checkpoint string bound");std::string s(bytes.data()+cursor,n);cursor+=n;return s;}
 void Values(const State::Values &v){Vector(v.volume);Vector(v.scri);Vector(v.endpoints);}
 State::Values Values(){return {Vector(),Vector(),Vector()};}
 uint64_t Checksum(size_t end) const{uint64_t h=1469598103934665603ULL;for(size_t i=0;i<end;++i){h^=static_cast<unsigned char>(bytes[i]);h*=1099511628211ULL;}return h;}
};
std::string InitialReceipt(const std::string &path){
 std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("missing native wormhole initial data file");
 uint64_t hash=1469598103934665603ULL,bytes=0;char c;
 while(file.get(c)){hash^=static_cast<unsigned char>(c);hash*=1099511628211ULL;++bytes;if(bytes>100000000)throw std::runtime_error("native wormhole initial receipt byte bound");}
 if(!file.eof())throw std::runtime_error("native wormhole initial receipt read failed");
 std::ostringstream out;out<<"fnv1a64:"<<std::hex<<hash<<std::dec<<":"<<bytes;return out.str();
}
}
struct NativeCompositeEvolution::Impl {
 Z4c &z;MeshBlockPack *pack;ParameterInput *parameters;bool adaptive_native=false;int validation_refine=-1,validation_coarsen=-1;std::unique_ptr<CompositeTensorEvolution> engine;
 std::unique_ptr<NativeCompositeFields> fields;std::unique_ptr<State> stage;
 std::vector<NativeCompositeFields::Block> mapping;
 std::unique_ptr<NativeCartesianCoreMap> core_map;std::vector<int> composite_to_native;int core_bulk=1;bool static_multilevel=false;size_t hierarchy_terminal_cells=0;
 State::Epoch epoch{};State::Values rate;NativeCompositeRK bound_rk{};bool has_bound_rk=false,initialized=false,mesh_receipt_written=false;
 V expected_native,expected_read,expected_beta;
 Real dt_ceiling=1e-4,half=.25,initial_amplitude=0;bool initial_angular=false;int output_every=1;V radii;std::string prefix,checkpoint_tag;
 bool wormhole=false;int positive_storage=0;std::string initial_file,initial_receipt;
 inline static const std::string wormhole_slice="isotropic-Schwarzschild-M1_3-hyperboloidal-outer-v1";
 inline static const std::string wormhole_gauge="Minkowski-reference-physical1log-restoring1-nu1-referenceflux-tau.02-kappa.005-taper.24-clockA-v1";
 struct Pending {
  std::unique_ptr<CompositeTensorEvolution> engine;
  std::unique_ptr<NativeCompositeFields> fields;
  std::unique_ptr<NativeCartesianCoreMap> map;
  std::unique_ptr<State> stage;
  std::vector<NativeCompositeFields::Block> mapping;
  std::vector<int> inverse;
  std::vector<NativeCartesianBlock> records;
  size_t copied=0,interpolated=0,terminal=0;
 };
 std::unique_ptr<Pending> pending;
 uint64_t begins=0,rhs_calls=0,updates=0,last_report=std::numeric_limits<uint64_t>::max();
 Impl(Z4c &native,MeshBlockPack *p,ParameterInput *pin):z(native),pack(p),parameters(pin){
#if defined(KOKKOS_ENABLE_SERIAL)
  if(!std::is_same<DevExeSpace,Kokkos::Serial>::value)throw std::invalid_argument("experimental composite requires Kokkos Serial");
#else
  throw std::invalid_argument("experimental composite requires Kokkos Serial");
#endif
  const auto &ind=p->pmesh->mb_indcs;const auto integrator=pin->GetString("time","integrator");
  wormhole=pin->GetString("problem","pgen_name")=="z4c_composite_wormhole";
  static_multilevel=p->pmesh->multilevel;
  adaptive_native=p->pmesh->adaptive&&pin->GetOrAddBoolean("z4c","composite_adaptive_experimental",false)&&!wormhole;
  if(adaptive_native)for(const auto &block:pin->block)
   if(block.block_name.compare(0,13,"amr_criterion")==0&&pin->GetString(block.block_name,"method")=="cyclic_zoom")
    throw std::invalid_argument("native adaptive cyclic zoom is unsupported; criterion precedes late zoom object creation");
  if(adaptive_native){validation_refine=pin->GetOrAddInteger("z4c","composite_validation_refine_cycle",-1);
   validation_coarsen=pin->GetOrAddInteger("z4c","composite_validation_coarsen_cycle",-1);
   if((validation_refine>=0||validation_coarsen>=0)&&!(validation_refine>=1&&validation_coarsen>validation_refine))throw std::invalid_argument("invalid explicit native AMR validation schedule");}
  if(SINGLE_PRECISION_ENABLED||MPI_PARALLEL_ENABLED||global_variable::nranks!=1||(!static_multilevel&&p->nmb_thispack!=1&&p->nmb_thispack!=8)||p->pmesh->nmb_total!=p->nmb_thispack||(p->pmesh->adaptive&&!adaptive_native)||!p->pmesh->three_d
      ||ind.ng!=(static_multilevel?4:3)||ind.nx1!=ind.nx2||ind.nx1!=ind.nx3||ind.nx1<6||ind.nx1>16||z.opt.spatial_order!=4||z.opt.chi_psi_power!=-4||z.opt.floor_chi||z.opt.enable_driftcontrol||z.opt.telegraph_lapse
      ||z.nrad||!z.ptracker.empty()||!z.pfastflow.empty()||!z.phorizon_dump.empty()||p->pmesh->pzoom
      ||pin->GetOrAddBoolean("z4c","hyperboloidal",false)||pin->GetOrAddBoolean("cubed_sphere","enabled",false)
      ||!pin->GetOrAddBoolean("adm","separate_z4c_gauge",false)||pin->GetString("problem","pgen_name")!=(wormhole?"z4c_composite_wormhole":"z4c_composite_minkowski")
      ||pin->GetOrAddReal("problem","mass",0)!=(wormhole?Real(1./3):Real(0))||pin->GetOrAddReal("problem","lapse_pulse",0)!=0||!(integrator=="rk3"||integrator=="rk4")
      ||pin->GetOrAddInteger("cce","num_radii",0)||pin->DoesBlockExist("hydro")||pin->DoesBlockExist("mhd")||pin->DoesBlockExist("radiation")||pin->DoesBlockExist("ion-neutral")||pin->DoesBlockExist("gravity")||pin->DoesBlockExist("particles"))
    throw std::invalid_argument("experimental composite native path: one/eight uniform ng3 or complete static-SMR cubic ng4 Serial double vacuum core, order4, chi^-4, separate ADM gauge, RK3/RK4, dedicated Minkowski or isotropic-M1/3 pgen; MPI/device/wormhole-adaptive/unopted-dynamic-regrid/matter/floors/tracking/Cauchy gauge/restart midstage unsupported");
  half=pin->GetOrAddReal("z4c","composite_half_width",.25);dt_ceiling=pin->GetOrAddReal("z4c","composite_dt",1e-4);
  initial_amplitude=pin->GetOrAddReal("z4c","composite_initial_lapse_amplitude",0);
  initial_angular=pin->GetOrAddBoolean("z4c","composite_initial_angular",false);
  output_every=pin->GetOrAddInteger("z4c","composite_output_every",1);
  if(!std::isfinite(initial_amplitude)||std::abs(initial_amplitude)>1e-3||output_every<1)throw std::invalid_argument("experimental native pulse/output controls");
  prefix=pin->GetOrAddString("z4c","composite_output","native_composite");checkpoint_tag=pin->GetOrAddString("z4c","composite_checkpoint_tag","native-composite-minkowski-v1");
  if(wormhole){
   if(ind.nx1%2)throw std::invalid_argument("native wormhole puncture must not collocate an active point");
   positive_storage=pin->GetOrAddInteger("z4c","composite_positive_storage",2);
   if(positive_storage!=2||pin->GetOrAddReal("z4c","composite_telegraph_width",.24)!=.24)
    throw std::invalid_argument("native selected wormhole requires positive log mode2 and verified taper .24");
   initial_file=pin->GetOrAddString("z4c","composite_initial_file","");
   initial_receipt=pin->GetOrAddString("z4c","composite_initial_receipt","");
  }
  std::stringstream list(pin->GetOrAddString("z4c","composite_radii",".65,1"));std::string word;while(std::getline(list,word,','))radii.push_back(std::stod(word));
  if(radii.empty()||radii.size()>16||radii.back()!=1||!(dt_ceiling>0&&dt_ceiling<=.0005)||!std::isfinite(dt_ceiling))throw std::invalid_argument("experimental composite shell/time controls");
  core_bulk=static_multilevel?p->pmesh->nmb_rootx1:(p->nmb_thispack==1?1:2);
  if(static_multilevel&&(p->pmesh->nmb_rootx2!=core_bulk||p->pmesh->nmb_rootx3!=core_bulk))
    throw std::invalid_argument("native static composite requires cubic root block subdivisions");
  const auto records=NativeRecords();
  if(static_multilevel){
   const auto &box=p->pmesh->mesh_size;
   if(box.x1min!=-half||box.x2min!=-half||box.x3min!=-half||box.x1max!=half||box.x2max!=half||box.x3max!=half)
    throw std::invalid_argument("native static composite physical domain differs from inner cube");
   for(int m=0;m<p->nmb_thispack;++m){const auto &box=p->pmb->mb_size.h_view(m);
    const Real xmax=std::max(std::abs(box.x1min),std::abs(box.x1max))+(ind.ng-.5)*box.dx1;
    const Real ymax=std::max(std::abs(box.x2min),std::abs(box.x2max))+(ind.ng-.5)*box.dx2;
    const Real zmax=std::max(std::abs(box.x3min),std::abs(box.x3max))+(ind.ng-.5)*box.dx3;
    const Real maximum_radius2=xmax*xmax+ymax*ymax+zmax*zmax;
    if(maximum_radius2>=1)throw std::invalid_argument("native static core ghosts reach scri");
   }
  }else{
   const Real ghostmax=half+(ind.ng-.5)*(2*half/(ind.nx1*core_bulk));
   if(std::sqrt(3.)*ghostmax>=1)throw std::invalid_argument("native core ghost ADM samples must remain strictly inside scri");
  }
  auto mesh=static_multilevel?NativeCartesianCoreMap::FromNativeLeaves(half,radii,1,ind.nx1,ind.ng,core_bulk,records):
      cubed_sphere::CompositeMesh(half,radii,1,ind.nx1,ind.ng,core_bulk);
  if(adaptive_native){const int revision=pin->GetOrAddInteger("z4c","composite_mesh_revision",0);if(revision<0)throw std::invalid_argument("invalid native mesh revision");mesh.revision=revision;}
  if(static_multilevel){const auto hierarchy=multigrid::CompositeCoarseningPlan::Build(mesh);
   if(!hierarchy.dense_supported)throw std::invalid_argument("native static composite hierarchy: "+hierarchy.unsupported_reason);
   hierarchy_terminal_cells=hierarchy.TerminalCells();}
  core_map=std::make_unique<NativeCartesianCoreMap>(mesh,NativeRecords(),ind.nx1,ind.ng);
  composite_to_native.assign(mesh.blocks.size(),-1);
  for(const auto &match:core_map->matches){mapping.push_back({match.native,match.composite});composite_to_native[match.composite]=match.native;}
  if(mesh.blocks.size()*size_t(ind.nx1)*ind.nx1*ind.nx1>50000)throw std::invalid_argument("experimental native composite CPU cell bound");
  if(wormhole){
   engine=std::make_unique<CompositeTensorEvolution>(std::move(mesh),true,pin->GetOrAddReal("z4c","composite_tolerance",1e-11),4,true,true,0,6,true,2,.24,true,true,true);
   fields=std::make_unique<NativeCompositeFields>(engine->shift.grid,true,0,-4,2);
  }else{
  engine=std::make_unique<CompositeTensorEvolution>(std::move(mesh),true,pin->GetOrAddReal("z4c","composite_tolerance",1e-11),4,true,false,0,6,true,0,.24,false,true,static_multilevel||core_bulk>1);
  fields=std::make_unique<NativeCompositeFields>(engine->shift.grid,false,0,-4);
  }
 }
 void WriteMeshReceipt()const{
  const auto records=NativeRecords();std::ofstream out(prefix+".mesh.dat");
  out<<std::setprecision(17)<<"NATIVE_STATIC_COMPOSITE_MESH_V1 "<<engine->shift.grid.n<<' '<<engine->shift.grid.ng<<' '<<half<<' '<<core_bulk<<' '<<records.size()<<' '<<radii.size()<<'\n';
  for(Real radius:radii)out<<radius<<' ';out<<'\n';
  for(const auto &b:records){out<<b.native<<' '<<b.refinement;for(int a=0;a<3;++a)out<<' '<<b.lo[a]<<' '<<b.hi[a];out<<'\n';}
  out<<hierarchy_terminal_cells<<'\n';if(!out)throw std::runtime_error("native static mesh receipt write failed");
  std::ofstream points(prefix+".coordinates.csv");points<<std::setprecision(17)<<"x,y,z,r\n";
  for(const auto &p:engine->shift.grid.geometry){const auto &x=p.x;points<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<'\n';}
  if(!points)throw std::runtime_error("native static coordinate receipt write failed");
 }
 std::vector<NativeCartesianBlock> NativeRecords()const{
  std::vector<NativeCartesianBlock> records;
  for(int m=0;m<pack->nmb_thispack;++m){const auto box=pack->pmb->mb_size.h_view(m);
   const int relative=pack->pmb->mb_lev.h_view(m)-pack->pmesh->root_level;
   if(relative<0||relative>20||(!static_multilevel&&relative!=0))throw std::invalid_argument("native composite relative refinement level unsupported");
   records.push_back({m,relative,{box.x1min,box.x2min,box.x3min},{box.x1max,box.x2max,box.x3max}});}
  return records;
 }
 void ValidateCore()const{core_map->ValidateCurrent(engine->shift.grid.mesh,NativeRecords());}
 std::array<Real,3> CoreCoordinate(int native,int i,int j,int k)const{
  const int n=engine->shift.grid.n,ng=engine->shift.grid.ng;
  if(!static_multilevel&&core_bulk==1){const Real h=2*half/n;return {-half+(i-ng+.5)*h,-half+(j-ng+.5)*h,-half+(k-ng+.5)*h};}
  const auto box=pack->pmb->mb_size.h_view(native);
  return {box.x1min+(i-ng+.5)*box.dx1,box.x2min+(j-ng+.5)*box.dx2,box.x3min+(k-ng+.5)*box.dx3};
 }
 std::string SidecarMagic()const{
  if(adaptive_native)return "ATHENAK_NATIVE_COMPOSITE_ADAPTIVE_REGULAR_V1";
  if(static_multilevel)return wormhole?"ATHENAK_NATIVE_COMPOSITE_WORMHOLE_STATIC_SMR_V1":"ATHENAK_NATIVE_COMPOSITE_STATIC_SMR_V1";
  if(core_bulk>1)return wormhole?"ATHENAK_NATIVE_COMPOSITE_WORMHOLE_STATIC_CORE_V1":"ATHENAK_NATIVE_COMPOSITE_STATIC_CORE_V1";
  return wormhole?"ATHENAK_NATIVE_COMPOSITE_WORMHOLE_V1":"ATHENAK_NATIVE_COMPOSITE_V1";
 }
 void WriteTopology(Buffer &b)const{
  if(!static_multilevel&&core_bulk==1)return;ValidateCore();b.Put(core_bulk);b.Put(pack->nmb_thispack);b.Put(uint64_t(mapping.size()));
  const auto records=NativeRecords();for(size_t i=0;i<mapping.size();++i){const auto &match=mapping[i];const auto &record=records[match.native];
   b.Put(match.native);b.Put(match.composite);b.Put(record.refinement);for(int d=0;d<3;++d){b.Put(record.lo[d]);b.Put(record.hi[d]);}}
 }
 void ReadTopology(Buffer &b)const{
  if(!static_multilevel&&core_bulk==1)return;ValidateCore();if(b.Get<int>()!=core_bulk||b.Get<int>()!=pack->nmb_thispack||b.Get<uint64_t>()!=mapping.size())
   throw std::runtime_error("native composite restart core topology/count mismatch");
  const auto records=NativeRecords();for(const auto &match:mapping){const auto &record=records[match.native];
   if(b.Get<int>()!=match.native||b.Get<int>()!=match.composite||b.Get<int>()!=record.refinement)throw std::runtime_error("native composite restart core pack map mismatch");
   for(int d=0;d<3;++d)if(b.Get<Real>()!=record.lo[d]||b.Get<Real>()!=record.hi[d])throw std::runtime_error("native composite restart core bounds mismatch");}
 }
 State::Metadata Metadata(int planned_count=-1,unsigned long planned_revision=std::numeric_limits<unsigned long>::max())const{
  State::Metadata m;if(wormhole){m.chart="wormhole-positive-log22-v1";m.chart_parameters={Real(1./3),2,0,.24,1,1,.02,.005,initial_amplitude,initial_angular?Real(1):Real(0)};}
  else{m.chart="regular-conformal-deviation22";m.chart_parameters={initial_amplitude,initial_angular?Real(1):Real(0)};}
  const int count=planned_count<0?pack->nmb_thispack:planned_count;
  if(adaptive_native){m.chart+="-adaptive-core";m.chart_parameters.insert(m.chart_parameters.end(),{Real(core_bulk),Real(count),Real(validation_refine),Real(validation_coarsen)});m.mesh_revision=planned_revision==std::numeric_limits<unsigned long>::max()?engine->shift.grid.mesh.revision:planned_revision;}
  else if(static_multilevel){m.chart+="-static-smr";m.chart_parameters.push_back(core_bulk);m.chart_parameters.push_back(pack->nmb_thispack);}
  else if(core_bulk>1){m.chart+="-static-core";m.chart_parameters.push_back(core_bulk);m.chart_parameters.push_back(pack->nmb_thispack);}return m;
 }
 void ValidatePreparedMirror(const CompositeTensorEvolution &e,const std::vector<NativeCartesianBlock>&records)const{
  const auto &g=e.shift.grid;const int n=g.n,ng=g.ng;
  for(const auto &b:records)for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){
   const int ijk[3]={i,j,k};Real x[3];for(int a=0;a<3;++a)x[a]=b.lo[a]+(ijk[a]-ng+.5)*((b.hi[a]-b.lo[a])/n);
   const bool active=i>=ng&&i<ng+n&&j>=ng&&j<ng+n&&k>=ng&&k<ng+n;
   V q(22,0);Real beta[3]={-x[0],-x[1],-x[2]};const Real *reference=x;
   if(active){const size_t p=((size_t(b.native)*n+k-ng)*n+j-ng)*n+i-ng;reference=g.geometry[p].x;
    std::copy_n(e.state.begin()+22*p,22,q.begin());for(int a=0;a<3;++a)beta[a]=e.beta_delta[3*p+a]-reference[a];
   }else for(const auto &entry:g.Interpolate(x,6)){
    const int block=entry.point/(g.padded*g.padded*g.padded),ii=entry.point%g.padded-ng,jj=entry.point/g.padded%g.padded-ng,kk=entry.point/(g.padded*g.padded)%g.padded-ng;
    const size_t p=((size_t(block)*n+kk)*n+jj)*n+ii;for(int f=0;f<22;++f)q[f]+=entry.weight*e.state[22*p+f];for(int a=0;a<3;++a)beta[a]+=entry.weight*e.beta_delta[3*p+a];}
   for(Real value:q)if(!std::isfinite(value))throw std::runtime_error("native regrid PREPARE nonfinite planned physical mirror");
   Z4cJet<Real> u{};u.chi.value=q[0]+1;u.alpha.value=q[18]+(1+reference[0]*reference[0]+reference[1]*reference[1]+reference[2]*reference[2])/2;u.trace.value=q[13]-3;u.theta.value=q[14];
   for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],c=CompositeTensorEvolution::tensor_j[t];u.metric.g[a][c]=u.metric.g[c][a]=q[1+t]+(a==c?1:0);u.a.k[a][c]=u.a.k[c][a]=q[7+t];}
   for(int a=0;a<3;++a)u.beta.value[a]=beta[a];const Real omega=(1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;
   if(!ToPhysicalADM(u,omega).valid)throw std::runtime_error("native regrid PREPARE invalid planned ADM active/ghost");
  }
 }
 void PrepareRegrid(const LogicalLocation *leaves,int count){
  if(!adaptive_native||wormhole||!initialized||!stage||stage->Active()||stage->StageOpen()||pending||!has_bound_rk)
   throw std::logic_error("native adaptive transaction requires opted regular accepted state");
  Verify();if(count<1||count>pack->pmesh->nmb_maxperrank)throw std::invalid_argument("native adaptive planned leaf capacity");
  auto next=std::make_unique<Pending>();const auto &domain=pack->pmesh->mesh_size;
  const Real lower[3]={domain.x1min,domain.x2min,domain.x3min},upper[3]={domain.x1max,domain.x2max,domain.x3max};
  for(int m=0;m<count;++m){const int relative=leaves[m].level-pack->pmesh->root_level;
   if(relative<0||relative>20)throw std::invalid_argument("native adaptive planned level");
   const int subdivisions=core_bulk<<relative,index[3]={leaves[m].lx1,leaves[m].lx2,leaves[m].lx3};NativeCartesianBlock b;b.native=m;b.refinement=relative;
   for(int a=0;a<3;++a){if(index[a]<0||index[a]>=subdivisions)throw std::invalid_argument("native adaptive planned logical index");
    b.lo[a]=index[a]==0?lower[a]:LeftEdgeX(index[a],subdivisions,lower[a],upper[a]);
    b.hi[a]=index[a]+1==subdivisions?upper[a]:LeftEdgeX(index[a]+1,subdivisions,lower[a],upper[a]);
    const Real end=std::max(std::abs(b.lo[a]),std::abs(b.hi[a]))+(engine->shift.grid.ng-.5)*(b.hi[a]-b.lo[a])/engine->shift.grid.n;
    if(end>=1/std::sqrt(Real(3)))throw std::invalid_argument("native adaptive planned ghost bound");}
   next->records.push_back(b);
  }
  auto mesh=NativeCartesianCoreMap::FromNativeLeaves(half,radii,1,engine->shift.grid.n,engine->shift.grid.ng,core_bulk,next->records);
  if(engine->shift.grid.mesh.revision>=unsigned(std::numeric_limits<int>::max()))throw std::overflow_error("native adaptive mesh revision");
  mesh.revision=engine->shift.grid.mesh.revision+1;const auto hierarchy=multigrid::CompositeCoarseningPlan::Build(mesh);
  if(!hierarchy.dense_supported)throw std::invalid_argument("native adaptive coarse cap: "+hierarchy.unsupported_reason);next->terminal=hierarchy.TerminalCells();
  auto rebuilt=RebuildCompositeAtAcceptedState(*engine,std::move(mesh));next->copied=rebuilt.copied_cells;next->interpolated=rebuilt.interpolated_cells;next->engine=std::move(rebuilt.engine);
  // This globally coupled accepted solve is still PREPARE: failure cannot
  // mutate native Views, old coefficients or old adapters.
  next->engine->RHS(next->engine->state,0);++rhs_calls;
  next->engine->max_shift_residual=std::max(next->engine->max_shift_residual,engine->max_shift_residual);
  next->engine->max_shift_point_residual=std::max(next->engine->max_shift_point_residual,engine->max_shift_point_residual);
  if(engine->scri->values!=next->engine->scri->values)throw std::runtime_error("native adaptive transfer changed scri coefficients");
  const size_t per=size_t(engine->shift.grid.n)*engine->shift.grid.n*engine->shift.grid.n;
  for(size_t b=0;b<next->engine->shift.grid.mesh.blocks.size();++b)if(next->engine->shift.grid.mesh.blocks[b].face>=0){int old=-1;
   for(size_t a=0;a<engine->shift.grid.mesh.blocks.size();++a)if(composite_regrid_detail::Same(next->engine->shift.grid.mesh.blocks[b],engine->shift.grid.mesh.blocks[a]))old=a;
   if(old<0||!std::equal(next->engine->state.begin()+22*per*b,next->engine->state.begin()+22*per*(b+1),engine->state.begin()+22*per*old))throw std::runtime_error("native adaptive changed shell coefficients");}
  next->map=std::make_unique<NativeCartesianCoreMap>(next->engine->shift.grid.mesh,next->records,engine->shift.grid.n,engine->shift.grid.ng);
  next->inverse.assign(next->engine->shift.grid.mesh.blocks.size(),-1);for(const auto &m:next->map->matches){next->mapping.push_back({m.native,m.composite});next->inverse[m.composite]=m.native;}
  next->fields=std::make_unique<NativeCompositeFields>(next->engine->shift.grid,false,0,-4);
  next->stage=std::make_unique<State>(*stage);
  // Both native RK3/RK4 overwrite scratch registers at BeginStage1. The
  // accepted-only reset is explicit and independently bitwise-tested.
  next->stage->ReplaceAcceptedState({next->engine->state,next->engine->scri->values,{}},next->engine->beta_delta,Metadata(count,next->engine->shift.grid.mesh.revision));
  next->stage->InstallShift(next->stage->CurrentEpoch(),next->engine->beta_delta);
  const auto a=stage->Save(),b=next->stage->Save();if(a.time!=b.time||a.cycle!=b.cycle||a.completed_cycles!=b.completed_cycles||!(a.rk==b.rk)||a.has_rk!=b.has_rk)throw std::logic_error("native adaptive lost accepted time/cycle/RK binding");
  ValidatePreparedMirror(*next->engine,next->records);
  pending=std::move(next);
 }
 void CommitRegrid(){
  if(!pending)throw std::logic_error("native adaptive commit without prepare");
  pending->map->ValidateCurrent(pending->engine->shift.grid.mesh,NativeRecords());
  // Keep old grid owner alive until every adapter referencing it is released.
  auto old_engine=std::move(engine);fields.reset();core_map.reset();
  engine=std::move(pending->engine);fields=std::move(pending->fields);core_map=std::move(pending->map);stage=std::move(pending->stage);
  mapping=std::move(pending->mapping);composite_to_native=std::move(pending->inverse);hierarchy_terminal_cells=pending->terminal;
  parameters->SetInteger("z4c","composite_mesh_revision",int(engine->shift.grid.mesh.revision));
  Scatter();ADM();
  const auto saved_prefix=prefix;const auto saved_report=last_report;prefix+=".regrid-"+std::to_string(engine->shift.grid.mesh.revision);last_report=std::numeric_limits<uint64_t>::max();
  WriteMeshReceipt();Diagnostics();Save(prefix+".composite");prefix=saved_prefix;last_report=saved_report;
  std::ofstream report(prefix+".regrid.jsonl",std::ios::app);report<<"{\"cycle\":"<<stage->CompletedCycles()<<",\"revision\":"<<engine->shift.grid.mesh.revision<<",\"native_leaves\":"<<pack->nmb_thispack<<",\"cells\":"<<engine->shift.grid.active.size()<<",\"copied_cells\":"<<pending->copied<<",\"interpolated_cells\":"<<pending->interpolated<<",\"coarse_cells\":"<<hierarchy_terminal_cells<<",\"shell_scri_coefficients_bitwise\":true,\"accepted_register_reset\":true,\"source\":\"canonical22_not_native_coarse_u0\"}\n";
  if(!report)throw std::runtime_error("native adaptive event output failed");pending.reset();
 }
 int ValidationFlag(int m)const{
  if(!adaptive_native||validation_refine<0)return 2;
  const int cycle=pack->pmesh->ncycle,relative=pack->pmb->mb_lev.h_view(m)-pack->pmesh->root_level;
  const auto &b=pack->pmb->mb_size.h_view(m);const bool negative=b.x1max<=0&&b.x2max<=0&&b.x3max<=0;
  if(cycle>=validation_coarsen)return negative&&relative>0?-1:0;
  return cycle>=validation_refine&&negative&&relative==0?1:0;
 }
 void WriteRK(const NativeCompositeRK &rk)const{
  std::ofstream file(prefix+".rk.dat");file<<std::setprecision(17)<<"NATIVE_COMPOSITE_RK_V1 "<<rk.integrator<<' '<<rk.stages<<'\n';
  for(const auto *a:{&rk.gam0,&rk.gam1,&rk.beta,&rk.delta}){for(Real x:*a)file<<x<<' ';file<<'\n';}
  if(!file)throw std::runtime_error("native wormhole RK receipt write failed");
 }
 State::Values Values() const{return {engine->state,engine->scri->values,{}};}
 void Load(){engine->state=stage->Coefficients().volume;engine->scri->values=stage->Coefficients().scri;engine->beta_delta=stage->ShiftWarmStart();}
 void SolveAccepted(){Load();engine->RHS(engine->state,0);++rhs_calls;stage->InstallShift(stage->CurrentEpoch(),engine->beta_delta);}
 void Scatter(){ValidateCore();fields->Write(z.u0,mapping,engine->state,engine->beta_delta);auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);const auto &g=engine->shift.grid;const int ng=g.ng,n=g.n;
  for(const auto &block:mapping)for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){if(i>=ng&&i<ng+n&&j>=ng&&j<ng+n&&k>=ng&&k<ng+n)continue;
   const auto location=CoreCoordinate(block.native,i,j,k);const Real *x=location.data();V q(22,0);Real beta[3]={-x[0],-x[1],-x[2]};for(const auto &entry:g.Interpolate(x,6)){
    const int m=entry.point/(g.padded*g.padded*g.padded),ii=entry.point%g.padded-ng,jj=entry.point/g.padded%g.padded-ng,kk=entry.point/(g.padded*g.padded)%g.padded-ng;const size_t p=((m*n+kk)*n+jj)*n+ii;
    for(int f=0;f<22;++f)q[f]+=entry.weight*engine->state[22*p+f];for(int a=0;a<3;++a)beta[a]+=entry.weight*engine->beta_delta[3*p+a];}
   if(wormhole){
    std::vector<std::array<Real,3>> zero(22);std::vector<std::array<std::array<Real,3>,3>> zz(22);
    const auto u=UnpackComposite(x,q,zero,zz,true,0,positive_storage);
    q[0]=u.chi.value;q[13]=u.trace.value;q[14]=u.theta.value;q[18]=u.alpha.value;
    for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],b=CompositeTensorEvolution::tensor_j[t];q[1+t]=u.metric.g[a][b];q[7+t]=u.a.k[a][b];}
   }else{
   q[0]+=1;q[13]-=3;q[18]+=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;for(int t=0;t<6;++t)if(CompositeTensorEvolution::tensor_i[t]==CompositeTensorEvolution::tensor_j[t])q[1+t]+=1;
   }
   for(int f=0;f<22;++f)host(block.native,NativeCompositeFields::scalar_native[f],k,j,i)=q[f];for(int a=0;a<3;++a)host(block.native,Z4c::I_Z4C_BETAX+a,k,j,i)=beta[a];
  }
  Kokkos::deep_copy(z.u0,host);expected_native.assign(host.data(),host.data()+host.size());expected_read=engine->state;expected_beta=engine->beta_delta;fields->Read(z.u0,mapping,expected_read,expected_beta);
 }
 void Verify(){if(!initialized)return;ValidateCore();const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);if(host.size()!=expected_native.size())throw std::runtime_error("native composite mirror shape changed");for(size_t i=0;i<host.size();++i)if(host.data()[i]!=expected_native[i])throw std::runtime_error("external native u0 edit: composite canonical mirror mismatch");
  auto actual=engine->state,beta=engine->beta_delta;fields->Read(z.u0,mapping,actual,beta);const size_t core=size_t(engine->shift.grid.n)*engine->shift.grid.n*engine->shift.grid.n;
  for(const auto &block:mapping)for(size_t p=core*block.composite;p<core*(block.composite+1);++p){
   for(int f=0;f<22;++f)if(actual[22*p+f]!=expected_read[22*p+f])throw std::runtime_error("native field-adapter gather mismatch");
   for(int a=0;a<3;++a)if(beta[3*p+a]!=expected_beta[3*p+a])throw std::runtime_error("native algebraic beta gather mismatch");}
 }
 void ADM(){if(!initialized)return;ValidateCore();auto *a=pack->padm;a->EnsureSeparateGaugeStorage();if(a->gauge_is_shared||!a->separate_z4c_gauge)throw std::runtime_error("native composite physical ADM gauge aliases conformal state");
  auto dst=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),a->u_adm);const auto q=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);const int n=engine->shift.grid.n,ng=engine->shift.grid.ng;
  for(const auto &block:mapping)for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){Z4cJet<Real> u{};u.chi.value=q(block.native,Z4c::I_Z4C_CHI,k,j,i);u.alpha.value=q(block.native,Z4c::I_Z4C_ALPHA,k,j,i);u.trace.value=q(block.native,Z4c::I_Z4C_KHAT,k,j,i);u.theta.value=q(block.native,Z4c::I_Z4C_THETA,k,j,i);
   for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],b=CompositeTensorEvolution::tensor_j[t];u.metric.g[a][b]=u.metric.g[b][a]=q(block.native,Z4c::I_Z4C_GXX+t,k,j,i);u.a.k[a][b]=u.a.k[b][a]=q(block.native,Z4c::I_Z4C_AXX+t,k,j,i);}for(int a=0;a<3;++a)u.beta.value[a]=q(block.native,Z4c::I_Z4C_BETAX+a,k,j,i);
   const auto location=CoreCoordinate(block.native,i,j,k);const Real *x=location.data();const Real omega=(1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;const auto physical=ToPhysicalADM(u,omega);if(!physical.valid)throw std::runtime_error("invalid native composite physical ADM conversion");
   dst(block.native,adm::ADM::I_ADM_PSI4,k,j,i)=physical.psi4;dst(block.native,adm::ADM::I_ADM_ALPHA,k,j,i)=physical.alpha;for(int a=0;a<3;++a)dst(block.native,adm::ADM::I_ADM_BETAX+a,k,j,i)=physical.beta[a];for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],b=CompositeTensorEvolution::tensor_j[t];dst(block.native,adm::ADM::I_ADM_GXX+t,k,j,i)=physical.metric[a][b];dst(block.native,adm::ADM::I_ADM_KXX+t,k,j,i)=physical.curvature[a][b];}
  }Kokkos::deep_copy(a->u_adm,dst);
 }
 void Diagnostics(){if(!initialized||stage->Active()||(adaptive_native&&last_report==stage->CompletedCycles()))return;Verify();auto con=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u_con);Kokkos::deep_copy(con,Real(0));const auto &g=engine->shift.grid;Real h2=0,m2=0,t2=0,z2=0,volume=0,maxh=0,maxm=0,drift=0;
  const bool write_snapshot=stage->CompletedCycles()%output_every==0;std::ostringstream number;number<<std::setw(5)<<std::setfill('0')<<stage->CompletedCycles();std::ofstream out;if(write_snapshot){out.open(prefix+"."+number.str()+".csv");out<<std::setprecision(17)<<"x,y,z,Omega,weight,alpha,chi,H,M,Theta,Z,beta_x,beta_y,beta_z";for(int f=0;f<22;++f)out<<",state_"<<f;if(wormhole)for(int a=0;a<3;++a)out<<",delta_beta_"<<a;out<<'\n';}
  for(size_t p=0;p<g.active.size();++p){const auto &u=engine->jets[p];const auto &x=g.geometry[p].x;const auto o=CompositeOmegaAt(x,u);const auto c=EvolvedConstraints(u,o);if(!c.valid)throw std::runtime_error("invalid native composite full-domain constraints");const Real w=g.volumes[p]*std::sqrt(engine->shift.geometry[p].determinant)/std::pow(u.chi.value,1.5);volume+=w;h2+=w*c.hamiltonian*c.hamiltonian;m2+=w*c.momentum_conformal_norm2;t2+=w*u.theta.value*u.theta.value;z2+=w*c.z4.z_conformal_norm2;maxh=std::max(maxh,std::abs(c.hamiltonian));maxm=std::max(maxm,std::sqrt(c.momentum_conformal_norm2));for(int f=0;f<22;++f)drift=std::max(drift,std::abs(engine->state[22*p+f]));if(write_snapshot){out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<o.omega<<','<<w<<','<<u.alpha.value<<','<<u.chi.value<<','<<c.hamiltonian<<','<<std::sqrt(c.momentum_conformal_norm2)<<','<<u.theta.value<<','<<std::sqrt(c.z4.z_conformal_norm2);for(int a=0;a<3;++a)out<<','<<u.beta.value[a];for(int f=0;f<22;++f){out<<','<<engine->state[22*p+f];drift=std::max(drift,std::abs(engine->state[22*p+f]));}if(wormhole)for(int a=0;a<3;++a)out<<','<<engine->beta_delta[3*p+a];out<<'\n';}
   const int native_block=composite_to_native[p/(size_t(g.n)*g.n*g.n)];if(native_block>=0){const int i=p%g.n+g.ng,j=p/g.n%g.n+g.ng,k=p/(g.n*g.n)%g.n+g.ng;con(native_block,Z4c::I_CON_H,k,j,i)=c.hamiltonian;con(native_block,Z4c::I_CON_M,k,j,i)=c.momentum_conformal_norm2;con(native_block,Z4c::I_CON_Z,k,j,i)=c.z4.z_conformal_norm2;con(native_block,Z4c::I_CON_C,k,j,i)=c.hamiltonian*c.hamiltonian+c.momentum_conformal_norm2+c.z4.z_conformal_norm2+u.theta.value*u.theta.value;for(int a=0;a<3;++a)con(native_block,Z4c::I_CON_MX+a,k,j,i)=c.momentum[a];}
  }Kokkos::deep_copy(z.u_con,con);if(write_snapshot){if(!out)throw std::runtime_error("native composite diagnostic output failed");std::ofstream scri(prefix+"."+number.str()+".scri.csv");scri<<std::setprecision(17)<<"x,y,z";for(int f=0;f<22;++f)scri<<",state_"<<f;for(int a=0;a<3;++a)scri<<",delta_beta_"<<a;scri<<'\n';for(size_t p=0;p<engine->scri->rows.size();++p){const auto &x=engine->scri->rows[p].geometry.x;scri<<x[0]<<','<<x[1]<<','<<x[2];for(int f=0;f<22;++f)scri<<','<<engine->scri->values[22*p+f];for(int a=0;a<3;++a)scri<<','<<engine->scri->beta[3*p+a];scri<<'\n';}}
  Real ghost_clock=0,ghost_null=0,ghost_tangent=0;const auto padded=engine->Padded(engine->state);
  for(size_t p=0;p<engine->scri->rows.size();++p){const auto &x=engine->scri->rows[p].geometry.x;const auto sample=engine->scri->Sample(p,padded,engine->scri->values,22);const auto u=wormhole?UnpackComposite(x,sample.value,sample.d,sample.dd,true,0,positive_storage):UnpackComposite(x,sample.value,sample.d,sample.dd);const auto geometry=Geometry(u.metric);Real norm=0,bn=0,beta[3];
    const Real alpha=wormhole?engine->DecodePositive(18,engine->scri->ReconstructFromGhost(p,padded,18),x):1+engine->scri->ReconstructFromGhost(p,padded,18);for(int i=0;i<3;++i){beta[i]=-x[i]+engine->scri->ReconstructFromGhost(p,engine->beta_padded,i);bn+=beta[i]*x[i];for(int j=0;j<3;++j)norm+=u.chi.value*geometry.inverse[i][j]*x[i]*x[j];}
    ghost_clock=std::max(ghost_clock,std::abs(alpha-1));ghost_null=std::max(ghost_null,std::abs(bn+alpha*std::sqrt(norm)));for(int i=0;i<3;++i)ghost_tangent=std::max(ghost_tangent,std::abs(beta[i]-bn*x[i]));
  }
  if(last_report!=stage->CompletedCycles()){std::ofstream report(prefix+".jsonl",std::ios::app);report<<std::setprecision(17)<<"{\"cycle\":"<<stage->CompletedCycles()<<",\"time\":"<<stage->Time()<<",\"cells\":"<<g.active.size()<<",\"shells\":"<<radii.size()<<",\"native_begin_calls\":"<<begins<<",\"rhs_calls\":"<<rhs_calls<<",\"native_update_calls\":"<<updates<<",\"H_rms\":"<<std::sqrt(h2/volume)<<",\"M_rms\":"<<std::sqrt(m2/volume)<<",\"Theta_rms\":"<<std::sqrt(t2/volume)<<",\"Z_rms\":"<<std::sqrt(z2/volume)<<",\"H_max\":"<<maxh<<",\"M_max\":"<<maxm<<",\"state_deviation_max\":"<<drift<<",\"dt_ceiling\":"<<dt_ceiling<<",\"full_grid_selected_dt\":"<<CFLStep()<<",\"shift_original_rms_max\":"<<engine->max_shift_residual<<",\"shift_point_max\":"<<engine->max_shift_point_residual<<",\"scri_tensor_pole\":"<<engine->boundary_pole<<",\"scri_ghost_clock\":"<<ghost_clock<<",\"scri_ghost_null\":"<<ghost_null<<",\"scri_ghost_tangent\":"<<ghost_tangent<<",\"canonical_global_state\":true,\"native_physical_mirror_verified\":true,\"beta_excluded_from_RK\":true,\"live_Driver_RK\":\""<<(has_bound_rk?bound_rk.integrator:"not_yet_started")<<"\"}\n";last_report=stage->CompletedCycles();}
 }
 // Full global grid selection uses the SAME characteristic speeds as RHS.
 // Accepted-state refresh is not a time advance: dt=0 evaluates coefficients,
 // then this limit selects dt before the native Driver advances any stage.
 Real CFLStep() const {
  if(engine->jets.size()!=engine->shift.grid.active.size())throw std::runtime_error("native composite CFL needs frozen accepted geometry");
  const auto &g=engine->shift.grid;const Real safety=std::min(Real(.4),pack->pmesh->cfl_no);Real limit=dt_ceiling;
  if(!(safety>0)||!std::isfinite(safety))throw std::runtime_error("invalid native composite CFL number");
  for(size_t p=0;p<g.active.size();++p){const auto &u=engine->jets[p];const auto &x=g.geometry[p].x;const auto omega=CompositeOmegaAt(x,u);const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);const auto support=ReferenceTelegraphSupport(radius,.8,engine->telegraph_width);
   if(!support.valid)throw std::runtime_error("invalid native composite CFL support");const Real F=u.alpha.value*u.alpha.value+4*omega.omega*omega.omega*u.alpha.value;Real speed=0;
   for(int A=0;A<3;++A){Real transport=0,norm=0;for(int i=0;i<3;++i){transport+=g.geometry[p].J[A][i]*u.beta.value[i];for(int j=0;j<3;++j)norm+=u.chi.value*engine->shift.geometry[p].inverse[i][j]*g.geometry[p].J[A][i]*g.geometry[p].J[A][j];}speed+=std::abs(transport)+std::sqrt(norm*(F+(engine->telegraph?.25*support.value*support.value:0)));}
   if(!(speed>0)||!std::isfinite(speed))throw std::runtime_error("invalid native composite full-grid characteristic speed");limit=std::min(limit,safety/(g.n*speed));
  }
  const Real penalties=engine->pole_sat+engine->regularity_sat;
  if(penalties>0)for(const auto &row:engine->scri->rows)limit=std::min(limit,safety*row.radial_spacing/penalties);
  if(engine->telegraph)limit=std::min(limit,Real(.004)); // 80% of tau*.25, tau=.02
  if(!(limit>0)||!std::isfinite(limit))throw std::runtime_error("invalid native composite selected timestep");return limit;
 }
 void Save(const std::string &path){if(!initialized||stage->Active())throw std::logic_error("native composite restart only at accepted cycle boundary");Verify();if(std::abs(stage->Time()-pack->pmesh->time)>1e-12)throw std::runtime_error("native/core accepted checkpoint time mismatch");Buffer b;b.String(SidecarMagic());b.String(checkpoint_tag);b.Put(engine->shift.grid.n);b.Put(engine->shift.grid.ng);b.Put(half);b.Vector(radii);WriteTopology(b);if(wormhole){b.String(wormhole_slice);b.String(wormhole_gauge);b.String(initial_receipt);b.Put(positive_storage);b.Put(engine->telegraph_width);b.Put(engine->elliptic_atol);b.Put(dt_ceiling);}const auto s=stage->Save();b.Put(s.version);b.Values(s.values);b.Values(s.registers);b.String(s.metadata.chart);b.Vector(s.metadata.chart_parameters);b.Put(s.metadata.mesh_revision);b.Put(s.metadata.chart_revision);b.Put(s.metadata.puncture_revision);b.String(s.rk.integrator);b.Put(s.rk.stages);for(const auto *a:{&s.rk.gam0,&s.rk.gam1,&s.rk.beta,&s.rk.delta})for(Real x:*a)b.Put(x);b.Vector(s.shift_warm_start);b.Put(s.cycle);b.Put(s.completed_cycles);b.Put(s.geometry);b.Put(s.next_stage);b.Put(s.open_stage);b.Put(s.active);b.Put(s.stage_open);b.Put(s.has_rk);b.Put(s.time);b.Put(s.dt);b.Put(s.abscissa);b.Put(s.register_abscissa);const uint64_t checksum=b.Checksum(b.bytes.size());b.Put(checksum);std::ofstream f(path,std::ios::binary);f.write(b.bytes.data(),b.bytes.size());if(!f)throw std::runtime_error("native composite checkpoint write failed");}
 void Restore(const std::string &path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("missing paired native composite restart sidecar");Buffer b;b.bytes.assign(std::istreambuf_iterator<char>(f),{});if(b.bytes.size()<8||b.bytes.size()>100000000)throw std::runtime_error("native composite restart size");uint64_t checksum;std::memcpy(&checksum,b.bytes.data()+b.bytes.size()-8,8);if(checksum!=b.Checksum(b.bytes.size()-8))throw std::runtime_error("native composite restart checksum mismatch");if(b.String()!=SidecarMagic()||b.String()!=checkpoint_tag||b.Get<int>()!=engine->shift.grid.n||b.Get<int>()!=engine->shift.grid.ng||b.Get<Real>()!=half||b.Vector()!=radii)throw std::runtime_error("native composite restart config/tag mismatch");ReadTopology(b);if(wormhole&&(b.String()!=wormhole_slice||b.String()!=wormhole_gauge||b.String()!=initial_receipt||initial_receipt.empty()||b.Get<int>()!=positive_storage||b.Get<Real>()!=engine->telegraph_width||b.Get<Real>()!=engine->elliptic_atol||b.Get<Real>()!=dt_ceiling))throw std::runtime_error("native wormhole restart slice/gauge/receipt/chart controls mismatch");State::Snapshot s;s.version=b.Get<uint32_t>();s.values=b.Values();s.registers=b.Values();s.metadata.chart=b.String();s.metadata.chart_parameters=b.Vector();s.metadata.mesh_revision=b.Get<uint64_t>();s.metadata.chart_revision=b.Get<uint64_t>();s.metadata.puncture_revision=b.Get<uint64_t>();s.rk.integrator=b.String();s.rk.stages=b.Get<int>();for(auto *a:{&s.rk.gam0,&s.rk.gam1,&s.rk.beta,&s.rk.delta})for(auto &x:*a)x=b.Get<Real>();s.shift_warm_start=b.Vector();s.cycle=b.Get<uint64_t>();s.completed_cycles=b.Get<uint64_t>();s.geometry=b.Get<uint64_t>();s.next_stage=b.Get<int>();s.open_stage=b.Get<int>();s.active=b.Get<bool>();s.stage_open=b.Get<bool>();s.has_rk=b.Get<bool>();s.time=b.Get<Real>();s.dt=b.Get<Real>();s.abscissa=b.Get<Real>();s.register_abscissa=b.Get<Real>();if(s.metadata.chart!=Metadata().chart||s.metadata.chart_parameters!=Metadata().chart_parameters)throw std::runtime_error("native composite restart pulse/chart configuration mismatch");if(b.cursor!=b.bytes.size()-8||s.active||s.stage_open||s.values.volume.size()!=engine->state.size()||s.values.scri.size()!=engine->scri->values.size()||s.time!=pack->pmesh->time||s.completed_cycles!=uint64_t(pack->pmesh->ncycle))throw std::runtime_error("native composite restart accepted-state layout/time mismatch");
  if(s.has_rk){bound_rk=s.rk;has_bound_rk=true;stage=std::make_unique<State>(State::Restore(s,s.rk));}else{if(s.cycle||s.completed_cycles||s.time!=0)throw std::runtime_error("unbound native restart must be initial");stage=std::make_unique<State>(s.values,s.shift_warm_start.size(),s.metadata);stage->InstallShift(stage->CurrentEpoch(),s.shift_warm_start);}Load();
 }
};
NativeCompositeEvolution::NativeCompositeEvolution(Z4c &z,MeshBlockPack *p,ParameterInput *pin):p_(std::make_unique<Impl>(z,p,pin)){}
NativeCompositeEvolution::~NativeCompositeEvolution()=default;
void NativeCompositeEvolution::Initialize(ParameterInput *pin,bool restart){auto &p=*p_;V loaded;if(restart){const auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.z.u0);loaded.assign(h.data(),h.data()+h.size());}if(!restart){p.engine->Initialize(p.initial_amplitude,p.initial_angular);if(p.wormhole){const auto receipt=InitialReceipt(p.initial_file);if(!p.initial_receipt.empty()&&p.initial_receipt!=receipt)throw std::runtime_error("native wormhole supplied initial receipt mismatch");p.engine->InitializeWormhole(p.initial_file,p.initial_amplitude,p.initial_angular);if(InitialReceipt(p.initial_file)!=receipt)throw std::runtime_error("native wormhole initializer changed during read");p.initial_receipt=receipt;pin->SetString("z4c","composite_initial_receipt",receipt);}}auto m=p.Metadata();p.stage=std::make_unique<State>(p.Values(),p.engine->beta_delta.size(),m);if(restart)p.Restore(pin->GetString("z4c","composite_restart_file"));p.SolveAccepted();p.Scatter();if(restart&&loaded!=p.expected_native)throw std::runtime_error("native restart u0 disagrees with paired global composite state");p.initialized=true;p.ADM();}
void NativeCompositeEvolution::BeginStage(Driver *d,int s){auto &p=*p_;p.Verify();const auto rk=NativeCompositeRK::FromDriver(*d);if(p.has_bound_rk&&!(p.bound_rk==rk))throw std::runtime_error("native composite live Driver RK differs from restart");if((p.wormhole||p.static_multilevel)&&!p.has_bound_rk)p.WriteRK(rk);p.bound_rk=rk;p.has_bound_rk=true;if(s==1)p.stage->BeginCycle(rk,p.pack->pmesh->dt,p.pack->pmesh->time);p.epoch=p.stage->BeginStage(s);p.Load();++p.begins;}
void NativeCompositeEvolution::EvaluateRHS(Driver *,int){auto &p=*p_;p.rate.volume=p.engine->RHS(p.engine->state,p.pack->pmesh->dt);p.rate.scri=p.engine->boundary_rhs;p.rate.endpoints.clear();p.stage->InstallShift(p.epoch,p.engine->beta_delta);p.fields->Write(p.z.u_rhs,p.mapping,p.engine->state,p.engine->beta_delta,&p.rate.volume);++p.rhs_calls;}
void NativeCompositeEvolution::AdvanceStage(Driver *,int){auto &p=*p_;p.stage->AdvanceStage(p.epoch,p.rate);p.Load();++p.updates;if(!p.stage->Active())p.SolveAccepted();p.Scatter();}
void NativeCompositeEvolution::PrepareNative(){if(p_->static_multilevel&&!p_->mesh_receipt_written){p_->WriteMeshReceipt();p_->mesh_receipt_written=true;}p_->Verify();p_->Scatter();p_->Diagnostics();}
void NativeCompositeEvolution::ToADM(){p_->ADM();}
void NativeCompositeEvolution::Constraints(){p_->Diagnostics();}
Real NativeCompositeEvolution::TimeStep()const{return p_->CFLStep()/p_->pack->pmesh->cfl_no;}
void NativeCompositeEvolution::WriteRestart(const std::string &path){if(p_->pending)throw std::logic_error("restart during native regrid transaction");p_->Save(path);}
void NativeCompositeEvolution::PrepareRegrid(const LogicalLocation *leaves,int count){p_->PrepareRegrid(leaves,count);}
void NativeCompositeEvolution::CommitRegrid(){p_->CommitRegrid();}
int NativeCompositeEvolution::ValidationRefinementFlag(int native)const{return p_->ValidationFlag(native);}
}} // namespace z4c::hyperboloidal
