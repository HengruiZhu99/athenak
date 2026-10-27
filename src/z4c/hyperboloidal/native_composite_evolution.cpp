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
#include "driver/driver.hpp"
#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "z4c/hyperboloidal/native_composite_fields.hpp"
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
}
struct NativeCompositeEvolution::Impl {
 Z4c &z;MeshBlockPack *pack;std::unique_ptr<CompositeTensorEvolution> engine;
 std::unique_ptr<NativeCompositeFields> fields;std::unique_ptr<State> stage;
 std::vector<NativeCompositeFields::Block> mapping{{0,0}};
 State::Epoch epoch{};State::Values rate;NativeCompositeRK bound_rk{};bool has_bound_rk=false,initialized=false;
 V expected_native,expected_read,expected_beta;
 Real dt_ceiling=1e-4,half=.25,initial_amplitude=0;bool initial_angular=false;int output_every=1;V radii;std::string prefix,checkpoint_tag;
 uint64_t begins=0,rhs_calls=0,updates=0,last_report=std::numeric_limits<uint64_t>::max();
 Impl(Z4c &native,MeshBlockPack *p,ParameterInput *pin):z(native),pack(p){
#if defined(KOKKOS_ENABLE_SERIAL)
  if(!std::is_same<DevExeSpace,Kokkos::Serial>::value)throw std::invalid_argument("experimental composite requires Kokkos Serial");
#else
  throw std::invalid_argument("experimental composite requires Kokkos Serial");
#endif
  const auto &ind=p->pmesh->mb_indcs;const auto integrator=pin->GetString("time","integrator");
  if(SINGLE_PRECISION_ENABLED||MPI_PARALLEL_ENABLED||global_variable::nranks!=1||p->nmb_thispack!=1||p->pmesh->nmb_total!=1||p->pmesh->multilevel||!p->pmesh->three_d
      ||ind.ng!=3||ind.nx1!=ind.nx2||ind.nx1!=ind.nx3||ind.nx1<6||ind.nx1>16||z.opt.spatial_order!=4||z.opt.chi_psi_power!=-4||z.opt.floor_chi||z.opt.enable_driftcontrol||z.opt.telegraph_lapse
      ||z.nrad||!z.ptracker.empty()||!z.pfastflow.empty()||!z.phorizon_dump.empty()||p->pmesh->pzoom
      ||pin->GetOrAddBoolean("z4c","hyperboloidal",false)||pin->GetOrAddBoolean("cubed_sphere","enabled",false)
      ||!pin->GetOrAddBoolean("adm","separate_z4c_gauge",false)||pin->GetString("problem","pgen_name")!="z4c_composite_minkowski"
      ||pin->GetOrAddReal("problem","mass",0)!=0||pin->GetOrAddReal("problem","lapse_pulse",0)!=0||!(integrator=="rk3"||integrator=="rk4")
      ||pin->GetOrAddInteger("cce","num_radii",0)||pin->DoesBlockExist("hydro")||pin->DoesBlockExist("mhd")||pin->DoesBlockExist("radiation")||pin->DoesBlockExist("ion-neutral")||pin->DoesBlockExist("gravity")||pin->DoesBlockExist("particles"))
    throw std::invalid_argument("experimental composite native path: one static cubic Serial double vacuum block, ng3/order4, chi^-4, separate ADM gauge, RK3/RK4, dedicated Minkowski pgen; MPI/device/AMR/matter/floors/tracking/Cauchy gauge/restart midstage unsupported");
  half=pin->GetOrAddReal("z4c","composite_half_width",.25);dt_ceiling=pin->GetOrAddReal("z4c","composite_dt",1e-4);
  initial_amplitude=pin->GetOrAddReal("z4c","composite_initial_lapse_amplitude",0);
  initial_angular=pin->GetOrAddBoolean("z4c","composite_initial_angular",false);
  output_every=pin->GetOrAddInteger("z4c","composite_output_every",1);
  if(!std::isfinite(initial_amplitude)||std::abs(initial_amplitude)>1e-3||output_every<1)throw std::invalid_argument("experimental native pulse/output controls");
  prefix=pin->GetOrAddString("z4c","composite_output","native_composite");checkpoint_tag=pin->GetOrAddString("z4c","composite_checkpoint_tag","native-composite-minkowski-v1");
  std::stringstream list(pin->GetOrAddString("z4c","composite_radii",".65,1"));std::string word;while(std::getline(list,word,','))radii.push_back(std::stod(word));
  if(radii.empty()||radii.size()>16||radii.back()!=1||!(dt_ceiling>0&&dt_ceiling<=.0005)||!std::isfinite(dt_ceiling))throw std::invalid_argument("experimental composite shell/time controls");
  const auto size=p->pmb->mb_size.h_view(0);const Real lo[3]={size.x1min,size.x2min,size.x3min},hi[3]={size.x1max,size.x2max,size.x3max};
  for(int a=0;a<3;++a)if(std::abs(lo[a]+half)>1e-13||std::abs(hi[a]-half)>1e-13)throw std::invalid_argument("native core bounds must exactly match composite Cartesian box");
  const Real ghostmax=half+(ind.ng-.5)*(2*half/ind.nx1);if(std::sqrt(3.)*ghostmax>=1)throw std::invalid_argument("native core ghost ADM samples must remain strictly inside scri");
  auto mesh=cubed_sphere::CompositeMesh(half,radii,1,ind.nx1,ind.ng,1);
  if(mesh.blocks.size()*size_t(ind.nx1)*ind.nx1*ind.nx1>50000)throw std::invalid_argument("experimental native composite CPU cell bound");
  engine=std::make_unique<CompositeTensorEvolution>(std::move(mesh),true,pin->GetOrAddReal("z4c","composite_tolerance",1e-11),4,true,false,0,6,true,0,.24,false,true,false);
  fields=std::make_unique<NativeCompositeFields>(engine->shift.grid,false,0,-4);
 }
 State::Values Values() const{return {engine->state,engine->scri->values,{}};}
 void Load(){engine->state=stage->Coefficients().volume;engine->scri->values=stage->Coefficients().scri;engine->beta_delta=stage->ShiftWarmStart();}
 void SolveAccepted(){Load();engine->RHS(engine->state,0);++rhs_calls;stage->InstallShift(stage->CurrentEpoch(),engine->beta_delta);}
 void Scatter(){fields->Write(z.u0,mapping,engine->state,engine->beta_delta);auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);const auto &g=engine->shift.grid;const int ng=g.ng,n=g.n;const Real h=2*half/n;
  for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){if(i>=ng&&i<ng+n&&j>=ng&&j<ng+n&&k>=ng&&k<ng+n)continue;
   const Real x[3]={-half+(i-ng+.5)*h,-half+(j-ng+.5)*h,-half+(k-ng+.5)*h};V q(22,0);Real beta[3]={-x[0],-x[1],-x[2]};for(const auto &entry:g.Interpolate(x,6)){
    const int m=entry.point/(g.padded*g.padded*g.padded),ii=entry.point%g.padded-ng,jj=entry.point/g.padded%g.padded-ng,kk=entry.point/(g.padded*g.padded)%g.padded-ng;const size_t p=((m*n+kk)*n+jj)*n+ii;
    for(int f=0;f<22;++f)q[f]+=entry.weight*engine->state[22*p+f];for(int a=0;a<3;++a)beta[a]+=entry.weight*engine->beta_delta[3*p+a];}
   q[0]+=1;q[13]-=3;q[18]+=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;for(int t=0;t<6;++t)if(CompositeTensorEvolution::tensor_i[t]==CompositeTensorEvolution::tensor_j[t])q[1+t]+=1;
   for(int f=0;f<22;++f)host(0,NativeCompositeFields::scalar_native[f],k,j,i)=q[f];for(int a=0;a<3;++a)host(0,Z4c::I_Z4C_BETAX+a,k,j,i)=beta[a];
  }
  Kokkos::deep_copy(z.u0,host);expected_native.assign(host.data(),host.data()+host.size());expected_read=engine->state;expected_beta=engine->beta_delta;fields->Read(z.u0,mapping,expected_read,expected_beta);
 }
 void Verify(){if(!initialized)return;const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);if(host.size()!=expected_native.size())throw std::runtime_error("native composite mirror shape changed");for(size_t i=0;i<host.size();++i)if(host.data()[i]!=expected_native[i])throw std::runtime_error("external native u0 edit: composite canonical mirror mismatch");
  auto actual=engine->state,beta=engine->beta_delta;fields->Read(z.u0,mapping,actual,beta);const size_t core=size_t(engine->shift.grid.n)*engine->shift.grid.n*engine->shift.grid.n;
  for(size_t i=0;i<22*core;++i)if(actual[i]!=expected_read[i])throw std::runtime_error("native field-adapter gather mismatch");for(size_t i=0;i<3*core;++i)if(beta[i]!=expected_beta[i])throw std::runtime_error("native algebraic beta gather mismatch");
 }
 void ADM(){if(!initialized)return;auto *a=pack->padm;a->EnsureSeparateGaugeStorage();if(a->gauge_is_shared||!a->separate_z4c_gauge)throw std::runtime_error("native composite physical ADM gauge aliases conformal state");
  auto dst=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),a->u_adm);const auto q=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u0);const int n=engine->shift.grid.n,ng=engine->shift.grid.ng;const Real h=2*half/n;
  for(int k=0;k<n+2*ng;++k)for(int j=0;j<n+2*ng;++j)for(int i=0;i<n+2*ng;++i){Z4cJet<Real> u{};u.chi.value=q(0,Z4c::I_Z4C_CHI,k,j,i);u.alpha.value=q(0,Z4c::I_Z4C_ALPHA,k,j,i);u.trace.value=q(0,Z4c::I_Z4C_KHAT,k,j,i);u.theta.value=q(0,Z4c::I_Z4C_THETA,k,j,i);
   for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],b=CompositeTensorEvolution::tensor_j[t];u.metric.g[a][b]=u.metric.g[b][a]=q(0,Z4c::I_Z4C_GXX+t,k,j,i);u.a.k[a][b]=u.a.k[b][a]=q(0,Z4c::I_Z4C_AXX+t,k,j,i);}for(int a=0;a<3;++a)u.beta.value[a]=q(0,Z4c::I_Z4C_BETAX+a,k,j,i);
   const Real x[3]={-half+(i-ng+.5)*h,-half+(j-ng+.5)*h,-half+(k-ng+.5)*h};const Real omega=(1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;const auto physical=ToPhysicalADM(u,omega);if(!physical.valid)throw std::runtime_error("invalid native composite physical ADM conversion");
   dst(0,adm::ADM::I_ADM_PSI4,k,j,i)=physical.psi4;dst(0,adm::ADM::I_ADM_ALPHA,k,j,i)=physical.alpha;for(int a=0;a<3;++a)dst(0,adm::ADM::I_ADM_BETAX+a,k,j,i)=physical.beta[a];for(int t=0;t<6;++t){const int a=CompositeTensorEvolution::tensor_i[t],b=CompositeTensorEvolution::tensor_j[t];dst(0,adm::ADM::I_ADM_GXX+t,k,j,i)=physical.metric[a][b];dst(0,adm::ADM::I_ADM_KXX+t,k,j,i)=physical.curvature[a][b];}
  }Kokkos::deep_copy(a->u_adm,dst);
 }
 void Diagnostics(){if(!initialized||stage->Active())return;Verify();auto con=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),z.u_con);Kokkos::deep_copy(con,Real(0));const auto &g=engine->shift.grid;Real h2=0,m2=0,t2=0,z2=0,volume=0,maxh=0,maxm=0,drift=0;
  const bool write_snapshot=stage->CompletedCycles()%output_every==0;std::ostringstream number;number<<std::setw(5)<<std::setfill('0')<<stage->CompletedCycles();std::ofstream out;if(write_snapshot){out.open(prefix+"."+number.str()+".csv");out<<std::setprecision(17)<<"x,y,z,Omega,weight,alpha,chi,H,M,Theta,Z,beta_x,beta_y,beta_z";for(int f=0;f<22;++f)out<<",state_"<<f;out<<'\n';}
  for(size_t p=0;p<g.active.size();++p){const auto &u=engine->jets[p];const auto &x=g.geometry[p].x;const auto o=CompositeOmegaAt(x,u);const auto c=EvolvedConstraints(u,o);if(!c.valid)throw std::runtime_error("invalid native composite full-domain constraints");const Real w=g.volumes[p]*std::sqrt(engine->shift.geometry[p].determinant)/std::pow(u.chi.value,1.5);volume+=w;h2+=w*c.hamiltonian*c.hamiltonian;m2+=w*c.momentum_conformal_norm2;t2+=w*u.theta.value*u.theta.value;z2+=w*c.z4.z_conformal_norm2;maxh=std::max(maxh,std::abs(c.hamiltonian));maxm=std::max(maxm,std::sqrt(c.momentum_conformal_norm2));for(int f=0;f<22;++f)drift=std::max(drift,std::abs(engine->state[22*p+f]));if(write_snapshot){out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<o.omega<<','<<w<<','<<u.alpha.value<<','<<u.chi.value<<','<<c.hamiltonian<<','<<std::sqrt(c.momentum_conformal_norm2)<<','<<u.theta.value<<','<<std::sqrt(c.z4.z_conformal_norm2);for(int a=0;a<3;++a)out<<','<<u.beta.value[a];for(int f=0;f<22;++f){out<<','<<engine->state[22*p+f];drift=std::max(drift,std::abs(engine->state[22*p+f]));}out<<'\n';}
   if(p<size_t(g.n)*g.n*g.n){const int i=p%g.n+g.ng,j=p/g.n%g.n+g.ng,k=p/(g.n*g.n)+g.ng;con(0,Z4c::I_CON_H,k,j,i)=c.hamiltonian;con(0,Z4c::I_CON_M,k,j,i)=c.momentum_conformal_norm2;con(0,Z4c::I_CON_Z,k,j,i)=c.z4.z_conformal_norm2;con(0,Z4c::I_CON_C,k,j,i)=c.hamiltonian*c.hamiltonian+c.momentum_conformal_norm2+c.z4.z_conformal_norm2+u.theta.value*u.theta.value;for(int a=0;a<3;++a)con(0,Z4c::I_CON_MX+a,k,j,i)=c.momentum[a];}
  }Kokkos::deep_copy(z.u_con,con);if(write_snapshot){if(!out)throw std::runtime_error("native composite diagnostic output failed");std::ofstream scri(prefix+"."+number.str()+".scri.csv");scri<<std::setprecision(17)<<"x,y,z";for(int f=0;f<22;++f)scri<<",state_"<<f;for(int a=0;a<3;++a)scri<<",delta_beta_"<<a;scri<<'\n';for(size_t p=0;p<engine->scri->rows.size();++p){const auto &x=engine->scri->rows[p].geometry.x;scri<<x[0]<<','<<x[1]<<','<<x[2];for(int f=0;f<22;++f)scri<<','<<engine->scri->values[22*p+f];for(int a=0;a<3;++a)scri<<','<<engine->scri->beta[3*p+a];scri<<'\n';}}
  Real ghost_clock=0,ghost_null=0,ghost_tangent=0;const auto padded=engine->Padded(engine->state);
  for(size_t p=0;p<engine->scri->rows.size();++p){const auto &x=engine->scri->rows[p].geometry.x;const auto sample=engine->scri->Sample(p,padded,engine->scri->values,22);const auto u=UnpackComposite(x,sample.value,sample.d,sample.dd);const auto geometry=Geometry(u.metric);Real norm=0,bn=0,beta[3];
    const Real alpha=1+engine->scri->ReconstructFromGhost(p,padded,18);for(int i=0;i<3;++i){beta[i]=-x[i]+engine->scri->ReconstructFromGhost(p,engine->beta_padded,i);bn+=beta[i]*x[i];for(int j=0;j<3;++j)norm+=u.chi.value*geometry.inverse[i][j]*x[i]*x[j];}
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
 void Save(const std::string &path){if(!initialized||stage->Active())throw std::logic_error("native composite restart only at accepted cycle boundary");Verify();if(std::abs(stage->Time()-pack->pmesh->time)>1e-12)throw std::runtime_error("native/core accepted checkpoint time mismatch");Buffer b;b.String("ATHENAK_NATIVE_COMPOSITE_V1");b.String(checkpoint_tag);b.Put(engine->shift.grid.n);b.Put(engine->shift.grid.ng);b.Put(half);b.Vector(radii);const auto s=stage->Save();b.Put(s.version);b.Values(s.values);b.Values(s.registers);b.String(s.metadata.chart);b.Vector(s.metadata.chart_parameters);b.Put(s.metadata.mesh_revision);b.Put(s.metadata.chart_revision);b.Put(s.metadata.puncture_revision);b.String(s.rk.integrator);b.Put(s.rk.stages);for(const auto *a:{&s.rk.gam0,&s.rk.gam1,&s.rk.beta,&s.rk.delta})for(Real x:*a)b.Put(x);b.Vector(s.shift_warm_start);b.Put(s.cycle);b.Put(s.completed_cycles);b.Put(s.geometry);b.Put(s.next_stage);b.Put(s.open_stage);b.Put(s.active);b.Put(s.stage_open);b.Put(s.has_rk);b.Put(s.time);b.Put(s.dt);b.Put(s.abscissa);b.Put(s.register_abscissa);const uint64_t checksum=b.Checksum(b.bytes.size());b.Put(checksum);std::ofstream f(path,std::ios::binary);f.write(b.bytes.data(),b.bytes.size());if(!f)throw std::runtime_error("native composite checkpoint write failed");}
 void Restore(const std::string &path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("missing paired native composite restart sidecar");Buffer b;b.bytes.assign(std::istreambuf_iterator<char>(f),{});if(b.bytes.size()<8||b.bytes.size()>100000000)throw std::runtime_error("native composite restart size");uint64_t checksum;std::memcpy(&checksum,b.bytes.data()+b.bytes.size()-8,8);if(checksum!=b.Checksum(b.bytes.size()-8))throw std::runtime_error("native composite restart checksum mismatch");if(b.String()!="ATHENAK_NATIVE_COMPOSITE_V1"||b.String()!=checkpoint_tag||b.Get<int>()!=engine->shift.grid.n||b.Get<int>()!=engine->shift.grid.ng||b.Get<Real>()!=half||b.Vector()!=radii)throw std::runtime_error("native composite restart config/tag mismatch");State::Snapshot s;s.version=b.Get<uint32_t>();s.values=b.Values();s.registers=b.Values();s.metadata.chart=b.String();s.metadata.chart_parameters=b.Vector();s.metadata.mesh_revision=b.Get<uint64_t>();s.metadata.chart_revision=b.Get<uint64_t>();s.metadata.puncture_revision=b.Get<uint64_t>();s.rk.integrator=b.String();s.rk.stages=b.Get<int>();for(auto *a:{&s.rk.gam0,&s.rk.gam1,&s.rk.beta,&s.rk.delta})for(auto &x:*a)x=b.Get<Real>();s.shift_warm_start=b.Vector();s.cycle=b.Get<uint64_t>();s.completed_cycles=b.Get<uint64_t>();s.geometry=b.Get<uint64_t>();s.next_stage=b.Get<int>();s.open_stage=b.Get<int>();s.active=b.Get<bool>();s.stage_open=b.Get<bool>();s.has_rk=b.Get<bool>();s.time=b.Get<Real>();s.dt=b.Get<Real>();s.abscissa=b.Get<Real>();s.register_abscissa=b.Get<Real>();if(s.metadata.chart!="regular-conformal-deviation22"||s.metadata.chart_parameters!=V{initial_amplitude,initial_angular?Real(1):Real(0)})throw std::runtime_error("native composite restart pulse/chart configuration mismatch");if(b.cursor!=b.bytes.size()-8||s.active||s.stage_open||s.values.volume.size()!=engine->state.size()||s.values.scri.size()!=engine->scri->values.size()||s.time!=pack->pmesh->time||s.completed_cycles!=uint64_t(pack->pmesh->ncycle))throw std::runtime_error("native composite restart accepted-state layout/time mismatch");
  if(s.has_rk){bound_rk=s.rk;has_bound_rk=true;stage=std::make_unique<State>(State::Restore(s,s.rk));}else{if(s.cycle||s.completed_cycles||s.time!=0)throw std::runtime_error("unbound native restart must be initial");stage=std::make_unique<State>(s.values,s.shift_warm_start.size(),s.metadata);stage->InstallShift(stage->CurrentEpoch(),s.shift_warm_start);}Load();
 }
};
NativeCompositeEvolution::NativeCompositeEvolution(Z4c &z,MeshBlockPack *p,ParameterInput *pin):p_(std::make_unique<Impl>(z,p,pin)){}
NativeCompositeEvolution::~NativeCompositeEvolution()=default;
void NativeCompositeEvolution::Initialize(ParameterInput *pin,bool restart){auto &p=*p_;V loaded;if(restart){const auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.z.u0);loaded.assign(h.data(),h.data()+h.size());}if(!restart)p.engine->Initialize(p.initial_amplitude,p.initial_angular);State::Metadata m;m.chart="regular-conformal-deviation22";m.chart_parameters={p.initial_amplitude,p.initial_angular?Real(1):Real(0)};p.stage=std::make_unique<State>(p.Values(),p.engine->beta_delta.size(),m);if(restart)p.Restore(pin->GetString("z4c","composite_restart_file"));p.SolveAccepted();p.Scatter();if(restart&&loaded!=p.expected_native)throw std::runtime_error("native restart u0 disagrees with paired global composite state");p.initialized=true;p.ADM();}
void NativeCompositeEvolution::BeginStage(Driver *d,int s){auto &p=*p_;p.Verify();const auto rk=NativeCompositeRK::FromDriver(*d);if(p.has_bound_rk&&!(p.bound_rk==rk))throw std::runtime_error("native composite live Driver RK differs from restart");p.bound_rk=rk;p.has_bound_rk=true;if(s==1)p.stage->BeginCycle(rk,p.pack->pmesh->dt,p.pack->pmesh->time);p.epoch=p.stage->BeginStage(s);p.Load();++p.begins;}
void NativeCompositeEvolution::EvaluateRHS(Driver *,int){auto &p=*p_;p.rate.volume=p.engine->RHS(p.engine->state,p.pack->pmesh->dt);p.rate.scri=p.engine->boundary_rhs;p.rate.endpoints.clear();p.stage->InstallShift(p.epoch,p.engine->beta_delta);p.fields->Write(p.z.u_rhs,p.mapping,p.engine->state,p.engine->beta_delta,&p.rate.volume);++p.rhs_calls;}
void NativeCompositeEvolution::AdvanceStage(Driver *,int){auto &p=*p_;p.stage->AdvanceStage(p.epoch,p.rate);p.Load();++p.updates;if(!p.stage->Active())p.SolveAccepted();p.Scatter();}
void NativeCompositeEvolution::PrepareNative(){p_->Verify();p_->Scatter();p_->Diagnostics();}
void NativeCompositeEvolution::ToADM(){p_->ADM();}
void NativeCompositeEvolution::Constraints(){p_->Diagnostics();}
Real NativeCompositeEvolution::TimeStep()const{return p_->CFLStep()/p_->pack->pmesh->cfl_no;}
void NativeCompositeEvolution::WriteRestart(const std::string &path){p_->Save(path);}
}} // namespace z4c::hyperboloidal
