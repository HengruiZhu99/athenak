// Opt-in mapped research adapter. Regular Z4c and production hyperboloidal
// guards are unchanged. This probe does not validate an exact-scri closure or AMR.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/extension.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/conformal_bridge.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "mesh/cubed_sphere/probe_surfaces.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "z4c/hyperboloidal/layer_wormhole.hpp"

namespace cubed_sphere {
namespace {
namespace hyp=z4c::hyperboloidal;
using Z=z4c::Z4c;
struct Cell {
  int m,k,j,i;
  Point geometry;
  hyp::LayerPoint<Real> reference;
  hyp::Z4cRHS<Real> reserved_zero_rhs;  // Keep serialized Cell layout; never subtracted.
  Real volume;
  Real dq[3];
  int radial_depth=-1,radial_last=0;
};
using ReferenceView=Kokkos::View<const Real*****,Kokkos::LayoutRight,DevMemSpace>;
struct ProbePack {
  MeshBlockPack *pack;
  DvceArray5D<Real> original,dev;
  ReferenceView values;
  ReferenceView baseline;
  Kokkos::View<const hyp::Z4cJet<Real>*,DevMemSpace> initial_jets;
  Kokkos::View<const Cell*,DevMemSpace> device_cells;
  std::shared_ptr<const std::vector<Cell>> reference_cells;
  const std::vector<Cell> &Cells() const {return *reference_cells;}
};
// The reference fields and jets are constructed once from the descriptor,
// independent of the initial disturbance and of every evolved field.
ProbePack MakePack(MeshBlockPack *pack,const Map &map,const hyp::LayerReference<Real> &ref,
                   Real amplitude,Real width,Real damping,Real mass,bool interior_radial=false,bool quadrupole=false) {
  auto device_u=pack->pz4c->u0;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),device_u);
  auto ind=pack->pmesh->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  auto geometry=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                 pack->pcoord->z4c_geometry);
  ProbePack result; result.pack=pack;
  std::vector<Cell> cells;
  std::vector<hyp::Z4cJet<Real>> initial_jets;
  std::unique_ptr<const hyp::LayerWormhole<Real>> wormhole;
  if (mass>0) wormhole=std::make_unique<const hyp::LayerWormhole<Real>>(ref,mass);
  result.original=DvceArray5D<Real>("conformal saved regular state",u.extent(0),25,
      u.extent(2),u.extent(3),u.extent(4));
  result.dev=DvceArray5D<Real>("conformal deviations",u.extent(0),25,
      u.extent(2),u.extent(3),u.extent(4));
  Kokkos::deep_copy(result.original,device_u);
  DvceArray5D<Real> values("immutable Minkowski reference values",u.extent(0),25,
      u.extent(2),u.extent(3),u.extent(4));
  auto host_values=Kokkos::create_mirror_view(values);
  for (int m=0;m<pack->nmb_thispack;++m)
  for (int k=0;k<u.extent_int(2);++k) for (int j=0;j<u.extent_int(3);++j)
  for (int i=0;i<u.extent_int(4);++i) {
    auto size=sizes(m); Point p;
    if (geometry.extent(0)) p=geometry(m,k,j,i);
    else {
      Real lo[3]={size.x1min,size.x2min,size.x3min};
      Real dq[3]={size.dx1,size.dx2,size.dx3}; int ijk[3]={i,j,k};
      for (int a=0;a<3;++a) {p.x[a]=lo[a]+(ijk[a]-ind.ng+0.5)*dq[a]; p.J[a][a]=1;}
    }
    const auto background=ref.At(p.x[0],p.x[1],p.x[2]);
    for (int n=0;n<25;++n) u(m,n,k,j,i)=host_values(m,n,k,j,i)=hyp::ReferenceComponent(n,background);
    if (i<ind.is || i>ind.ie || j<ind.js || j>ind.je || k<ind.ks || k>ind.ke) continue;
    if (!(background.omega>0)) throw std::runtime_error("probe active cell reaches Omega<=0");
    Cell cell{m,k,j,i,p,background,{ },size.dx1*size.dx2*size.dx3,
              {size.dx1,size.dx2,size.dx3}};
    Real det=p.J[0][0]*(p.J[1][1]*p.J[2][2]-p.J[1][2]*p.J[2][1])
        -p.J[0][1]*(p.J[1][0]*p.J[2][2]-p.J[1][2]*p.J[2][0])
        +p.J[0][2]*(p.J[1][0]*p.J[2][1]-p.J[1][1]*p.J[2][0]);
    cell.volume/=fabs(det);
    if (interior_radial && ind.ie-i<3) {cell.radial_depth=ind.ie-i;cell.radial_last=ind.ie;}
    cells.push_back(cell);
    if (wormhole) {
      if (!(background.radius>0)) throw std::invalid_argument("puncture probe must exclude r=0");
      auto initial=background;
      initial.state=wormhole->At(p.x[0],p.x[1],p.x[2]);
      initial_jets.push_back(initial.state);
      for (int n=0;n<25;++n) u(m,n,k,j,i)=hyp::ReferenceComponent(n,initial);
    }
    // Same tapered angular perturbation as the archived native wave-map run.
    Real r2=background.radius*background.radius;
    Real pulse=pow(1-r2/(ref.scri_radius*ref.scri_radius),4)*exp(-r2/(width*width));
    Real x=p.x[0],y=p.x[1],z=p.x[2];
    if (quadrupole) {
      // Pure trace-free Cartesian quadratic: zero spherical mean, regular origin.
      const Real w2=width*width;
      const Real peak_r2=2*w2/(1+5*w2+sqrt(SQR(1+5*w2)-4*w2));
      const Real peak=peak_r2/w2*exp(-peak_r2/w2)*pow(1-peak_r2,4);
      const Real angular_scale=(1+sqrt(1.36))/2;
      const Real relative=amplitude*(x*x-y*y+0.6*y*z)*pulse/(angular_scale*w2*peak);
      u(m,Z::I_Z4C_ALPHA,k,j,i)+=background.state.alpha.value*relative;
    } else {
    u(m,Z::I_Z4C_ALPHA,k,j,i)+=amplitude*pulse*(1+0.2*x+0.3*y*z);
    u(m,Z::I_Z4C_BETAX,k,j,i)+=amplitude/2*pulse*(1+0.3*y*z);
    u(m,Z::I_Z4C_BETAY,k,j,i)+=amplitude/2*pulse*0.2*x;
    u(m,Z::I_Z4C_BETAZ,k,j,i)+=amplitude/2*pulse*0.1*x*y;
    }
  }
  Kokkos::deep_copy(device_u,u);
  Kokkos::deep_copy(values,host_values);
  Kokkos::View<Cell*,DevMemSpace> device_cells("immutable reference cells",cells.size());
  auto host_cells=Kokkos::create_mirror_view(device_cells);
  for (size_t i=0;i<cells.size();++i) host_cells(i)=cells[i];
  Kokkos::deep_copy(device_cells,host_cells);
  result.device_cells=device_cells;
  result.values=values;  // No mutable reference handle survives this function.
  result.baseline=result.values;
  if (wormhole) {
    Kokkos::View<hyp::Z4cJet<Real>*,DevMemSpace> jets("fixed initial wormhole jets",initial_jets.size());
    auto host_jets=Kokkos::create_mirror_view(jets);
    for (size_t i=0;i<initial_jets.size();++i) host_jets(i)=initial_jets[i];
    Kokkos::deep_copy(jets,host_jets);result.initial_jets=jets;
  }
  result.reference_cells=std::make_shared<const std::vector<Cell>>(std::move(cells));
  return result;
}
struct ProbeComponent {
 DvceArray5D<Real> data;int component;
 KOKKOS_INLINE_FUNCTION Real operator()(int m,int k,int j,int i) const {return data(m,component,k,j,i);}
};
KOKKOS_INLINE_FUNCTION hyp::ScalarJet<Real> LoadRadialScalar(
    const DvceArray5D<Real> &dev,int component,const Cell &c) {
 ProbeComponent q{dev,component};hyp::ScalarJet<Real> v{};
 Real idx[3]={1/c.dq[0],1/c.dq[1],1/c.dq[2]};v.value=q(c.m,c.k,c.j,c.i);
 auto sample=[&](int node){return q(c.m,c.k,c.j,c.radial_last-5+node);};
 v.d[0]=InteriorRadialFirst(sample,c.radial_depth,c.dq[0]);
 v.dd[0][0]=InteriorRadialSecond(sample,c.radial_depth,c.dq[0]);
 for(int a=1;a<3;++a) {
  v.d[a]=::Dx<3>(a,idx,q,c.m,c.k,c.j,c.i);
  v.dd[a][a]=::Dxx<3>(a,idx,q,c.m,c.k,c.j,c.i);
  auto mixed=[&](int node){return ::Dx<3>(a,idx,q,c.m,c.k,c.j,c.radial_last-5+node);};
  v.dd[0][a]=v.dd[a][0]=InteriorRadialFirst(mixed,c.radial_depth,c.dq[0]);
 }
 v.dd[1][2]=v.dd[2][1]=::Dxy<3>(1,2,idx,q,c.m,c.k,c.j,c.i);
 return v;
}
KOKKOS_INLINE_FUNCTION hyp::Z4cJet<Real> LoadRadialJet(const DvceArray5D<Real> &dev,const Cell &c) {
 hyp::Z4cJet<Real> u{};
 u.chi=LoadRadialScalar(dev,0,c);u.trace=LoadRadialScalar(dev,7,c);
 u.theta=LoadRadialScalar(dev,17,c);u.alpha=LoadRadialScalar(dev,18,c);
 const int ii[6]={0,0,0,1,1,2},jj[6]={0,1,2,1,2,2};
 for(int a=0;a<3;++a) {
  auto b=LoadRadialScalar(dev,19+a,c),l=LoadRadialScalar(dev,14+a,c);
  u.beta.value[a]=b.value;u.lambda.value[a]=l.value;
  for(int d=0;d<3;++d) {
   u.beta.d[d][a]=b.d[d];u.lambda.d[d][a]=l.d[d];
   for(int e=0;e<3;++e){u.beta.dd[d][e][a]=b.dd[d][e];u.lambda.dd[d][e][a]=l.dd[d][e];}
  }
 }
 for(int n=0;n<6;++n) {
  int a=ii[n],b=jj[n];auto g=LoadRadialScalar(dev,1+n,c),A=LoadRadialScalar(dev,8+n,c);
  u.metric.g[a][b]=u.metric.g[b][a]=g.value;u.a.k[a][b]=u.a.k[b][a]=A.value;
  for(int d=0;d<3;++d) {
   u.metric.dg[d][a][b]=u.metric.dg[d][b][a]=g.d[d];u.a.dk[d][a][b]=u.a.dk[d][b][a]=A.d[d];
   for(int e=0;e<3;++e)u.metric.ddg[d][e][a][b]=u.metric.ddg[d][e][b][a]=g.dd[d][e];
  }
 }
 return u;
}
KOKKOS_INLINE_FUNCTION hyp::Z4cJet<Real> LoadProbeJet(
    const Z::Z4c_vars &fields,const DvceArray5D<Real> &dev,const Cell &c,const hyp::Z4cJet<Real> &background) {
  Real idx[3]={1/c.dq[0],1/c.dq[1],1/c.dq[2]};
  auto u=c.radial_depth>=0 ? LoadRadialJet(dev,c) : hyp::LoadMeshJet<3>(fields,idx,c.m,c.k,c.j,c.i);
  CartesianJet(u,c.geometry);
  hyp::AddBackgroundJet(u,background);
  return u;
}
void Project(ProbePack &pack) {
  auto q=hyp::BindCartesianFields(pack.pack->pz4c->u0);
  auto cells=pack.device_cells; int failures=0;
  Kokkos::parallel_reduce("mapped conformal projection",cells.extent_int(0),
      KOKKOS_LAMBDA(int index,int &bad) {
    const auto &c=cells(index);
    hyp::MetricJet<Real> metric{};
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      metric.g[a][b]=q.g_dd(c.m,a,b,c.k,c.j,c.i);
    const auto g=hyp::Geometry(metric);
    if (!g.valid || !(metric.g[0][0]>0) ||
        !(metric.g[0][0]*metric.g[1][1]-SQR(metric.g[0][1])>0)) {++bad;return;}
    Real scale=1/cbrt(g.determinant),trace=0;
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      trace+=g.inverse[a][b]*q.vA_dd(c.m,a,b,c.k,c.j,c.i);
    for (int a=0;a<3;++a) for (int b=a;b<3;++b) {
      q.g_dd(c.m,a,b,c.k,c.j,c.i)=scale*metric.g[a][b];
      q.vA_dd(c.m,a,b,c.k,c.j,c.i)-=metric.g[a][b]*trace/3;
    }
  },failures);
  if (failures) throw std::runtime_error("probe invalid metric before projection");
}
struct Sample {
  Real h,m2,z2,theta,alpha,chi,dev,null_deviation,lapse_relative,gauge2;
  ProbeSurfacePoint surface;
  bool valid;
};
std::uint64_t CheckpointHash(std::istream &input,std::uint64_t bytes) {
  std::uint64_t hash=1469598103934665603ULL;char buffer[8192];
  while (bytes) {
    auto count=std::min<std::uint64_t>(bytes,sizeof(buffer));input.read(buffer,count);
    if (!input) throw std::runtime_error("truncated probe checkpoint checksum");
    for (std::uint64_t i=0;i<count;++i) {hash^=static_cast<unsigned char>(buffer[i]);hash*=1099511628211ULL;}
    bytes-=count;
  }
  return hash;
}
void RecordRHSFailure(const ProbePack &p,int index,int region,Real time,int cycle,int stage) {
  const auto &c=p.Cells().at(index);
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.pack->pz4c->u0);
  auto dev=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.dev);
  auto rhs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.pack->pz4c->u_rhs);
  std::ofstream record("cubed_sphere_conformal_rhs_failure.txt");record<<std::setprecision(17);
  record<<"region="<<region<<" time="<<time<<" cycle="<<cycle<<" stage="<<stage
        <<" cell="<<index<<" m,k,j,i="<<c.m<<","<<c.k<<","<<c.j<<","<<c.i
        <<" xyz="<<c.geometry.x[0]<<","<<c.geometry.x[1]<<","<<c.geometry.x[2]
        <<" Omega="<<c.reference.omega<<"\n";
  std::ofstream stencil("cubed_sphere_conformal_rhs_failure_stencil.csv");
  stencil<<std::setprecision(17)<<"m,k,j,i,field,state,deviation,rhs\n";
  for(int k=std::max(0,c.k-4);k<=std::min(u.extent_int(2)-1,c.k+4);++k)
  for(int j=std::max(0,c.j-4);j<=std::min(u.extent_int(3)-1,c.j+4);++j)
  for(int i=std::max(0,c.i-4);i<=std::min(u.extent_int(4)-1,c.i+4);++i)
  for(int n=0;n<25;++n)
    stencil<<c.m<<","<<k<<","<<j<<","<<i<<","<<n<<","<<u(c.m,n,k,j,i)
           <<","<<dev(c.m,n,k,j,i)<<","<<rhs(c.m,n,k,j,i)<<"\n";
}

}

void Extension::ConformalProbe(ParameterInput *pin,Real tlim) {
  auto *mesh=bulk_->pmesh;
  // The physical boundary may be truncated for finite-Omega diagnostics.
  // The unit-scri Minkowski descriptor is independent of that mesh boundary.
  constexpr Real reference_scri_radius=1;
  const Real domain_radius_squared=SQR(map_.R);
  if (!std::isfinite(tlim) || tlim<=0 || tlim>1000 || mesh->multilevel || initial_data_!=0 ||
      !(map_.R>0 && map_.R<=reference_scri_radius) || analytic_outer_ || pin->GetInteger("time","nlim")!=0)
    throw std::invalid_argument("conformal probe requires uniform Minkowski storage, 0<R<=1, "
                               "sommerfeld placeholder, nlim=0 and 0<tlim<=1000");
  Real amp=pin->GetOrAddReal("cubed_sphere","conformal_amplitude",0);
  Real cfl=pin->GetOrAddReal("cubed_sphere","conformal_cfl",0.025);
  Real pole_cfl=pin->GetOrAddReal("cubed_sphere","conformal_pole_cfl",0.03);
  Real damping=pin->GetOrAddReal("cubed_sphere","conformal_kappa1",0);
  Real epsilon=pin->GetOrAddReal("cubed_sphere","conformal_diss",0.1);
  Real mass=pin->GetOrAddReal("cubed_sphere","conformal_mass",0);
  Real fixed_dt=pin->GetOrAddReal("cubed_sphere","conformal_fixed_dt",0);
  int requested_steps=pin->GetOrAddInteger("cubed_sphere","conformal_probe_steps",0);
  bool quadrupole=pin->GetOrAddBoolean("cubed_sphere","conformal_quadrupole",false);
  bool native_gauge=pin->GetOrAddBoolean("cubed_sphere","conformal_native_layer_gauge",false);
  int max_cycles=pin->GetOrAddInteger("cubed_sphere","conformal_max_cycles",10000000);
  int degree=pin->GetOrAddInteger("cubed_sphere","conformal_outer_degree",2);
  bool interior_radial=pin->GetOrAddBoolean("cubed_sphere","conformal_interior_radial",false);
  if (interior_radial && shell_->mb_indcs.nx1<6) throw std::invalid_argument("interior radial closure requires six active radial cells");
  bool binary_reference=pin->GetOrAddBoolean("cubed_sphere","conformal_binary_reference",false);
  auto checkpoint=pin->GetOrAddString("cubed_sphere","conformal_checkpoint","none");
  auto restart=pin->GetOrAddString("cubed_sphere","conformal_restart","none");
  const auto checkpoint_tag=pin->GetOrAddString("cubed_sphere","conformal_checkpoint_tag","none");
  if (checkpoint=="none") checkpoint.clear();
  if (restart=="none") restart.clear();
  const Real wall_seconds=pin->GetOrAddReal("cubed_sphere","conformal_wall_seconds",0);
  if (!std::isfinite(wall_seconds) || wall_seconds<0 || wall_seconds>360 ||
      ((!checkpoint.empty() || !restart.empty()) && checkpoint_tag.size()!=64) ||
      (wall_seconds>0 && checkpoint.empty()))
    throw std::invalid_argument("probe checkpoints need a 64-character executable tag and 0<=wall_seconds<=360");
  if (!std::isfinite(amp) || fabs(amp)>0.5 || !std::isfinite(cfl) || cfl<=0 || cfl>0.15 ||
      !std::isfinite(pole_cfl) || pole_cfl<=0 || pole_cfl>0.1 || damping<0 ||
      !std::isfinite(damping) || epsilon<0 || !std::isfinite(epsilon) || max_cycles<=0 ||
      degree<2 || degree>5)
    throw std::invalid_argument("invalid conformal probe parameters");
  {std::ofstream parameters("cubed_sphere_conformal_parameters.txt"); pin->ParameterDump(parameters);}
  const hyp::LayerReference<Real> reference{reference_scri_radius,
      pin->GetOrAddReal("cubed_sphere","conformal_curvature_radius",0.5),
      {true,pin->GetOrAddReal("cubed_sphere","conformal_layer_r0",0.05),
            pin->GetOrAddReal("cubed_sphere","conformal_layer_r1",0.95)}};
  reference.Validate();
  if (!(reference.layer.r1<map_.R))
    throw std::invalid_argument("conformal probe outer radius must exceed layer end");
  {std::ofstream descriptor("cubed_sphere_conformal_domain.json");
   descriptor<<std::setprecision(17)<<"{\"domain_radius\":"<<map_.R
             <<",\"reference_scri_radius\":"<<reference.scri_radius<<"}\n";}
  if (!std::isfinite(mass) || mass<0 || !std::isfinite(fixed_dt) || fixed_dt<0 ||
      requested_steps<0 || requested_steps>max_cycles || (mass>0 && (amp!=0 || !native_gauge)))
    throw std::invalid_argument("invalid puncture probe controls; mass requires native gauge and no pulse");
  hyp::LayerGaugeParameters layer_gauge;
  layer_gauge.physical_trace_lapse=true;layer_gauge.preferred_source=false;
  layer_gauge.Validate(reference.scri_radius);
  ProbePack packs[2]={MakePack(bulk_,map_,reference,amp,pulse_width_,damping,mass,false,quadrupole),
      MakePack(shell_->pmb_pack,map_,reference,amp,pulse_width_,damping,mass,interior_radial,quadrupole)};
  // Uniform probe bypasses ordinary neighbor exchange; all halos use the same
  // active-only donor atlas as AMR. Only deviations are interpolated.
  scalar_probe_active_=true; mesh_sequence_=-1; RebuildPlans();
  auto bp=bulk_plan_;
  auto sp=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),shell_plan_);
  // Preserve a configurable quadratic native-like extrapolation control at
  // the radial outer boundary. Internal interface/seam interpolation remains
  // six point. This is still a different atlas closure from the old ray fit.
  auto si=shell_->mb_indcs;
  auto shell_geometry=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                      shell_->pmb_pack->pcoord->z4c_geometry);
  for (int l=0;l<sp.extent_int(0);++l) {
    auto &row=sp(l);
    Real x[3];
    for (int a=0;a<3;++a) x[a]=shell_geometry(row.tm,row.tk,row.tj,row.ti).x[a];
    if (SQR(x[0])+SQR(x[1])+SQR(x[2])<SQR(map_.R)) continue;
    Real q[3]; map_.Inverse(row.sm,x,q);
    int n[3]={si.nx1,si.nx2,si.nx3},start[3];
    Real lo[3]={0,-1,-1},dq[3]={1.0/si.nx1,2.0/si.nx2,2.0/si.nx3};
    for (int A=0;A<3;++A) {
      row.count[A]=degree+1; Real z=(q[A]-lo[A])/dq[A]-0.5;
      start[A]=std::clamp(static_cast<int>(floor(z))-degree/2,0,n[A]-degree-1);
      for (int a=0;a<=degree;++a) {
        row.w[A][a]=1;
        for (int b=0;b<=degree;++b) if (a!=b) row.w[A][a]*=(z-start[A]-b)/Real(a-b);
      }
      start[A]+=si.ng;
    }
    row.si=start[0]; row.sj=start[1]; row.sk=start[2];
  }
  Plan probe_shell_plan("mapped probe outer closure",sp.extent_int(0));
  Kokkos::deep_copy(probe_shell_plan,sp);
  auto fill=[&]() {
    for (auto &p:packs) {
      auto u=p.pack->pz4c->u0,dev=p.dev; auto rv=p.baseline;
      par_for("mapped conformal deviations",DevExeSpace(),0,u.extent_int(0)-1,0,24,
          0,u.extent_int(2)-1,0,u.extent_int(3)-1,0,u.extent_int(4)-1,
          KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
        dev(m,n,k,j,i)=u(m,n,k,j,i)-rv(m,n,k,j,i);
      });
    }
    auto bulk_dev=packs[0].dev,shell_dev=packs[1].dev;
    for (int region=0;region<2;++region) {
      auto plan=region ? probe_shell_plan : bp; auto &target=packs[region];
      auto target_dev=target.dev,state=target.pack->pz4c->u0; auto rv=target.baseline;
      Kokkos::parallel_for("mapped conformal donor fill",plan.extent_int(0)*25,
          KOKKOS_LAMBDA(int index) {
        const auto &r=plan(index/25);int n=index%25;
        auto source=r.source_shell ? shell_dev : bulk_dev;
        Real anchor=source(r.sm,n,r.sk,r.sj,r.si),value=0;
        for (int c=0;c<r.count[2];++c) for (int b=0;b<r.count[1];++b) {
          Real line=0;
          for (int a=0;a<r.count[0];++a)
            line+=r.w[0][a]*(source(r.sm,n,r.sk+c,r.sj+b,r.si+a)-anchor);
          value+=r.w[2][c]*r.w[1][b]*line;
        }
        value+=anchor;
        target_dev(r.tm,n,r.tk,r.tj,r.ti)=value;
        state(r.tm,n,r.tk,r.tj,r.ti)=value+rv(r.tm,n,r.tk,r.tj,r.ti);
      });
    }
  };
  if (mass>0) {
    // Exactly the native wormhole reconstruction policy: first extend initial
    // deviations from Minkowski, then freeze that initial profile including
    // its ghosts. All later derivatives/KO/interpolation use evolving
    // deviations from it. Gauge and Omega use the independent immutable
    // Minkowski reference. The full native RHS is evolved without subtraction.
    fill();
    for (auto &p:packs) {
      auto state=p.pack->pz4c->u0;
      DvceArray5D<Real> baseline("fixed initial wormhole values",state.extent(0),25,
          state.extent(2),state.extent(3),state.extent(4));
      Kokkos::deep_copy(baseline,state);p.baseline=baseline;
    }
  }
  std::ofstream out("cubed_sphere_conformal.csv"); out<<std::setprecision(17);
  out<<"time,cycle,region,volume,min_Omega,min_alpha,min_chi,max_deviation,H_l2,H_linf,"
       "M_conformal_l2,Z_conformal_l2,Theta_l2,max_null_deviation\n";
  std::ofstream profile("cubed_sphere_conformal_regions.csv"); profile<<std::setprecision(17);
  profile<<"time,cycle,region,subset,cells,volume,H_l2,H_linf,M_conformal_l2,"
           "Z_conformal_l2,max_H_x,max_H_y,max_H_z\n";
  std::ofstream maxima("cubed_sphere_conformal_maxima.csv");
  maxima<<std::setprecision(17);
  maxima<<"time,cycle,region,subset,cells,M_conformal_linf,Z_conformal_linf,"
          "max_M_x,max_M_y,max_M_z,max_Z_x,max_Z_y,max_Z_z\n";
  std::ofstream steps("cubed_sphere_conformal_steps.csv"); steps<<std::setprecision(17);
  steps<<"time,cycle,dt\n";
  std::ofstream surfaces("cubed_sphere_surfaces.csv"),horizons("cubed_sphere_horizon_candidates.csv");
  surfaces<<std::setprecision(17);horizons<<std::setprecision(17);
  surfaces<<"time,cycle,r,areal_radius,Hawking_mass,theta_plus_mean,theta_plus_min,theta_plus_max,physical_lapse_mean,proper_radial_factor,proper_distance_from_first_radius\n";
  horizons<<"time,cycle,found,r,areal_radius,irreducible_mass,Hawking_mass,expansion_max_abs,area_quadrature_relative_difference\n";
  Real next_surface_time=0;
  std::ofstream gauge_monitor("cubed_sphere_gauge_monitor.csv");
  gauge_monitor<<std::setprecision(17)<<"time,cycle,region,subset,gauge_deviation_RMS,relative_lapse_RMS,relative_lapse_max\n";
  int last_measured_cycle=-1;
  auto measure=[&](Real time,int cycle) {
    fill();
    for (int region=0;region<2;++region) {
      auto &p=packs[region]; Real vol=0,min_o=1e100,min_a=1e100,min_c=1e100;
      Real max_d=0,h2=0,hmax=0,m2=0,z2=0,t2=0,max_null=0;
      struct Sum {
        int count=0;
        Real vol=0,h2=0,hmax=0,m2=0,z2=0,x[3]{},gauge2=0,lapse2=0,lapsemax=0;
        Real mmax2=0,zmax2=0,mx[3]{},zx[3]{};
      };
      std::map<std::string,Sum> sums;
      auto cells=p.device_cells;auto dev=p.dev;auto initial=p.initial_jets;
      auto fields=hyp::BindCartesianFields(dev);
      Kokkos::View<Sample*,DevMemSpace> samples("mapped conformal diagnostics",cells.extent(0));
      Kokkos::parallel_for("mapped conformal constraint samples",cells.extent_int(0),
          KOKKOS_LAMBDA(int index) {
        const auto &c=cells(index);const auto &r=c.reference;
        const auto u=LoadProbeJet(fields,dev,c,initial.extent(0) ? initial(index) : r.state);
        const auto constraint=hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,r));
        const auto ref_constraint=hyp::EvolvedConstraints(r.state,hyp::CartesianOmega(r.state,r));
        auto &v=samples(index);v.valid=constraint.valid && constraint.z4.valid;
        v.valid=v.valid && u.metric.g[0][0]>0 &&
            u.metric.g[0][0]*u.metric.g[1][1]-SQR(u.metric.g[0][1])>0;
        if (mass>0 && region==0) v.surface=SurfacePoint(u,hyp::CartesianOmega(u,r),c.geometry.x);
        v.h=constraint.hamiltonian;v.m2=constraint.momentum_conformal_norm2;
        v.z2=constraint.z4.z_conformal_norm2;v.theta=u.theta.value;
        v.alpha=u.alpha.value;v.chi=u.chi.value;
        v.lapse_relative=(u.alpha.value-r.state.alpha.value)/r.state.alpha.value;
        v.gauge2=SQR(v.lapse_relative);
        for(int a=0;a<3;++a)v.gauge2+=SQR(u.beta.value[a]-r.state.beta.value[a]);
        v.null_deviation=fabs(constraint.null_residual-ref_constraint.null_residual);v.dev=0;
        for (int n=0;n<25;++n) v.dev=Kokkos::fmax(v.dev,fabs(dev(c.m,n,c.k,c.j,c.i)));
      });
      auto host_samples=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),samples);
      for (size_t index=0;index<p.Cells().size();++index) {
        const auto &c=p.Cells()[index];const auto &r=c.reference;
        const auto &sample=host_samples(index);
        if (!sample.valid) throw std::runtime_error("invalid mapped conformal constraints");
        Real v=c.volume; vol+=v; min_o=std::min(min_o,r.omega);
        min_a=std::min(min_a,sample.alpha); min_c=std::min(min_c,sample.chi);
        h2+=v*SQR(sample.h); hmax=std::max(hmax,fabs(sample.h));
        m2+=v*sample.m2; z2+=v*sample.z2;
        t2+=v*SQR(sample.theta);
        max_null=std::max(max_null,sample.null_deviation);
        max_d=std::max(max_d,sample.dev);
        auto add=[&](const std::string &name) {
          auto &s=sums[name]; ++s.count; s.vol+=v;
          s.h2+=v*SQR(sample.h); s.m2+=v*sample.m2;
          s.z2+=v*sample.z2;
          s.gauge2+=v*sample.gauge2;s.lapse2+=v*SQR(sample.lapse_relative);
          s.lapsemax=std::max(s.lapsemax,fabs(sample.lapse_relative));
          if (fabs(sample.h)>s.hmax) {
            s.hmax=fabs(sample.h);
            for (int a=0;a<3;++a) s.x[a]=c.geometry.x[a];
          }
          if (s.count==1 || sample.m2>s.mmax2) {
            s.mmax2=sample.m2;
            for (int a=0;a<3;++a) s.mx[a]=c.geometry.x[a];
          }
          if (s.count==1 || sample.z2>s.zmax2) {
            s.zmax2=sample.z2;
            for (int a=0;a<3;++a) s.zx[a]=c.geometry.x[a];
          }
        };
        if (r.radius>=reference.layer.r0 && r.radius<reference.layer.r1) add("fixed_transition");
        add("all"); add("radial_"+std::to_string(std::min(15,int(16*r.radius/reference.scri_radius))));
        Real extent=std::max({fabs(c.geometry.x[0]),fabs(c.geometry.x[1]),fabs(c.geometry.x[2])});
        if (!region) add(extent<0.18 ? "fixed_interior" : "fixed_interface");
        else {add(r.radius<0.5 ? "fixed_inner_shell" : "fixed_outer_shell");
              if (r.radius>0.9) add("fixed_outer_collar");}
      }
      if (mass>0 && region==0 && (cycle==0 || time>=next_surface_time ||
          time>=tlim || (requested_steps && cycle==requested_steps))) {
        const int n=int(std::round(std::cbrt(Real(p.Cells().size()))));
        if (n<4 || size_t(n)*n*n!=p.Cells().size()) throw std::runtime_error("surface bulk must be uniform cube");
        BulkSurfaceSampler sampler{n,map_.L,std::vector<ProbeSurfacePoint>(n*n*n)};
        const Real dx=2*map_.L/n;
        for(size_t v=0;v<p.Cells().size();++v) {
          const auto &c=p.Cells()[v];int q[3];
          for(int a=0;a<3;++a)q[a]=int(std::round((c.geometry.x[a]+map_.L)/dx-.5));
          sampler.points[(q[2]*n+q[1])*n+q[0]]=host_samples(v).surface;
        }
        ProbeSphere previous{};Real distance=0;bool found=false;ProbeSphere candidate{};
        for(int k=0;k<48;++k) {
          const Real radius=dx+(map_.L*.999-dx)*k/47;
          const auto sphere=MeasureProbeSphere(sampler,radius);
          if(k) distance+=(sphere.r-previous.r)*(sphere.proper+previous.proper)/2;
          surfaces<<time<<","<<cycle<<","<<sphere.r<<","<<sphere.areal<<","<<sphere.mass
                  <<","<<sphere.plus<<","<<sphere.lo<<","<<sphere.hi<<","<<sphere.lapse
                  <<","<<sphere.proper<<","<<distance<<"\n";
          if(k && previous.plus<=0 && sphere.plus>=0) {
            Real lo=previous.r,hi=sphere.r;
            for(int iteration=0;iteration<24;++iteration) {
              auto trial=MeasureProbeSphere(sampler,(lo+hi)/2);
              if(trial.plus>0)hi=trial.r;else lo=trial.r;
            }
            candidate=MeasureProbeSphere(sampler,(lo+hi)/2);found=true;
          }
          previous=sphere;
        }
        horizons<<time<<","<<cycle<<","<<int(found);
        if(found) {
          const auto doubled=MeasureProbeSphere(sampler,candidate.r,32);
          horizons<<","<<candidate.r<<","<<candidate.areal<<","<<candidate.areal/2
                  <<","<<candidate.mass<<","<<std::max(fabs(candidate.lo),fabs(candidate.hi))
                  <<","<<fabs(doubled.area/candidate.area-1);
        } else horizons<<",nan,nan,nan,nan,nan,nan";
        horizons<<"\n";surfaces.flush();horizons.flush();next_surface_time=time+.1;
      }
      out<<time<<","<<cycle<<","<<(region ? "shell" : "bulk")<<","<<vol<<","<<min_o
         <<","<<min_a<<","<<min_c<<","<<max_d<<","<<sqrt(h2/vol)<<","<<hmax
         <<","<<sqrt(m2/vol)<<","<<sqrt(z2/vol)<<","<<sqrt(t2/vol)<<","<<max_null<<"\n";
      for (const auto &entry:sums) {
        const auto &s=entry.second;
        profile<<time<<","<<cycle<<","<<(region ? "shell" : "bulk")<<","<<entry.first
               <<","<<s.count<<","<<s.vol<<","<<sqrt(s.h2/s.vol)<<","<<s.hmax
               <<","<<sqrt(s.m2/s.vol)<<","<<sqrt(s.z2/s.vol)
               <<","<<s.x[0]<<","<<s.x[1]<<","<<s.x[2]<<"\n";
        gauge_monitor<<time<<","<<cycle<<","<<(region ? "shell" : "bulk")<<","<<entry.first
                     <<","<<sqrt(s.gauge2/s.vol)<<","<<sqrt(s.lapse2/s.vol)<<","<<s.lapsemax<<"\n";
        maxima<<time<<","<<cycle<<","<<(region ? "shell" : "bulk")<<","<<entry.first
              <<","<<s.count<<","<<sqrt(s.mmax2)<<","<<sqrt(s.zmax2)
              <<","<<s.mx[0]<<","<<s.mx[1]<<","<<s.mx[2]
              <<","<<s.zx[0]<<","<<s.zx[1]<<","<<s.zx[2]<<"\n";
      }
    }
    out.flush();
    profile.flush();
    maxima.flush();gauge_monitor.flush();
    last_measured_cycle=cycle;
  };
  // Emit all consumed reference values/first/second jets through a stable text
  // serialization before and after evolution. External receipts hash both files.
  auto serialize=[&](const char *path) {
    if (binary_reference) {
      std::string name(path);name.replace(name.size()-4,4,".bin");
      std::ofstream saved(name,std::ios::binary);
      saved<<"CUBED_SPHERE_REFERENCE_V1\n";
      for (int region=0;region<2;++region) {
        const auto &p=packs[region];
        auto cells=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.device_cells);
        auto values=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.values);
        size_t n=cells.extent(0),shape[5];
        for (int a=0;a<5;++a) shape[a]=values.extent(a);
        if (!values.span_is_contiguous() || values.span()!=values.size())
          throw std::runtime_error("noncontiguous immutable reference serialization");
        saved.write(reinterpret_cast<const char*>(&n),sizeof(n));
        saved.write(reinterpret_cast<const char*>(shape),sizeof(shape));
        saved.write(reinterpret_cast<const char*>(cells.data()),n*sizeof(Cell));
        saved.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Real));
      }
      if (!saved) throw std::runtime_error("failed immutable reference binary serialization");
      return;
    }
    std::ofstream saved(path); saved<<std::hexfloat;
    for (int region=0;region<2;++region) {
      auto cells=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),packs[region].device_cells);
      for (size_t index=0;index<cells.extent(0);++index) {
      const auto &c=cells(index);
      saved<<region<<" "<<c.m<<" "<<c.k<<" "<<c.j<<" "<<c.i<<" "<<c.reference.omega;
      auto emit=[&](const Real *v,int n){for (int a=0;a<n;++a) saved<<" "<<v[a];};
      const auto &u=c.reference.state;
      for (const auto *s:{&u.chi,&u.alpha,&u.trace,&u.theta}) {
        saved<<" "<<s->value; emit(s->d,3);
        for (int a=0;a<3;++a) emit(s->dd[a],3);
      }
      for (const auto *v:{&u.beta,&u.lambda}) {
        emit(v->value,3);
        for (int a=0;a<3;++a) {emit(v->d[a],3); for (int b=0;b<3;++b) emit(v->dd[a][b],3);}
      }
      for (int a=0;a<3;++a) {
        emit(u.metric.g[a],3); emit(u.a.k[a],3);
        for (int b=0;b<3;++b) {
          emit(u.metric.dg[a][b],3); emit(u.a.dk[a][b],3);
          for (int d=0;d<3;++d) emit(u.metric.ddg[a][b][d],3);
        }
      }
      emit(c.reference.domega,3);
      for (int a=0;a<3;++a) emit(c.reference.omega_hessian[a],3);
      const auto &r=c.reference;
      saved<<" "<<r.radius<<" "<<r.L<<" "<<r.b<<" "<<r.k_bar<<" "<<r.k_physical<<" "<<r.alpha;
      emit(r.beta,3); emit(r.dalpha,3);
      saved<<"\n";
      }
    }
  };
  auto serialize_initial=[&](const char *path) {
    if (mass==0) return;
    std::ofstream saved(path,std::ios::binary);saved<<"CUBED_SPHERE_INITIAL_PROFILE_V1\n";
    for (const auto &p:packs) {
      auto jets=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.initial_jets);
      auto values=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.baseline);
      size_t n=jets.extent(0),shape[5];
      for (int a=0;a<5;++a) shape[a]=values.extent(a);
      if (!values.span_is_contiguous() || values.span()!=values.size())
        throw std::runtime_error("noncontiguous initial profile serialization");
      saved.write(reinterpret_cast<const char*>(&n),sizeof(n));
      saved.write(reinterpret_cast<const char*>(shape),sizeof(shape));
      saved.write(reinterpret_cast<const char*>(jets.data()),n*sizeof(hyp::Z4cJet<Real>));
      saved.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Real));
    }
    if (!saved) throw std::runtime_error("failed initial profile serialization");
  };
  serialize("cubed_sphere_reference_before.txt");
  serialize_initial("cubed_sphere_initial_profile_before.bin");
  Real time=0; int cycle=0;
  // Research checkpoint only: immutable references/initial jets are reconstructed
  // afresh. The file contains evolved fields, not a replacement reference.
  const std::array<Real,17> signature={map_.L,map_.R,map_.radial_stretch,
      reference.curvature_radius,reference.layer.r0,reference.layer.r1,mass,amp,
      pulse_width_,damping,epsilon,Real(native_gauge),Real(degree+100*int(interior_radial)+1000*int(quadrupole)),cfl,pole_cfl,fixed_dt,Real(sizeof(Real))};
  std::uint64_t fingerprint=1469598103934665603ULL;
  auto hash_real=[&](Real value) {
    unsigned char bytes[sizeof(Real)];std::memcpy(bytes,&value,sizeof(Real));
    for (auto byte:bytes) {fingerprint^=byte;fingerprint*=1099511628211ULL;}
  };
  for (const auto &p:packs) for (const auto &c:p.Cells()) {
    for (int a=0;a<3;++a) {
      hash_real(c.geometry.x[a]);hash_real(c.dq[a]);hash_real(c.reference.domega[a]);
      for (int b=0;b<3;++b) {
        hash_real(c.geometry.J[a][b]);hash_real(c.reference.omega_hessian[a][b]);
        for (int d=0;d<3;++d) hash_real(c.geometry.H[a][b][d]);
      }
    }
    for (int n=0;n<25;++n) hash_real(hyp::ReferenceComponent(n,c.reference));
    hash_real(c.reference.omega);
  }
  if (!restart.empty()) {
    std::ifstream input(restart,std::ios::binary);
    input.seekg(0,std::ios::end);const auto length=input.tellg();
    if (!input || length<8) throw std::runtime_error("missing/truncated probe checkpoint");
    input.seekg(0);const auto checksum=CheckpointHash(input,static_cast<std::uint64_t>(length)-8);
    std::uint64_t stored_checksum;input.read(reinterpret_cast<char*>(&stored_checksum),8);
    if (!input || checksum!=stored_checksum) throw std::runtime_error("probe checkpoint checksum mismatch");
    input.seekg(0);
    char magic[16]{},tag[64];std::array<Real,17> saved_signature;
    std::uint64_t saved_fingerprint;std::int64_t saved_cycle;
    input.read(magic,16);input.read(tag,64);
    input.read(reinterpret_cast<char*>(saved_signature.data()),sizeof(saved_signature));
    input.read(reinterpret_cast<char*>(&saved_fingerprint),sizeof(saved_fingerprint));
    input.read(reinterpret_cast<char*>(&time),sizeof(time));
    input.read(reinterpret_cast<char*>(&saved_cycle),sizeof(saved_cycle));
    if (!input || std::string(magic,16)!="ATHENA_CSP_CK02!" || std::string(tag,64)!=checkpoint_tag ||
        saved_signature!=signature || saved_fingerprint!=fingerprint ||
        !std::isfinite(time) || time<0 || time>tlim || saved_cycle<0 || saved_cycle>max_cycles)
      throw std::runtime_error("probe checkpoint descriptor/reference/build mismatch");
    cycle=static_cast<int>(saved_cycle);
    for (auto &p:packs) {
      auto state=p.pack->pz4c->u0;
      std::array<std::uint64_t,5> shape,saved_shape;
      for (int a=0;a<5;++a) shape[a]=state.extent(a);
      input.read(reinterpret_cast<char*>(saved_shape.data()),sizeof(saved_shape));
      if (!input || shape!=saved_shape) throw std::runtime_error("probe checkpoint shape mismatch");
      auto host=Kokkos::create_mirror_view(state);
      input.read(reinterpret_cast<char*>(host.data()),host.size()*sizeof(Real));
      if (!input) throw std::runtime_error("truncated probe checkpoint");
      for (const auto &c:p.Cells()) {
        for (int n=0;n<25;++n) if (!std::isfinite(host(c.m,n,c.k,c.j,c.i)))
          throw std::runtime_error("nonfinite probe checkpoint field");
        hyp::MetricJet<Real> metric{};
        const int indices[3][3]={{1,2,3},{2,4,5},{3,5,6}};
        for (int a=0;a<3;++a) for (int b=0;b<3;++b)
          metric.g[a][b]=host(c.m,indices[a][b],c.k,c.j,c.i);
        if (!(host(c.m,Z::I_Z4C_ALPHA,c.k,c.j,c.i)>0) ||
            !(host(c.m,Z::I_Z4C_CHI,c.k,c.j,c.i)>0) || !hyp::Geometry(metric).valid)
          throw std::runtime_error("inadmissible probe checkpoint state");
      }
      Kokkos::deep_copy(state,host);
    }
    input.read(reinterpret_cast<char*>(&stored_checksum),8);
    if (!input || input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing probe checkpoint data");
    std::cout<<"Probe checkpoint restored: time="<<std::setprecision(17)<<time<<", cycle="<<cycle<<std::endl;
  }
  measure(time,cycle);
  const auto wall_start=std::chrono::steady_clock::now();bool wall_stop=false;
  while (time<tlim && cycle<max_cycles && (!requested_steps || cycle<requested_steps)) {
    if (wall_seconds>0 && std::chrono::duration<Real>(std::chrono::steady_clock::now()-wall_start).count()>=wall_seconds) {
      wall_stop=true;if (last_measured_cycle!=cycle) measure(time,cycle);break;
    }
    fill(); Real dt=tlim-time;
    for (const auto &p:packs) {
      auto cells=p.device_cells;auto fields=hyp::BindCartesianFields(p.dev);auto initial=p.initial_jets;
      Real pack_dt=std::numeric_limits<Real>::max();
      Kokkos::parallel_reduce("mapped conformal CFL",cells.extent_int(0),
          KOKKOS_LAMBDA(int index,Real &minimum) {
        const auto &c=cells(index);
        // CFL needs values, not spatial derivatives or curvature. Reconstruct
        // deviation + background in the same order as LoadProbeJet; using u0
        // directly instead would change rounding. Full jets are still checked
        // by the native RHS before any stage update.
        const auto &background=initial.extent(0) ? initial(index) : c.reference.state;
        hyp::Z4cJet<Real> u{};
        u.alpha.value=fields.alpha(c.m,c.k,c.j,c.i)+background.alpha.value;
        u.chi.value=fields.chi(c.m,c.k,c.j,c.i)+background.chi.value;
        for (int a=0;a<3;++a) {
          u.beta.value[a]=fields.beta_u(c.m,a,c.k,c.j,c.i)+background.beta.value[a];
          if (!Kokkos::isfinite(u.beta.value[a])) {minimum=0;return;}
          for (int b=0;b<3;++b)
            u.metric.g[a][b]=fields.g_dd(c.m,a,b,c.k,c.j,c.i)+background.metric.g[a][b];
        }
        auto g=hyp::Geometry(u.metric);
        if (!g.valid || !(u.alpha.value>0) || !(u.chi.value>0)
            || !Kokkos::isfinite(u.alpha.value) || !Kokkos::isfinite(u.chi.value)) {minimum=0;return;}
        Real inverse_dt=0;
        for (int A=0;A<3;++A) {
          Real shift=0,norm=0;
          for (int a=0;a<3;++a) {
            shift+=c.geometry.J[A][a]*u.beta.value[a];
            for (int b=0;b<3;++b) norm+=c.geometry.J[A][a]*c.geometry.J[A][b]*g.inverse[a][b];
          }
          Real speed=native_gauge
              ? sqrt(hyp::LayerCoefficients(c.reference.radius,u.alpha.value,layer_gauge).alpha2f*u.chi.value*norm)
              : u.alpha.value*sqrt(u.chi.value*norm);
          inverse_dt+=(fabs(shift)+speed)/c.dq[A];
        }
        minimum=Kokkos::fmin(minimum,Kokkos::fmin(cfl/inverse_dt,pole_cfl*c.reference.omega));
      },Kokkos::Min<Real>(pack_dt));
      dt=std::min(dt,pack_dt);
    }
    if (fixed_dt>0) dt=std::min(dt,fixed_dt);
    if (!(dt>0) || !std::isfinite(dt)) throw std::runtime_error("probe inadmissible before CFL");
    mesh->dt=dt; mesh->time=time; mesh->ncycle=cycle;
    steps<<time<<","<<cycle<<","<<dt<<"\n"; steps.flush();
    shell_->dt=dt; shell_->time=time;
    for (auto &p:packs) Kokkos::deep_copy(p.pack->pz4c->u1,p.pack->pz4c->u0);
    for (int stage=1;stage<=3;++stage) {
      fill();
      for (int region=0;region<2;++region) {
        auto &p=packs[region]; auto *pack=p.pack; auto state=pack->pz4c->u0;
        auto &snapshot=region ? stage_shell_ : stage_bulk_;
        if (snapshot.extent(0)!=state.extent(0))
          Kokkos::realloc(snapshot,state.extent(0),25,state.extent(2),state.extent(3),state.extent(4));
        Kokkos::deep_copy(snapshot,state);
        auto q=hyp::BindCartesianFields(state),output=hyp::BindCartesianFields(pack->pz4c->u_rhs);
        auto cells=p.device_cells;auto dev=p.dev;auto rhs_data=pack->pz4c->u_rhs;
        auto initial=p.initial_jets;
        auto fields=hyp::BindCartesianFields(dev);
        auto geometry=pack->pcoord->z4c_geometry;bool mapped=region;
        Kokkos::View<int,DevMemSpace> first_bad("mapped conformal first bad RHS");
        Kokkos::deep_copy(first_bad,std::numeric_limits<int>::max());int failures=0;
        Kokkos::parallel_reduce("mapped original conformal RHS",cells.extent_int(0),
            KOKKOS_LAMBDA(int index,int &bad) {
          const auto &c=cells(index);const auto &r=c.reference;
          auto u=LoadProbeJet(fields,dev,c,initial.extent(0) ? initial(index) : r.state);
          hyp::Z4cRHS<Real> rhs{};hyp::GaugeRHS<Real> gauge{};
          auto conn=rwm::ReferenceConnection(r,c.geometry.x);
          if (!(u.metric.g[0][0]>0) ||
              !(u.metric.g[0][0]*u.metric.g[1][1]-SQR(u.metric.g[0][1])>0) ||
              !hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,r),
                  damping/u.alpha.value,Real(0)),r.omega,rhs) ||
              !(native_gauge ? hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(r,u,layer_gauge),r.omega,gauge)
                             : rwm::Assemble(rwm::Gauge(r,u,conn),r.omega,gauge))) {
            ++bad;Kokkos::atomic_min(&first_bad(),index);return;
          }
          hyp::StoreMeshRHS(output,c.m,c.k,c.j,c.i,rhs,gauge);
          Stencil s; s.mapped=mapped; s.p=c.geometry;
          Real dq[3]={c.dq[0],c.dq[1],c.dq[2]};
          for (int A=0;A<3;++A) s.idx[A]=1/dq[A];
          for (int n=0;n<Z::I_Z4C_BX;++n) {
            Real extra=0;
            for (int A=0;A<3;++A) {
              if(c.radial_depth<0) {
                extra+=::Lx<3>(A,s,q.beta_u,dev,c.m,A,n,c.k,c.j,c.i)
                    -q.beta_u(c.m,A,c.k,c.j,c.i)*::Dx<3>(A,s,dev,c.m,n,c.k,c.j,c.i);
              } else if(A>0) {
                // Radial advection already uses the new D1 in the native jet.
                // Retain existing sign-biased angular advection on this layer.
                Real velocity=0;for(int a=0;a<3;++a)velocity+=c.geometry.J[A][a]*q.beta_u(c.m,a,c.k,c.j,c.i);
                UnitVelocity unit{velocity<0 ? Real(-1):Real(1)};
                extra+=fabs(velocity)*::Lx<3>(A,s.idx,unit,dev,c.m,A,n,c.k,c.j,c.i)
                    -velocity*::Dx<3>(A,s.idx,dev,c.m,n,c.k,c.j,c.i);
              }
              // Same four-node active-line KO closure as the native cut-sphere
              // code, in logical directions. Mapped variable coefficients and
              // interpolation mean its Cartesian energy proof does not transfer.
              Real norm=0; for (int a=0;a<3;++a) norm+=SQR(c.geometry.J[A][a]);
              const Real coeff[4]={-1,3,-3,1};
              for (int l=0;l<4;++l) {
                Real third=0; bool inside=true;
                for (int b=0;b<4;++b) {
                  int ijk[3]={c.i,c.j,c.k}; ijk[A]+=b-l;
                  Real xyz[3];
                  if (mapped) for (int a=0;a<3;++a)
                    xyz[a]=geometry(c.m,ijk[2],ijk[1],ijk[0]).x[a];
                  else {
                    int delta[3]={ijk[0]-c.i,ijk[1]-c.j,ijk[2]-c.k};
                    for (int a=0;a<3;++a) xyz[a]=c.geometry.x[a]+delta[a]*dq[a];
                  }
                  if (SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2])>=domain_radius_squared) inside=false;
                  third+=coeff[b]*dev(c.m,n,ijk[2],ijk[1],ijk[0]);
                }
                if (inside) extra-=epsilon*sqrt(norm)*coeff[l]*third/(64*dq[A]);
              }
            }
            rhs_data(c.m,n,c.k,c.j,c.i)+=extra;
          }
        },failures);
        if (failures) {
          auto first=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),first_bad);
          RecordRHSFailure(p,first(),region,time,cycle,stage);
          throw std::runtime_error("invalid mapped original-formalism RHS");
        }
      }
      for (auto &p:packs) {
        auto u=p.pack->pz4c->u0,reg=p.pack->pz4c->u1,rhs=p.pack->pz4c->u_rhs;
        Real g0=stage==1 ? 0 : (stage==2 ? 0.25 : 2.0/3);
        Real beta=stage==2 ? 0.25 : (stage==3 ? 2.0/3 : 1);
        auto cells=p.device_cells;
        Kokkos::parallel_for("mapped conformal RK update",cells.extent_int(0)*25,
            KOKKOS_LAMBDA(int index) {
          const auto &c=cells(index/25);int n=index%25;
          u(c.m,n,c.k,c.j,c.i)=g0*u(c.m,n,c.k,c.j,c.i)+(1-g0)*reg(c.m,n,c.k,c.j,c.i)
                              +beta*dt*rhs(c.m,n,c.k,c.j,c.i);
        });
      }
      CheckState(bulk_,stage,time+(stage==2 ? 0.5 : 1)*dt,"conformal_bulk");
      CheckState(shell_->pmb_pack,stage,time+(stage==2 ? 0.5 : 1)*dt,"conformal_shell");
    }
    for (auto &p:packs) Project(p);
    time+=dt; ++cycle;
    if (cycle%20==0 || time>=tlim || cycle==max_cycles || cycle==requested_steps) {
      measure(time,cycle);
      std::cout<<"Mapped original wave-map probe: t="<<time<<", cycles="<<cycle<<", dt="<<dt<<std::endl;
    }
  }
  serialize("cubed_sphere_reference_after.txt");
  serialize_initial("cubed_sphere_initial_profile_after.bin");
  if (!checkpoint.empty()) {
    const auto temporary=checkpoint+".tmp";
    std::ofstream saved(temporary,std::ios::binary|std::ios::trunc);
    const char magic[]="ATHENA_CSP_CK02!";
    const std::int64_t saved_cycle=cycle;
    saved.write(magic,16);saved.write(checkpoint_tag.data(),64);
    saved.write(reinterpret_cast<const char*>(signature.data()),sizeof(signature));
    saved.write(reinterpret_cast<const char*>(&fingerprint),sizeof(fingerprint));
    saved.write(reinterpret_cast<const char*>(&time),sizeof(time));
    saved.write(reinterpret_cast<const char*>(&saved_cycle),sizeof(saved_cycle));
    for (const auto &p:packs) {
      auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.pack->pz4c->u0);
      std::array<std::uint64_t,5> shape;
      for (int a=0;a<5;++a) shape[a]=host.extent(a);
      saved.write(reinterpret_cast<const char*>(shape.data()),sizeof(shape));
      saved.write(reinterpret_cast<const char*>(host.data()),host.size()*sizeof(Real));
    }
    saved.close();
    if (!saved) throw std::runtime_error("failed probe checkpoint payload write");
    std::ifstream verify(temporary,std::ios::binary);verify.seekg(0,std::ios::end);
    const auto length=verify.tellg();verify.seekg(0);
    const auto checksum=CheckpointHash(verify,static_cast<std::uint64_t>(length));verify.close();
    std::ofstream footer(temporary,std::ios::binary|std::ios::app);
    footer.write(reinterpret_cast<const char*>(&checksum),8);footer.close();
    if (!footer || std::rename(temporary.c_str(),checkpoint.c_str())!=0)
      throw std::runtime_error("failed atomic probe checkpoint write");
    std::cout<<"Probe checkpoint written: time="<<std::setprecision(17)<<time<<", cycle="<<cycle
             <<", bounded_wall_stop="<<(wall_stop ? "true" : "false")<<std::endl;
  }
  if (time<tlim && !(requested_steps && cycle==requested_steps) && !wall_stop)
    throw std::runtime_error("conformal probe reached bounded cycle cap");
  for (auto &p:packs) Kokkos::deep_copy(p.pack->pz4c->u0,p.original);
  mesh->time=0; mesh->ncycle=0; mesh->dt=0;
  scalar_probe_active_=false; mesh_sequence_=-1; Fill(0);
}
}  // namespace cubed_sphere
