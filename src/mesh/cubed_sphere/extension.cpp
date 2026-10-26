#include "mesh/cubed_sphere/extension.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/test_state.hpp"
#include "coordinates/adm.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace z4c {
void Z4c::InitializeCubedSphere(ParameterInput *pin, bool restart) {
  if (!pin->GetOrAddBoolean("cubed_sphere","enabled",false)) return;
  if (restart) throw std::invalid_argument("cubed-sphere restart is not implemented");
  cubed_sphere=std::make_unique<::cubed_sphere::Extension>(pmy_pack,pin);
}
}  // namespace z4c

namespace cubed_sphere {
Extension::~Extension() = default;

Extension::Extension(MeshBlockPack *bulk, ParameterInput *pin) : bulk_(bulk) {
  using Z=z4c::Z4c;
  Mesh *mesh=bulk->pmesh;
  if (global_variable::nranks!=1 || !mesh->three_d || bulk->pdyngr ||
      bulk->phydro || bulk->pmhd || bulk->pz4c->hyperboloidal_patch || mesh->pzoom)
    throw std::invalid_argument("cubed sphere currently requires one rank, 3D vacuum "
        "regular Z4c, and no cyclic zoom");
  if (pin->GetString("time","integrator")!="rk3" || bulk->pz4c->opt.fd_stencil!=3)
    throw std::invalid_argument("cubed sphere currently requires RK3 and spatial_order=4");
  static_assert(Z::nz4c==25,"update cubed sphere test-state and stage storage");
  if (bulk->pz4c->opt.chi_psi_power!=-4)
    throw std::invalid_argument("cubed sphere test data requires chi_psi_power=-4");
  const auto &opt=bulk->pz4c->opt;
  if (opt.telegraph_lapse || opt.slow_start_lapse || opt.shift_hh!=0 ||
      opt.lapse_harmonic<0 || opt.lapse_oplog<0 || opt.lapse_harmonicf<0 ||
      opt.shift_ggamma<0 || opt.shift_alpha2ggamma<0)
    throw std::invalid_argument("cubed sphere supports harmonic/1+log lapse and "
        "Gamma shift; telegraph, slow start and harmonic shift need separate CFL validation");
  if (bulk->pz4c->nrad || !bulk->pz4c->ptracker.empty() ||
      pin->GetOrAddInteger("fastflow","num_horizons",0) ||
      pin->GetOrAddInteger("cce","num_radii",0) || bulk->pz4c->opt.enable_driftcontrol)
    throw std::invalid_argument("cubed sphere: extraction/trackers are not patch aware");
  if (pin->GetString("problem","pgen_name")!="z4c_cubed_sphere")
    throw std::invalid_argument("cubed sphere currently supports z4c_cubed_sphere test data");
  for (const auto &b:pin->block) if (b.block_name.compare(0,6,"output")==0 &&
      pin->GetString(b.block_name,"file_type")=="rst")
    throw std::invalid_argument("cubed sphere: restart outputs would omit shell state");
  const auto ms=mesh->mesh_size;
  map_.L=ms.x1max;
  map_.R=pin->GetReal("cubed_sphere","outer_radius");
  map_.radial_stretch=pin->GetOrAddReal("cubed_sphere","radial_stretch",0);
  if (!std::isfinite(map_.radial_stretch) || map_.radial_stretch<0 || map_.radial_stretch>6 ||
      (map_.radial_stretch!=0 && map_.radial_stretch<0.01))
    throw std::invalid_argument("radial_stretch must be zero or in [0.01,6]");
  if (!(map_.L>0) || ms.x1min!=-map_.L || ms.x2min!=-map_.L ||
      ms.x3min!=-map_.L || ms.x2max!=map_.L || ms.x3max!=map_.L ||
      !std::isfinite(map_.R) || !(map_.R>sqrt(3.0)*map_.L))
    throw std::invalid_argument("cubed sphere needs a centered cubic bulk and R>sqrt(3)*L");
  for (int a=0;a<6;++a) if (mesh->mesh_bcs[a]!=BoundaryFlag::outflow)
    throw std::invalid_argument("cubed sphere needs outflow placeholders on all bulk faces");
  const auto bi=mesh->mb_indcs;
  if (std::min({bi.nx1,bi.nx2,bi.nx3})<6)
    throw std::invalid_argument("cubed sphere needs at least six cells per bulk block axis");
  int nr=pin->GetOrAddInteger("cubed_sphere","nr",16);
  int na=pin->GetOrAddInteger("cubed_sphere","nangular",16);
  if (nr<6 || na<6 || map_.L+map_.RadialForward(-(bi.ng-0.5)/nr)*(map_.R-map_.L)<=0)
    throw std::invalid_argument("cubed sphere radial ghost map crosses the origin; increase nr");
  Real angular_ghost=1+Real(2*bi.ng-1)/na;
  if (map_.R<=map_.L*sqrt(1+2*angular_ghost*angular_ghost))
    throw std::invalid_argument("cubed sphere angular ghost map folds; increase outer radius");
  std::string bc=pin->GetOrAddString("cubed_sphere","outer_boundary","sommerfeld");
  if (bc!="sommerfeld" && bc!="analytic")
    throw std::invalid_argument("outer_boundary must be sommerfeld or analytic");
  analytic_outer_=bc=="analytic";
  std::string data=pin->GetOrAddString("problem","initial_data","minkowski");
  initial_data_=data=="gauge_wave" ? 1 : (data=="angular_pulse" ? 2 : 0);
  if (data!="minkowski" && data!="gauge_wave" && data!="angular_pulse")
    throw std::invalid_argument("unknown cubed sphere initial_data");
  if (analytic_outer_ && initial_data_==2)
    throw std::invalid_argument("angular pulse has no prescribed analytic outer solution");
  amplitude_=pin->GetOrAddReal("problem","amplitude",0.001);
  wave_number_=pin->GetOrAddReal("problem","wave_number",1.0);
  pulse_width_=pin->GetOrAddReal("problem","pulse_width",0.35);

  // Reuse the ordinary Z4c allocation/kernels without altering the bulk octree.
  // This temporary Cartesian tree only allocates six equal tensor-product blocks;
  // none of its neighbor relations are used by the shell evolution.
  pin_=std::make_unique<ParameterInput>();
  for (const auto &b:pin->block)
    if (b.block_name=="z4c" || b.block_name=="time" || b.block_name=="problem")
      pin_->block.push_back(b);
  pin_->SetString("mesh_refinement","refinement","none");
  pin_->SetInteger("mesh","nx1",6*nr);
  pin_->SetInteger("mesh","nx2",na); pin_->SetInteger("mesh","nx3",na);
  pin_->SetInteger("mesh","nghost",bi.ng);
  pin_->SetInteger("meshblock","nx1",nr);
  pin_->SetInteger("meshblock","nx2",na); pin_->SetInteger("meshblock","nx3",na);
  pin_->SetReal("mesh","x1min",0); pin_->SetReal("mesh","x1max",6);
  pin_->SetReal("mesh","x2min",-1); pin_->SetReal("mesh","x2max",1);
  pin_->SetReal("mesh","x3min",-1); pin_->SetReal("mesh","x3max",1);
  for (const char *f:{"ix1_bc","ox1_bc","ix2_bc","ox2_bc","ix3_bc","ox3_bc"})
    pin_->SetString("mesh",f,"outflow");
  shell_=std::make_unique<Mesh>(pin_.get());
  shell_->BuildTreeFromScratch(pin_.get());
  shell_->AddCoordinatesAndPhysics(pin_.get());
  auto *sp=shell_->pmb_pack;
  auto &sz=sp->pmb->mb_size;
  for (int m=0;m<6;++m) {
    sz.h_view(m)={0,-1,-1,1,1,1,1.0/nr,2.0/na,2.0/na};
    for (int n=0;n<6;++n) sp->pmb->mb_bcs.h_view(m,n)=BoundaryFlag::block;
  }
  sz.modify<HostMemSpace>(); sz.sync<DevExeSpace>();
  sp->pmb->mb_bcs.modify<HostMemSpace>(); sp->pmb->mb_bcs.sync<DevExeSpace>();
  auto si=shell_->mb_indcs;
  auto &geometry=sp->pcoord->z4c_geometry;
  Kokkos::realloc(geometry,6,na+2*si.ng,na+2*si.ng,nr+2*si.ng);
  Map map=map_;
  par_for("cubed sphere geometry",DevExeSpace(),0,5,0,na+2*si.ng-1,
      0,na+2*si.ng-1,0,nr+2*si.ng-1,
      KOKKOS_LAMBDA(int m,int k,int j,int i) {
    Real q[3]={(i-si.is+0.5)/si.nx1,
               -1+2*(j-si.js+0.5)/si.nx2,-1+2*(k-si.ks+0.5)/si.nx3};
    geometry(m,k,j,i)=map.Geometry(m,q);
  });
  shell_->pgen=std::make_unique<ProblemGenerator>(pin_.get(),shell_.get());
  sp->pz4c->Z4cToADM(sp);
  RebuildPlans();
  Fill(mesh->time);
  std::cout << "Cubed sphere: six fixed patches, Cartesian AMR bulk, Cartesian components, "
            << "R=" << map_.R << ", nr=" << nr << ", nangular=" << na
            << ", outer=" << bc << std::endl;
  Real scalar_tlim=pin->GetOrAddReal("cubed_sphere","scalar_probe_tlim",0);
  std::ofstream parameters("cubed_sphere_parameters.txt");
  pin->ParameterDump(parameters);
  parameters.close();
  if (scalar_tlim>0) ScalarProbe(scalar_tlim,pin->GetReal("time","cfl_number"));
  Real conformal_tlim=pin->GetOrAddReal("cubed_sphere","conformal_probe_tlim",0);
  if (conformal_tlim>0) {
#ifdef ATHENA_CUBED_SPHERE_CONFORMAL_PROBE
    ConformalProbe(pin,conformal_tlim);
#else
    throw std::invalid_argument("conformal probe requires its explicit Serial CMake option");
#endif
  }
}

void Extension::RebuildPlans() {
  Mesh *bm=bulk_->pmesh;
  if (mesh_sequence_==bm->GetAMRLoadBalanceUpdateSeq()) return;
  std::vector<Row> bp,cp,sp;
  const auto bi=bm->mb_indcs, si=shell_->mb_indcs;
  const auto &sizes=bulk_->pmb->mb_size.h_view;
  auto weights=[](Real z,int n,int ng,int &start,Real w[6],int &count) {
    const Real nearest=round(z);
    if (fabs(z-nearest)<1e-11 && nearest>=0 && nearest<n) {
      start=static_cast<int>(nearest)+ng; w[0]=1; count=1; return;
    }
    count=6;
    start=std::clamp(static_cast<int>(floor(z))-2,0,n-6);
    for (int a=0;a<6;++a) {
      w[a]=1;
      for (int b=0;b<6;++b) if (a!=b) w[a]*=(z-start-b)/Real(a-b);
    }
    start+=ng;
  };
  auto donor=[&](Row &r,const Real x[3]) {
    Real extent=std::max({fabs(x[0]),fabs(x[1]),fabs(x[2])});
    Real q[3],lo[3],dx[3]; int n[3];
    if (extent<map_.L*(1-1e-13)) {
      r.source_shell=0; r.sm=-1;
      for (int m=0;m<bulk_->nmb_thispack;++m) {
        const auto z=sizes(m);
        if (x[0]>=z.x1min-1e-13 && x[0]<=z.x1max+1e-13 &&
            x[1]>=z.x2min-1e-13 && x[1]<=z.x2max+1e-13 &&
            x[2]>=z.x3min-1e-13 && x[2]<=z.x3max+1e-13) {
          if (r.sm<0 || z.dx1<sizes(r.sm).dx1) r.sm=m;
        }
      }
      if (r.sm<0) throw std::runtime_error("cubed sphere missing Cartesian AMR donor");
      auto z=sizes(r.sm);
      lo[0]=z.x1min; lo[1]=z.x2min; lo[2]=z.x3min;
      dx[0]=z.dx1; dx[1]=z.dx2; dx[2]=z.dx3;
      n[0]=bi.nx1; n[1]=bi.nx2; n[2]=bi.nx3;
      for (int a=0;a<3;++a) q[a]=x[a];
    } else {
      r.source_shell=1; r.sm=map_.Owner(x); map_.Inverse(r.sm,x,q);
      lo[0]=0; lo[1]=lo[2]=-1;
      dx[0]=1.0/si.nx1; dx[1]=2.0/si.nx2; dx[2]=2.0/si.nx3;
      n[0]=si.nx1; n[1]=si.nx2; n[2]=si.nx3;
    }
    int start[3];
    for (int a=0;a<3;++a) weights((q[a]-lo[a])/dx[a]-0.5,n[a],bi.ng,start[a],r.w[a],r.count[a]);
    r.si=start[0]; r.sj=start[1]; r.sk=start[2];
  };
  auto bulk_targets=[&](bool coarse,std::vector<Row> &rows) {
    int n[3]={bi.nx1,bi.nx2,bi.nx3};
    if (coarse) for (int &v:n) v/=2;
    for (int m=0;m<bulk_->nmb_thispack;++m) {
      auto z=sizes(m); Real lo[3]={z.x1min,z.x2min,z.x3min};
      Real dx[3]={z.dx1,z.dx2,z.dx3};
      if (coarse) for (Real &v:dx) v*=2;
      for (int k=0;k<n[2]+2*bi.ng;++k) for (int j=0;j<n[1]+2*bi.ng;++j)
      for (int i=0;i<n[0]+2*bi.ng;++i) {
        Real x[3]={lo[0]+(i-bi.ng+0.5)*dx[0],lo[1]+(j-bi.ng+0.5)*dx[1],
                   lo[2]+(k-bi.ng+0.5)*dx[2]};
        bool outside=std::max({fabs(x[0]),fabs(x[1]),fabs(x[2])})>map_.L;
        bool active=i>=bi.ng && i<bi.ng+n[0] && j>=bi.ng && j<bi.ng+n[1] &&
                    k>=bi.ng && k<bi.ng+n[2];
        // The interface owns all adaptive bulk halos. Resolve each halo directly
        // against the current leaf hierarchy, including newly allocated blocks;
        // no donor reads a halo or depends on a previous exchange/prolongation.
        if (!outside && (coarse || (!bm->adaptive && !scalar_probe_active_) || active)) continue;
        Real radius=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        if (radius>=map_.R) throw std::invalid_argument(
            "cubed sphere outer radius must enclose Cartesian fine/coarse ghost corners");
        Row r{}; r.tm=m; r.tk=k; r.tj=j; r.ti=i; donor(r,x); rows.push_back(r);
      }
    }
  };
  bulk_targets(false,bp);
  if (bm->multilevel) bulk_targets(true,cp);
  for (int m=0;m<6;++m) for (int k=0;k<si.nx3+2*si.ng;++k)
  for (int j=0;j<si.nx2+2*si.ng;++j) for (int i=0;i<si.nx1+2*si.ng;++i) {
    if (i>=si.is && i<=si.ie && j>=si.js && j<=si.je && k>=si.ks && k<=si.ke) continue;
    Real q[3]={(i-si.is+0.5)/si.nx1,-1+2*(j-si.js+0.5)/si.nx2,
               -1+2*(k-si.ks+0.5)/si.nx3},x[3];
    map_.Forward(m,q,x);
    Row r{}; r.tm=m; r.tk=k; r.tj=j; r.ti=i; donor(r,x); sp.push_back(r);
  }
  auto upload=[](const std::vector<Row> &rows,Plan &plan) {
    Kokkos::realloc(plan,rows.size()); auto host=Kokkos::create_mirror_view(plan);
    for (size_t n=0;n<rows.size();++n) host(n)=rows[n];
    Kokkos::deep_copy(plan,host);
  };
  upload(bp,bulk_plan_); upload(cp,coarse_plan_); upload(sp,shell_plan_);
  mesh_sequence_=bm->GetAMRLoadBalanceUpdateSeq(); ++plan_builds_;
  std::cout << "Cubed sphere donor plan " << plan_builds_ << ": bulk blocks="
            << bulk_->nmb_thispack << ", halos=" << bp.size() << "/" << cp.size()
            << "/" << sp.size() << std::endl;
}

void Extension::Apply(const Plan &plan,const DvceArray5D<Real> &target) {
  auto bulk=bulk_->pz4c->u0, shell=shell_->pmb_pack->pz4c->u0;
  int last=scalar_probe_active_ ? 0 : z4c::Z4c::nz4c-1;
  par_for("cubed sphere interpolation",DevExeSpace(),0,plan.extent_int(0)-1,
      0,last,KOKKOS_LAMBDA(int l,int n) {
    const auto &r=plan(l); auto source=r.source_shell ? shell : bulk;
    const Real anchor=source(r.sm,n,r.sk,r.sj,r.si);
    Real value=0;
    for (int c=0;c<r.count[2];++c) for (int b=0;b<r.count[1];++b) {
      Real line=0;
      for (int a=0;a<r.count[0];++a) line+=r.w[0][a]*(source(r.sm,n,r.sk+c,r.sj+b,r.si+a)-anchor);
      value+=r.w[2][c]*r.w[1][b]*line;
    }
    target(r.tm,n,r.tk,r.tj,r.ti)=anchor+value;
  });
}

void Extension::Fill(Real time) {
  RebuildPlans();
  Apply(bulk_plan_,bulk_->pz4c->u0);
  Apply(shell_plan_,shell_->pmb_pack->pz4c->u0);
  if (analytic_outer_) {
    auto *sp=shell_->pmb_pack; auto ind=sp->pmesh->mb_indcs;
    auto geometry=sp->pcoord->z4c_geometry; auto u=sp->pz4c->u0;
    int kind=initial_data_; Real amp=amplitude_,wn=wave_number_,width=pulse_width_;
    par_for("cubed sphere analytic outer ghosts",DevExeSpace(),0,5,
        0,ind.nx3+2*ind.ng-1,0,ind.nx2+2*ind.ng-1,ind.ie+1,ind.ie+ind.ng,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real value[25]; TestState(geometry(m,k,j,i).x,time,kind,amp,wn,width,value);
      for (int n=0;n<25;++n) u(m,n,k,j,i)=value[n];
    });
  }
}
void Extension::FillCoarse() {
  RebuildPlans(); Apply(coarse_plan_,bulk_->pz4c->coarse_u0);
}

void Extension::BoundaryRHS() {
  if (analytic_outer_) return;
  // Same Sommerfeld subset/speeds as regular Cartesian Z4c, now with the true
  // spherical radial derivative. This is a finite-radius test BC, not a CPBC.
  using Z=z4c::Z4c;
  auto *sp=shell_->pmb_pack; auto ind=shell_->mb_indcs;
  auto u=sp->pz4c->u0, rhs=sp->pz4c->u_rhs;
  auto geometry=sp->pcoord->z4c_geometry; auto sizes=sp->pmb->mb_size;
  par_for("cubed sphere Sommerfeld",DevExeSpace(),0,5,ind.ks,ind.ke,
      ind.js,ind.je,KOKKOS_LAMBDA(int m,int k,int j) {
    int i=ind.ie; const auto &p=geometry(m,k,j,i);
    Real r=sqrt(p.x[0]*p.x[0]+p.x[1]*p.x[1]+p.x[2]*p.x[2]);
    auto s=MakeStencil(geometry,sizes.d_view(m),ind,m,k,j,i);
    for (int n=0;n<Z::nz4c;++n) {
      if (!(n==Z::I_Z4C_KHAT || n==Z::I_Z4C_THETA ||
            (n>=Z::I_Z4C_AXX && n<=Z::I_Z4C_GAMZ))) continue;
      Real radial=0;
      for (int a=0;a<3;++a) radial+=p.x[a]/r*::Dx<2>(a,s,u,m,n,k,j,i);
      Real speed=n==Z::I_Z4C_KHAT ? sqrt(2.0) : 1.0;
      rhs(m,n,k,j,i)=-speed*(radial+u(m,n,k,j,i)/r);
    }
  });
}
void Extension::BeginStage(Driver *driver,int stage) {
  Real c=stage==2 ? 1.0 : (stage==3 ? 0.5 : 0.0);
  shell_->dt=bulk_->pmesh->dt; shell_->time=bulk_->pmesh->time;
  Fill(shell_->time+c*shell_->dt);
  auto bu=bulk_->pz4c->u0,su=shell_->pmb_pack->pz4c->u0;
  if (stage_bulk_.extent(0)!=bu.extent(0))
    Kokkos::realloc(stage_bulk_,bu.extent(0),25,bu.extent(2),bu.extent(3),bu.extent(4));
  if (stage_shell_.extent(0)!=su.extent(0))
    Kokkos::realloc(stage_shell_,su.extent(0),25,su.extent(2),su.extent(3),su.extent(4));
  Kokkos::deep_copy(DevExeSpace(),stage_bulk_,bu);
  Kokkos::deep_copy(DevExeSpace(),stage_shell_,su);
  auto *z=shell_->pmb_pack->pz4c;
  z->CopyU(driver,stage);
  z->CalcRHS<3>(driver,stage);
  BoundaryRHS();
}
void Extension::Update(Driver *driver,int stage) {
  shell_->pmb_pack->pz4c->ExpRKUpdate(driver,stage);
  Real c=stage==2 ? 0.5 : 1.0;
  CheckState(bulk_,stage,bulk_->pmesh->time+c*bulk_->pmesh->dt,"bulk");
  CheckState(shell_->pmb_pack,stage,shell_->time+c*shell_->dt,"shell");
}
void Extension::FinishStage(Driver *driver,int stage) {
  auto *sp=shell_->pmb_pack;
  if (stage==driver->nexp_stages) sp->pz4c->AlgConstr(sp);
  sp->pz4c->Z4cToADM(sp);
}

Real Extension::TimeStep(Driver *driver) {
  Real minimum=std::numeric_limits<Real>::max();
  for (auto *sp:{bulk_,shell_->pmb_pack}) {
  auto ind=sp->pmesh->mb_indcs;
  auto sizes=sp->pmb->mb_size;
  auto z=sp->pz4c->z4c; auto opt=sp->pz4c->opt;
  auto geometry=sp->pcoord->z4c_geometry;
  Real dt=std::numeric_limits<Real>::max();
  int cells=sp->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  Kokkos::parallel_reduce("cubed sphere dt",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
      KOKKOS_LAMBDA(int l,Real &min_dt) {
    int i=l%ind.nx1+ind.is,j=(l/ind.nx1)%ind.nx2+ind.js;
    int k=(l/(ind.nx1*ind.nx2))%ind.nx3+ind.ks;
    int m=l/(ind.nx1*ind.nx2*ind.nx3);
    Point p;
    if (geometry.extent(0)) p=geometry(m,k,j,i);
    else for (int a=0;a<3;++a) p.J[a][a]=1;
    Real inv[3][3],det=adm::SpatialDet(z.g_dd(m,0,0,k,j,i),z.g_dd(m,0,1,k,j,i),
        z.g_dd(m,0,2,k,j,i),z.g_dd(m,1,1,k,j,i),z.g_dd(m,1,2,k,j,i),z.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1/det,z.g_dd(m,0,0,k,j,i),z.g_dd(m,0,1,k,j,i),
        z.g_dd(m,0,2,k,j,i),z.g_dd(m,1,1,k,j,i),z.g_dd(m,1,2,k,j,i),z.g_dd(m,2,2,k,j,i),
        &inv[0][0],&inv[0][1],&inv[0][2],&inv[1][1],&inv[1][2],&inv[2][2]);
    inv[1][0]=inv[0][1]; inv[2][0]=inv[0][2]; inv[2][1]=inv[1][2];
    Real alpha=z.alpha(m,k,j,i), chi=pow(z.chi(m,k,j,i),-4/opt.chi_psi_power);
    Real lapse_factor=fmax(1.0,sqrt(fabs(opt.lapse_harmonic+
        opt.lapse_oplog*opt.lapse_harmonicf/alpha)));
    Real advect=fmax(1.0,fmax(fabs(opt.lapse_advect),fabs(opt.shift_advect)));
    Real inverse_dt=0;
    for (int A=0;A<3;++A) {
      Real norm=0,shift=0;
      for (int a=0;a<3;++a) {
        shift+=p.J[A][a]*z.beta_u(m,a,k,j,i);
        for (int b=0;b<3;++b) norm+=p.J[A][a]*p.J[A][b]*inv[a][b];
      }
      auto size=sizes.d_view(m);
      Real dq=A==0 ? size.dx1 : (A==1 ? size.dx2 : size.dx3);
      Real light=fabs(alpha)*sqrt(fmax(chi*norm,0.0));
      // Gamma-driver waves use the conformal inverse metric; their speed does
      // not acquire a factor alpha*sqrt(chi) when the lapse collapses.
      Real gamma=sqrt((4.0/3)*(opt.shift_ggamma+opt.shift_alpha2ggamma*alpha*alpha)
                       *fmax(norm,0.0));
      inverse_dt+=(advect*fabs(shift)+fmax(lapse_factor*light,gamma))/dq;
    }
    min_dt=fmin(min_dt,1/inverse_dt);
  },Kokkos::Min<Real>(dt));
  minimum=std::min(minimum,dt);
  }
  return minimum;
}

void Extension::CheckState(MeshBlockPack *pack,int stage,Real time,const char *where) {
  using Z=z4c::Z4c;
  auto ind=pack->pmesh->mb_indcs; auto u=pack->pz4c->u0;
  int cells=pack->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  int bad=cells;
  Kokkos::parallel_reduce("cubed sphere RK admissibility",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
      KOKKOS_LAMBDA(int l,int &first) {
    int i=l%ind.nx1+ind.is,j=(l/ind.nx1)%ind.nx2+ind.js;
    int k=(l/(ind.nx1*ind.nx2))%ind.nx3+ind.ks;
    int m=l/(ind.nx1*ind.nx2*ind.nx3);
    bool ok=u(m,Z::I_Z4C_ALPHA,k,j,i)>0 && u(m,Z::I_Z4C_CHI,k,j,i)>0;
    for (int n=0;n<Z::nz4c;++n) ok=ok && Kokkos::isfinite(u(m,n,k,j,i));
    Real xx=u(m,Z::I_Z4C_GXX,k,j,i),xy=u(m,Z::I_Z4C_GXY,k,j,i);
    Real xz=u(m,Z::I_Z4C_GXZ,k,j,i),yy=u(m,Z::I_Z4C_GYY,k,j,i);
    Real yz=u(m,Z::I_Z4C_GYZ,k,j,i),zz=u(m,Z::I_Z4C_GZZ,k,j,i);
    ok=ok && xx>0 && xx*yy-xy*xy>0 &&
        adm::SpatialDet(xx,xy,xz,yy,yz,zz)>0;
    if (!ok && l<first) first=l;
  },Kokkos::Min<int>(bad));
  if (bad>=cells) return;
  int i=bad%ind.nx1+ind.is,j=(bad/ind.nx1)%ind.nx2+ind.js;
  int k=(bad/(ind.nx1*ind.nx2))%ind.nx3+ind.ks;
  int m=bad/(ind.nx1*ind.nx2*ind.nx3);
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
  auto input=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
      pack==bulk_ ? stage_bulk_ : stage_shell_);
  auto previous=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u1);
  auto rhs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u_rhs);
  std::ofstream out("cubed_sphere_first_bad_stage.csv"); out<<std::setprecision(17);
  out<<"# region="<<where<<", time="<<time<<", dt="<<bulk_->pmesh->dt
     <<", cycle="<<bulk_->pmesh->ncycle<<", stage="<<stage<<", block="<<m
     <<", first_kji="<<k<<","<<j<<","<<i<<"\n";
  std::ofstream coefficients("cubed_sphere_first_bad_stage_geometry.csv");
  coefficients<<std::setprecision(17)<<"k,j,i,dx1,dx2,dx3,x,y,z";
  for (int A=0;A<3;++A) for (int a=0;a<3;++a) coefficients<<",J"<<A<<a;
  for (int A=0;A<3;++A) for (int a=0;a<3;++a) for (int b=0;b<3;++b)
    coefficients<<",H"<<A<<a<<b;
  coefficients<<"\n";
  auto sizes=pack->pmb->mb_size.h_view;
  auto geom=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pcoord->z4c_geometry);
  out<<"k,j,i,x,y,z,variable,input,current,rk_register,rhs\n";
  for (int c=std::max(0,k-ind.ng);c<=std::min(ind.ke+ind.ng,k+ind.ng);++c)
  for (int b=std::max(0,j-ind.ng);b<=std::min(ind.je+ind.ng,j+ind.ng);++b)
  for (int a=std::max(0,i-ind.ng);a<=std::min(ind.ie+ind.ng,i+ind.ng);++a)
  {
    auto s=sizes(m);
    Real x[3]={s.x1min+(a-ind.is+0.5)*s.dx1,s.x2min+(b-ind.js+0.5)*s.dx2,
               s.x3min+(c-ind.ks+0.5)*s.dx3};
    Point p;
    if (geom.extent(0)) p=geom(m,c,b,a);
    else for (int A=0;A<3;++A) {p.x[A]=x[A]; p.J[A][A]=1;}
    coefficients<<c<<","<<b<<","<<a<<","<<s.dx1<<","<<s.dx2<<","<<s.dx3
                <<","<<p.x[0]<<","<<p.x[1]<<","<<p.x[2];
    for (int A=0;A<3;++A) for (int d=0;d<3;++d) coefficients<<","<<p.J[A][d];
    for (int A=0;A<3;++A) for (int d=0;d<3;++d) for (int e=0;e<3;++e)
      coefficients<<","<<p.H[A][d][e];
    coefficients<<"\n";
    for (int n=0;n<Z::nz4c;++n)
      out<<c<<","<<b<<","<<a<<","<<p.x[0]<<","<<p.x[1]<<","<<p.x[2]<<","<<n<<","<<input(m,n,c,b,a)
         <<","<<host(m,n,c,b,a)<<","<<previous(m,n,c,b,a)<<","<<rhs(m,n,c,b,a)<<"\n";
  }
  // An uncaught exception may terminate without unwinding the stream buffer.
  // Persist the complete record before reporting the inadmissible state.
  out.flush();
  coefficients.flush();
  if (!out || !coefficients) throw std::runtime_error("cubed sphere failed to write RK-stage stencil");
  out.close();
  coefficients.close();
  throw std::runtime_error("cubed sphere inadmissible RK stage; saved full stencil");
}

void Extension::Report(ParameterInput *pin) {
  using Z=z4c::Z4c;
  std::ofstream out("cubed_sphere_errors.csv"); out<<std::setprecision(17);
  out<<"region,time,blocks,plan_builds,l2,linf,H_l2,H_linf,M_l2,Z_l2,weyl_l2\n";
  std::ofstream regions("cubed_sphere_constraint_regions.csv");
  regions<<std::setprecision(17)
         <<"region,subset,time,cells,volume,H_l2,H_linf,M_l2,Z_l2,weyl_l2,"
         <<"max_H_x,max_H_y,max_H_z,min_alpha,min_chi\n";
  for (int region=0;region<2;++region) {
    auto *pack=region ? shell_->pmb_pack : bulk_;
    pack->pz4c->Z4cToADM(pack);
    pack->pz4c->ADMConstraints<3>(pack);
    pack->pz4c->Z4cWeyl<3>(pack);
    ReportRegions(pack,region ? "shell" : "bulk",regions);
    auto constraints=pack->pz4c->u_con, weyl=pack->pz4c->u_weyl;
    auto ind=pack->pmesh->mb_indcs; auto sizes=pack->pmb->mb_size;
    auto geometry=pack->pcoord->z4c_geometry; auto u=pack->pz4c->u0;
    int cells=pack->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
    int kind=initial_data_; Real amp=amplitude_,wn=wave_number_,width=pulse_width_;
    Real time=bulk_->pmesh->time,l2=0,linf=0,volume=0;
    Real h2=0,hmax=0,m2=0,z2=0,w2=0;
    Kokkos::parallel_reduce("cubed sphere error",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
        KOKKOS_LAMBDA(int l,Real &sum,Real &max,Real &vol,Real &hs,Real &hm,
                      Real &ms,Real &zs,Real &ws) {
      int i=l%ind.nx1+ind.is,j=(l/ind.nx1)%ind.nx2+ind.js;
      int k=(l/(ind.nx1*ind.nx2))%ind.nx3+ind.ks;
      int m=l/(ind.nx1*ind.nx2*ind.nx3);
      auto s=sizes.d_view(m); Real x[3]={s.x1min+(i-ind.is+0.5)*s.dx1,
        s.x2min+(j-ind.js+0.5)*s.dx2,s.x3min+(k-ind.ks+0.5)*s.dx3};
      Real v=s.dx1*s.dx2*s.dx3;
      if (region) {
        auto p=geometry(m,k,j,i);
        for (int a=0;a<3;++a) x[a]=p.x[a];
        Real det=p.J[0][0]*(p.J[1][1]*p.J[2][2]-p.J[1][2]*p.J[2][1])
          -p.J[0][1]*(p.J[1][0]*p.J[2][2]-p.J[1][2]*p.J[2][0])
          +p.J[0][2]*(p.J[1][0]*p.J[2][1]-p.J[1][1]*p.J[2][0]);
        v/=fabs(det);
      }
      Real exact[25]; TestState(x,time,kind,amp,wn,width,exact);
      if (kind==2) TestState(x,0,0,amp,wn,width,exact); // report disturbance, not an error
      Real err=0;
      for (int n=0;n<Z::nz4c;++n) {
        Real e=fabs(u(m,n,k,j,i)-exact[n]); err+=e*e; max=fmax(max,e);
      }
      sum+=v*err; vol+=v;
      Real H=constraints(m,Z::I_CON_H,k,j,i);
      hs+=v*H*H; hm=fmax(hm,fabs(H));
      Real M=constraints(m,Z::I_CON_M,k,j,i),Zc=constraints(m,Z::I_CON_Z,k,j,i);
      ms+=v*M; zs+=v*Zc;
      ws+=v*(SQR(weyl(m,0,k,j,i))+SQR(weyl(m,1,k,j,i)));
    },Kokkos::Sum<Real>(l2),Kokkos::Max<Real>(linf),Kokkos::Sum<Real>(volume),
       Kokkos::Sum<Real>(h2),Kokkos::Max<Real>(hmax),Kokkos::Sum<Real>(m2),
       Kokkos::Sum<Real>(z2),Kokkos::Sum<Real>(w2));
    if (linf>0.01) {
      auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
      Real max_error=0; int mm=0,nn=0,kk=0,jj=0,ii=0;
      for (int m=0;m<pack->nmb_thispack;++m)
      for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
      for (int i=ind.is;i<=ind.ie;++i) for (int n=0;n<25;++n) {
        Real baseline=(n==Z::I_Z4C_CHI || n==Z::I_Z4C_ALPHA || n==Z::I_Z4C_GXX ||
                       n==Z::I_Z4C_GYY || n==Z::I_Z4C_GZZ) ? 1 : 0;
        if (fabs(host(m,n,k,j,i)-baseline)>max_error) {
          max_error=fabs(host(m,n,k,j,i)-baseline); mm=m;nn=n;kk=k;jj=j;ii=i;
        }
      }
      std::cout<<"Largest disturbance in region "<<region<<": "<<Z::Z4c_names[nn]
               <<"="<<host(mm,nn,kk,jj,ii)<<", m/k/j/i="<<mm<<"/"<<kk<<"/"<<jj<<"/"<<ii<<std::endl;
    }
    out<<(region ? "shell" : "bulk")<<","<<time<<","<<pack->nmb_thispack
       <<","<<plan_builds_<<","<<sqrt(l2/volume)<<","<<linf<<","<<sqrt(h2/volume)
       <<","<<hmax<<","<<sqrt(m2/volume)<<","<<sqrt(z2/volume)<<","<<sqrt(w2/volume)<<"\n";
  }
}
}  // namespace cubed_sphere
