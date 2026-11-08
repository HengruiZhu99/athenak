// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Actual native AMR lifecycle; see amr_lifecycle_contract.md for predeclared gates.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"

namespace h=z4c_hyperboloidal;
namespace {
int operation=0;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
bool Selected(const chart_atlas::BlockMap &map) {
  if (map.chart_id==0)
    return map.hi[0]<=-.1/3+1e-15 && map.hi[1]<=-.1/3+1e-15 &&
           map.hi[2]<=-.1/3+1e-15;
  return map.chart_id==1 && (map.layer==0 || map.layer==11);
}
void Criterion(MeshBlockPack *pack) {
  auto &mesh=*pack->pmesh;
  for (int m=0;m<pack->nmb_thispack;++m) {
    const int gid=pack->gids+m;
    const auto map=pack->pmb->mb_map.h_view(m);
    const int level=mesh.lloc_eachmb[gid].level;
    mesh.pmr->refine_flag.h_view(gid)=Selected(map)
        ? (operation==1 && level==mesh.root_level ? 1
           : operation==2 && level==mesh.root_level+1 ? -1 : 0) : 0;
  }
}
Real Polynomial(const chart_atlas::BlockMap &map,const Real local[3]) {
  Real q[3];for (int a=0;a<3;++a)q[a]=map.lo[a]+local[a]*(map.hi[a]-map.lo[a]);
  return .001*(1+map.chart_id+.2*q[0]+.3*q[1]+.4*q[2]+
               .1*q[0]*q[0]+.03*q[1]*q[1]*q[2]+.02*q[2]*q[2]*q[2]);
}
h::ChiRecord History(int gid,int i,int j,int k) {
  const int key=(gid*19+i*7+j*11+k*13)%256;
  return {.125+key/4096.,.25+key/4096.,Real(1+key%4),Real(2+key%3),
          Real(3+key%2),.5+key/512.,1+key/8.,.125+key/8192.};
}
void Seed(Mesh &mesh) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  auto chi=Kokkos::create_mirror_view(p.chi_records);
  auto schi=Kokkos::create_mirror_view(p.surface_chi_records);
  const auto maps=mesh.pmb_pack->pmb->mb_map.h_view;
  const auto values=[&](const chart_atlas::BlockMap &map,const Real q[3],Real out[3]) {
    Real x[3];map.Forward(q,x);
    out[0]=0x1p-8;out[1]=Polynomial(map,q);
    out[2]=.001*(1+x[0]+x[0]*x[0]+x[1]*x[2]);
  };
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);const auto map=maps(c.m);
    const Real q[3]={(c.i-p.origin+map.node_offset[0])/p.n[0],
                    (c.j-p.origin+.5)/p.n[1],(c.k-p.origin+.5)/p.n[2]};
    Real v[3];values(map,q,v);
    for (int a=0;a<3;++a)bulk(c.m,19+a,c.k,c.j,c.i)=v[a];
    // Dormant owned values expose copying all slots on unchanged leaves.
    for (int a=0;a<3;++a)bulk(c.m,28+a,c.k,c.j,c.i)=(a+1)*0x1p-16;
    chi(point)=History(mesh.pmb_pack->gids+c.m,c.i,c.j,c.k);
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);
    const int m=p.face_blocks.h_view(c.face);const auto map=maps(m);
    const Real q[3]={1,(c.j+.5)/p.n[1],(c.k+.5)/p.n[2]};
    Real v[3];values(map,q,v);
    for (int a=0;a<3;++a)edge(c.face,19+a,c.k,c.j)=v[a];
    schi(point)=History(mesh.pmb_pack->gids+m,p.n[0],c.j,c.k);
  }
  Kokkos::deep_copy(p.u0,bulk);Kokkos::deep_copy(p.s0,edge);
  Kokkos::deep_copy(p.chi_records,chi);Kokkos::deep_copy(p.surface_chi_records,schi);
}
void Poison(h::Z4cHyperboloidal &p) {
  const Real nan=std::numeric_limits<Real>::quiet_NaN();
  auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  for (int m=0;m<p.nblocks;++m) {
    const auto b=p.evolved_bounds.h_view(m);
    for (int k=0;k<p.storage[2];++k)for (int j=0;j<p.storage[1];++j)
    for (int i=0;i<p.storage[0];++i)
      if (i<b.lo[0] || i>b.hi[0] || j<b.lo[1] || j>b.hi[1] ||
          k<b.lo[2] || k>b.hi[2])for (int f=0;f<31;++f)bulk(m,f,k,j,i)=nan;
  }
  Kokkos::deep_copy(p.u0,bulk);
  for (auto *v:{&p.u1,&p.u_rhs,&p.u_accum,&p.residual})Kokkos::deep_copy(*v,nan);
  for (auto *v:{&p.s1,&p.s_rhs,&p.s_accum})Kokkos::deep_copy(*v,nan);
}
std::array<Real,8> Totals(const h::Z4cHyperboloidal &p,bool surface) {
  const auto data=Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      surface ? p.surface_chi_records : p.chi_records);
  std::array<Real,8> out={1e300,1e300,0,0,0,0,0,1e300};
  for (std::size_t n=0;n<data.extent(0);++n) {
    Real q[8];h::PackChiRecord(data(n),q);
    for (int a=0;a<8;++a) {
      Require(std::isfinite(q[a]),"unwritten chi history member after AMR");
      if (a==0 || a==1 || a==7)out[a]=std::min(out[a],q[a]);
      else if (a==5)out[a]=std::max(out[a],q[a]);
      else out[a]+=q[a];
    }
  }
  for (int a=0;a<8;++a)MPI_Allreduce(MPI_IN_PLACE,&out[a],1,MPI_DOUBLE,
      a==0 || a==1 || a==7 ? MPI_MIN : a==5 ? MPI_MAX : MPI_SUM,MPI_COMM_WORLD);
  return out;
}
void DumpAndCheck(Mesh &mesh,int phase) {
  const auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  const auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto schi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  const auto maps=mesh.pmb_pack->pmb->mb_map.h_view;
  const std::string stem="phase"+std::to_string(phase)+"-rank"+
      std::to_string(global_variable::my_rank);
  std::ofstream file(stem+".bin",std::ios::binary),meta(stem+".txt");
  Require(bool(file) && bool(meta),"cannot create AMR snapshot");
  std::uint64_t checks[4]={0,0,0,std::uint64_t(p.nfaces==0)};
  Real maximum=0;
  const auto write=[&](int m,int kind,int i,int j,int k,const Real *v) {
    const auto &loc=mesh.lloc_eachmb[mesh.pmb_pack->gids+m];
    const std::int32_t key[9]={loc.chart_id,loc.level,loc.lx1,loc.lx2,loc.lx3,
                               kind,k,j,i};
    file.write(reinterpret_cast<const char *>(key),sizeof(key));
    file.write(reinterpret_cast<const char *>(v),39*sizeof(Real));
    for (int f=0;f<39;++f)Require(std::isfinite(v[f]),"nonfinite AMR owned field/record");
    Require(std::abs(v[19]-0x1p-8)<=512*std::numeric_limits<Real>::epsilon(),
            "AMR constant reproduction failed");
    ++checks[kind];
    const auto map=maps(m);
    if (phase && Selected(map)) {
      const Real q[3]={kind ? 1 : (i-p.origin+map.node_offset[0])/p.n[0],
          (j-(kind ? 0 : p.origin)+.5)/p.n[1],
          (k-(kind ? 0 : p.origin)+.5)/p.n[2]};
      bool check=kind || (map.chart_id>0 && (phase==1 ||
          (i>=p.origin && i<p.origin+p.n[0] && j>=p.origin && j<p.origin+p.n[1] &&
           k>=p.origin && k<p.origin+p.n[2])));
      if (map.chart_id==0) {
        Real x[3];map.Forward(q,x);check=true;
        for (int a=0;a<3;++a)check&=x[a]>=-.1+.25*(.2/3) &&
                                         x[a]<=-.1+.75*(.2/3);
      }
      if (check) {
        const Real error=std::abs(v[20]-Polynomial(map,q));
        maximum=std::max(maximum,error);++checks[2];
        Require(error<=256*std::numeric_limits<Real>::epsilon()*32768*.02,
                "AMR same-family polynomial reproduction failed");
      }
    }
  };
  for (int m=0;m<p.nblocks;++m) {
    const auto &loc=mesh.lloc_eachmb[mesh.pmb_pack->gids+m];
    meta<<loc.chart_id<<' '<<loc.level<<' '<<loc.lx1<<' '<<loc.lx2<<' '<<loc.lx3<<'\n';
  }
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);Real v[39];
    for (int f=0;f<31;++f)v[f]=bulk(c.m,f,c.k,c.j,c.i);
    h::PackChiRecord(chi(point),v+31);write(c.m,0,c.i,c.j,c.k,v);
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);Real v[39];
    for (int f=0;f<31;++f)v[f]=edge(c.face,f,c.k,c.j);
    h::PackChiRecord(schi(point),v+31);
    write(p.face_blocks.h_view(c.face),1,p.n[0],c.j,c.k,v);
  }
  Require(bool(file) && bool(meta),"AMR snapshot write failed");
  MPI_Allreduce(MPI_IN_PLACE,checks,4,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&maximum,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  Real compatibility=p.StageMaxCompatibility();
  MPI_Allreduce(MPI_IN_PLACE,&compatibility,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  Require(std::isfinite(compatibility) && compatibility<=1e-7,"AMR compatibility gate failed");
  Require(global_variable::nranks<8 || checks[3]>0,"missing zero-scri-rank AMR case");
  Require(phase==0 || checks[2]>1000,"insufficient changed AMR polynomial coverage");
  if (!global_variable::my_rank)std::cout<<std::setprecision(17)
      <<"AMR_REPORT {\"phase\":"<<phase<<",\"blocks\":"<<mesh.nmb_total
      <<",\"bulk_points\":"<<checks[0]<<",\"scri_points\":"<<checks[1]
      <<",\"polynomial_checks\":"<<checks[2]<<",\"zero_scri_ranks\":"<<checks[3]
      <<",\"max_polynomial_error\":"<<maximum<<",\"compatibility\":"<<compatibility<<"}\n";
}
// Optional accepted-state diagnostic cache gate. It uses actual AMR below, not
// synthetic maps or direct storage reset, and keeps the mathematical oracle
// separate: a new Diagnostics instance checks only cache/plan reconstruction.
std::vector<Real> OwnedState(const h::Z4cHyperboloidal &p) {
  const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto schi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  std::vector<Real> values(39*(p.npoints+p.nsurface_points));
  for (int q=0;q<p.npoints;++q) {
    const auto c=p.points.h_view(q);auto *v=values.data()+39*q;
    for (int f=0;f<31;++f)v[f]=u(c.m,f,c.k,c.j,c.i);
    h::PackChiRecord(chi(q),v+31);
  }
  for (int q=0;q<p.nsurface_points;++q) {
    const auto c=p.surface_points.h_view(q);auto *v=values.data()+39*(p.npoints+q);
    for (int f=0;f<31;++f)v[f]=s(c.face,f,c.k,c.j);
    h::PackChiRecord(schi(q),v+31);
  }
  return values;
}
std::vector<Real> DiagnosticValues(const h::Diagnostics &diagnostic) {
  const auto points=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostic.Points());
  const auto weights=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostic.Weights());
  std::vector<Real> values;values.reserve(23*points.extent(0));
  for (std::size_t q=0;q<points.extent(0);++q) {
    const auto &p=points(q);const auto &c=p.constraints;const auto &z=c.z4;
    // Compare named scalars, excluding implementation-dependent struct padding.
    values.insert(values.end(),{Real(p.status),Real(c.valid),Real(z.valid),
        p.chi,p.lapse,p.theta,c.hamiltonian,c.momentum[0],c.momentum[1],c.momentum[2],
        c.momentum_conformal_norm2,c.momentum_physical_norm2,c.physical_trace,
        c.null_residual,z.determinant_residual,z.tracefree_residual,z.theta_physical,
        z.z_covector[0],z.z_covector[1],z.z_covector[2],z.z_conformal_norm2,
        z.z_physical_norm2,weights(q)});
  }
  return values;
}
void Same(const std::vector<Real> &a,const std::vector<Real> &b,const char *message) {
  Require(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(Real))==0,
          message);
}
template<class View>void Sentinel(const View &view,Real expected) {
  const auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),view);
  for (std::size_t q=0;q<host.size();++q)
    Require(host.data()[q]==expected,"AMR diagnostics changed RK/RHS workspace");
}
void CheckDiagnostics(ParameterInput &pin,Mesh &mesh,int phase) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  auto &cached=p.GetDiagnostics();
  Require(cached.GetStatistics().refreshes==std::uint64_t(phase),
          "diagnostic refresh count before AMR phase changed");
  Kokkos::deep_copy(p.u_rhs,13);Kokkos::deep_copy(p.s_rhs,14);
  Kokkos::deep_copy(p.u1,15);Kokkos::deep_copy(p.u_accum,16);
  Kokkos::deep_copy(p.s1,17);Kokkos::deep_copy(p.s_accum,18);
  Kokkos::deep_copy(p.rhs_status,73);Kokkos::deep_copy(p.rhs_gap,19);
  Kokkos::deep_copy(p.rhs_cfl,20);
  const auto before=OwnedState(p);const auto version=p.StateVersion();
  const Real compatibility=p.StageMaxCompatibility(),clock=p.StageMaxClock();
  cached.Refresh();
  Require(cached.GetStatistics().refreshes==std::uint64_t(phase+1),
          "accepted AMR state did not invalidate diagnostic cache");
  Require(cached.Points().extent_int(0)==p.npoints && cached.Weights().extent_int(0)==p.npoints
          && cached.ExteriorMask().extent_int(0)==p.npoints,
          "diagnostic extents still describe the old mesh");
  const auto values=DiagnosticValues(cached);
  cached.Refresh();
  Require(cached.GetStatistics().refreshes==std::uint64_t(phase+1),
          "unchanged AMR state missed diagnostic cache");
  Same(values,DiagnosticValues(cached),"cached diagnostics changed on repeat");
  {
    h::Diagnostics cold(mesh.pmb_pack,&p,&pin);cold.Refresh();
    Require(cold.GetStatistics().refreshes==1,"fresh diagnostics did not refresh once");
    Same(values,DiagnosticValues(cold),"cached/fresh AMR constraints or weights differ");
  }
  Same(before,OwnedState(p),"AMR diagnostics changed owned fields or chi history");
  Require(p.StateVersion()==version && p.StageMaxCompatibility()==compatibility
          && p.StageMaxClock()==clock,"AMR diagnostics changed accepted-stage bookkeeping");
  Sentinel(p.u_rhs,13);Sentinel(p.s_rhs,14);Sentinel(p.u1,15);Sentinel(p.u_accum,16);
  Sentinel(p.s1,17);Sentinel(p.s_accum,18);Sentinel(p.rhs_status,73);
  Sentinel(p.rhs_gap,19);Sentinel(p.rhs_cfl,20);
  std::uint64_t counts[3]={std::uint64_t(values.size()),std::uint64_t(before.size()),
      std::uint64_t(p.u_rhs.size()+p.s_rhs.size()+p.u1.size()+p.u_accum.size()
          +p.s1.size()+p.s_accum.size()+p.rhs_status.size()+p.rhs_gap.size()+p.rhs_cfl.size())};
  MPI_Allreduce(MPI_IN_PLACE,counts,3,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  if (!global_variable::my_rank)std::cout<<"DIAGNOSTICS_REPORT {\"phase\":"<<phase
      <<",\"refreshes\":"<<cached.GetStatistics().refreshes
      <<",\"constraint_weight_scalars\":"<<counts[0]<<",\"owned_scalars\":"<<counts[1]
      <<",\"rk_status_values\":"<<counts[2]
      <<",\"fresh_exact\":true,\"repeat_cached\":true,\"readonly\":true}\n";
}
}  // namespace
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);
  try {{
    Require(argc==2 || (argc==3 && std::string(argv[2])=="--diagnostics"),
            "usage: amr_lifecycle N12-initial-input [--diagnostics]");
    const bool diagnostics=argc==3;
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetString("mesh_refinement","refinement","adaptive");
    pin.SetInteger("mesh_refinement","num_levels",2);
    pin.SetInteger("mesh_refinement","max_nmb_per_rank",160/global_variable::nranks+8);
    pin.SetInteger("mesh_refinement","refinement_interval",0);
    pin.SetString("amr_criterion0","method","user");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pmr=new MeshRefinement(&mesh,&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    mesh.pgen->user_ref_func=Criterion;
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    Seed(mesh);
    pin.SetString("time","evolution","dynamic");
    pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",1);pin.SetRealExact("time","tlim",10*p.requested_dt);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);Outputs output(&pin,&mesh);
    driver.Initialize(&mesh,&pin,&output,false);
    const auto bulk_history=Totals(p,false),surface_history=Totals(p,true);
    Require(mesh.nmb_total==106,"initial N12 AMR fixture topology changed");
    if (diagnostics)CheckDiagnostics(pin,mesh,0);
    DumpAndCheck(mesh,0);
    for (operation=1;operation<=2;++operation) {
      Poison(p);
      const auto sequence=mesh.GetAMRLoadBalanceUpdateSeq();
      mesh.pmr->AdaptiveMeshRefinement(&driver,&pin);
      Require(mesh.nmb_total==(operation==1 ? 127 : 106),"AMR refinement/derefinement missing");
      Require(mesh.GetAMRLoadBalanceUpdateSeq()==sequence+1,"AMR topology sequence not advanced");
      Require(p.StagePlansReady() && p.IsInitialized(),"AMR plans/state not ready");
      Require(p.dtnew==p.requested_dt,"AMR failed to refresh fixed timestep");
      Require(Totals(p,false)==bulk_history,"AMR changed bulk historical totals/extrema");
      Require(Totals(p,true)==surface_history,"AMR changed scri historical totals/extrema");
      if (diagnostics)CheckDiagnostics(pin,mesh,operation);
      DumpAndCheck(mesh,operation);
    }
    if (!global_variable::my_rank)std::cout<<"PASS native field AMR lifecycle\n";
  }} catch (const std::exception &error) {
    std::cerr<<"AMR lifecycle rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
