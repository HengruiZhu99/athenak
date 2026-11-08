// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only saved N12 replay. Bulk input is read once and scattered; production
// diagnostics retain native owned storage and sparse transport throughout.
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
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/remesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"
namespace native=z4c_hyperboloidal;
namespace old=z4c::hyperboloidal;
constexpr int n=12,nf=31,nedge=6*n*n,nc=19;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
void Read(const std::string &path,std::vector<Real> &values) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  Require(file.good() && file.tellg()==std::streampos(values.size()*sizeof(Real)),
          "saved checkpoint extent mismatch");
  file.seekg(0);file.read(reinterpret_cast<char *>(values.data()),values.size()*sizeof(Real));
  Require(file.good(),"saved checkpoint read failed");
}
std::vector<Real> JsonNumbers(const std::string &text,const std::string &key,int count) {
  auto pos=text.find('"'+key+'"');Require(pos!=std::string::npos,"missing reference key");
  pos=text.find(':',pos);Require(pos!=std::string::npos,"malformed reference key");++pos;
  std::vector<Real> out;
  for (int i=0;i<count;++i) {
    pos=text.find_first_not_of(" \t\r\n[,",pos);Require(pos!=std::string::npos,"truncated reference");
    std::size_t used=0;const Real value=std::stod(text.substr(pos),&used);
    Require(std::isfinite(value),"nonfinite reference");out.push_back(value);pos+=used;
  }
  if (count>1) {pos=text.find_first_not_of(" \t\r\n",pos);Require(text.at(pos)==']',"reference extent mismatch");}
  return out;
}
template<class C>std::array<Real,nc> Pack(const C &c,Real chi,Real lapse,Real theta) {
  return {chi,lapse,theta,c.hamiltonian,c.momentum[0],c.momentum[1],c.momentum[2],
      c.momentum_conformal_norm2,c.momentum_physical_norm2,c.physical_trace,
      c.null_residual,c.z4.determinant_residual,c.z4.tracefree_residual,c.z4.theta_physical,
      c.z4.z_covector[0],c.z4.z_covector[1],c.z4.z_covector[2],
      c.z4.z_conformal_norm2,c.z4.z_physical_norm2};
}
std::vector<Real> Owned(const native::Z4cHyperboloidal &p) {
  const auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto chi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto schi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  std::vector<Real> out(39*(p.npoints+p.nsurface_points));
  for (int q=0;q<p.npoints;++q) {
    const auto c=p.points.h_view(q);auto *v=out.data()+39*q;
    for (int f=0;f<31;++f)v[f]=u(c.m,f,c.k,c.j,c.i);native::PackChiRecord(chi(q),v+31);
  }
  for (int q=0;q<p.nsurface_points;++q) {
    const auto c=p.surface_points.h_view(q);auto *v=out.data()+39*(p.npoints+q);
    for (int f=0;f<31;++f)v[f]=s(c.face,f,c.k,c.j);native::PackChiRecord(schi(q),v+31);
  }
  return out;
}
template<class V>bool Sentinel(const V &view,Real value) {
  auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),view);
  for (std::size_t i=0;i<host.size();++i)if (host.data()[i]!=value)return false;
  return true;
}
struct DeviceSums {
  struct value_type {Real sum[10];};
  DvceArray1D<native::ConstraintPoint> points;
  DvceArray1D<native::mapped::Point> geometry;
  DvceArray1D<Real> weights,radius;
  DvceArray1D<int> outer,control;
  bool host_control=false;
  KOKKOS_INLINE_FUNCTION void init(value_type &r) const {for (int a=0;a<10;++a)r.sum[a]=0;}
  KOKKOS_INLINE_FUNCTION void join(value_type &a,const value_type &b) const {
    for (int f=0;f<10;++f)a.sum[f]+=b.sum[f];
  }
  KOKKOS_INLINE_FUNCTION void operator()(int q,value_type &result) const {
    const auto map=geometry(q);const Real *x=map.x;
    // Match legacy Diagnose literally, on the device, including its strict cut.
    const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    const Real weight=weights(q);const auto point=points(q);const auto c=point.constraints;
    if (!host_control) {radius(q)=r;outer(q)=r>.94;}
    const bool selected=host_control?control(q)!=0:r>.94;
    result.sum[0]+=weight;if (selected)result.sum[5]+=weight;
    const Real row[4]={weight*c.hamiltonian*c.hamiltonian,
        weight*c.momentum_physical_norm2,weight*point.theta*point.theta,
        weight*c.z4.z_physical_norm2};
    for (int f=0;f<4;++f) {
      result.sum[1+f]+=row[f];if(selected)result.sum[6+f]+=row[f];
    }
  }
};

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  int final_status=0;
  try {{
    Require(argc==3,"native_saved_state_gpu input checkpoint_directory (output in cwd)");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetString("job","basename","native_saved3551");
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;p.BuildStagePlans();mesh.dt=p.requested_dt;
    native::NativeDonorSelector selector(mesh,p.origin);
    std::vector<int> offset(mesh.nmb_total+1);
    for (int gid=0;gid<mesh.nmb_total;++gid) {
      int count=1;for (int a=0;a<3;++a) count*=selector.Upper(gid,a)-selector.Lower(gid,a);
      offset[gid+1]=offset[gid]+count;
    }
    auto canonical=[&](int gid,int i,int j,int k) {
      const int wi=selector.Upper(gid,0)-selector.Lower(gid,0),
                wj=selector.Upper(gid,1)-selector.Lower(gid,1);
      return offset[gid]+((k-selector.Lower(gid,2))*wj+j-selector.Lower(gid,1))*wi
          +i-selector.Lower(gid,0);
    };
    std::vector<int> counts(global_variable::nranks),displacements(global_variable::nranks);
    for (int rank=0;rank<global_variable::nranks;++rank) {
      const int first=mesh.gids_eachrank[rank],end=first+mesh.nmb_eachrank[rank];
      counts[rank]=nf*(offset[end]-offset[first]);displacements[rank]=nf*offset[first];
    }
    Require(offset.back()==229392,"native N12 workload differs");
    Require(counts[global_variable::my_rank]==nf*p.npoints,"native local point ordering differs");
    std::unique_ptr<old::SphericalTransfer3D> model;std::vector<int> native_id;
    std::uint64_t coordinate_mismatches=0;double coordinate_error=0;
    std::vector<double> global_weights,weights(p.npoints),surface_weights(nedge);
    std::vector<int> global_old_ids,old_ids(p.npoints);
    if (global_variable::my_rank==0) {
      cubed_sphere::CompositeMesh oldmesh(.1,
          {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,n,4,3,.08);
      oldmesh.evolved_overlap=3;oldmesh.RefineCartesian(13);
      oldmesh.amr_cartesian_core=true;oldmesh.outer_radial_endpoint_grid=true;
      old::CentralZ4cConfiguration config;config.inner_z4c_radius_M=20;
      config.compact_slicing_end=.97;auto parameters=config.Resolve().gauge;
      parameters.outer_lapse_damping=.25;
      model=std::make_unique<old::SphericalTransfer3D>(std::move(oldmesh),parameters);
      model->ko=p.ko;model->penalty=p.penalty;model->experimental_timestep=true;
      Require(model->grid.active.size()==229392&&model->scri.rows.size()==nedge,
              "legacy N12 workload differs");
      native_id.assign(model->grid.mesh.blocks.size(),-1);
      for (std::size_t old=0;old<native_id.size();++old) {
        const auto &b=model->grid.mesh.blocks[old];
        for (int gid=0;gid<mesh.nmb_total;++gid) {
          const auto &a=selector.Map(gid);bool match=a.face==b.face&&a.layer==b.layer;
          for (int d=0;d<3;++d) match=match&&std::abs(a.lo[d]-b.lo[d])<1e-14
              &&std::abs(a.hi[d]-b.hi[d])<1e-14;
          if (match) {Require(native_id[old]<0,"ambiguous legacy/native block");native_id[old]=gid;}
        }
        Require(native_id[old]>=0,"missing legacy/native block");
      }
      for (std::size_t point=0;point<model->grid.cells.size();++point) {
        const auto c=model->grid.cells[point];const auto &map=selector.Map(native_id[c.block]);
        const Real local[3]={(c.i+map.node_offset[0])/n,(c.j+map.node_offset[1])/n,
                             (c.k+map.node_offset[2])/n};
        Real x[3];map.Forward(local,x);
        for (int a=0;a<3;++a) {
          const double old=model->grid.geometry[point].x[a];
          coordinate_mismatches+=x[a]!=old;
          coordinate_error=std::max(coordinate_error,std::abs(x[a]-old));
        }
      }
      global_weights.resize(offset.back());global_old_ids.resize(offset.back());
      for (std::size_t point=0;point<model->grid.cells.size();++point) {
        const auto c=model->grid.cells[point];
        const int target=canonical(native_id[c.block],c.i,c.j,c.k);
        global_weights[target]=model->grid.DiagnosticVolume(point);global_old_ids[target]=point;
      }
      for (const auto &row:model->scri.rows) {
        const int face=model->grid.mesh.blocks[row.block].face;
        surface_weights[(face*n+row.k)*n+row.j]=
            cubed_sphere::PhysicalCellVolume(row.geometry,n)/row.radial_spacing;
      }
      std::cout<<std::setprecision(17)<<"coordinate_mismatches="<<coordinate_mismatches
          <<" coordinate_max_absolute="<<coordinate_error<<std::endl;
    }
    MPI_Bcast(&coordinate_mismatches,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    Require(coordinate_mismatches==0,"N12 native and legacy bulk sampling coordinates differ");
    std::vector<int> weight_counts=counts,weight_offsets=displacements;
    for (int rank=0;rank<global_variable::nranks;++rank) {weight_counts[rank]/=nf;weight_offsets[rank]/=nf;}
    MPI_Scatterv(global_weights.data(),weight_counts.data(),weight_offsets.data(),MPI_DOUBLE,
        weights.data(),weights.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(surface_weights.data(),surface_weights.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatterv(global_old_ids.data(),weight_counts.data(),weight_offsets.data(),MPI_INT,
        old_ids.data(),old_ids.size(),MPI_INT,0,MPI_COMM_WORLD);
      const Real poison=std::numeric_limits<Real>::quiet_NaN();
      std::vector<Real> global,local(nf*p.npoints),edge(nf*nedge);
      if (model) {
        Read(std::string(argv[2])+"/fields.bin",model->state);
        Read(std::string(argv[2])+"/boundary.bin",model->scri.values);
        global.resize(nf*offset.back());
        for (std::size_t q=0;q<model->grid.cells.size();++q) {
          const auto c=model->grid.cells[q];const int target=canonical(native_id[c.block],c.i,c.j,c.k);
          for (int f=0;f<nf;++f)global[nf*target+f]=model->state[nf*q+f];
        }
        for (int q=0;q<nedge;++q) {
          const auto &r=model->scri.rows[q];const int face=model->grid.mesh.blocks[r.block].face;
          for (int f=0;f<nf;++f)edge[nf*((face*n+r.k)*n+r.j)+f]=model->scri.values[nf*q+f];
        }
      }
      MPI_Scatterv(global.data(),counts.data(),displacements.data(),MPI_DOUBLE,
                   local.data(),local.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(edge.data(),edge.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      auto state=Kokkos::create_mirror_view(p.u0);Kokkos::deep_copy(state,poison);
      auto boundary=Kokkos::create_mirror_view(p.s0);
      for (int q=0;q<p.npoints;++q) {
        const auto c=p.points.h_view(q);const int gid=mesh.pmb_pack->gids+c.m;
        const int index=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)-offset[mesh.pmb_pack->gids];
        for (int f=0;f<nf;++f)state(c.m,f,c.k,c.j,c.i)=local[nf*index+f];
      }
      for (int q=0;q<p.nsurface_points;++q) {
        const auto c=p.surface_points.h_view(q);
        const int face=selector.Map(mesh.pmb_pack->gids+p.face_blocks.h_view(c.face)).face;
        for (int f=0;f<nf;++f)boundary(c.face,f,c.k,c.j)=edge[nf*((face*n+c.k)*n+c.j)+f];
      }
      Kokkos::deep_copy(p.u0,state);Kokkos::deep_copy(p.s0,boundary);
      Kokkos::deep_copy(p.residual,poison);p.InvalidateDiagnostics();
      mesh.ncycle=3551;mesh.time=1.6669886746517166;
      // Discard the test decoder's complete state before invoking native diagnostics.
      model.reset();global.clear();global.shrink_to_fit();
      Kokkos::deep_copy(p.u_rhs,13);Kokkos::deep_copy(p.s_rhs,14);
      Kokkos::deep_copy(p.u1,15);Kokkos::deep_copy(p.u_accum,16);
      Kokkos::deep_copy(p.s1,17);Kokkos::deep_copy(p.s_accum,18);
      const auto before=Owned(p);const auto version=p.StateVersion();
      const Real stage_compatibility=p.StageMaxCompatibility(),stage_clock=p.StageMaxClock();
      auto &diagnostics=p.GetDiagnostics();diagnostics.Refresh();
      const auto refreshes=diagnostics.GetStatistics().refreshes;
      diagnostics.Refresh();Require(diagnostics.GetStatistics().refreshes==refreshes,"cache miss");
      DvceArray1D<Real> radius("device cut radius",p.npoints);
      DvceArray1D<int> outer("device fixed cut",p.npoints),host_cut("host fixed cut",p.npoints);
      DeviceSums::value_type raw{},controlled{};
      Kokkos::parallel_reduce("native saved full and fixed-outer replay",
          Kokkos::RangePolicy<>(0,p.npoints),
          DeviceSums{diagnostics.Points(),p.geometry,diagnostics.Weights(),radius,outer,host_cut,false},raw);
      auto rgpu=Kokkos::create_mirror_view_and_copy(HostMemSpace(),radius);
      auto mgpu=Kokkos::create_mirror_view_and_copy(HostMemSpace(),outer);
      auto mhost=Kokkos::create_mirror_view(host_cut);
      auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
      auto points=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Points());
      auto w=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Weights());
      std::vector<Real> rhost(p.npoints);
      // Counts: all, positive_weight, device_outer, host_outer, near_any,
      // near_positive, device1_host0, device0_host1, weighted_1_0, weighted_0_1,
      // invalid_any, invalid_zero_weight, exact_device_cut, exact_host_cut.
      std::uint64_t audit[14]{};Real weight_delta=0,max_weight_delta=0;
      std::uint64_t weight_differences=0;
      const std::string rank=std::to_string(global_variable::my_rank);
      std::ofstream near("near-cut-rank"+rank+".csv");
      Require(bool(near),"cannot open cut audit");
      near<<"old_index,gid,k,j,i,x,y,z,device_radius,host_radius,weight,device_mask,host_mask\n";
      for (int q=0;q<p.npoints;++q) {
        const auto c=p.points.h_view(q);const int gid=mesh.pmb_pack->gids+c.m;
        const int index=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)-offset[mesh.pmb_pack->gids];
        const auto &x=geometry(q).x;
        rhost[q]=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);mhost(q)=rhost[q]>.94;
        const bool positive=w(q)>0;
        const bool close=std::abs(rgpu(q)-.94)<=32*std::numeric_limits<Real>::epsilon()
            || std::abs(rhost[q]-.94)<=32*std::numeric_limits<Real>::epsilon();
        const bool invalid=points(q).status!=native::PointStatus::success;
        ++audit[0];audit[1]+=positive;audit[2]+=mgpu(q);audit[3]+=mhost(q);
        audit[4]+=close;audit[5]+=close&&positive;
        audit[6]+=mgpu(q)&&!mhost(q);audit[7]+=!mgpu(q)&&mhost(q);
        audit[8]+=positive&&mgpu(q)&&!mhost(q);audit[9]+=positive&&!mgpu(q)&&mhost(q);
        audit[10]+=invalid;audit[11]+=invalid&&!positive;
        audit[12]+=rgpu(q)==.94;audit[13]+=rhost[q]==.94;
        weight_delta+=(Real(mgpu(q))-mhost(q))*w(q);
        if (w(q)!=weights[index])++weight_differences;
        max_weight_delta=std::max(max_weight_delta,std::abs(w(q)-weights[index]));
        if (close || mgpu(q)!=mhost(q)) {
          near<<old_ids[index]<<','<<gid<<','<<c.k-p.origin<<','<<c.j-p.origin<<','<<c.i-p.origin;
          near<<std::hexfloat;for (auto a:x)near<<','<<a;
          near<<','<<rgpu(q)<<','<<rhost[q]<<','<<w(q)<<std::defaultfloat
              <<','<<mgpu(q)<<','<<mhost(q)<<'\n';
        }
      }
      near.close();Require(bool(near),"cut audit write failed");
      Kokkos::deep_copy(host_cut,mhost);
      Kokkos::parallel_reduce("diagnostic host-cut control on device",
          Kokkos::RangePolicy<>(0,p.npoints),
          DeviceSums{diagnostics.Points(),p.geometry,diagnostics.Weights(),radius,outer,host_cut,true},controlled);
      MPI_Allreduce(MPI_IN_PLACE,raw.sum,10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,controlled.sum,10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,audit,14,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&weight_delta,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&weight_differences,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&max_weight_delta,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      auto history=p.DiagnosticHistorySums();
      MPI_Allreduce(MPI_IN_PLACE,history.data(),10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      const auto &horizon=diagnostics.FindHorizon();
      const auto mask=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.ExteriorMask());
      const auto after=Owned(p);
      int readonly=before.size()==after.size()
          && std::memcmp(before.data(),after.data(),before.size()*sizeof(Real))==0
          && p.StateVersion()==version && p.StageMaxCompatibility()==stage_compatibility
          && p.StageMaxClock()==stage_clock && Sentinel(p.u_rhs,13) && Sentinel(p.s_rhs,14)
          && Sentinel(p.u1,15) && Sentinel(p.u_accum,16) && Sentinel(p.s1,17) && Sentinel(p.s_accum,18);
      MPI_Allreduce(MPI_IN_PLACE,&readonly,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
      // Stable per-point records, explicit arrays avoid structure padding.
      // Header: 16-byte magic, u64 local count, u32 key-count=8,value-count=26.
      // Keys: old_active,gid,k,j,i,device_fixed,host_fixed,horizon_exterior.
      // Values: xyz,19 constraint/primitive scalars,weight,device_r,host_r,legacy_weight.
      std::ofstream dump("points-rank"+rank+".bin",std::ios::binary);
      const char magic[16]="NATIVE_DIAG_V1";const std::uint64_t count=p.npoints;
      const std::uint32_t shape[2]={8,26};
      dump.write(magic,16);dump.write(reinterpret_cast<const char *>(&count),8);
      dump.write(reinterpret_cast<const char *>(shape),8);
      for (int q=0;q<p.npoints;++q) {
        const auto c=p.points.h_view(q);const int gid=mesh.pmb_pack->gids+c.m;
        const int index=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)-offset[mesh.pmb_pack->gids];
        const std::int32_t keys[8]={old_ids[index],gid,c.k-p.origin,c.j-p.origin,c.i-p.origin,
                                    mgpu(q),mhost(q),mask(q)};
        Real values[26];for (int a=0;a<3;++a)values[a]=geometry(q).x[a];
        const auto packed=Pack(points(q).constraints,points(q).chi,points(q).lapse,points(q).theta);
        for (int f=0;f<19;++f)values[3+f]=packed[f];
        values[22]=w(q);values[23]=rgpu(q);values[24]=rhost[q];values[25]=weights[index];
        dump.write(reinterpret_cast<const char *>(keys),sizeof(keys));
        dump.write(reinterpret_cast<const char *>(values),sizeof(values));
      }
      dump.close();Require(bool(dump),"diagnostic point dump failed");
      std::ifstream input(std::string(argv[2])+"/time.json");
      const std::string text((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
      Require(JsonNumbers(text,"steps",1)[0]==3551 && JsonNumbers(text,"t",1)[0]==mesh.time,
              "pinned saved-state step/time mismatch");
      auto expected=JsonNumbers(text,"constraint_RMS",4),outside=JsonNumbers(text,"outer_constraint_RMS",4);
      expected.insert(expected.end(),outside.begin(),outside.end());
      const Real history_reference[8]={.0073257648025702423,.0039666959397409406,
          .00063679744011741466,.00074930798279592369,.0066650464364187327,
          .003785788884853934,.00062887813299441033,.0007488764374473728};
      bool strict=readonly&&audit[0]==229392&&audit[10]==0,history_pass=true;
      std::ofstream report;
      if (!global_variable::my_rank) {
        report.open("results.json");report<<std::setprecision(17)
          <<"{\"backend\":\""<<DevExeSpace::name()<<"\",\"ranks\":"<<global_variable::nranks
          <<",\"step\":3551,\"time\":"<<mesh.time<<",\"readonly\":"<<(readonly?"true":"false")
          <<",\"raw_device\":[";
        for (int f=0;f<10;++f)report<<(f?",":"")<<raw.sum[f];
        report<<"],\"raw_device_host_mask_control\":[";
        for (int f=0;f<10;++f)report<<(f?",":"")<<controlled.sum[f];
        report<<"],\"mask_counts\":[";for (int f=0;f<14;++f)report<<(f?",":"")<<audit[f];
        report<<"],\"mask_weight_difference\":"<<weight_delta
          <<",\"legacy_weight_differences\":"<<weight_differences
          <<",\"legacy_weight_max_absolute\":"<<max_weight_delta<<",\"strict_checks\":[";
      }
      for (int f=0;f<8;++f) {
        const int region=f/4,field=f%4;
        const Real value=std::sqrt(raw.sum[5*region+1+field]/raw.sum[5*region]);
        const Real delta=std::abs(value-expected[f]),gate=5e-11*(1+std::abs(expected[f]));
        const bool pass=std::isfinite(value)&&delta<=gate;strict&=pass;
        if (!global_variable::my_rank)report<<(f?",":"")<<"{\"region\":"<<region
          <<",\"field\":"<<field<<",\"actual\":"<<value<<",\"reference\":"<<expected[f]
          <<",\"difference\":"<<delta<<",\"gate\":"<<gate<<",\"passed\":"<<(pass?"true":"false")<<'}';
      }
      if (!global_variable::my_rank) report<<"],\"history_checks\":[";
      for (int f=0;f<8;++f) {
        const int region=f/4,field=f%4;
        const Real value=std::sqrt(history[5*region+1+field]/history[5*region]);
        const Real delta=std::abs(value-history_reference[f]),gate=5e-10+1e-6*std::abs(history_reference[f]);
        const bool pass=std::isfinite(value)&&delta<=gate;history_pass&=pass;
        if (!global_variable::my_rank)report<<(f?",":"")<<"{\"region\":"<<region
          <<",\"field\":"<<field<<",\"actual\":"<<value<<",\"reference\":"<<history_reference[f]
          <<",\"difference\":"<<delta<<",\"gate\":"<<gate<<",\"passed\":"<<(pass?"true":"false")<<'}';
      }
      const Real areal=3*horizon.areal_radius,mass=3*horizon.irreducible_mass;
      const bool ah=horizon.found&&horizon.complete_surface
          && std::abs(areal-1.9999860425879357)<=1e-10+1e-7*1.9999860425879357
          && std::abs(mass-.99999302129396783)<=1e-10+1e-7*.99999302129396783;
      const bool passed=strict&&history_pass&&ah;
      if (!global_variable::my_rank) {
        report<<"],\"horizon\":{\"found\":"<<(horizon.found?"true":"false")
          <<",\"complete\":"<<(horizon.complete_surface?"true":"false")
          <<",\"areal_radius_M\":"<<areal<<",\"irreducible_mass_M\":"<<mass
          <<",\"mean_coordinate_radius\":"<<horizon.mean_radius
          <<",\"expansion_rms\":"<<horizon.expansion_rms<<",\"iterations\":"<<horizon.iterations
          <<",\"passed\":"<<(ah?"true":"false")<<"},\"strict_replay_passed\":"<<(strict?"true":"false")
          <<",\"history_passed\":"<<(history_pass?"true":"false")
          <<",\"passed\":"<<(passed?"true":"false")<<"}\n";
        report.close();Require(bool(report),"result JSON write failed");
        const auto &s=diagnostics.Surface();std::ofstream shape_file("surface.txt");
        shape_file<<std::setprecision(17)<<s.lmax<<'\n';for (auto v:s.center)shape_file<<v<<' ';shape_file<<'\n';
        for (const auto *v:{&s.a0,&s.ac,&s.as}) {for (auto x:*v)shape_file<<x<<' ';shape_file<<'\n';}
        shape_file.close();Require(bool(shape_file),"harmonic surface write failed");
        std::cout<<(passed?"PASS":"FAIL")<<" native saved3551 device replay strict="<<strict
            <<" history="<<history_pass<<" horizon="<<ah<<" readonly="<<readonly<<'\n';
      }
      final_status=passed?0:2;
  }} catch (const std::exception &e) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return final_status;
}
