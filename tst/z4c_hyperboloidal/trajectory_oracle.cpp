// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Actual native Driver trajectory versus the preserved N12 CPU RK4 model.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "outputs/outputs.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/chi_policy.hpp"

namespace native=z4c_hyperboloidal;
namespace old=z4c::hyperboloidal;
constexpr int n=12,nf=31,nedge=6*n*n;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
struct Error {
  struct Site {
    int f=0,gid=0,i=0,j=0,k=0,step=0,rank=0;
    double actual=0,expected=0,x[3]{};
    void Broadcast(int root) {
      int index[7]={f,gid,i,j,k,step,rank};
      double values[5]={actual,expected,x[0],x[1],x[2]};
      MPI_Bcast(index,7,MPI_INT,root,MPI_COMM_WORLD);
      MPI_Bcast(values,5,MPI_DOUBLE,root,MPI_COMM_WORLD);
      f=index[0];gid=index[1];i=index[2];j=index[3];k=index[4];step=index[5];rank=index[6];
      actual=values[0];expected=values[1];for(int a=0;a<3;++a)x[a]=values[a+2];
    }
  } absolute_site,scaled_site;
  double absolute=0,scaled=0,ratio=0;
  double weighted_square=0,weight_sum=0;
  std::uint64_t count=0,failures=0;
  std::array<std::uint64_t,nf> field_failures{};
  std::array<double,nf> field_absolute{},field_scaled{},field_square{},field_weight{};
  void Compare(double actual,double expected,int f,int gid,int i,int j,int k,
               int step,const char *label,const double x[3],double weight) {
    const double tolerance=(f==0?1e-14:1e-12)+5e-10*std::abs(expected);
    const double delta=std::abs(actual-expected);
    const bool finite=std::isfinite(actual)&&std::isfinite(expected);
    if (!finite||delta>tolerance) {
      if (failures<8) std::cerr<<std::setprecision(17)<<label<<" mismatch step="<<step
          <<" rank="<<global_variable::my_rank<<" gid="<<gid<<" ijk="<<i<<','<<j<<','<<k
          <<" field="<<f<<" actual="<<actual<<" expected="<<expected
          <<" delta="<<delta<<" tolerance="<<tolerance<<'\n';
      ++failures;++field_failures[f];
    }
    const double ae=finite?delta:std::numeric_limits<double>::infinity(),
                 se=finite?delta/(1+std::abs(expected)):std::numeric_limits<double>::infinity();
    const Site site{f,gid,i,j,k,step,global_variable::my_rank,actual,expected,{x[0],x[1],x[2]}};
    if (count==0||ae>absolute)absolute_site=site;
    if (count==0||se>scaled)scaled_site=site;
    absolute=std::max(absolute,ae);scaled=std::max(scaled,se);
    ratio=std::max(ratio,finite?delta/tolerance:std::numeric_limits<double>::infinity());++count;
    field_absolute[f]=std::max(field_absolute[f],finite?delta:std::numeric_limits<double>::infinity());
    field_scaled[f]=std::max(field_scaled[f],se);
    Require(std::isfinite(weight)&&weight>0,"invalid reference error quadrature weight");
    weighted_square+=weight*delta*delta;weight_sum+=weight;
    field_square[f]+=weight*delta*delta;field_weight[f]+=weight;
  }
  void Global() {
    const double local_absolute=absolute,local_scaled=scaled;
    const auto local_count=count;
    MPI_Allreduce(MPI_IN_PLACE,&absolute,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&scaled,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ratio,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&failures,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_failures.data(),nf,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_absolute.data(),nf,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_scaled.data(),nf,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_square.data(),nf,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_weight.data(),nf,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&weighted_square,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&weight_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    int owner=local_absolute==absolute&&local_count?global_variable::my_rank:global_variable::nranks;
    MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);absolute_site.Broadcast(owner);
    owner=local_scaled==scaled&&local_count?global_variable::my_rank:global_variable::nranks;
    MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);scaled_site.Broadcast(owner);
  }
  void Print(const char *label) const {
    std::cout<<label<<" comparisons="<<count<<" max_scaled="<<scaled<<" max_absolute="<<absolute
        <<" max_tolerance_ratio="<<ratio<<" failures="<<failures<<" weighted_rms="
        <<std::sqrt(weighted_square/weight_sum)<<" weight_sum="<<weight_sum<<'\n';
    for (int f=0;f<nf;++f) std::cout<<label<<" field="<<f<<" max_absolute="<<field_absolute[f]
        <<" max_scaled="<<field_scaled[f]<<" weighted_rms="<<std::sqrt(field_square[f]/field_weight[f])
        <<" failures="<<field_failures[f]<<'\n';
    for (bool scaled_metric:{false,true}) {
      const auto &s=scaled_metric?scaled_site:absolute_site;
      std::cout<<label<<" worst="<<(scaled_metric?"scaled":"absolute")<<" field="<<s.f
          <<" gid="<<s.gid<<" ijk="<<s.i<<','<<s.j<<','<<s.k<<" xyz="<<s.x[0]<<','<<s.x[1]<<','<<s.x[2]
          <<" step="<<s.step<<" rank="<<s.rank<<" actual="<<s.actual<<" expected="<<s.expected<<'\n';
    }
  }
};

constexpr int nr=8;
template<class Record>
std::array<Real,nr> RecordValues(const Record &r) {
  return {r.minimum,r.accepted_minimum,r.floor_count,r.nonpositive_count,
          r.nonfinite_count,r.max_correction,r.sum_correction,r.first_floor};
}
struct Snapshot {
  int step=0;Real time=0;
  std::vector<Real> bulk,edge,records,surface_records;
};
Snapshot Capture(const native::Z4cHyperboloidal &p,int step,Real time) {
  Snapshot out;out.step=step;out.time=time;
  out.bulk.resize(nf*p.npoints);out.edge.resize(nf*p.nsurface_points);
  out.records.resize(nr*p.npoints);out.surface_records.resize(nr*p.nsurface_points);
  const auto bulk=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
  const auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto records=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.chi_records);
  const auto surface_records=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  for (int point=0;point<p.npoints;++point) {
    const auto c=p.points.h_view(point);const auto record=RecordValues(records(point));
    for (int f=0;f<nf;++f)out.bulk[nf*point+f]=bulk(c.m,f,c.k,c.j,c.i);
    for (int f=0;f<nr;++f)out.records[nr*point+f]=record[f];
  }
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);const auto record=RecordValues(surface_records(point));
    for (int f=0;f<nf;++f)out.edge[nf*point+f]=edge(c.face,f,c.k,c.j);
    for (int f=0;f<nr;++f)out.surface_records[nr*point+f]=record[f];
  }
  return out;
}
TaskID AllTasks(const std::shared_ptr<TaskList> &list) {
  TaskID result(0);for(int i=1;i<=list->Size();++i)result=result|TaskID(i);return result;
}
struct StageRecord {Real compatibility=0,clock=0,chi=1e300,rhs_time=0,state_time=0;};
struct Observe {
  Mesh &mesh;native::Z4cHyperboloidal &p;
  int before=0,after=0,accepted=0;std::vector<StageRecord> stages;
  std::vector<Snapshot> snapshots;
  TaskStatus Poison(Driver *driver,int stage) {
    Require(stage==before%4+1&&mesh.ncycle==before/4,"native Driver stage order changed");++before;
    Require(driver->UsesClassicalRK4()&&mesh.dt==p.requested_dt,"native integrator/dt changed");
    const Real poison=std::numeric_limits<Real>::quiet_NaN();
    Kokkos::deep_copy(p.u_rhs,poison);Kokkos::deep_copy(p.s_rhs,poison);
    Kokkos::deep_copy(p.residual,poison);Kokkos::deep_copy(p.rhs_status,-777);
    const auto state=p.u0;const auto bounds=p.evolved_bounds.d_view;
    const int nk=p.storage[2],nj=p.storage[1],ni=p.storage[0];
    Kokkos::parallel_for("trajectory poison ghosts",Kokkos::RangePolicy<>(0,p.nblocks*nk*nj*ni),
      KOKKOS_LAMBDA(int q) {
        const int i=q%ni,j=q/ni%nj,k=q/(ni*nj)%nk,m=q/(ni*nj*nk);const auto b=bounds(m);
        if(i<b.lo[0]||i>b.hi[0]||j<b.lo[1]||j>b.hi[1]||k<b.lo[2]||k>b.hi[2])
          for(int f=0;f<nf;++f)state(m,f,k,j,i)=poison;
      });
    return TaskStatus::complete;
  }
  TaskStatus Record(Driver *driver,int stage) {
    Require(stage==after%4+1&&mesh.ncycle==after/4&&before==after+1,
            "native Driver skipped or repeated a stage");++after;
    const Real rhs_fraction[4]={0,.5,.5,1},state_fraction[4]={.5,.5,1,1};
    Require(driver->stage_time==mesh.time+mesh.dt*rhs_fraction[stage-1]
        &&driver->stage_state_time==mesh.time+mesh.dt*state_fraction[stage-1],
        "native classical RK4 stage clocks changed");
    StageRecord record{p.StageMaxCompatibility(),p.StageMaxClock(),1e300,
                       driver->stage_time,driver->stage_state_time};
    const auto bulk=p.u0;const auto edge=p.s0;const auto points=p.points.d_view;
    const auto faces=p.surface_points.d_view;Real bulk_min=1e300,edge_min=1e300;
    Kokkos::parallel_reduce("trajectory bulk minimum",Kokkos::RangePolicy<>(0,p.npoints),
      KOKKOS_LAMBDA(int point,Real &v) {const auto c=points(point);
        v=fmin(v,bulk(c.m,0,c.k,c.j,c.i));},Kokkos::Min<Real>(bulk_min));
    Kokkos::parallel_reduce("trajectory surface minimum",Kokkos::RangePolicy<>(0,p.nsurface_points),
      KOKKOS_LAMBDA(int point,Real &v) {const auto c=faces(point);
        v=fmin(v,edge(c.face,0,c.k,c.j));},Kokkos::Min<Real>(edge_min));
    record.chi=std::min(bulk_min,edge_min);stages.push_back(record);
    Require(std::isfinite(record.compatibility)&&record.compatibility<=1e-7,
            "native stage normal compatibility rejected");
    return TaskStatus::complete;
  }
  TaskStatus Accepted(Driver *driver,int stage) {
    Require(stage==1&&mesh.ncycle==accepted&&after==4*(accepted+1),
            "native accepted-step callback order changed");++accepted;
    if(accepted==1||accepted==8)snapshots.push_back(Capture(p,accepted,driver->stage_state_time));
    return TaskStatus::complete;
  }
};

void WriteBinary(const std::string &path,const std::vector<Real> &values) {
  std::ofstream file(path,std::ios::binary);file.write(
      reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Real));
  Require(file.good(),"cannot write complete trajectory snapshot");
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    native::NativeDonorSelector selector(mesh,p.origin);
    std::vector<int> offset(mesh.nmb_total+1);
    for(int gid=0;gid<mesh.nmb_total;++gid) {int count=1;
      for(int a=0;a<3;++a)count*=selector.Upper(gid,a)-selector.Lower(gid,a);
      offset[gid+1]=offset[gid]+count;}
    auto canonical=[&](int gid,int i,int j,int k) {
      const int wi=selector.Upper(gid,0)-selector.Lower(gid,0),
                wj=selector.Upper(gid,1)-selector.Lower(gid,1);
      return offset[gid]+((k-selector.Lower(gid,2))*wj+j-selector.Lower(gid,1))*wi
          +i-selector.Lower(gid,0);
    };
    std::vector<int> counts(global_variable::nranks),displacements(global_variable::nranks),
        edge_counts(global_variable::nranks),edge_offsets(global_variable::nranks);
    for(int rank=0;rank<global_variable::nranks;++rank) {
      const int first=mesh.gids_eachrank[rank],end=first+mesh.nmb_eachrank[rank];
      counts[rank]=offset[end]-offset[first];displacements[rank]=offset[first];
      for(int gid=first;gid<end;++gid)if(selector.Map(gid).outer_surface)edge_counts[rank]+=n*n;
      if(rank)edge_offsets[rank]=edge_offsets[rank-1]+edge_counts[rank-1];
    }
    Require(offset.back()==229392&&counts[global_variable::my_rank]==p.npoints,
            "native N12 workload changed");
    Require(p.nsurface_points==edge_counts[global_variable::my_rank]
        &&edge_offsets.back()+edge_counts.back()==nedge,
        "native N12 surface workload changed");
    std::vector<int> surface_coverage(nedge);
    for(int point=0;point<p.nsurface_points;++point) {
      const auto c=p.surface_points.h_view(point);
      Require(c.face>=0&&c.face<int(p.face_blocks.h_view.extent(0))
          &&c.j>=0&&c.j<n&&c.k>=0&&c.k<n,"invalid native surface point");
      const int gid=mesh.pmb_pack->gids+p.face_blocks.h_view(c.face);
      Require(gid>=mesh.pmb_pack->gids&&gid<mesh.pmb_pack->gids+p.nblocks,
          "surface point belongs to another rank");
      const auto &map=selector.Map(gid);
      Require(map.outer_surface&&map.face>=0&&map.face<6,"surface point has no outer leaf");
      ++surface_coverage[(map.face*n+c.k)*n+c.j];
    }
    MPI_Allreduce(MPI_IN_PLACE,surface_coverage.data(),nedge,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    for(int count:surface_coverage)Require(count==1,"duplicate or missing canonical surface point");
    const auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
    const auto sgeom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
    Observe observer{mesh,p};observer.snapshots.push_back(Capture(p,0,mesh.time));
    Outputs output(&pin,&mesh);Kokkos::Timer timer;Driver driver(&pin,&mesh,1000,&timer);
    Require(driver.UsesClassicalRK4()&&driver.nlim==8,"trajectory requires ordinary eight-step input");
    driver.Initialize(&mesh,&pin,&output,false);
    Require(p.StagePlansReady()&&mesh.dt==p.requested_dt,"native initialization omitted plans/exactdt");
    auto before=mesh.pmb_pack->tl_map.at("before_stagen"),after=mesh.pmb_pack->tl_map.at("after_stagen");
    auto accepted=mesh.pmb_pack->tl_map.at("after_timeintegrator");
    auto dep_before=AllTasks(before),dep_after=AllTasks(after),dep_accepted=AllTasks(accepted);
    before->AddTask([&](Driver *d,int stage){return observer.Poison(d,stage);},dep_before);
    after->AddTask([&](Driver *d,int stage){return observer.Record(d,stage);},dep_after);
    accepted->AddTask([&](Driver *d,int stage){return observer.Accepted(d,stage);},dep_accepted);
    driver.Execute(&mesh,&pin,&output,false);
    Require(mesh.ncycle==8&&observer.before==32&&observer.after==32&&observer.accepted==8&&observer.snapshots.size()==3,
            "original Driver did not complete exactly eight full RK4 steps");
    Require(mesh.dt_last_completed==p.requested_dt&&mesh.dt==p.requested_dt,"native dt changed");

    std::unique_ptr<old::SphericalTransfer3D> model;std::vector<int> native_id;
    std::vector<EndpointChiRecord> old_records,old_surface_records;
    std::vector<StageRecord> old_stages;std::vector<Real> global_weights,weights(p.npoints),surface_weights(nedge);
    if(!global_variable::my_rank) {
      cubed_sphere::CompositeMesh oldmesh(.1,
          {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,n,4,3,.08);
      oldmesh.evolved_overlap=3;oldmesh.RefineCartesian(13);oldmesh.amr_cartesian_core=true;
      oldmesh.outer_radial_endpoint_grid=true;
      old::CentralZ4cConfiguration config;config.inner_z4c_radius_M=20;config.compact_slicing_end=.97;
      auto parameters=config.Resolve().gauge;parameters.outer_lapse_damping=.25;
      model=std::make_unique<old::SphericalTransfer3D>(std::move(oldmesh),parameters);
      model->ko=p.ko;model->penalty=p.penalty;model->experimental_timestep=true;
      Require(model->grid.active.size()==229392&&model->scri.rows.size()==nedge,"legacy workload changed");
      old_records.resize(model->grid.active.size());old_surface_records.resize(nedge);
      model->trajectory_rhs_observer=[&](double compatibility,double clock) {
        const int stage=old_stages.size()%4;const Real fractions[4]={0,.5,.5,1};
        old_stages.push_back({compatibility,clock,1e300,model->time+p.requested_dt*fractions[stage],0});
      };
      model->trajectory_candidate_observer=[&](old::SphericalTransfer3D::V &bulk,
          old::SphericalTransfer3D::V &edge,Real time,bool accepted) {
        Require(!old_stages.empty()&&accepted==(old_stages.size()%4==0),"legacy candidate stage mismatch");
        Real minimum=1e300;
        for(size_t point=0;point<old_records.size();++point) {
          bulk[nf*point]=EndpointChiCandidate(bulk[nf*point],p.chi_floor,time,accepted,old_records[point]);
          minimum=std::min(minimum,bulk[nf*point]);
        }
        for(size_t point=0;point<old_surface_records.size();++point) {
          edge[nf*point]=EndpointChiCandidate(edge[nf*point],p.chi_floor,time,accepted,old_surface_records[point]);
          minimum=std::min(minimum,edge[nf*point]);
        }
        old_stages.back().chi=minimum;old_stages.back().state_time=time;
      };
      native_id.assign(model->grid.mesh.blocks.size(),-1);
      for(size_t old=0;old<native_id.size();++old) {
        const auto &b=model->grid.mesh.blocks[old];
        for(int gid=0;gid<mesh.nmb_total;++gid) {const auto &a=selector.Map(gid);
          bool match=a.face==b.face&&a.layer==b.layer;
          for(int d=0;d<3;++d)match=match&&a.lo[d]==b.lo[d]&&a.hi[d]==b.hi[d];
          if(match){Require(native_id[old]<0,"ambiguous leaf identity");native_id[old]=gid;}}
        Require(native_id[old]>=0,"native/legacy leaf bounds are not exact");
      }
      auto unique_ids=native_id;std::sort(unique_ids.begin(),unique_ids.end());
      Require(unique_ids.size()==size_t(mesh.nmb_total),"native/legacy leaf counts differ");
      for(int gid=0;gid<mesh.nmb_total;++gid)
        Require(unique_ids[gid]==gid,"native/legacy leaf matching is not one to one");
      global_weights.resize(offset.back());
      for(size_t point=0;point<model->grid.cells.size();++point) {
        const auto c=model->grid.cells[point];const int gid=native_id[c.block];const auto &map=selector.Map(gid);
        const Real local[3]={(c.i+map.node_offset[0])/n,(c.j+map.node_offset[1])/n,(c.k+map.node_offset[2])/n};
        Real x[3];map.Forward(local,x);for(int a=0;a<3;++a)
          Require(x[a]==model->grid.geometry[point].x[a],"native/legacy sampling mismatch");
        global_weights[canonical(gid,c.i,c.j,c.k)]=model->grid.volumes[point];
      }
      for(const auto &row:model->scri.rows) {
        const int face=model->grid.mesh.blocks[row.block].face;
        surface_weights[(face*n+row.k)*n+row.j]=
            cubed_sphere::PhysicalCellVolume(row.geometry,n)/row.radial_spacing;
      }
    }
    MPI_Scatterv(global_weights.data(),counts.data(),displacements.data(),MPI_DOUBLE,
        weights.data(),weights.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(surface_weights.data(),surface_weights.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    auto scatter=[&](const std::vector<Real> &global,std::vector<Real> &local,int width) {
      auto c=counts,d=displacements;for(auto &v:c)v*=width;for(auto &v:d)v*=width;
      MPI_Scatterv(global.data(),c.data(),d.data(),MPI_DOUBLE,local.data(),local.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    };
    auto dump=[&](const std::vector<Real> &local,int width,bool surface,const std::string &path) {
      auto c=surface?edge_counts:counts,d=surface?edge_offsets:displacements;
      for(auto &v:c)v*=width;for(auto &v:d)v*=width;
      std::vector<Real> global;if(!global_variable::my_rank)global.resize(width*(surface?nedge:offset.back()));
      MPI_Gatherv(local.data(),local.size(),MPI_DOUBLE,global.data(),c.data(),d.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      if(!global_variable::my_rank)WriteBinary(path,global);
    };
    std::uint64_t all_failures=0;
    for(const auto &snapshot:observer.snapshots) {
      std::vector<Real> global,expected(nf*p.npoints),expected_edge(nf*nedge),
          global_records,expected_records(nr*p.npoints),expected_surface_records(nr*nedge);
      Real reference_time=0;
      if(model) {
        while(model->steps<size_t(snapshot.step))model->Step3D(p.requested_dt);
        reference_time=model->time;global.resize(nf*offset.back());global_records.resize(nr*offset.back());
        for(size_t point=0;point<model->grid.cells.size();++point) {
          const auto c=model->grid.cells[point];const int target=canonical(native_id[c.block],c.i,c.j,c.k);
          const auto record=RecordValues(old_records[point]);
          for(int f=0;f<nf;++f)global[nf*target+f]=model->state[nf*point+f];
          for(int f=0;f<nr;++f)global_records[nr*target+f]=record[f];
        }
        for(size_t point=0;point<model->scri.rows.size();++point) {
          const auto &row=model->scri.rows[point];const int face=model->grid.mesh.blocks[row.block].face;
          const int target=(face*n+row.k)*n+row.j;const auto record=RecordValues(old_surface_records[point]);
          for(int f=0;f<nf;++f)expected_edge[nf*target+f]=model->scri.values[nf*point+f];
          for(int f=0;f<nr;++f)expected_surface_records[nr*target+f]=record[f];
        }
      }
      scatter(global,expected,nf);scatter(global_records,expected_records,nr);
      MPI_Bcast(expected_edge.data(),expected_edge.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(expected_surface_records.data(),expected_surface_records.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&reference_time,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      Require(snapshot.time==reference_time,"accepted native/reference times differ");
      Error bulk_error,edge_error;std::uint64_t record_failures=0,record_comparisons=0;
      Real native_floor_count=0,reference_floor_count=0,record_max_delta=0;
      Real minima[6]={1e300,1e300,1e300,1e300,1e300,1e300};
      auto compare_record=[&](const Real *actual,const Real *reference) {
        for(int f=0;f<nr;++f) {
          const Real tolerance=f<2?1e-14+5e-10*std::abs(reference[f]):0;
          const Real delta=std::abs(actual[f]-reference[f]);
          record_failures+=!std::isfinite(actual[f])||!std::isfinite(reference[f])||delta>tolerance;
          record_max_delta=std::max(record_max_delta,delta);++record_comparisons;
        }
        native_floor_count+=actual[2];reference_floor_count+=reference[2];
        minima[0]=std::min(minima[0],actual[0]);minima[1]=std::min(minima[1],reference[0]);
        minima[2]=std::min(minima[2],actual[1]);minima[3]=std::min(minima[3],reference[1]);
      };
      for(int point=0;point<p.npoints;++point) {
        const auto c=p.points.h_view(point);const int gid=mesh.pmb_pack->gids+c.m;
        Require(canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)-offset[mesh.pmb_pack->gids]==point,
                "native point order changed");
        for(int f=0;f<nf;++f)bulk_error.Compare(snapshot.bulk[nf*point+f],expected[nf*point+f],f,gid,
            c.i-p.origin,c.j-p.origin,c.k-p.origin,snapshot.step,"bulk",geom(point).x,weights[point]);
        compare_record(&snapshot.records[nr*point],&expected_records[nr*point]);
        minima[4]=std::min(minima[4],snapshot.bulk[nf*point]);minima[5]=std::min(minima[5],expected[nf*point]);
      }
      for(int point=0;point<p.nsurface_points;++point) {
        const auto c=p.surface_points.h_view(point);const int gid=mesh.pmb_pack->gids+p.face_blocks.h_view(c.face),
                  face=selector.Map(gid).face,target=(face*n+c.k)*n+c.j;
        for(int f=0;f<nf;++f)edge_error.Compare(snapshot.edge[nf*point+f],expected_edge[nf*target+f],f,gid,
            n,c.j,c.k,snapshot.step,"surface",sgeom(point).x,surface_weights[target]);
        compare_record(&snapshot.surface_records[nr*point],&expected_surface_records[nr*target]);
        minima[4]=std::min(minima[4],snapshot.edge[nf*point]);minima[5]=std::min(minima[5],expected_edge[nf*target]);
      }
      bulk_error.Global();edge_error.Global();
      MPI_Allreduce(MPI_IN_PLACE,&record_failures,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&record_comparisons,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&native_floor_count,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&reference_floor_count,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&record_max_delta,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,minima,6,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
      all_failures+=bulk_error.failures+edge_error.failures+record_failures;
      if(snapshot.step==0)Require(bulk_error.absolute==0&&edge_error.absolute==0,"initial states not bitwise equal");
      Require(native_floor_count==0&&reference_floor_count==0,"short trajectory unexpectedly activates chi floor");
      const auto name="step"+std::to_string(snapshot.step);
      if(!global_variable::my_rank) {
        std::cout<<std::setprecision(17);bulk_error.Print((name+"_bulk").c_str());edge_error.Print((name+"_surface").c_str());
        std::cout<<name<<" chi_records="<<record_comparisons<<" failures="<<record_failures
            <<" max_absolute="<<record_max_delta<<" native_floor_events="<<native_floor_count
            <<" reference_floor_events="<<reference_floor_count<<" time="<<snapshot.time<<std::endl;
        std::cout<<name<<" native_candidate_min="<<minima[0]<<" reference_candidate_min="<<minima[1]
            <<" native_accepted_min="<<minima[2]<<" reference_accepted_min="<<minima[3]
            <<" native_state_min="<<minima[4]<<" reference_state_min="<<minima[5]<<std::endl;
        if(snapshot.step) {WriteBinary("reference-"+name+"-bulk.bin",global);
          WriteBinary("reference-"+name+"-surface.bin",expected_edge);}
      }
      if(snapshot.step) {
        dump(snapshot.bulk,nf,false,"native-"+name+"-bulk.bin");
        dump(snapshot.edge,nf,true,"native-"+name+"-surface.bin");
        dump(snapshot.records,nr,false,"native-"+name+"-chi.bin");
        dump(snapshot.surface_records,nr,true,"native-"+name+"-surface-chi.bin");
      }
    }
    std::vector<Real> compatibility(32),clock(32),chi(32),reference_history(5*32);
    for(int i=0;i<32;++i) {compatibility[i]=observer.stages[i].compatibility;
      clock[i]=observer.stages[i].clock;chi[i]=observer.stages[i].chi;}
    MPI_Allreduce(MPI_IN_PLACE,compatibility.data(),32,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,clock.data(),32,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,chi.data(),32,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    if(model) {Require(old_stages.size()==32,"legacy skipped an RK stage");
      for(int i=0;i<32;++i) {const auto &r=old_stages[i];const Real v[5]={r.compatibility,r.clock,r.chi,r.rhs_time,r.state_time};
        for(int f=0;f<5;++f)reference_history[5*i+f]=v[f];}}
    MPI_Bcast(reference_history.data(),reference_history.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(int i=0;i<32;++i) {
      const auto *r=&reference_history[5*i];
      Require(compatibility[i]<=1e-7&&r[0]<=1e-7,"normal compatibility exceeds source acceptance");
      Require(observer.stages[i].rhs_time==r[3]&&observer.stages[i].state_time==r[4],"stage time mismatch");
      const Real actual[3]={compatibility[i],clock[i],chi[i]};
      for(int f=0;f<3;++f)all_failures+=std::abs(actual[f]-r[f])>(f==2?1e-14:1e-12)+5e-10*std::abs(r[f]);
      if(!global_variable::my_rank)std::cout<<std::setprecision(17)<<"stage_history step="<<i/4+1<<" stage="<<i%4+1
          <<" native_compatibility="<<compatibility[i]<<" reference_compatibility="<<r[0]
          <<" native_clock="<<clock[i]<<" reference_clock="<<r[1]<<" native_chi="<<chi[i]
          <<" reference_chi="<<r[2]<<" rhs_time="<<r[3]<<" state_time="<<r[4]<<'\n';
    }
    Require(all_failures==0,"complete trajectory exceeds adopted short gate");
    if(!global_variable::my_rank)std::cout<<"PASS original native Driver N12 trajectory through eight RK4 steps\n";
  }} catch(const std::exception &e) {std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);}
  Kokkos::finalize();MPI_Finalize();return 0;
}
