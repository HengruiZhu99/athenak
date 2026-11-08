// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Full native stage RHS versus the independent preserved N12 CPU model.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"

namespace native=z4c_hyperboloidal;
namespace old=z4c::hyperboloidal;
constexpr int n=12,nf=31,nedge=6*n*n;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
double Perturb(const double *x,int f,int scenario) {
  if (!scenario)return 0;
  return 2e-6*(f+1)/nf*(x[0]+.3*x[1]*x[1]-.2*x[2]*x[2]*x[2]
      +.1*x[0]*x[1]+std::sin(3*x[2]));
}
struct Error {
  struct Site {
    int f=0,gid=0,i=0,j=0,k=0,scenario=0,rank=0;
    double actual=0,expected=0,x[3]{};
    void Broadcast(int root) {
      int index[7]={f,gid,i,j,k,scenario,rank};
      double values[5]={actual,expected,x[0],x[1],x[2]};
      MPI_Bcast(index,7,MPI_INT,root,MPI_COMM_WORLD);
      MPI_Bcast(values,5,MPI_DOUBLE,root,MPI_COMM_WORLD);
      f=index[0];gid=index[1];i=index[2];j=index[3];k=index[4];scenario=index[5];rank=index[6];
      actual=values[0];expected=values[1];for(int a=0;a<3;++a)x[a]=values[a+2];
    }
  } absolute_site,scaled_site;
  double absolute=0,scaled=0,ratio=0;
  double weighted_square=0,weight_sum=0;
  std::uint64_t count=0,failures=0;
  std::array<std::uint64_t,nf> field_failures{};
  std::array<double,nf> field_absolute{},field_scaled{},field_square{},field_weight{};
  void Compare(double actual,double expected,int f,int gid,int i,int j,int k,
               int scenario,const char *label,const double x[3],double weight) {
    const double tolerance=(f==0?1e-14:1e-12)+5e-10*std::abs(expected);
    const double delta=std::abs(actual-expected);
    const bool finite=std::isfinite(actual)&&std::isfinite(expected);
    if (!finite||delta>tolerance) {
      if (failures<8) std::cerr<<std::setprecision(17)<<label<<" mismatch scenario="<<scenario
          <<" rank="<<global_variable::my_rank<<" gid="<<gid<<" ijk="<<i<<','<<j<<','<<k
          <<" field="<<f<<" actual="<<actual<<" expected="<<expected
          <<" delta="<<delta<<" tolerance="<<tolerance<<'\n';
      ++failures;++field_failures[f];
    }
    const double ae=finite?delta:std::numeric_limits<double>::infinity(),
                 se=finite?delta/(1+std::abs(expected)):std::numeric_limits<double>::infinity();
    const Site site{f,gid,i,j,k,scenario,global_variable::my_rank,actual,expected,{x[0],x[1],x[2]}};
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
          <<" scenario="<<s.scenario<<" rank="<<s.rank<<" actual="<<s.actual<<" expected="<<s.expected<<'\n';
    }
  }
};

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;p.BuildStagePlans();
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1000,&timer);
    driver.integrator="rk4_classical";driver.nexp_stages=4;mesh.dt=p.requested_dt;
    driver.stage_time=mesh.time;driver.stage_state_time=mesh.time+mesh.dt/2;
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
      global_weights.resize(offset.back());
      for (std::size_t point=0;point<model->grid.cells.size();++point) {
        const auto c=model->grid.cells[point];
        global_weights[canonical(native_id[c.block],c.i,c.j,c.k)]=model->grid.volumes[point];
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
    Error initial_bulk,initial_surface,bulk_rhs,surface_rhs;
    double normal_defect=0;
    for (int scenario=0;scenario<2;++scenario) {
      auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
      const auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
      const auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.reference);
      const auto sgeom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
      const auto srefs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
      const double poison=std::numeric_limits<double>::quiet_NaN();
      Kokkos::deep_copy(state,poison);Kokkos::deep_copy(p.residual,poison);
      Kokkos::deep_copy(p.u_rhs,poison);Kokkos::deep_copy(p.s_rhs,poison);
      for (int point=0;point<p.npoints;++point) {
        const auto c=p.points.h_view(point);const auto &x=geom(point).x;
        const double r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
        for (int f=0;f<nf;++f) state(c.m,f,c.k,c.j,c.i)=refs(point)[f].f
            +(native::Fields::Independent(f)?Perturb(x,f,scenario)*(f==0?r2:1):0);
      }
      for (int point=0;point<p.nsurface_points;++point) {
        const auto c=p.surface_points.h_view(point);
        for (int f=0;f<nf;++f) edge(c.face,f,c.k,c.j)=srefs(point)[f].f
            +(native::Fields::Independent(f)?Perturb(sgeom(point).x,f,scenario):0);
      }
      Kokkos::deep_copy(p.u0,state);Kokkos::deep_copy(p.s0,edge);p.EnforceAlgebraicConstraints();
      std::vector<double> global,expected(p.npoints*nf),expected_edge(nedge*nf);
      auto scatter=[&]() {
        MPI_Scatterv(global.data(),counts.data(),displacements.data(),MPI_DOUBLE,
            expected.data(),expected.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(expected_edge.data(),expected_edge.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      };
      auto reorder=[&](const std::vector<double> &bulk,const std::vector<double> &boundary) {
        global.assign(nf*offset.back(),poison);
        for (std::size_t point=0;point<model->grid.cells.size();++point) {
          const auto c=model->grid.cells[point];const int gid=native_id[c.block];
          const int target=canonical(gid,c.i,c.j,c.k);
          for (int f=0;f<nf;++f) global[nf*target+f]=bulk[nf*point+f];
        }
        for (std::size_t point=0;point<model->scri.rows.size();++point) {
          const auto &r=model->scri.rows[point];const int face=model->grid.mesh.blocks[r.block].face;
          const int target=(face*n+r.k)*n+r.j;
          for (int f=0;f<nf;++f) expected_edge[nf*target+f]=boundary[nf*point+f];
        }
      };
      if (model) {
        for (std::size_t point=0;point<model->grid.active.size();++point) {
          const auto &x=model->grid.geometry[point].x;
          const double r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
          for (int f=0;f<nf;++f) model->state[nf*point+f]=model->refs[point][f].f
              +(old::CartesianIndependent(f)?Perturb(x,f,scenario)*(f==0?r2:1):0);
        }
        for (int point=0;point<nedge;++point) for (int f=0;f<nf;++f)
          model->scri.values[nf*point+f]=model->edge_refs[point][f].f
              +(old::CartesianIndependent(f)?Perturb(model->scri.rows[point].geometry.x,f,scenario):0);
        model->Materialize(model->state);model->Materialize(model->scri.values);
        reorder(model->state,model->scri.values);
      }
      scatter();
      auto compare=[&](const auto &actual,const auto &surface,Error &bulk_error,Error &edge_error,
                       const char *label) {
        for (int point=0;point<p.npoints;++point) {
          const auto c=p.points.h_view(point);const int gid=mesh.pmb_pack->gids+c.m;
          const int local=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)
              -offset[mesh.pmb_pack->gids];
          for (int f=0;f<nf;++f) bulk_error.Compare(actual(c.m,f,c.k,c.j,c.i),
              expected[nf*local+f],f,gid,c.i-p.origin,c.j-p.origin,c.k-p.origin,scenario,label,
              geom(point).x,weights[local]);
        }
        for (int point=0;point<p.nsurface_points;++point) {
          const auto c=p.surface_points.h_view(point);const int m=p.face_blocks.h_view(c.face),
                    gid=mesh.pmb_pack->gids+m,face=selector.Map(gid).face;
          const int canonical=(face*n+c.k)*n+c.j;
          for (int f=0;f<nf;++f) edge_error.Compare(surface(c.face,f,c.k,c.j),
              expected_edge[nf*canonical+f],f,gid,n,c.j,c.k,scenario,label,
              sgeom(point).x,surface_weights[canonical]);
        }
      };
      state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
      compare(state,edge,initial_bulk,initial_surface,"initial");

      // Invoke the actual module wrappers in the native task dependency order.
      // This computes one stage RHS only; no RK update changes the inputs.
      p.InitRecv(&driver,1);p.CopyU(&driver,1);p.PrepareStage(&driver,1);
      p.SendBulk(&driver,1);p.SendSurface(&driver,1);
      p.CalcRHS(&driver,1,native::RHSSubset::Interior);
      p.FillLocalHalos(&driver,1);p.CalcRHS(&driver,1,native::RHSSubset::LocalHalo);
      while (p.RecvSurface(&driver,1)!=TaskStatus::complete) {}
      p.SolveNormalCompatibility(&driver,1);p.CalcSurfaceRHS(&driver,1);
      while (p.RecvBulk(&driver,1)!=TaskStatus::complete) {}
      p.FillRemoteHalos(&driver,1);p.CalcRHS(&driver,1,native::RHSSubset::RemoteHalo);
      p.CalcRHS(&driver,1,native::RHSSubset::Surface);
      while (p.ClearSend(&driver,1)!=TaskStatus::complete) {}
      while (p.ClearRecv(&driver,1)!=TaskStatus::complete) {}
      if (model) {
        std::vector<double> edge_rate;
        const auto bulk_rate=model->RHS3D(model->state,p.requested_dt,edge_rate);
        normal_defect=std::max(normal_defect,model->normal_defect);
        reorder(bulk_rate,edge_rate);
      }
      scatter();
      const auto rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u_rhs);
      const auto srhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_rhs);
      compare(rhs,srhs,bulk_rhs,surface_rhs,"RHS");
      if (!global_variable::my_rank) std::cout<<"Completed coupled scenario "<<scenario<<std::endl;
    }
    const char *names[]={"initial_bulk","initial_surface","bulk_rhs","surface_rhs"};
    Error *errors[]={&initial_bulk,&initial_surface,&bulk_rhs,&surface_rhs};
    for (auto *e:errors)e->Global();
    if (!global_variable::my_rank) {
      std::cout<<std::setprecision(17);
      for (int a=0;a<4;++a) errors[a]->Print(names[a]);
      std::cout<<"legacy_normal_compatibility="<<normal_defect<<'\n';
    }
    for (auto *e:errors) Require(e->failures==0,"complete coupled oracle exceeds adopted short gate");
    if (!global_variable::my_rank) std::cout<<"PASS full native N12 coupled RHS, two scenarios\n";
  }} catch (const std::exception &e) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
