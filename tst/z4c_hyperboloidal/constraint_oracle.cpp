// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Accepted-state native constraints versus the independent preserved N12 CPU model.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <cstring>
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
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"

namespace native=z4c_hyperboloidal;
namespace old=z4c::hyperboloidal;
constexpr int n=12,nf=31,nedge=6*n*n,nc=19;
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
  std::array<std::uint64_t,nc> field_failures{};
  std::array<double,nc> field_absolute{},field_scaled{},field_square{},field_weight{};
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
    MPI_Allreduce(MPI_IN_PLACE,field_failures.data(),nc,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_absolute.data(),nc,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_scaled.data(),nc,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_square.data(),nc,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,field_weight.data(),nc,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
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
    for (int f=0;f<nc;++f) std::cout<<label<<" field="<<f<<" max_absolute="<<field_absolute[f]
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

template<class C>std::array<Real,nc> Pack(const C &c,Real chi,Real lapse,Real theta) {
  return {chi,lapse,theta,c.hamiltonian,c.momentum[0],c.momentum[1],c.momentum[2],
      c.momentum_conformal_norm2,c.momentum_physical_norm2,c.physical_trace,
      c.null_residual,c.z4.determinant_residual,c.z4.tracefree_residual,c.z4.theta_physical,
      c.z4.z_covector[0],c.z4.z_covector[1],c.z4.z_covector[2],
      c.z4.z_conformal_norm2,c.z4.z_physical_norm2};
}
void Read(const std::string &path,std::vector<Real> &values) {
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  Require(file.good() && file.tellg()==std::streampos(values.size()*sizeof(Real)),
          "saved checkpoint has wrong extent");
  file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(Real));
  Require(file.good(),"saved checkpoint read failed");
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==3,"constraint_oracle input checkpoint_directory");
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
        global_weights[canonical(native_id[c.block],c.i,c.j,c.k)]=model->grid.DiagnosticVolume(point);
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
    std::uint64_t total_failures=0,weight_checks=0,cache_checks=0;
    const auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
    std::vector<int> constraint_counts=weight_counts,constraint_offsets=weight_offsets;
    for (auto &v:constraint_counts)v*=nc;
    for (auto &v:constraint_offsets)v*=nc;
    for (int scenario=0;scenario<3;++scenario) {
      const Real poison=std::numeric_limits<Real>::quiet_NaN();
      std::vector<Real> global,local(nf*p.npoints),edge(nf*nedge);
      if (model) {
        if (scenario<2) {
          for (std::size_t q=0;q<model->grid.active.size();++q) {
            const auto &x=model->grid.geometry[q].x;const Real r=old::SphericalTransfer3D::Radius(x);
            for (int f=0;f<nf;++f)model->state[nf*q+f]=model->refs[q][f].f
                +(old::CartesianIndependent(f)?Perturb(x,f,scenario)*(f==0?r*r:1):0);
          }
          for (int q=0;q<nedge;++q)for (int f=0;f<nf;++f)
            model->scri.values[nf*q+f]=model->edge_refs[q][f].f
                +(old::CartesianIndependent(f)?Perturb(model->scri.rows[q].geometry.x,f,scenario):0);
          model->Materialize(model->state);model->MaterializeBoundary(model->state);
        } else {
          Read(std::string(argv[2])+"/fields.bin",model->state);
          Read(std::string(argv[2])+"/boundary.bin",model->scri.values);
        }
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
      mesh.ncycle=scenario==2?3551:0;mesh.time=scenario==2?1.6669886746517166:0;
      auto &diagnostics=p.GetDiagnostics();diagnostics.Refresh();
      const auto refreshes=diagnostics.GetStatistics().refreshes;
      diagnostics.Refresh();Require(diagnostics.GetStatistics().refreshes==refreshes,"cache reran refresh");
      ++cache_checks;
      const auto actual=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Points());
      const auto actual_weights=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Weights());
      std::vector<Real> expected(nc*p.npoints);std::array<Real,10> legacy_sums{};
      if (model) {
        global.resize(nc*offset.back());
        const auto residual=model->Residuals(model->state);model->SolveBoundary(residual.first,residual.second);
        for (std::size_t q=0;q<model->grid.cells.size();++q) {
          const auto c=model->grid.cells[q];const auto &x=model->grid.geometry[q].x;
          const auto jets=model->BulkJet(q,residual.first);
          const auto u=old::SphericalTransfer3D::PhysicalJet(x,jets);
          auto omega=model->parameters.profile.At(x);
          old::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
          const auto constraints=old::EvolvedConstraints(u,omega);
          Require(constraints.valid,"invalid legacy constraint geometry");
          const auto values=Pack(constraints,u.chi.value,u.alpha.value,u.theta.value);
          const int target=canonical(native_id[c.block],c.i,c.j,c.k);
          for (int f=0;f<nc;++f)global[nc*target+f]=values[f];
          const Real w=model->grid.DiagnosticVolume(q),r=old::SphericalTransfer3D::Radius(x);
          const Real sums[5]={w,w*constraints.hamiltonian*constraints.hamiltonian,
            w*constraints.momentum_physical_norm2,w*u.theta.value*u.theta.value,
            w*constraints.z4.z_physical_norm2};
          for (int f=0;f<5;++f) {legacy_sums[f]+=sums[f];if(r>.94)legacy_sums[5+f]+=sums[f];}
        }
      }
      MPI_Scatterv(global.data(),constraint_counts.data(),constraint_offsets.data(),MPI_DOUBLE,
                   expected.data(),expected.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(legacy_sums.data(),10,MPI_DOUBLE,0,MPI_COMM_WORLD);
      Error error;std::array<Real,10> sums{};
      for (int q=0;q<p.npoints;++q) {
        const auto c=p.points.h_view(q);const int gid=mesh.pmb_pack->gids+c.m;
        const int index=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)-offset[mesh.pmb_pack->gids];
        Require(actual(q).status==native::PointStatus::success,"invalid native constraint point");
        const auto &v=actual(q);const auto values=Pack(v.constraints,v.chi,v.lapse,v.theta);
        const Real w=actual_weights(q);
        Require(w==weights[index],"diagnostic weight differs from independent legacy");++weight_checks;
        for (int f=0;f<nc;++f)error.Compare(values[f],expected[nc*index+f],f,gid,
            c.i-p.origin,c.j-p.origin,c.k-p.origin,scenario,"constraint",geom(q).x,w>0?w:1e-30);
        const Real row[5]={w,w*v.constraints.hamiltonian*v.constraints.hamiltonian,
            w*v.constraints.momentum_physical_norm2,w*v.theta*v.theta,w*v.constraints.z4.z_physical_norm2};
        const auto &x=geom(q).x;
        const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        for (int f=0;f<5;++f) {sums[f]+=row[f];if (r>.94)sums[5+f]+=row[f];}
      }
      error.Global();total_failures+=error.failures;
      MPI_Allreduce(MPI_IN_PLACE,sums.data(),10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      if (!global_variable::my_rank)error.Print(("constraints"+std::to_string(scenario)).c_str());
      for (int region=0;region<2;++region)for (int f=1;f<5;++f) {
        const Real a=std::sqrt(sums[5*region+f]/sums[5*region]);
        const Real b=std::sqrt(legacy_sums[5*region+f]/legacy_sums[5*region]);
        Require(std::abs(a-b)<=5e-11*(1+std::abs(b)),"full/fixed-outer constraint replay failed");
        if (!global_variable::my_rank)std::cout<<"norm scenario="<<scenario<<" region="<<region
            <<" field="<<f<<" actual="<<a<<" expected="<<b<<'\n';
      }
      // Run the same full-harmonic native finder used by actual history output.
      if (scenario!=1) {
        auto history=p.DiagnosticHistorySums();
        MPI_Allreduce(MPI_IN_PLACE,history.data(),10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        const auto &horizon=diagnostics.FindHorizon();
        if (!global_variable::my_rank) {
          std::cout<<"horizon scenario="<<scenario<<" mean="<<horizon.mean_radius
              <<" areal_M="<<3*horizon.areal_radius<<" mass_M="<<3*horizon.irreducible_mass
              <<" expansion="<<horizon.expansion_rms<<" iterations="<<horizon.iterations<<'\n';
          std::ofstream shape("surface-"+std::to_string(scenario)+".txt");shape<<std::setprecision(17);
          const auto &surface=diagnostics.Surface();shape<<surface.lmax<<'\n';
          for (auto c:surface.center)shape<<c<<' ';shape<<'\n';
          for (const auto *values:{&surface.a0,&surface.ac,&surface.as}) {
            for (auto c:*values)shape<<c<<' ';shape<<'\n';
          }
        }
        const Real reference[8]={.0073257648025702423,.0039666959397409406,
          .00063679744011741466,.00074930798279592369,.0066650464364187327,
          .003785788884853934,.00062887813299441033,.0007488764374473728};
        for (int region=0;region<2;++region)for (int f=1;f<5;++f) {
          const Real value=std::sqrt(history[5*region+f]/history[5*region]);
          if (!global_variable::my_rank)std::cout<<"history scenario="<<scenario<<" region="<<region
              <<" field="<<f<<" value="<<value<<'\n';
          if (scenario==2)Require(std::abs(value-reference[4*region+f-1])<=
              5e-10+1e-6*std::abs(reference[4*region+f-1]),"saved N12 full/exterior history mismatch");
        }
        if (scenario==2) {
          Require(std::abs(3*horizon.areal_radius-1.9999860425879357)<=
              1e-10+1e-7*1.9999860425879357,"saved horizon radius mismatch");
          Require(std::abs(3*horizon.irreducible_mass-.99999302129396783)<=
              1e-10+1e-7*.99999302129396783,"saved horizon mass mismatch");
        }
      }
      if (!global_variable::my_rank)std::cout<<"completed_constraint_scenario="<<scenario<<std::endl;
    }
    MPI_Allreduce(MPI_IN_PLACE,&weight_checks,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(weight_checks==3*229392,"pointwise diagnostic quadrature coverage incomplete");
    Require(total_failures==0,"pointwise constraints exceed adopted arithmetic gate");
    if (!global_variable::my_rank)std::cout<<"PASS native constraints and saved N12 diagnostics weights="
        <<weight_checks<<" cache_checks="<<cache_checks<<'\n';
  }} catch (const std::exception &e) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
