// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Independent predecessor sampler versus native sparse, block-owned sampling.
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
#include <unordered_map>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/horizon_sampling.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"
#include "z4c/hyperboloidal/horizon_geometry.hpp"
namespace h=z4c::hyperboloidal;
#include "legacy_horizon_sampler.hpp"  // Exact class extracted by the runner.
namespace native=z4c_hyperboloidal;
using Geometry=z4c::HorizonGeometry<Real>;
constexpr int n=12,nf=31,components=45;
void Require(bool yes,const char *message) {if (!yes)throw std::runtime_error(message);}
std::array<double,components> Pack(const Geometry &q) {
  std::array<double,components> v{};int f=0;
  for (int i=0;i<3;++i)for (int j=0;j<3;++j)v[f++]=q.metric[i][j];
  for (int i=0;i<3;++i)for (int j=0;j<3;++j)v[f++]=q.curvature[i][j];
  for (int a=0;a<3;++a)for (int i=0;i<3;++i)for (int j=0;j<3;++j)
    v[f++]=q.derivative[a][i][j];
  return v;
}
struct Error {
  std::array<double,components> absolute{},scaled{},square{},ratio{};
  std::array<std::uint64_t,components> count{},failed{};
  double worst_ratio=0;int worst_point=-1,worst_field=-1;Real worst_x[3]{};
  void Compare(const Geometry &actual,const double *expected,int point,const Real *x) {
    const auto a=Pack(actual);
    for (int f=0;f<components;++f) {
      const double delta=std::abs(a[f]-expected[f]),tol=1e-12+5e-10*std::abs(expected[f]);
      const bool finite=std::isfinite(a[f]) && std::isfinite(expected[f]);
      const double r=finite?delta/tol:std::numeric_limits<double>::infinity();
      ++count[f];failed[f]+=!finite || delta>tol;
      absolute[f]=std::max(absolute[f],finite?delta:std::numeric_limits<double>::infinity());
      scaled[f]=std::max(scaled[f],finite?delta/(1+std::abs(expected[f])):
                                              std::numeric_limits<double>::infinity());
      square[f]+=delta*delta;ratio[f]=std::max(ratio[f],r);
      if (r>worst_ratio) {
        worst_ratio=r;worst_point=point;worst_field=f;
        for (int a=0;a<3;++a)worst_x[a]=x[a];
      }
      if ((!finite || delta>tol) && failed[f]<=2)
        std::cerr<<std::setprecision(17)<<"sample mismatch point="<<point<<" component="<<f
                 <<" actual="<<a[f]<<" expected="<<expected[f]<<" ratio="<<r<<'\n';
    }
  }
  std::uint64_t Report(int scenario) {
    MPI_Allreduce(MPI_IN_PLACE,absolute.data(),components,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,scaled.data(),components,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,ratio.data(),components,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,square.data(),components,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,count.data(),components,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,failed.data(),components,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    double maximum=*std::max_element(ratio.begin(),ratio.end());
    int rank=worst_ratio==maximum?global_variable::my_rank:global_variable::nranks;
    MPI_Allreduce(MPI_IN_PLACE,&rank,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Bcast(&worst_point,1,MPI_INT,rank,MPI_COMM_WORLD);
    MPI_Bcast(&worst_field,1,MPI_INT,rank,MPI_COMM_WORLD);
    MPI_Bcast(worst_x,3,MPI_DOUBLE,rank,MPI_COMM_WORLD);
    std::uint64_t failures=0,comparisons=0;
    for (int f=0;f<components;++f) {failures+=failed[f];comparisons+=count[f];}
    if (global_variable::my_rank==0) {
      std::cout<<std::setprecision(17)<<"scenario="<<scenario<<" comparisons="<<comparisons
          <<" failures="<<failures<<" max_tolerance_ratio="<<maximum<<" worst_point="
          <<worst_point<<" worst_component="<<worst_field<<" worst_xyz="
          <<worst_x[0]<<','<<worst_x[1]<<','<<worst_x[2]<<'\n';
      for (int f=0;f<components;++f)std::cout<<"scenario="<<scenario<<" component="<<f
          <<" count="<<count[f]<<" failures="<<failed[f]<<" max_absolute="<<absolute[f]
          <<" max_scaled="<<scaled[f]<<" max_tolerance_ratio="<<ratio[f]
          <<" rms="<<std::sqrt(square[f]/count[f])<<'\n';
    }
    return failures;
  }
};
std::vector<Real> Queries() {
  std::vector<Real> q;
  auto add=[&](Real x,Real y,Real z) {q.insert(q.end(),{x,y,z});};
  for (Real r:{.02,.030549496850187822})for (int j=0;j<12;++j)for (int i=0;i<24;++i) {
    const Real theta=M_PI*(j+.5)/12,phi=2*M_PI*(i+.37)/24;
    add(r*std::sin(theta)*std::cos(phi),r*std::sin(theta)*std::sin(phi),r*std::cos(theta));
  }
  // Both sides of Cartesian child seams and triple-corner stencils.
  for (Real x:{-.00017,.00017})for (Real y:{-.00023,.00023})
    for (Real z:{-.02011,-.00019,.00019,.02011})add(x,y,z);
  // Core/shell and radial-layer seams, every face, and angular face corners.
  for (int face=0;face<6;++face)for (Real r:{.0799,.0801,.099,.1001,.1599,.1601,.4,.8,.93}) {
    Real x[3]{};const int a=face/2,b=(a+1)%3,c=(a+2)%3;
    x[a]=(face%2?-1:1)*r/std::sqrt(1+.17*.17+.11*.11);
    x[b]=.17*std::abs(x[a]);x[c]=-.11*std::abs(x[a]);add(x[0],x[1],x[2]);
  }
  for (Real r:{.12,.4})for (Real sign:{-1.,1.}) {
    const Real a=r/std::sqrt(3.);add(a,sign*a,a);add(-a,sign*a,-a);
  }
  // Outside, singular, and invalid inputs must not become valid geometry.
  add(1.1,0,0);add(1,0,0);add(.999,0,0);add(0,0,0);
  add(std::numeric_limits<Real>::quiet_NaN(),0,0);
  return q;
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==3,"horizon_sampling_oracle input saved_fields.bin");
    ParameterInput pin;IOWrapper input;input.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(input);input.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    native::NativeDonorSelector selector(mesh,p.origin);
    native::HorizonSampler sampler(mesh.pmb_pack,&p);
    std::vector<int> offsets(mesh.nmb_total+1);
    for (int gid=0;gid<mesh.nmb_total;++gid) {
      int size=1;for (int a=0;a<3;++a)size*=selector.Upper(gid,a)-selector.Lower(gid,a);
      offsets[gid+1]=offsets[gid]+size;
    }
    auto canonical=[&](int gid,int i,int j,int k) {
      return offsets[gid]+((k-selector.Lower(gid,2))
          *(selector.Upper(gid,1)-selector.Lower(gid,1))+j-selector.Lower(gid,1))
          *(selector.Upper(gid,0)-selector.Lower(gid,0))+i-selector.Lower(gid,0);
    };
    std::vector<int> counts(global_variable::nranks),displacements(counts.size());
    for (int rank=0;rank<global_variable::nranks;++rank) {
      const int first=mesh.gids_eachrank[rank],end=first+mesh.nmb_eachrank[rank];
      counts[rank]=nf*(offsets[end]-offsets[first]);displacements[rank]=nf*offsets[first];
    }
    Require(offsets.back()==229392,"N12 point coverage changed");
    std::unique_ptr<h::SphericalTransfer3D> legacy;
    std::unique_ptr<EndpointHorizonSampler> oracle;
    std::vector<int> native_ids;
    if (global_variable::my_rank==0) {
      cubed_sphere::CompositeMesh oldmesh(.1,
          {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,n,4,3,.08);
      oldmesh.evolved_overlap=3;oldmesh.RefineCartesian(13);
      oldmesh.amr_cartesian_core=true;oldmesh.outer_radial_endpoint_grid=true;
      h::CentralZ4cConfiguration config;config.inner_z4c_radius_M=20;config.compact_slicing_end=.97;
      auto parameters=config.Resolve().gauge;parameters.outer_lapse_damping=.25;
      legacy=std::make_unique<h::SphericalTransfer3D>(std::move(oldmesh),parameters);
      oracle=std::make_unique<EndpointHorizonSampler>(*legacy);
      native_ids.assign(legacy->grid.mesh.blocks.size(),-1);
      for (int old=0;old<int(native_ids.size());++old) {
        const auto &b=legacy->grid.mesh.blocks[old];
        for (int gid=0;gid<mesh.nmb_total;++gid) {
          const auto &a=selector.Map(gid);bool same=a.face==b.face && a.layer==b.layer;
          for (int d=0;d<3;++d)same=same && a.lo[d]==b.lo[d] && a.hi[d]==b.hi[d];
          if (same) {Require(native_ids[old]<0,"ambiguous native/legacy leaf");native_ids[old]=gid;}
        }
        Require(native_ids[old]>=0,"missing native/legacy leaf");
      }
      Require(legacy->grid.active.size()==std::size_t(offsets.back()),"legacy point count changed");
    }
    const auto positions=Queries();const int nq=positions.size()/3;
    std::vector<Geometry> actual(nq),repeat(nq);
    std::vector<int> owners(nq),repeated_owners(nq),expected_owner(nq),expected_valid(nq);
    std::vector<Real> expected(nq*components);
    std::uint64_t failures=0,ghosts=0,remote=0,unchanged=0,cache_checks=0,zero_owner_batches=0;
    std::uint64_t tensor_corners=0;
    const Real poison=std::numeric_limits<Real>::quiet_NaN();
    for (int scenario=0;scenario<3;++scenario) {
      std::vector<Real> global,local(p.npoints*nf);
      if (legacy) {
        if (scenario<2) {
          for (std::size_t point=0;point<legacy->grid.active.size();++point) {
            const auto &x=legacy->grid.geometry[point].x;
            const Real r=h::SphericalTransfer3D::Radius(x);
            for (int f=0;f<nf;++f) {
              const Real perturb=scenario*2e-6*(f+1)/nf
                  *(x[0]+.3*x[1]*x[1]-.2*x[2]*x[2]*x[2]+.1*x[0]*x[1]+std::sin(3*x[2]));
              legacy->state[nf*point+f]=legacy->refs[point][f].f
                  +(h::CartesianIndependent(f)?perturb*(f==0?r*r:1):0);
            }
          }
          legacy->Materialize(legacy->state);
        } else {
          std::ifstream saved(argv[2],std::ios::binary|std::ios::ate);
          Require(saved.good() && saved.tellg()==std::streampos(legacy->state.size()*sizeof(Real)),
                  "saved N12 field extent differs");
          saved.seekg(0);saved.read(reinterpret_cast<char*>(legacy->state.data()),
                                  legacy->state.size()*sizeof(Real));
          Require(saved.good(),"failed to read saved N12 fields");
        }
        global.resize(nf*offsets.back());
        for (std::size_t point=0;point<legacy->grid.cells.size();++point) {
          const auto c=legacy->grid.cells[point];const int gid=native_ids[c.block];
          const auto &map=selector.Map(gid);
          const Real q[3]={(c.i+map.node_offset[0])/n,(c.j+map.node_offset[1])/n,
                           (c.k+map.node_offset[2])/n};
          Real x[3];map.Forward(q,x);
          for (int a=0;a<3;++a)Require(x[a]==legacy->grid.geometry[point].x[a],
                                     "native/legacy sample coordinate differs");
          const int target=canonical(gid,c.i,c.j,c.k);
          for (int f=0;f<nf;++f)global[nf*target+f]=legacy->state[nf*point+f];
        }
        oracle->Set(legacy->state);
        for (int point=0;point<nq;++point) {
          const Real *x=positions.data()+3*point;Geometry q{};
          bool finite=std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
          const int old=finite?legacy->grid.mesh.Owner(x):-1;
          expected_owner[point]=old>=0?mesh.rank_eachmb[native_ids[old]]:-1;
          if (finite)oracle->Sample(x,q);
          expected_valid[point]=q.valid;const auto packed=Pack(q);
          std::copy(packed.begin(),packed.end(),expected.begin()+point*components);
        }
      }
      MPI_Scatterv(global.data(),counts.data(),displacements.data(),MPI_DOUBLE,
                   local.data(),local.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(expected_owner.data(),nq,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(expected_valid.data(),nq,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(expected.data(),expected.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      auto state=Kokkos::create_mirror_view(p.u0);Kokkos::deep_copy(state,poison);
      for (int point=0;point<p.npoints;++point) {
        const auto c=p.points.h_view(point);const int gid=mesh.pmb_pack->gids+c.m;
        const int index=canonical(gid,c.i-p.origin,c.j-p.origin,c.k-p.origin)
            -offsets[mesh.pmb_pack->gids];
        for (int f=0;f<nf;++f)state(c.m,f,c.k,c.j,c.i)=local[nf*index+f];
      }
      Kokkos::deep_copy(p.u0,state);Kokkos::deep_copy(p.residual,poison);
      p.InvalidateDiagnostics();p.FormResidual();
      // A single trial point forces all other ranks through the collective
      // callback with no owned queries, including the first sparse fetch.
      Geometry seed{};int seed_owner=0;
      sampler.SampleBatch(1,positions.data(),&seed,&seed_owner);
      zero_owner_batches+=seed_owner==0;
      sampler.SampleBatch(nq,positions.data(),actual.data(),owners.data());
      Error error;
      for (int point=0;point<nq;++point) {
        const int owned=expected_owner[point]==global_variable::my_rank;
        Require(owners[point]==owned,"horizon sample owner differs from legacy");
        if (!owned) {Require(!actual[point].valid,"nonowner returned valid geometry");continue;}
        if (actual[point].valid!=bool(expected_valid[point])) {
          std::cerr<<"validity mismatch scenario="<<scenario<<" point="<<point
              <<" xyz="<<positions[3*point]<<','<<positions[3*point+1]<<','<<positions[3*point+2]
              <<" native="<<actual[point].valid<<" old="<<expected_valid[point]<<'\n';
          ++failures;continue;
        }
        if (actual[point].valid) {
          error.Compare(actual[point],expected.data()+point*components,
                        point,positions.data()+3*point);
          // Independently enumerate required full tensor corners, rather than
          // trusting the implementation's generic ghost-node counter.
          const int gid=selector.Owner(positions.data()+3*point);
          const auto &map=selector.Map(gid);Real q[3];map.Inverse(positions.data()+3*point,q);
          int first[3];
          for (int a=0;a<3;++a)first[a]=std::clamp(int(std::floor(n*q[a]-map.node_offset[a]))-2,
                                               -p.nghost,n+p.nghost-6);
          for (int k=0;k<6;++k)for (int j=0;j<6;++j)for (int i=0;i<6;++i) {
            const int index[3]={first[0]+i,first[1]+j,first[2]+k};int outside=0;
            for (int a=0;a<3;++a)outside+=index[a]<selector.Lower(gid,a)
                || index[a]>=selector.Upper(gid,a);
            tensor_corners+=outside==3;
          }
        }
      }
      const auto compared_failures=error.Report(scenario);
      if (global_variable::my_rank==0)failures+=compared_failures;
      const auto stats=sampler.GetStatistics();ghosts+=stats.ghost_nodes;remote+=stats.remote_donors;
      sampler.SampleBatch(nq,positions.data(),repeat.data(),repeated_owners.data());
      Require(sampler.GetStatistics().exchanges==stats.exchanges,"cached batch exchanged again");
      for (int point=0;point<nq;++point) {
        const auto a=Pack(actual[point]),b=Pack(repeat[point]);
        Require(owners[point]==repeated_owners[point] && actual[point].valid==repeat[point].valid
            && std::memcmp(a.data(),b.data(),sizeof(a))==0,"cached horizon geometry changed");
        ++cache_checks;
      }
      auto after=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      Require(std::memcmp(state.data(),after.data(),state.size()*sizeof(Real))==0,
              "horizon sampler modified accepted state or poisoned ghosts");
      unchanged+=state.size();
      std::vector<Real> sample_record(nq*(components+2),0);
      for (int point=0;point<nq;++point)if (owners[point]) {
        const int offset=point*(components+2);sample_record[offset]=1;
        sample_record[offset+1]=actual[point].valid?1:0;
        const auto a=Pack(actual[point]);
        std::copy(a.begin(),a.end(),sample_record.begin()+offset+2);
      }
      // Only these small query outputs are collected, never evolution fields.
      MPI_Allreduce(MPI_IN_PLACE,sample_record.data(),sample_record.size(),MPI_DOUBLE,
                    MPI_SUM,MPI_COMM_WORLD);
      if (global_variable::my_rank==0) {
        std::ofstream snapshot("samples-"+std::to_string(scenario)+".bin",std::ios::binary);
        snapshot.write(reinterpret_cast<const char*>(sample_record.data()),
                       sample_record.size()*sizeof(Real));
        Require(snapshot.good(),"cannot save complete query outputs");
      }
      if (global_variable::my_rank==0)std::cout<<"completed scenario="<<scenario<<std::endl;
    }
    // Select an actual ghost-stencil dependency whose evolved donor is on a
    // different rank from the query owner. Only that donor owner invalidates.
    int mutation[5]={-1,0,0,0,-1};  // gid,k,j,i,query
    if (global_variable::my_rank==0) {
      for (int point=0;point<nq && mutation[0]<0;++point) {
        if (!expected_valid[point])continue;
        const int gid=selector.Owner(positions.data()+3*point);
        const auto &map=selector.Map(gid);Real local[3];map.Inverse(positions.data()+3*point,local);
        int first[3];Real weights[3][6]{};
        for (int a=0;a<3;++a) {
          const Real t=n*local[a]-map.node_offset[a];
          first[a]=std::clamp(int(std::floor(t))-2,-p.nghost,n+p.nghost-6);
          for (int i=0;i<6;++i) {
            weights[a][i]=1;
            for (int j=0;j<6;++j)if (j!=i)weights[a][i]*=(t-first[a]-j)/Real(i-j);
          }
        }
        for (int k=0;k<6 && mutation[0]<0;++k)for (int j=0;j<6 && mutation[0]<0;++j)
          for (int i=0;i<6 && mutation[0]<0;++i) {
            const bvals::DonorPoint node{gid,p.origin+first[2]+k,p.origin+first[1]+j,
                                         p.origin+first[0]+i};
            if (selector.IsEvolved(node))continue;
            const Real outer=weights[0][i]*weights[1][j]*weights[2][k];
            if (std::abs(outer)<1e-8)continue;
            const Real q[3]={(first[0]+i+map.node_offset[0])/n,
                             (first[1]+j+map.node_offset[1])/n,
                             (first[2]+k+map.node_offset[2])/n};
            Real x[3];map.Forward(q,x);
            for (const auto &d:selector.Interpolate(x))
              if (mesh.rank_eachmb[d.point.gid]!=expected_owner[point]
                  && std::abs(outer*d.weight)>1e-7) {
                mutation[0]=d.point.gid;mutation[1]=d.point.k;mutation[2]=d.point.j;
                mutation[3]=d.point.i;mutation[4]=point;break;
              }
          }
      }
    }
    MPI_Bcast(mutation,5,MPI_INT,0,MPI_COMM_WORLD);
    Require(mutation[0]>=0,"no remote dependency found for one-rank invalidation");
    const int donor_owner=mesh.rank_eachmb[mutation[0]],reader_owner=expected_owner[mutation[4]];
    Require(donor_owner!=reader_owner,"cache invalidation test donor is local");
    const auto version_before=p.StateVersion();
    if (global_variable::my_rank==donor_owner) {
      auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      state(mutation[0]-mesh.pmb_pack->gids,13,mutation[1],mutation[2],mutation[3])+=1e-3;
      Kokkos::deep_copy(p.u0,state);p.InvalidateDiagnostics();p.FormResidual();
    }
    Require((p.StateVersion()!=version_before)==(global_variable::my_rank==donor_owner),
            "test did not invalidate exactly one owner");
    std::vector<Geometry> refreshed(nq),fresh(nq);
    std::vector<int> refreshed_owners(nq),fresh_owners(nq);
    sampler.SampleBatch(nq,positions.data(),refreshed.data(),refreshed_owners.data());
    native::HorizonSampler fresh_sampler(mesh.pmb_pack,&p);
    fresh_sampler.SampleBatch(nq,positions.data(),fresh.data(),fresh_owners.data());
    std::uint64_t remote_invalidation_checks=0,changed_remote_query=0;
    for (int point=0;point<nq;++point) {
      const auto a=Pack(refreshed[point]),b=Pack(fresh[point]);
      Require(refreshed_owners[point]==fresh_owners[point]
          && refreshed[point].valid==fresh[point].valid
          && std::memcmp(a.data(),b.data(),sizeof(a))==0,
          "one-rank invalidation retained a stale remote donor value");
      ++remote_invalidation_checks;
      if (point==mutation[4] && global_variable::my_rank==reader_owner) {
        const auto old=Pack(actual[point]);
        Require(refreshed[point].valid && std::memcmp(a.data(),old.data(),sizeof(a))!=0,
                "remote mutation did not change its intended query");
        ++changed_remote_query;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&remote_invalidation_checks,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&changed_remote_query,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(changed_remote_query==1,"remote invalidation query coverage differs");
    if (global_variable::my_rank==0)
      std::cout<<"remote_invalidation checks="<<remote_invalidation_checks
          <<" changed_query="<<changed_remote_query<<" donor_rank="<<donor_owner
          <<" reader_rank="<<reader_owner<<" query="<<mutation[4]<<std::endl;
    MPI_Allreduce(MPI_IN_PLACE,&failures,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&ghosts,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&remote,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&unchanged,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&cache_checks,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&zero_owner_batches,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&tensor_corners,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    Require(ghosts>0 && remote>0 && zero_owner_batches>0 && tensor_corners>0,
            "sparse tensor-ghost/remote/zero-owner path was not exercised");
    if (global_variable::my_rank==0)std::cout<<"horizon_sampling queries="<<nq
        <<" scenarios=3 failures="<<failures<<" ghost_nodes="<<ghosts<<" remote_donors="
        <<remote<<" unchanged_values="<<unchanged<<" cache_checks="<<cache_checks
        <<" zero_owner_batches="<<zero_owner_batches<<" tensor_corners="<<tensor_corners<<std::endl;
    Require(failures==0,"native horizon sampling differs from predecessor");
  }} catch (const std::exception &e) {
    std::cerr<<"horizon sampler oracle: "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
