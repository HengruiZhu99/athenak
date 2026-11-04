// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_AMR_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_AMR_HPP_
#include <algorithm>
#include <cmath>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/refinement_criteria.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/amr_fields.hpp"

namespace z4c::hyperboloidal {
// Adapter to AthenaK's actual octree and transfer lifecycle. Only one Mesh may
// own a tree in a process (MeshBlockTree's pre-existing static root contract).
// Fixed spherical shells are external to this Mesh and never enter its octree.
class CartesianAMR {
 public:
  ParameterInput input;
  std::unique_ptr<Mesh> mesh;
  DvceArray5D<Real> state,coarse,compensation,coarse_compensation;
  const int n,ng=4;
  const std::vector<Real> shell_radii;
  const Real half_width,inner_radius;
  CartesianAMR(int cells,int subdivisions=2,int capacity=64,int levels=3,
               std::vector<Real> radii={.305,.45,.6,.75,.875,1},Real width=.18,Real inner=.16)
      :n(cells),shell_radii(std::move(radii)),half_width(width),inner_radius(inner){
    if(!(width>0 && inner>0 && inner<width))throw std::invalid_argument("invalid AMR core bounds");
    input.SetReal("time","cfl_number",.1);
    for(int a=1;a<=3;++a){const auto x=std::to_string(a);
      input.SetReal("mesh","x"+x+"min",-half_width);input.SetReal("mesh","x"+x+"max",half_width);
      input.SetInteger("mesh","nx"+x,subdivisions*n);input.SetInteger("meshblock","nx"+x,n);
      input.SetString("mesh","ix"+x+"_bc","outflow");input.SetString("mesh","ox"+x+"_bc","outflow");}
    input.SetInteger("mesh","nghost",ng);input.SetString("mesh_refinement","refinement","adaptive");
    input.SetString("amr_criterion0","method","location");
    input.SetInteger("mesh_refinement","num_levels",levels);input.SetInteger("mesh_refinement","max_nmb_per_rank",capacity);
    mesh=std::make_unique<Mesh>(&input);mesh->BuildTreeFromScratch(&input);
    mesh->pmb_pack->AddCoordinates(&input);mesh->pmr=new MeshRefinement(mesh.get(),&input);
    state=DvceArray5D<Real>("AMR native31",capacity,31,n+8,n+8,n+8);
    compensation=DvceArray5D<Real>("AMR compensation31",capacity,31,n+8,n+8,n+8);
    coarse=DvceArray5D<Real>("AMR coarse31",capacity,31,n/2+8,n/2+8,n/2+8);
    coarse_compensation=DvceArray5D<Real>("AMR coarse compensation31",capacity,31,n/2+8,n/2+8,n/2+8);
    mesh->pmr->RegisterCC("hyperboloidal.native31.v1",state,coarse);
    mesh->pmr->RegisterCC("hyperboloidal.compensation31.v1",compensation,coarse_compensation);
  }
  cubed_sphere::CompositeMesh Descriptor()const{
    cubed_sphere::CompositeMesh out(half_width,shell_radii,1,n,4,1,inner_radius);
    const auto shells=out.blocks;out.blocks.clear();
    for(int m=0;m<mesh->nmb_total;++m){const auto&l=mesh->lloc_eachmb[m];
      cubed_sphere::CompositeBlock b;b.refinement=l.level-mesh->root_level;
      const int loc[3]={l.lx1,l.lx2,l.lx3};
      const int root[3]={mesh->nmb_rootx1,mesh->nmb_rootx2,mesh->nmb_rootx3};
      for(int a=0;a<3;++a){const Real h=2*half_width/(root[a]*(1<<b.refinement));b.lo[a]=-half_width+loc[a]*h;b.hi[a]=-half_width+(loc[a]+1)*h;}
      out.blocks.push_back(b);
    }
    for(const auto&b:shells)if(b.face>=0)out.blocks.push_back(b);
    out.evolved_overlap=3;out.amr_cartesian_core=true;out.revision=mesh->GetAMRLoadBalanceUpdateSeq();return out;
  }
  // All input fields are complete ownership-independent accepted snapshots.
  // This boundary adapter supplies the four fine halos required by AthenaK's
  // native five-point prolongation; no rank-local donor assumptions are made.
  void Load(const cubed_sphere::CompositeDifference&g,const std::vector<Real>&u,
            const std::vector<Real>&rk){
    if(u.size()!=31*g.active.size()||rk.size()!=u.size())throw std::invalid_argument("AMR snapshot layout");
    auto h=Kokkos::create_mirror_view(state),r=Kokkos::create_mirror_view(compensation);
    const int first=mesh->gids_eachrank[global_variable::my_rank];
    for(int m=0;m<mesh->nmb_thisrank;++m){
      for(int k=-ng;k<n+ng;++k)for(int j=-ng;j<n+ng;++j)for(int i=-ng;i<n+ng;++i){
        std::vector<cubed_sphere::CompositeDifference::Entry> row;
        const int p=g.ActiveIndex(first+m,i,j,k);
        if(p>=0)row.push_back({g.active[p],1});else{
          Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n},x[3];g.mesh.Forward(first+m,q,x);row=g.Interpolate(x,6);}
        if(row.empty())throw std::runtime_error("missing AMR transfer donor");
        const int anchor=g.inverse_active[row[0].point];
        for(int f=0;f<31;++f){Real v=u[31*anchor+f],c=rk[31*anchor+f];
          for(const auto&e:row){const int p=g.inverse_active[e.point];if(p<0)throw std::runtime_error("unevolved AMR transfer donor");
            v+=e.weight*(u[31*p+f]-u[31*anchor+f]);c+=e.weight*(rk[31*p+f]-rk[31*anchor+f]);}
          h(m,f,k+ng,j+ng,i+ng)=v;r(m,f,k+ng,j+ng,i+ng)=c;
        }
      }
    }Kokkos::deep_copy(state,h);Kokkos::deep_copy(compensation,r);
  }
  // Caller collectively selects flags only at an accepted-step boundary.
  // AthenaK resolves sibling coarsening, 2:1 balance and rank redistribution.
  bool Regrid(const std::vector<int>&flags){
    int bad=flags.size()!=size_t(mesh->nmb_total);MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(bad)throw std::invalid_argument("AMR flag count");
    auto root=flags;MPI_Bcast(root.data(),root.size(),MPI_INT,0,MPI_COMM_WORLD);
    bad=root!=flags;for(int flag:flags)bad=bad||flag < -1||flag > 1;
    MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(bad)throw std::invalid_argument("inconsistent AMR refinement flags");
    auto&a=*mesh->pmr;Kokkos::realloc(a.refine_flag,mesh->nmb_total);
    for(int m=0;m<mesh->nmb_total;++m){const auto level=mesh->lloc_eachmb[m].level;
      if(flags[m]<-1||flags[m]>1)throw std::invalid_argument("AMR flag value");
      a.refine_flag.h_view(m)=(flags[m]>0&&level==mesh->max_level)||(flags[m]<0&&level==mesh->root_level)?0:flags[m];}
    a.refine_flag.modify<HostMemSpace>();a.refine_flag.sync<DevExeSpace>();
    int added=0,removed=0;a.UpdateMeshBlockTree(added,removed);
    if(!added&&!removed)return false;
    a.RedistAndRefineMeshBlocks(&input,added,removed);mesh->MarkMeshUpdated();return true;
  }
  // Existing AthenaK location criterion and refinement-interval policy. The
  // standalone driver may move the center using its accepted puncture tracker.
  bool RefineAround(const std::array<Real,3>&center,Real radius,int accepted_cycle){
    auto&criterion=mesh->pmr->pmrc->rcrit.at(0);
    criterion.rloc_x1=center[0];criterion.rloc_x2=center[1];criterion.rloc_x3=center[2];criterion.rloc_rad=radius;
    mesh->ncycle=accepted_cycle;mesh->pmr->CheckForRefinement(mesh->pmb_pack);
    std::vector<int> flags(mesh->nmb_total);for(int m=0;m<mesh->nmb_total;++m)flags[m]=mesh->pmr->refine_flag.h_view(m);
    return Regrid(flags);
  }
  // Initial static target only. AthenaK still performs octree balancing,
  // redistribution and registered field transfer through Regrid. Strict box
  // intersection avoids refining blocks that only touch the requested region.
  bool RefineBox(const std::array<Real,3>&half){
    for(Real x:half)if(!std::isfinite(x) || !(x>0 && x<inner_radius))
      throw std::invalid_argument("invalid static Cartesian refinement box");
    const auto d=Descriptor();std::vector<int> flags(mesh->nmb_total,0);
    for(int m=0;m<mesh->nmb_total;++m){bool hit=true;
      for(int a=0;a<3;++a)hit=hit && d.blocks[m].lo[a]<half[a] && d.blocks[m].hi[a]>-half[a];
      flags[m]=hit?1:0;
    }
    return Regrid(flags);
  }
  struct FinestSupport {Real radius=0,spacing=0,core_margin=0;int level=0;};
  // Radius of the largest coordinate ball about center entirely covered by
  // nominal finest Cartesian cells. Appended overlap/halo rows do not count.
  // Subtract stencil and horizon radii separately; this is not an error bound.
  FinestSupport SupportAt(const std::array<Real,3>&center)const{
    const auto d=Descriptor();FinestSupport out;out.core_margin=half_width;
    for(Real x:center){if(!std::isfinite(x))throw std::invalid_argument("nonfinite support center");
      out.core_margin=std::min(out.core_margin,half_width-std::abs(x));}
    for(const auto&b:d.blocks)if(b.face<0)out.level=std::max(out.level,b.refinement);
    out.radius=std::max(Real(0),out.core_margin);
    for(const auto&b:d.blocks)if(b.face<0){
      if(b.refinement==out.level){
        for(int a=0;a<3;++a)out.spacing=std::max(out.spacing,(b.hi[a]-b.lo[a])/n);
      }else{
        Real distance2=0;
        for(int a=0;a<3;++a){const Real dx=std::max({b.lo[a]-center[a],Real(0),center[a]-b.hi[a]});distance2+=dx*dx;}
        out.radius=std::min(out.radius,std::sqrt(distance2));
      }
    }
    return out;
  }
  std::vector<Real> Gather(bool rk=false)const{
    const auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rk?compensation:state);
    const int block=31*n*n*n;std::vector<Real> local(block*mesh->nmb_thisrank),all(block*mesh->nmb_total);
    for(int m=0;m<mesh->nmb_thisrank;++m)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
      for(int f=0;f<31;++f)local[((m*n+k)*n*n+j*n+i)*31+f]=h(m,f,k+ng,j+ng,i+ng);
    std::vector<int> counts(global_variable::nranks),offsets(counts.size());
    for(int r=0;r<global_variable::nranks;++r){counts[r]=block*mesh->nmb_eachrank[r];offsets[r]=block*mesh->gids_eachrank[r];}
    MPI_Allgatherv(local.data(),local.size(),MPI_ATHENA_REAL,all.data(),counts.data(),offsets.data(),MPI_ATHENA_REAL,MPI_COMM_WORLD);return all;
  }
};
}
#endif
