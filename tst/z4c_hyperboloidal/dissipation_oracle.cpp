// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// The old term-builder and old device KO expression are extracted at test build
// time. Only field/geometry storage is adapted to the native local pack.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c/hyperboloidal/composite_dissipation.hpp"

namespace nh=z4c_hyperboloidal;
void Require(bool ok,const char *message) {if (!ok) throw std::runtime_error(message);}

// A storage adapter for the unchanged legacy builder, with nominal cells first
// and then evolved overlap, as in CompositeDifference. No halo interpolation is
// performed in this operator test; interpolation has its own complete-row test.
struct OracleGrid {
  struct Block {int face; Real lo[3],hi[3];};
  struct Cell {int block,i,j,k;};
  struct Descriptor {bool amr_cartesian_core=true; Real half_width;
    int evolved_overlap;std::vector<Block> blocks;} mesh;
  int n,ng,width;
  std::vector<nh::mapped::Point> geometry;
  std::vector<Real> volumes;
  std::vector<Cell> cells;
  std::vector<int> active,inverse;
  std::vector<nh::EvolvedBounds> bounds;
  int Index(int m,int i,int j,int k) const {
    const int origin=ng+mesh.evolved_overlap;
    return ((m*width+k+origin)*width+j+origin)*width+i+origin;
  }
  int ShiftActive(int p,int axis,int offset) const {
    auto c=cells[p];int q[3]={c.i,c.j,c.k};q[axis]+=offset;
    for (int a=0;a<3;++a)
      if (q[a]<bounds[c.block].lo[a] || q[a]>bounds[c.block].hi[a]) return -1;
    return inverse[Index(c.block,q[0],q[1],q[2])];
  }
  Real DiagnosticVolume(int p) const {return volumes[p];}
  OracleGrid(const MeshBlockPack &pack,const nh::Z4cHyperboloidal &module)
      : n(module.n[0]),ng(module.nghost),width(module.storage[0]) {
    mesh.half_width=pack.pmesh->atlas->HalfWidth();mesh.evolved_overlap=module.overlap;
    const auto maps=pack.pmb->mb_map.h_view;
    auto g=Kokkos::create_mirror_view_and_copy(HostMemSpace(),module.geometry);
    inverse.assign(module.nblocks*width*width*width,-1);
    std::vector<int> native_index(inverse.size(),-1);
    for (int p=0;p<module.npoints;++p) {
      auto c=module.points.h_view(p);
      native_index[Index(c.m,c.i-module.origin,c.j-module.origin,c.k-module.origin)]=p;
    }
    for (int m=0;m<module.nblocks;++m) {
      Block b{};b.face=maps(m).face;
      auto limits=module.evolved_bounds.h_view(m);
      for (int a=0;a<3;++a) {
        b.lo[a]=maps(m).lo[a];b.hi[a]=maps(m).hi[a];
        limits.lo[a]-=module.origin;limits.hi[a]-=module.origin;
      }
      mesh.blocks.push_back(b);bounds.push_back(limits);
    }
    for (int pass=0;pass<2;++pass) for (int m=0;m<module.nblocks;++m)
      for (int k=bounds[m].lo[2];k<=bounds[m].hi[2];++k)
      for (int j=bounds[m].lo[1];j<=bounds[m].hi[1];++j)
      for (int i=bounds[m].lo[0];i<=bounds[m].hi[0];++i) {
        const bool nominal=i>=0&&i<n&&j>=0&&j<n&&k>=0&&k<n;
        if (nominal!=(pass==0)) continue;
        const int index=Index(m,i,j,k),p=native_index[index];
        Require(p>=0,"oracle point mapping");
        inverse[index]=cells.size();cells.push_back({m,i,j,k});active.push_back(index);
        geometry.push_back(g(p));
        const auto &a=g(p).J;
        const double det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
        volumes.push_back(1/(std::abs(det)*n*n*n)
                          *(maps(m).node_offset[0]==0&&i==0?.5:1.));
      }
  }
};

namespace z4c::hyperboloidal {
struct LegacyKO {
  struct KOTerm {int point[CompositeDissipation::points];int target_coefficient;Real weight;};
  using IntBuffer=DvceArray1D<int>;
  using Buffer=DvceArray1D<Real>;
  using GeometryBuffer=DvceArray1D<nh::mapped::Point>;
  using KOTermBuffer=DvceArray1D<KOTerm>;
  int first_=0,local_,n,width;
  IntBuffer active_,ko_offsets_;
  GeometryBuffer geometry_;
  Buffer volumes_,diagnostic_volumes_;
  KOTermBuffer ko_terms_;
  Real ko=.1;
  bool amr_cfl_=true;
  DvceArray5D<Real> state;
  LegacyKO(const OracleGrid &grid,const DvceArray5D<Real> &values)
      : local_(grid.active.size()),n(grid.n),width(grid.width),
        active_("old active",local_),ko_offsets_("old offsets",local_+1),
        geometry_("old geometry",local_),volumes_("old volumes",local_),
        diagnostic_volumes_("old diagnostic volumes",local_),state(values) {
#include "ko_legacy_builder.hpp"
  }
  Real PolicyPhysicalValue(int p,int f) const {
    const int i=p%width;p/=width;const int j=p%width;p/=width;
    const int k=p%width;return state(p/width,f,k,j,i);
  }
  Real Evaluate(int local,int f) const {
    const auto offsets=ko_offsets_;const auto terms=ko_terms_;const auto volume=volumes_;
#include "ko_legacy_evaluate.hpp"
  }
};
}  // namespace z4c::hyperboloidal

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  const std::string input=argv[1];Kokkos::initialize(argc,argv);
  try {
    {
      ParameterInput pin;IOWrapper file;file.Open(input.c_str(),IOWrapper::FileMode::read);
      pin.LoadFromFile(file);file.Close();
      Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
      auto &module=*mesh.pmb_pack->pz4c_hyperboloidal;
      nh::DissipationPlan plan;plan.Rebuild(*mesh.pmb_pack,module);
      const auto kernel=plan.Bind();
      OracleGrid grid(*mesh.pmb_pack,module);
      auto state=Kokkos::create_mirror_view(module.u0);
      // Nonsmooth deterministic physical values exercise all KO coefficients,
      // including dependent and dormant fields, and distinguish each halo node.
      for (int m=0;m<module.nblocks;++m) for (int f=0;f<31;++f)
        for (int k=0;k<module.storage[2];++k) for (int j=0;j<module.storage[1];++j)
        for (int i=0;i<module.storage[0];++i)
          state(m,f,k,j,i)=std::sin(.17*i+.39*j+.23*k+.47*f)+.001*m;
      Kokkos::deep_copy(module.u0,state);
      z4c::hyperboloidal::LegacyKO old(grid,module.u0);
      auto offsets=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.offsets);
      auto lines=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.lines);
      auto volumes=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.volumes);
      std::uint64_t terms=0,comparisons=0,bitwise=0;
      double error=0;
      DvceArray2D<Real> rates("native KO rates",module.npoints,31);
      const auto points=module.points.d_view;const auto values=module.u0;
      Kokkos::parallel_for("KO oracle",Kokkos::RangePolicy<>(0,module.npoints*31),
        KOKKOS_LAMBDA(int t) {const int p=t/31,f=t%31;const auto c=points(p);
          rates(p,f)=kernel.Evaluate(p,f,{values,c.m,c.k,c.j,c.i},.1);});
      auto output=Kokkos::create_mirror_view_and_copy(HostMemSpace(),rates);
      for (int p=0;p<module.npoints;++p) {
        const auto c=module.points.h_view(p);const int origin=module.origin;
        const int op=grid.inverse[grid.Index(c.m,c.i-origin,c.j-origin,c.k-origin)];
        Require(offsets(p+1)-offsets(p)==old.ko_offsets_(op+1)-old.ko_offsets_(op),
                "KO term count changed");
        Require(volumes(p)==old.volumes_(op),"KO volume changed");
        for (int t=offsets(p);t<offsets(p+1);++t) {
          const auto line=lines(t);const auto reference=old.ko_terms_(old.ko_offsets_(op)+t-offsets(p));
          Require(nh::PhysicalKO8::Coefficient(line.target)==reference.target_coefficient,
                  "KO transpose coefficient changed");
          if (line.weight!=reference.weight) {
            std::cerr<<std::setprecision(17)<<"KO weight block="<<c.m<<" point="<<p
                <<" axis="<<line.axis<<" target="<<line.target<<" native="
                <<line.weight<<" old="<<reference.weight<<'\n';
            throw std::runtime_error("KO geometry weight changed");
          }
          for (int a=0;a<5;++a) {
            int index[3]={c.i-origin,c.j-origin,c.k-origin};index[line.axis]+=a-line.target;
            Require(grid.Index(c.m,index[0],index[1],index[2])==reference.point[a],
                    "KO line donor/order changed");
          }
          ++terms;
        }
        for (int f=0;f<31;++f) {
          const double expected=old.Evaluate(op,f),actual=output(p,f);
          Require(std::isfinite(expected)&&std::isfinite(actual),"nonfinite KO value");
          error=std::max(error,std::abs(actual-expected)/(1+std::abs(expected)));
          Require(std::abs(actual-expected)<=1e-12+5e-10*std::abs(expected),"KO rate mismatch");
          ++comparisons;bitwise+=actual==expected;
        }
      }
      std::uint64_t counts[3]={terms,comparisons,bitwise};
      MPI_Allreduce(MPI_IN_PLACE,counts,3,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      if (global_variable::my_rank==0) std::cout<<std::setprecision(17)
        <<"PASS native KO: terms="<<counts[0]<<" rates="<<counts[1]
        <<" bitwise="<<counts[2]<<" max_scaled_error="<<error<<'\n';
    }
    Kokkos::finalize();MPI_Finalize();return 0;
  } catch (const std::exception &e) {
    std::cerr<<"KO oracle: "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
