// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Geometry-only independent interpolation checks; no field storage or remeshing.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/remesh.hpp"

namespace h=z4c_hyperboloidal;
namespace {
void Require(bool good,const char *why) {if(!good)throw std::runtime_error(why);}
void Input(ParameterInput &pin,const char *path,int n,bool refined) {
  IOWrapper file;file.Open(path,IOWrapper::FileMode::read);pin.LoadFromFile(file);file.Close();
  for(int a=1;a<=3;++a) {
    pin.SetInteger("mesh","nx"+std::to_string(a),3*n);
    pin.SetInteger("meshblock","nx"+std::to_string(a),n);
  }
  if(!refined)return;
  for(int region=0;region<3;++region) {
    const auto name="refined_region"+std::to_string(10+region);
    pin.SetInteger(name,"level",1);
    if(region==0) {
      for(int a=1;a<=3;++a) {
        pin.SetReal(name,"x"+std::to_string(a)+"min",-.095);
        pin.SetReal(name,"x"+std::to_string(a)+"max",-.075);
      }
    } else {
      pin.SetInteger(name,"chart_id",1);pin.SetString(name,"coordinate_system","radius_angles");
      pin.SetReal(name,"x1min",region==1?.09:.95);pin.SetReal(name,"x1max",region==1?.10:.96);
      for(int a=2;a<=3;++a) {
        pin.SetReal(name,"x"+std::to_string(a)+"min",.1);
        pin.SetReal(name,"x"+std::to_string(a)+"max",.2);
      }
    }
  }
}
bool Selected(const chart_atlas::BlockMap &m,int family) {
  if(family==0)return m.face<0&&m.hi[0]<-.03&&m.hi[1]<-.03&&m.hi[2]<-.03;
  return m.face==0&&m.layer==(family==1?0:11);
}
std::vector<chart_atlas::BlockMap> Maps(const Mesh &m) {
  std::vector<chart_atlas::BlockMap> out;
  for(int gid=0;gid<m.nmb_total;++gid)out.push_back(m.atlas->GetBlockMap(m.lloc_eachmb[gid]));
  return out;
}
std::array<Real,3> Chart(const chart_atlas::BlockMap &m,int n,int i,int j,int k) {
  const int index[3]={i,j,k};std::array<Real,3> q;
  for(int a=0;a<3;++a)q[a]=m.lo[a]+(index[a]+m.node_offset[a])/n*(m.hi[a]-m.lo[a]);
  return q;
}
Real Power(Real x,int p) {Real out=1;for(int i=0;i<p;++i)out*=x;return out;}
std::vector<std::array<int,3>> Powers() {
  std::vector<std::array<int,3>> out{{0,0,0},{1,2,3},{5,5,5}};
  for(int a=0;a<3;++a)for(int p=1;p<=5;++p) {std::array<int,3> q{};q[a]=p;out.push_back(q);}
  return out;
}
Real Polynomial(const std::array<Real,3> &q,const std::array<int,3> &p) {
  return Power(q[0],p[0])*Power(q[1],p[1])*Power(q[2],p[2]);
}
struct Error {
  std::uint64_t rows=0,fallback=0,polynomials=0,chart_rejections=0;
  Real max_ratio=0,smooth=0,global_smooth=0,remainder_ratio=0;
};
void CheckRow(const std::vector<h::WeightedDonor> &row,const h::RemeshLayout &l,
    const chart_atlas::BlockMap &parent,const std::array<Real,3> &target_chart,
    const Real target_x[3],bool fallback,Error &error) {
  Require(!row.empty(),"empty interpolation row");++error.rows;error.fallback+=fallback;
  std::vector<std::array<Real,3>> q;std::vector<Real> physical;
  Real sum=0,norm=0,chart_value=0,physical_value=0;
  for(const auto &term:row) {
    const auto &map=l.old_maps.at(term.point.gid);
    const auto at=Chart(map,l.cells,term.point.i-l.origin,term.point.j-l.origin,term.point.k-l.origin);
    std::array<Real,3> u;Real local[3],x[3];
    for(int a=0;a<3;++a) {u[a]=(at[a]-parent.lo[a])/(parent.hi[a]-parent.lo[a]);
      local[a]=(at[a]-map.lo[a])/(map.hi[a]-map.lo[a]);}
    map.Forward(local,x);q.push_back(u);
    const Real f=std::exp(40*x[0]+30*x[1]+20*x[2]);physical.push_back(f);
    sum+=term.weight;norm+=std::abs(term.weight);chart_value+=term.weight*map.chart_id;
    physical_value+=term.weight*f;
  }
  const Real eps=std::numeric_limits<Real>::epsilon();
  Require(std::abs(sum-1)<=4096*eps*std::max(Real(1),norm),"constant exceeds a priori roundoff gate");
  if(fallback) {
    if(std::abs(chart_value-parent.chart_id)>.1)++error.chart_rejections;
    const Real expected=std::exp(40*target_x[0]+30*target_x[1]+20*target_x[2]);
    error.global_smooth=std::max(error.global_smooth,std::abs(physical_value-expected)/expected);
    // Exact error identity for a globally smooth degree-six physical field.
    // Cartesian r^6 is x^6+y^6+z^6 plus tensor-reproducible mixed terms.
    // On spherical charts it is a degree-six polynomial of physical radius.
    const bool cartesian=l.old_maps.at(row.front().point.gid).face<0;
    std::set<Real> nodes[3];Real got6=0,norm6=0;
    for(const auto &term:row) {
      const auto &map=l.old_maps.at(term.point.gid);
      Require((map.face<0)==cartesian,"mixed Cartesian/shell row invalidates exact remainder identity");
      if(cartesian)Require(term.point.gid==row.front().point.gid,"Cartesian remainder needs one tensor leaf");
      const int index[3]={term.point.i-l.origin,term.point.j-l.origin,term.point.k-l.origin};
      Real local[3],x[3];for(int a=0;a<3;++a)local[a]=(index[a]+map.node_offset[a])/l.cells;
      map.Forward(local,x);
      const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
      const Real value=1e6*r2*r2*r2;
      got6+=term.weight*value;norm6+=std::abs(term.weight*value);
      if(cartesian)for(int a=0;a<3;++a)nodes[a].insert(x[a]);
      else nodes[0].insert(map.inner_radius+(map.lo[0]+local[0]*(map.hi[0]-map.lo[0]))
          *(map.outer_radius-map.inner_radius));
    }
    const Real target_r2=target_x[0]*target_x[0]+target_x[1]*target_x[1]+target_x[2]*target_x[2];
    const Real exact6=1e6*target_r2*target_r2*target_r2;Real predicted_error=0;
    for(int a=0;a<(cartesian?3:1);++a) {
      Require(nodes[a].size()==1||nodes[a].size()==6,"unexpected remainder node count");
      if(nodes[a].size()==1)continue;
      Real product=1;const Real at=cartesian?target_x[a]:std::sqrt(target_r2);
      for(Real node:nodes[a])product*=at-node;
      predicted_error-=1e6*product;
    }
    const Real bound=4096*eps*std::max({Real(1),norm6,std::abs(exact6)});
    const Real ratio=std::abs((got6-exact6)-predicted_error)/bound;
    error.remainder_ratio=std::max(error.remainder_ratio,ratio);
    Require(ratio<=1,"physical r^6 fallback error violates exact interpolation remainder identity");
    return;
  }
  std::array<Real,3> u;
  for(int a=0;a<3;++a)u[a]=(target_chart[a]-parent.lo[a])/(parent.hi[a]-parent.lo[a]);
  for(const auto &powers:Powers()) {
    Real actual=0,condition=0;
    for(size_t t=0;t<row.size();++t) {
      const Real term=row[t].weight*Polynomial(q[t],powers);
      actual+=term;condition+=std::abs(term);
    }
    const Real expected=Polynomial(u,powers);
    const Real bound=4096*eps*std::max({Real(1),condition,std::abs(expected)});
    const Real ratio=std::abs(actual-expected)/bound;
    error.max_ratio=std::max(error.max_ratio,ratio);++error.polynomials;
    Require(ratio<=1,"same-family polynomial exceeds a priori coefficient-norm bound");
  }
  Real actual=0;
  for(size_t t=0;t<row.size();++t)actual+=row[t].weight*std::exp(2*q[t][0]+3*q[t][1]+4*q[t][2]);
  const Real expected=std::exp(2*u[0]+3*u[1]+4*u[2]);
  error.smooth=std::max(error.smooth,std::abs(actual-expected)/expected);
}
std::vector<std::array<int,3>> Probes(const chart_atlas::BlockMap &map,int n) {
  std::vector<std::array<int,3>> points;
  std::vector<int> axes[3];
  for(int a=0;a<3;++a) {
    const int lo=-map.overlap_sides[2*a],hi=n+map.overlap_sides[2*a+1];
    axes[a]={lo,lo+1,lo+2,0,n/4,n/2-1,n/2,3*n/4,n-1,hi-3,hi-2,hi-1};
    std::sort(axes[a].begin(),axes[a].end());axes[a].erase(std::unique(axes[a].begin(),axes[a].end()),axes[a].end());
  }
  for(size_t a=0;a<axes[0].size();++a)for(size_t b=0;b<axes[1].size();++b)
    points.push_back({axes[0][a],axes[1][b],axes[2][(a+3*b)%axes[2].size()]});
  return points;
}
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Error results[3][2][3],overlap[3];
    for(int resolution=0;resolution<3;++resolution) {
      const int n=12<<resolution;ParameterInput base_input,fine_input;
      Input(base_input,argv[1],n,false);Input(fine_input,argv[1],n,true);
      Mesh base(&base_input);base.BuildTreeFromScratch(&base_input);
      Mesh fine(&fine_input);fine.BuildTreeFromScratch(&fine_input);
      Require(base.nmb_total==106&&fine.nmb_total==127,"unexpected metadata-only refinement topology");
      const auto base_maps=Maps(base),fine_maps=Maps(fine);
      h::NativeDonorSelector base_selector(base,7),fine_selector(fine,7);
      for(int family=0;family<3;++family) {
        int parent=-1;for(int gid=0;gid<base.nmb_total;++gid)if(Selected(base_maps[gid],family)) {
          Require(parent<0,"ambiguous selected parent");parent=gid;}
        Require(parent>=0,"missing selected parent");const auto loc=base.lloc_eachmb[parent];
        std::vector<int> children;
        for(int gid=0;gid<fine.nmb_total;++gid) {const auto c=fine.lloc_eachmb[gid];
          if(c.chart_id==loc.chart_id&&c.level==loc.level+1&&c.lx1/2==loc.lx1
              &&c.lx2/2==loc.lx2&&c.lx3/2==loc.lx3)children.push_back(gid);}
        Require(children.size()==8,"replacement family is not an octet");
        for(int mode=0;mode<2;++mode) {
          h::RemeshLayout l;l.cells=n;l.origin=7;
          l.old_maps=mode?fine_maps:base_maps;l.new_maps=mode?base_maps:fine_maps;
          const auto &selector=mode?fine_selector:base_selector;
          const std::vector<int> targets=mode?std::vector<int>{parent}:children;
          const std::vector<int> sources=mode?children:std::vector<int>{parent};
          for(int gid:targets)for(const auto &point:Probes(l.new_maps[gid],n)) {
            if(family==0&&point[0]>=0&&point[0]<n&&point[1]>=0&&point[1]<n
                &&point[2]>=0&&point[2]<n)continue;  // nominal core uses native CC kernels
            bool fallback=false;
            const auto row=h::BuildBulkTransferRow(l,selector,sources,gid,point[0],point[1],point[2],&fallback);
            const auto &map=l.new_maps[gid];const auto chart=Chart(map,n,point[0],point[1],point[2]);
            Real local[3],x[3];for(int a=0;a<3;++a)local[a]=(chart[a]-map.lo[a])/(map.hi[a]-map.lo[a]);
            map.Forward(local,x);
            CheckRow(row,l,base_maps[parent],chart,x,fallback,results[resolution][mode][family]);
          }
        }
        if(family==1)for(int bits=0;bits<8;++bits) {
          // Targets straddle all three internal child interfaces. They lie
          // outside the Cartesian cube and have no centered child stencil.
          Real q[3];for(int a=0;a<3;++a)q[a]=.5+((bits>>a)&1?1:-1)*.125/n;
          Real x[3];base_maps[parent].Forward(q,x);
          const auto row=fine_selector.Interpolate(x,6);const int owner=row.front().point.gid;
          Require(fine_maps[owner].layer==0&&fine_maps[owner].face==0,"first-shell test took another chart");
          for(const auto &term:row)Require(term.point.gid==owner,"one-sided row mixes leaves");
          Real local[3];fine_maps[owner].Inverse(x,local);bool one_sided=false;
          for(int a=0;a<3;++a) {const int first=std::floor(n*local[a]-fine_maps[owner].node_offset[a])-2;
            one_sided=one_sided||first<-fine_maps[owner].overlap_sides[2*a]
                ||first+6>n+fine_maps[owner].overlap_sides[2*a+1];}
          Require(one_sided,"first-shell test did not need the new one-sided fallback");
          std::array<Real,3> chart;for(int a=0;a<3;++a)chart[a]=fine_maps[owner].lo[a]
              +local[a]*(fine_maps[owner].hi[a]-fine_maps[owner].lo[a]);
          h::RemeshLayout l;l.cells=n;l.origin=7;l.old_maps=fine_maps;
          CheckRow(row,l,fine_maps[owner],chart,x,false,overlap[resolution]);
        }
      }
      if(!global_variable::my_rank)for(int mode=0;mode<2;++mode)for(int family=0;family<3;++family) {
        const auto &r=results[resolution][mode][family];
        std::cout<<std::setprecision(17)<<"bulk_rows n="<<n<<" mode="<<mode<<" family="<<family
          <<" rows="<<r.rows<<" fallback="<<r.fallback<<" polynomials="<<r.polynomials
          <<" chart_rejections="<<r.chart_rejections<<" max_ratio="<<r.max_ratio
          <<" smooth="<<r.smooth<<" global_smooth="<<r.global_smooth
          <<" remainder_ratio="<<r.remainder_ratio<<std::endl;
      }
      Require(results[resolution][1][1].fallback>0&&results[resolution][1][1].chart_rejections>0,
          "first-shell coarsening did not expose the chart-lineage/global-fallback distinction");
      if(!global_variable::my_rank)std::cout<<"overlap_rows n="<<n<<" rows="<<overlap[resolution].rows
          <<" max_ratio="<<overlap[resolution].max_ratio<<" smooth="<<overlap[resolution].smooth<<std::endl;
    }
    for(int mode=0;mode<2;++mode)for(int family=0;family<3;++family)
      for(int r=1;r<3;++r) {
        const Real coarse=results[r-1][mode][family].smooth,fine=results[r][mode][family].smooth;
        const Real order=std::log2(coarse/fine);
        Require(fine<4096*std::numeric_limits<Real>::epsilon()||order>=4.5,
            "same-family smooth interpolation did not converge at expected sixth order");
        if(!global_variable::my_rank)std::cout<<"convergence mode="<<mode<<" family="<<family
            <<" n="<<(12<<r)<<" order="<<order<<std::endl;
      }
    for(int r=1;r<3;++r) {
      const Real order=std::log2(overlap[r-1].smooth/overlap[r].smooth);
      Require(overlap[r].smooth<4096*std::numeric_limits<Real>::epsilon()||order>=4.5,
          "one-sided first-shell interpolation did not converge at expected sixth order");
      if(!global_variable::my_rank)std::cout<<"overlap_convergence n="<<(12<<r)<<" order="<<order<<std::endl;
    }
    for(int family=0;family<2;++family)for(int r=1;r<3;++r) {
      const Real fine=results[r][1][family].global_smooth;
      const Real order=std::log2(results[r-1][1][family].global_smooth/fine);
      Require(fine<4096*std::numeric_limits<Real>::epsilon()||order>=4.5,
          "global physical fallback did not meet the general smooth convergence gate");
      if(!global_variable::my_rank)std::cout<<"fallback_convergence family="<<family
          <<" n="<<(12<<r)<<" order="<<order<<std::endl;
    }
    if(!global_variable::my_rank)std::cout<<"PASS bulk family and first-shell one-sided interpolation metadata\n";
  }}catch(const std::exception &error) {std::cerr<<error.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
  Kokkos::finalize();MPI_Finalize();return 0;
}
