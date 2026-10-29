// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_PAIRED_TRACE_PENALTY_HPP_
#define MESH_CUBED_SPHERE_PAIRED_TRACE_PENALTY_HPP_
#include <map>
#include "mesh/cubed_sphere/composite_difference.hpp"
namespace cubed_sphere {
// Experimental ghost-plus-penalty operator, NOT an SBP-SAT closure.
// L = [T_C,-T_S], or h [N_C,-N_S], at the SAME physical trace point.
// P = -kappa W^-1 L^T A L, hence q^T W Pq = -kappa ||Lq||_A^2.
// W contains positive cell volumes on BOTH copies, not partition-of-unity
// diagnostic weights (which vanish at some points). A=surface area * core h.
// This identity applies to this additive term alone, not the bulk/ghost PDE.
class PairedTracePenalty {
 public:
  struct Term {int point;Real coefficient;};
  struct Row {std::vector<Term> terms;Real weight;int interface,normal;std::array<Real,3>x;};
  std::vector<Row> rows;
  std::vector<Real> volumes;
  Real upper_value=0,upper_normal=0;
  int core_traces=0,shell_traces=0;
 private:
  struct Basis {int point;Real value,normal;};
  static std::vector<Basis> Trace(const CompositeDifference&g,const Real*x,const Real*normal,bool core) {
    constexpr int width=6;
    int begin[3],lo[3],hi[3],block=-1;Real t[3],map[3][3]{};
    if(core){
      for(int a=0;a<3;++a){lo[a]=-g.mesh.CoreOverlap();hi[a]=g.cartesian_n+g.mesh.CoreOverlap();
        t[a]=(x[a]+g.mesh.half_width)/g.cartesian_h-.5;map[a][a]=1/g.cartesian_h;}
    }else{
      const int face=Map{}.Owner(x);
      for(int m=0;m<int(g.mesh.blocks.size());++m)if(g.mesh.blocks[m].layer==0&&g.mesh.blocks[m].face==face){block=m;break;}
      if(block<0)throw std::runtime_error("no first-shell trace block");
      Real q[3];g.mesh.Inverse(block,x,q);const auto geo=g.mesh.Geometry(block,q);
      for(int a=0;a<3;++a){lo[a]=g.Lower(block,a);hi[a]=g.Upper(block,a);t[a]=g.n*q[a]-.5;
        for(int d=0;d<3;++d)map[a][d]=g.n*geo.J[a][d];}
    }
    Real w[3][width]{},dw[3][width]{};
    for(int a=0;a<3;++a){
      // Interior, one-sided stencils near the artificial boundary are allowed;
      // extrapolation beyond the evolved sample hull is not.
      if(!std::isfinite(t[a])||t[a]<lo[a]-1e-10||t[a]>hi[a]-1+1e-10||hi[a]-lo[a]<width)
        throw std::runtime_error("paired trace outside donor sample hull");
      begin[a]=std::clamp(int(std::floor(t[a]))-2,lo[a],hi[a]-width);
      for(int i=0;i<width;++i){
        w[a][i]=1;for(int j=0;j<width;++j)if(j!=i)w[a][i]*=(t[a]-begin[a]-j)/Real(i-j);
        for(int k=0;k<width;++k)if(k!=i){Real v=1/Real(i-k);
          for(int j=0;j<width;++j)if(j!=i&&j!=k)v*=(t[a]-begin[a]-j)/Real(i-j);dw[a][i]+=v;}
      }
    }
    std::vector<Basis> out;
    for(int k=0;k<width;++k)for(int j=0;j<width;++j)for(int i=0;i<width;++i){
      const int p=core?g.CartesianPoint(begin[0]+i,begin[1]+j,begin[2]+k):g.ActiveIndex(block,begin[0]+i,begin[1]+j,begin[2]+k);
      if(p<0)throw std::runtime_error("paired trace reads ghost or missing donor");
      const int c[3]={i,j,k};Real dn=0;
      for(int a=0;a<3;++a){Real v=dw[a][c[a]];for(int b=0;b<3;++b)if(a!=b)v*=w[b][c[b]];
        for(int d=0;d<3;++d)dn+=normal[d]*map[a][d]*v;}
      out.push_back({p,w[0][i]*w[1][j]*w[2][k],dn});
    }
    return out;
  }
  void Add(const CompositeDifference&g,const Real*x,const Real*normal,Real area,int interface) {
    const auto C=Trace(g,x,normal,true),S=Trace(g,x,normal,false);
    for(int derivative=0;derivative<2;++derivative){
      Row row{{},area*g.cartesian_h,interface,derivative,{x[0],x[1],x[2]}};
      std::map<int,Real> values;
      for(const auto&t:C)values[t.point]+=derivative?g.cartesian_h*t.normal:t.value;
      for(const auto&t:S)values[t.point]-=derivative?g.cartesian_h*t.normal:t.value;
      for(const auto&t:values)if(std::abs(t.second)>1e-18)row.terms.push_back({t.first,t.second});
      rows.push_back(std::move(row));
    }
  }
 public:
  explicit PairedTracePenalty(const CompositeDifference&g):volumes(g.volumes){
    if(!g.mesh.continuous_cartesian_core||!g.mesh.spherical_inner_boundary||g.mesh.face_subdivisions!=1)
      throw std::invalid_argument("trace experiment requires uniform core and six spherical faces");
    const int o=g.mesh.CoreOverlap(),N=g.cartesian_n;const Real h=g.cartesian_h;
    // One evolved trace layer at each of the six outer Cartesian faces.
    // Edge/corner points appear with each face's normal and quadrature weight.
    for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side)
    for(int j=-o;j<N+o;++j)for(int k=-o;k<N+o;++k){
      int q[3];q[axis]=side?N+o-1:-o;q[(axis+1)%3]=j;q[(axis+2)%3]=k;
      const auto*x=g.geometry[g.CartesianPoint(q[0],q[1],q[2])].x;
      Real normal[3]{};normal[axis]=side?1:-1;
      Add(g,x,normal,h*h,0);++core_traces;
    }
    // First evolved spherical radial layer, with nominal angular ownership.
    // The annulus stays purely spherical; this is not a conforming cube seam.
    for(int m=0;m<int(g.mesh.blocks.size());++m)if(g.mesh.blocks[m].layer==0)
    for(int k=0;k<g.n;++k)for(int j=0;j<g.n;++j){
      const int p=g.ActiveIndex(m,0,j,k);const auto*x=g.geometry[p].x;
      const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      Real normal[3];for(int a=0;a<3;++a)normal[a]=x[a]/radius;
      const Real dr=(g.mesh.radii[0]-g.mesh.inner_radius)/g.n;
      Add(g,x,normal,g.volumes[p]/dr,1);++shell_traces;
    }
    // Infinity norm bound of W^-1/2 L^T A L W^-1/2; bounds its spectral radius.
    std::vector<Real> bounds[2];for(auto&b:bounds)b.assign(volumes.size(),0);
    for(const auto&r:rows){Real sum=0;for(const auto&t:r.terms)sum+=std::abs(t.coefficient)/std::sqrt(volumes[t.point]);
      for(const auto&t:r.terms)bounds[r.normal][t.point]+=r.weight*std::abs(t.coefficient)/std::sqrt(volumes[t.point])*sum;}
    upper_value=*std::max_element(bounds[0].begin(),bounds[0].end());
    upper_normal=*std::max_element(bounds[1].begin(),bounds[1].end());
  }
  std::vector<Real> Apply(const std::vector<Real>&q,Real kappa,Real normal_ratio,Real*negative_energy=nullptr)const{
    if(q.size()!=volumes.size()||!std::isfinite(kappa)||kappa<0||!std::isfinite(normal_ratio)||normal_ratio<0)
      throw std::invalid_argument("invalid paired penalty input");
    std::vector<Real> rate(q.size(),0);Real loss=0;
    for(const auto&r:rows){Real delta=0;for(const auto&t:r.terms)delta+=t.coefficient*q[t.point];
      const Real factor=kappa*r.weight*(r.normal?normal_ratio:1);loss-=factor*delta*delta;
      for(const auto&t:r.terms)rate[t.point]-=factor*t.coefficient*delta/volumes[t.point];}
    if(negative_energy)*negative_energy=loss;return rate;
  }
};
}
#endif
