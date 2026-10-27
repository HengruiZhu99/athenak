// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_SCRI_SURFACE_HPP_
#define MESH_CUBED_SPHERE_SCRI_SURFACE_HPP_
#include "mesh/cubed_sphere/composite_difference.hpp"

namespace cubed_sphere {
// An evolved physical-boundary atlas, separate from equal-shaped volume blocks.
// Every trace has five interior radial nodes plus the actual Omega=0 node.
// Components remain global Cartesian; all Cartesian first/second chain-rule
// terms are precomputed. Surface values communicate through physical ownership.
class ScriSurface {
 public:
  struct Term {
    int point;
    bool boundary;
    Real w{}, d[3]{}, dd[3][3]{};
  };
  struct Row {
    int block, j, k;
    Point geometry;
    Real radial_spacing;
    std::vector<Term> terms;
  };
  struct Jets {
    std::vector<Real> value;
    std::vector<std::array<Real,3>> d;
    std::vector<std::array<std::array<Real,3>,3>> dd;
  };
  const CompositeDifference &grid;
  std::vector<Row> rows;
  std::vector<int> base;
  std::vector<Real> values, beta;
  int components;

  explicit ScriSurface(const CompositeDifference &g,int fields)
      :grid(g),base(g.mesh.blocks.size(),-1),components(fields) {
    if(g.n<6)throw std::invalid_argument("regular tensor scri needs six nodes per block axis");
    const int n=g.n;
    for(size_t m=0;m<g.mesh.blocks.size();++m){
      if(g.mesh.blocks[m].layer!=static_cast<int>(g.mesh.radii.size())-1)continue;
      base[m]=rows.size();
      for(int k=0;k<n;++k)for(int j=0;j<n;++j){
        Row row;row.block=m;row.j=j;row.k=k;
        const Real q[3]={1,(j+.5)/n,(k+.5)/n};row.geometry=g.mesh.Geometry(m,q);
        Real inverse_radial=0,radius=0;
        for(int a=0;a<3;++a)radius+=row.geometry.x[a]*row.geometry.x[a];radius=std::sqrt(radius);
        for(int a=0;a<3;++a)inverse_radial+=row.geometry.J[0][a]*row.geometry.x[a]/radius;
        row.radial_spacing=1/(inverse_radial*n);
        if(!(row.radial_spacing>0))throw std::runtime_error("invalid scri normal map");
        rows.push_back(std::move(row));
      }
    }
    values.assign(rows.size()*fields,0);beta.assign(rows.size()*3,0);
    for(auto &row:rows)Build(row);
  }

  std::vector<Real> Interpolate(const std::vector<Real> &data,int fields,const Real x[3]) const {
    const int owner=grid.mesh.Owner(x),n=grid.n;
    if(owner<0||base[owner]<0||data.size()!=rows.size()*fields)
      throw std::runtime_error("scri surface donor/shape mismatch");
    Real q[3];grid.mesh.Inverse(owner,x,q);int first[2];Real weights[2][6];
    for(int a=0;a<2;++a){Real t=n*q[a+1]-.5;first[a]=std::clamp(int(std::floor(t))-2,0,n-6);
      for(int i=0;i<6;++i){weights[a][i]=1;for(int j=0;j<6;++j)if(i!=j)weights[a][i]*=(t-first[a]-j)/Real(i-j);}}
    const int anchor=base[owner]+first[0]+n*first[1];
    std::vector<Real> out(fields);
    for(int f=0;f<fields;++f){out[f]=data[anchor*fields+f];
      for(int k=0;k<6;++k)for(int j=0;j<6;++j)
        out[f]+=weights[0][j]*weights[1][k]*(data[(base[owner]+first[0]+j+n*(first[1]+k))*fields+f]-data[anchor*fields+f]);}
    return out;
  }

  Jets Sample(size_t point,const std::vector<Real> &volume,const std::vector<Real> &surface,int fields) const {
    if(volume.size()!=size_t(grid.count*fields)||surface.size()!=rows.size()*fields)
      throw std::invalid_argument("scri jet state shape");
    const auto &row=rows.at(point);Jets out;
    out.value.resize(fields);out.d.resize(fields);out.dd.resize(fields);
    auto entry=[&](const Term &t,int f){return t.boundary?surface[t.point*fields+f]:volume[f*grid.count+t.point];};
    for(int f=0;f<fields;++f){const Real anchor=entry(row.terms[0],f);out.value[f]=anchor;
      for(const auto &term:row.terms){Real difference=entry(term,f)-anchor;out.value[f]+=term.w*difference;
        for(int a=0;a<3;++a){out.d[f][a]+=term.d[a]*difference;
          for(int b=0;b<3;++b)out.dd[f][a][b]+=term.dd[a][b]*difference;}}}
    return out;
  }

  // Independent residual of the actual ghost-lift boundary equation. It reads
  // six active values and the first ghost, NEVER the boundary value supplied to
  // Fill. Active-only extrapolation remains a separate truncation diagnostic.
  Real ReconstructFromGhost(size_t point,const std::vector<Real> &volume,int f) const {
    const auto &row=rows.at(point);const int n=grid.n;
    const Real anchor=volume[f*grid.count+grid.Index(row.block,n-6,row.j,row.k)];Real out=anchor;
    for(int i=0;i<7;++i){const int node=n-6+i;Real weight=1;
      for(int j=0;j<7;++j)if(i!=j)weight*=(n-.5-(n-6+j))/Real(i-j);
      out+=weight*(volume[f*grid.count+grid.Index(row.block,node,row.j,row.k)]-anchor);}
    return out;
  }

 private:
  void Build(Row &row){
    const int n=grid.n;int first[3]={n-5,std::clamp(row.j-2,0,n-6),std::clamp(row.k-2,0,n-6)};
    const Real q[3]={1,(row.j+.5)/n,(row.k+.5)/n};Jet basis[3][6];
    for(int a=0;a<3;++a)for(int i=0;i<6;++i){
      const Real node=a==0&&i==5?1:Real(first[a]+i+.5)/n;basis[a][i]=Jet(1);
      for(int j=0;j<6;++j)if(i!=j){const Real other=a==0&&j==5?1:Real(first[a]+j+.5)/n;
        basis[a][i]=basis[a][i]*(Jet::Variable(q[a],a)-Jet(other))/Jet(node-other);}}
    for(int k=0;k<6;++k)for(int j=0;j<6;++j)for(int i=0;i<6;++i){
      const Jet w=basis[0][i]*basis[1][j]*basis[2][k];Term term;
      term.boundary=i==5;term.point=term.boundary?base[row.block]+first[1]+j+n*(first[2]+k)
        :grid.Index(row.block,first[0]+i,first[1]+j,first[2]+k);term.w=w.f;
      for(int a=0;a<3;++a)for(int A=0;A<3;++A){term.d[a]+=row.geometry.J[A][a]*w.d[A];
        for(int b=0;b<3;++b){term.dd[a][b]+=row.geometry.H[A][a][b]*w.d[A];
          for(int B=0;B<3;++B)term.dd[a][b]+=row.geometry.J[A][a]*row.geometry.J[B][b]*w.h[A][B];}}
      row.terms.push_back(term);
    }
  }
};
}  // namespace cubed_sphere
#endif  // MESH_CUBED_SPHERE_SCRI_SURFACE_HPP_
