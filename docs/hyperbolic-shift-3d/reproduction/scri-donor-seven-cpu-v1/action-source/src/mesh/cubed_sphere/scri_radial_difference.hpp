// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_SCRI_RADIAL_DIFFERENCE_HPP_
#define MESH_CUBED_SPHERE_SCRI_RADIAL_DIFFERENCE_HPP_

#include "mesh/cubed_sphere/scri_surface.hpp"

namespace cubed_sphere {
// Optional CPU research closure. In the last three radial cell rows, use six
// active centers plus the evolved endpoint directly, without exterior radial
// ghost donors. Angular derivatives remain centered five-point, with Cartesian
// components interpolated across faces. This is a consistency control, not an
// SBP operator or a proven stable boundary treatment.
class ScriRadialDifference {
 public:
  using Term=ScriSurface::Term;
  const CompositeDifference &grid;
  const ScriSurface &scri;
  std::vector<int> row_index;
  std::vector<std::vector<Term>> rows;

  ScriRadialDifference(const CompositeDifference &g,const ScriSurface &s)
      :grid(g),scri(s),row_index(g.active.size(),-1) {
    if(g.order!=4||s.radial_nodes!=7||!s.cross_face_angular||g.n<6)
      throw std::invalid_argument("donor radial closure requires fourth-order bulk and seven-node cross-face scri");
    const int n=g.n;
    for(size_t m=0;m<g.mesh.blocks.size();++m)if(s.base[m]>=0)
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=n-3;i<n;++i){
      const size_t p=m*n*n*n+(k*n+j)*n+i;
      row_index[p]=rows.size();rows.push_back(Build(m,i,j,k,p));
    }
  }

  bool Contains(size_t p)const{return row_index.at(p)>=0;}
  void Derivatives(const std::vector<Real>&volume,const std::vector<Real>&surface,
                   int f,size_t p,Real d[3],Real dd[3][3],bool second)const{
    const auto &row=rows.at(row_index.at(p));
    const Real anchor=volume[f*grid.count+grid.active[p]];
    for(int a=0;a<3;++a){d[a]=0;for(int b=0;b<3;++b)dd[a][b]=0;}
    for(const auto &term:row){
      const Real v=(term.boundary?surface[term.point*scri.components+f]
                                    :volume[f*grid.count+term.point])-anchor;
      for(int a=0;a<3;++a){d[a]+=term.d[a]*v;
        if(second)for(int b=0;b<3;++b)dd[a][b]+=term.dd[a][b]*v;}
    }
  }

 private:
  std::vector<Term> Build(int m,int i,int j,int k,size_t p)const{
    const int n=grid.n,first[3]={n-6,j-2,k-2},width[3]={7,5,5};
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Jet basis[3][7];
    for(int a=0;a<3;++a)for(int b=0;b<width[a];++b){
      const Real node=a==0&&b==6?1:Real(first[a]+b+.5)/n;basis[a][b]=Jet(1);
      for(int c=0;c<width[a];++c)if(c!=b){
        const Real other=a==0&&c==6?1:Real(first[a]+c+.5)/n;
        basis[a][b]=basis[a][b]*(Jet::Variable(q[a],a)-Jet(other))/Jet(node-other);}
    }
    std::vector<Term> out;std::map<std::pair<bool,int>,size_t> positions;
    auto add=[&](const Term&t,int point,Real weight){
      if(weight==0)return;
      const auto key=std::make_pair(t.boundary,point);auto inserted=positions.emplace(key,out.size());
      if(inserted.second){Term entry{};entry.boundary=t.boundary;entry.point=point;out.push_back(entry);}
      auto &entry=out[inserted.first->second];
      for(int a=0;a<3;++a){entry.d[a]+=weight*t.d[a];
        for(int b=0;b<3;++b)entry.dd[a][b]+=weight*t.dd[a][b];}
    };
    const auto &geometry=grid.geometry[p];
    for(int c=0;c<5;++c)for(int b=0;b<5;++b)for(int a=0;a<7;++a){
      const Jet w=basis[0][a]*basis[1][b]*basis[2][c];Term t{};t.boundary=a==6;
      for(int x=0;x<3;++x)for(int A=0;A<3;++A){t.d[x]+=geometry.J[A][x]*w.d[A];
        for(int y=0;y<3;++y){t.dd[x][y]+=geometry.H[A][x][y]*w.d[A];
          for(int B=0;B<3;++B)t.dd[x][y]+=geometry.J[A][x]*geometry.J[B][y]*w.h[A][B];}}
      bool nonzero=false;for(int x=0;x<3;++x){nonzero=nonzero||t.d[x]!=0;
        for(int y=0;y<3;++y)nonzero=nonzero||t.dd[x][y]!=0;}
      if(!nonzero)continue;
      const int aj=first[1]+b,ak=first[2]+c;
      if(!t.boundary){add(t,grid.Index(m,first[0]+a,aj,ak),1);continue;}
      if(aj>=0&&aj<n&&ak>=0&&ak<n){add(t,scri.base[m]+aj+n*ak,1);continue;}
      const Real local[3]={1,(aj+.5)/n,(ak+.5)/n};Real x[3],donor_q[3];
      grid.mesh.Forward(m,local,x);const int owner=grid.mesh.Owner(x);
      if(owner<0||scri.base[owner]<0)throw std::runtime_error("radial closure surface donor");
      grid.mesh.Inverse(owner,x,donor_q);int begin[2];Real weights[2][6];
      for(int A=0;A<2;++A){const Real z=n*donor_q[A+1]-.5;begin[A]=std::clamp(int(std::floor(z))-2,0,n-6);
        for(int b2=0;b2<6;++b2){weights[A][b2]=1;
          for(int c2=0;c2<6;++c2)if(b2!=c2)weights[A][b2]*=(z-begin[A]-c2)/Real(b2-c2);}}
      for(int c2=0;c2<6;++c2)for(int b2=0;b2<6;++b2)
        add(t,scri.base[owner]+begin[0]+b2+n*(begin[1]+c2),weights[0][b2]*weights[1][c2]);
    }
    return out;
  }
};
} // namespace cubed_sphere
#endif
