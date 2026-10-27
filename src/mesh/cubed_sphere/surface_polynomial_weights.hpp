// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_SURFACE_POLYNOMIAL_WEIGHTS_HPP_
#define MESH_CUBED_SPHERE_SURFACE_POLYNOMIAL_WEIGHTS_HPP_
#include <numeric>
#include <utility>
#include <limits>
#include "mesh/cubed_sphere/map.hpp"
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace cubed_sphere {
// Host geometry-only degree5/21-term surface interpolation weights.
// No field values, normal jets, or stage data are cached here.
// Uses the independently qualified pivoted twice-reorthogonalized QR plan.
class SurfacePolynomialWeights {
 public:
 struct Row {std::vector<int> ids;std::vector<Real> weights;int rank=0;Real condition=0,l1=0,reproduction=0;};
 int donors;std::vector<std::array<Real,3>> points;
 mutable size_t rows_built=0,identity_rows=0;mutable Real maximum_l1=0,maximum_condition=0,maximum_reproduction=0;
 explicit SurfacePolynomialWeights(std::vector<std::array<Real,3>> coordinates,int count=48):donors(count),points(std::move(coordinates)){if(count<21||count>128||points.size()<size_t(count))throw std::invalid_argument("surface polynomial donor count");for(const auto&x:points)for(Real v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite surface coordinate");}
 Row Build(const Real*x)const{Row row;std::vector<std::pair<Real,int>> distance;distance.reserve(points.size());for(size_t i=0;i<points.size();++i){Real d=0;for(int a=0;a<3;++a)d+=std::pow(points[i][a]-x[a],2);if(d<1e-26){row.ids={int(i)};row.weights={1};row.rank=21;row.condition=row.l1=1;++identity_rows;return row;}distance.push_back({d,int(i)});}std::partial_sort(distance.begin(),distance.begin()+donors,distance.end());Real h=std::sqrt(distance[donors-1].first);if(!(h>0&&std::isfinite(h)))throw std::runtime_error("MLS invalid scale");Real normal[3],t1[3]{},t2[3]{},r=0;for(int a=0;a<3;++a)r+=x[a]*x[a];r=std::sqrt(r);for(int a=0;a<3;++a)normal[a]=x[a]/r;int axis=0;for(int a=1;a<3;++a)if(std::abs(normal[a])<std::abs(normal[axis]))axis=a;t1[axis]=1;Real norm=0;for(int a=0;a<3;++a){t1[a]-=normal[axis]*normal[a];norm+=t1[a]*t1[a];}for(auto&a:t1)a/=std::sqrt(norm);for(int a=0;a<3;++a)t2[a]=normal[(a+1)%3]*t1[(a+2)%3]-normal[(a+2)%3]*t1[(a+1)%3];
  constexpr int p=21;std::vector<Real> original[p],v[p],Q[p],sw(donors);Real R[p][p]{};int perm[p];for(int k=0;k<p;++k){perm[k]=k;v[k].resize(donors);original[k].resize(donors);}for(int i=0;i<donors;++i){row.ids.push_back(distance[i].second);const auto &node=points[row.ids.back()];Real dot=0,u=0,z=0;for(int a=0;a<3;++a){dot+=node[a]*normal[a];u+=node[a]*t1[a];z+=node[a]*t2[a];}if(!(dot>.15*r))throw std::runtime_error("MLS donor outside local hemisphere");u/=dot*h/r;z/=dot*h/r;sw[i]=std::exp(-2*distance[i].first/(h*h));int k=0;for(int degree=0;degree<=5;++degree)for(int a=0;a<=degree;++a){original[k][i]=std::pow(u,a)*std::pow(z,degree-a);v[k][i]=sw[i]*original[k][i];++k;}}
  Real largest=0,smallest=std::numeric_limits<Real>::max();for(int k=0;k<p;++k){int pivot=k;Real biggest=-1;for(int j=k;j<p;++j){Real n=0;for(Real a:v[j])n+=a*a;if(n>biggest){biggest=n;pivot=j;}}if(pivot!=k){std::swap(v[k],v[pivot]);std::swap(perm[k],perm[pivot]);for(int i=0;i<k;++i)std::swap(R[i][k],R[i][pivot]);}R[k][k]=std::sqrt(biggest);largest=std::max(largest,R[k][k]);smallest=std::min(smallest,R[k][k]);if(!(R[k][k]>1e-12*largest))throw std::runtime_error("MLS rank deficient");++row.rank;Q[k]=v[k];for(auto&a:Q[k])a/=R[k][k];for(int j=k+1;j<p;++j)for(int pass=0;pass<2;++pass){Real dot=0;for(int i=0;i<donors;++i)dot+=Q[k][i]*v[j][i];R[k][j]+=dot;for(int i=0;i<donors;++i)v[j][i]-=dot*Q[k][i];}}
  Real inverse[p][p]{},rnorm=0,inorm=0;for(int col=0;col<p;++col)for(int i=p-1;i>=0;--i){Real rhs=i==col?1:0;for(int j=i+1;j<p;++j)rhs-=R[i][j]*inverse[j][col];inverse[i][col]=rhs/R[i][i];}for(int i=0;i<p;++i){Real a=0,b=0;for(int j=0;j<p;++j){a+=std::abs(R[i][j]);b+=std::abs(inverse[i][j]);}rnorm=std::max(rnorm,a);inorm=std::max(inorm,b);}row.condition=rnorm*inorm;if(!(row.condition<1e8))throw std::runtime_error("MLS condition guard");Real y[p]{};for(int k=0;k<p;++k){Real b=perm[k]==0?1:0;for(int j=0;j<k;++j)b-=R[j][k]*y[j];y[k]=b/R[k][k];}row.weights.resize(donors);for(int i=0;i<donors;++i){for(int k=0;k<p;++k)row.weights[i]+=sw[i]*Q[k][i]*y[k];}Real sum=0;for(Real w:row.weights)sum+=w;row.weights[0]+=1-sum;for(Real w:row.weights)row.l1+=std::abs(w);if(!(row.l1<=8&&std::isfinite(row.l1)))throw std::runtime_error("MLS value L1 guard");for(int k=0;k<p;++k){Real value=0;for(int i=0;i<donors;++i)value+=row.weights[i]*original[k][i];row.reproduction=std::max(row.reproduction,std::abs(value-(k==0?1.:0)));}if(!(row.reproduction<1e-10))throw std::runtime_error("MLS polynomial reproduction guard");++rows_built;maximum_l1=std::max(maximum_l1,row.l1);maximum_condition=std::max(maximum_condition,row.condition);maximum_reproduction=std::max(maximum_reproduction,row.reproduction);return row;
 }

};
} // namespace cubed_sphere
#endif
