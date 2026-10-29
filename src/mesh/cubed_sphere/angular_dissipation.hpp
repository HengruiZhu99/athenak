// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_ANGULAR_DISSIPATION_HPP_
#define MESH_CUBED_SPHERE_ANGULAR_DISSIPATION_HPP_

#include <climits>
#include <limits>
#include <map>
#include <numeric>
#include "mesh/cubed_sphere/scri_surface.hpp"

namespace cubed_sphere {

// Optional high-pass residual on actual global sphere nodes. This is an
// additive dissipative operator, not a boundary condition or SPD interpolant.
// Each incidence is b*(u[donor]-u[row]). Its transpose scatters the SAME b to
// the donor and -b to the anchor, without a separately rounded diagonal.
class AngularDissipationPlan {
 public:
  static constexpr int native_fields=31, free_fields=21;
  KOKKOS_INLINE_FUNCTION static constexpr int FreeField(int f) {return f<18?f:f+1;}
  struct Term {int point;Real coefficient;};
  struct Statistics {
    int min_neighbors=INT_MAX,max_neighbors=0,max_incoming=0,cross_face_rows=0;
    size_t incidences=0;
    Real max_condition_bound=0,max_moment_error=0,max_orthogonality_error=0;
    Real max_coefficient_l1=0,max_radius_over_h=0,min_area=0,max_area=0;
    Real spectral_upper_bound=0;
  };

 private:
  int n_,degree_,points_;
  std::vector<std::array<Real,3>> coordinates_;
  std::vector<Real> areas_;
  std::vector<int> offsets_,transpose_offsets_;
  std::vector<Term> terms_,transpose_terms_;
  Statistics statistics_;

  static Real Dot(const std::array<Real,3>&a,const std::array<Real,3>&b) {
    Real out=0;for(int d=0;d<3;++d)out+=a[d]*b[d];return out;
  }
  static Real Primitive(Real u,Real v) {return std::atan2(u*v,std::sqrt(1+u*u+v*v));}
  static std::vector<Real> Legendre(Real x,int degree) {
    std::vector<Real> p(degree+1,1);p[1]=x;
    for(int d=2;d<=degree;++d)p[d]=((2*d-1)*x*p[d-1]-(d-1)*p[d-2])/d;
    return p;
  }
  // Twice-orthogonalized modified Gram-Schmidt: V=QR. Then solve R^T z=e
  // and c=Qz, the unique minimum-norm solution of V^T c=e. No normal equations
  // or external linear-algebra dependency are introduced.
  std::vector<Real> Reconstruct(const std::vector<Real>&v,int rows,int columns,
                                const std::vector<Real>&evaluation) {
    std::vector<Real> q(v),r(columns*columns,0);
    for(int j=0;j<columns;++j) {
      Real original=0;for(int k=0;k<rows;++k)original+=v[k*columns+j]*v[k*columns+j];
      for(int pass=0;pass<2;++pass)for(int i=0;i<j;++i) {
        Real projection=0;for(int k=0;k<rows;++k)projection+=q[k*columns+i]*q[k*columns+j];
        r[i*columns+j]+=projection;
        for(int k=0;k<rows;++k)q[k*columns+j]-=projection*q[k*columns+i];
      }
      Real length=0;for(int k=0;k<rows;++k)length+=q[k*columns+j]*q[k*columns+j];
      length=std::sqrt(length);
      if(!(length>1e-12*std::sqrt(original))||!std::isfinite(length))
        throw std::runtime_error("angular polynomial moments are rank deficient");
      r[j*columns+j]=length;for(int k=0;k<rows;++k)q[k*columns+j]/=length;
    }
    Real orthogonality=0;
    for(int i=0;i<columns;++i)for(int j=0;j<columns;++j) {
      Real value=-(i==j);for(int k=0;k<rows;++k)value+=q[k*columns+i]*q[k*columns+j];
      orthogonality=std::max(orthogonality,std::abs(value));
    }
    statistics_.max_orthogonality_error=std::max(statistics_.max_orthogonality_error,orthogonality);
    std::vector<Real> inverse(columns*columns,0);
    for(int column=0;column<columns;++column)for(int i=columns-1;i>=0;--i) {
      Real value=i==column;for(int j=i+1;j<columns;++j)value-=r[i*columns+j]*inverse[j*columns+column];
      inverse[i*columns+column]=value/r[i*columns+i];
    }
    Real norm=0,inverse_norm=0;
    for(int i=0;i<columns;++i) {
      Real a=0,b=0;for(int j=0;j<columns;++j){a+=std::abs(r[i*columns+j]);b+=std::abs(inverse[i*columns+j]);}
      norm=std::max(norm,a);inverse_norm=std::max(inverse_norm,b);
    }
    const Real condition=norm*inverse_norm;
    if(!(condition<1e8)||orthogonality>1e-11)
      throw std::runtime_error("angular polynomial QR conditioning check failed");
    statistics_.max_condition_bound=std::max(statistics_.max_condition_bound,condition);
    std::vector<Real> z(columns,0),c(rows,0);
    for(int i=0;i<columns;++i) {
      Real value=evaluation[i];for(int j=0;j<i;++j)value-=r[j*columns+i]*z[j];
      z[i]=value/r[i*columns+i];
    }
    for(int k=0;k<rows;++k)for(int j=0;j<columns;++j)c[k]+=q[k*columns+j]*z[j];
    return c;
  }
  void CheckShape(const std::vector<Real>&values,int fields)const {
    if(values.size()!=size_t(points_)*fields)throw std::invalid_argument("angular dissipation state shape");
  }
  static void CheckEpsilon(Real epsilon) {
    if(!(epsilon>=0)||!std::isfinite(epsilon))throw std::invalid_argument("invalid angular dissipation epsilon");
  }

 public:
  AngularDissipationPlan(const ScriSurface&surface,int n,int degree=7)
      :n_(n),degree_(degree),points_(surface.rows.size()) {
    if(sizeof(Real)!=sizeof(double)||n<6||n!=surface.grid.n||(degree!=5&&degree!=7)
        ||surface.grid.mesh.face_subdivisions!=1||size_t(points_)!=size_t(6)*n*n
        ||size_t(points_)>size_t(INT_MAX)/native_fields)
      throw std::invalid_argument("unsupported angular dissipation grid/precision/degree");
    coordinates_.resize(points_);areas_.resize(points_);std::vector<int> faces(points_);
    statistics_.min_area=std::numeric_limits<Real>::max();
    for(int i=0;i<points_;++i) {
      const auto&row=surface.rows[i];const auto&block=surface.grid.mesh.blocks.at(row.block);
      if(block.layer!=int(surface.grid.mesh.radii.size())-1||block.face<0||row.j<0||row.j>=n||row.k<0||row.k>=n)
        throw std::invalid_argument("angular dissipation requires actual outer-sphere rows");
      for(int a=0;a<3;++a)coordinates_[i][a]=row.geometry.x[a];
      const Real radius_squared=Dot(coordinates_[i],coordinates_[i]);
      if(!std::isfinite(radius_squared)||std::abs(radius_squared-1)>2e-12)
        throw std::invalid_argument("angular dissipation requires unit reference sphere");
      faces[i]=block.face;
      const Real ul=surface.grid.mesh.AngularRatio(-1+Real(2*row.j)/n),uh=surface.grid.mesh.AngularRatio(-1+Real(2*(row.j+1))/n);
      const Real vl=surface.grid.mesh.AngularRatio(-1+Real(2*row.k)/n),vh=surface.grid.mesh.AngularRatio(-1+Real(2*(row.k+1))/n);
      areas_[i]=Primitive(uh,vh)-Primitive(ul,vh)-Primitive(uh,vl)+Primitive(ul,vl);
      if(!(areas_[i]>0)||!std::isfinite(areas_[i]))throw std::runtime_error("nonpositive angular reference area");
      statistics_.min_area=std::min(statistics_.min_area,areas_[i]);
      statistics_.max_area=std::max(statistics_.max_area,areas_[i]);
    }
    std::vector<std::array<int,2>> powers;
    for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b)powers.push_back({a,b});
    const int moments=powers.size(),target=2*moments+6;
    const auto zero=Legendre(0,degree);std::vector<Real> evaluation(moments);
    for(int f=0;f<moments;++f)evaluation[f]=zero[powers[f][0]]*zero[powers[f][1]];
    offsets_.push_back(0);std::vector<Real> distance(points_),ordered(points_);
    std::vector<int> incoming(points_,0);
    for(int i=0;i<points_;++i) {
      for(int j=0;j<points_;++j)distance[j]=2-2*Dot(coordinates_[i],coordinates_[j]);
      ordered=distance;std::nth_element(ordered.begin(),ordered.begin()+target-1,ordered.end());
      const Real cutoff=ordered[target-1];std::vector<int> neighbors;
      for(int j=0;j<points_;++j)if(distance[j]<=cutoff+2e-13)neighbors.push_back(j);
      const int count=neighbors.size();
      if(count>2*target)throw std::runtime_error("unbounded angular neighbor ties");
      statistics_.min_neighbors=std::min(statistics_.min_neighbors,count);
      statistics_.max_neighbors=std::max(statistics_.max_neighbors,count);
      statistics_.max_radius_over_h=std::max(statistics_.max_radius_over_h,n*std::sqrt(std::max(Real(0),cutoff)));
      const auto normal=coordinates_[i];int axis=0;
      for(int a=1;a<3;++a)if(std::abs(normal[a])<std::abs(normal[axis]))axis=a;
      std::array<Real,3> e1{},e2{};
      for(int a=0;a<3;++a)e1[a]=(a==axis)-normal[a]*normal[axis];
      const Real length=std::sqrt(Dot(e1,e1));for(auto&v:e1)v/=length;
      for(int a=0;a<3;++a)e2[a]=normal[(a+1)%3]*e1[(a+2)%3]-normal[(a+2)%3]*e1[(a+1)%3];
      std::vector<Real> xi(count),eta(count);Real radius=0;int center=-1;bool cross=false;
      for(int j=0;j<count;++j) {
        const auto point=coordinates_[neighbors[j]];std::array<Real,3> delta{};
        if(!(Dot(point,normal)>0))throw std::runtime_error("angular stencil leaves tangent hemisphere");
        for(int a=0;a<3;++a)delta[a]=point[a]-normal[a];
        xi[j]=Dot(delta,e1);eta[j]=Dot(delta,e2);radius=std::max(radius,std::hypot(xi[j],eta[j]));
        if(neighbors[j]==i)center=j;cross=cross||faces[neighbors[j]]!=faces[i];
      }
      if(center<0||!(radius>0))throw std::runtime_error("angular stencil has no center/scale");
      statistics_.cross_face_rows+=cross;
      std::vector<Real> v(count*moments);
      for(int j=0;j<count;++j) {
        const auto px=Legendre(xi[j]/radius,degree),py=Legendre(eta[j]/radius,degree);
        for(int f=0;f<moments;++f)v[j*moments+f]=px[powers[f][0]]*py[powers[f][1]];
      }
      const auto c=Reconstruct(v,count,moments,evaluation);
      Real one_norm=0,center_coefficient=0;
      for(int j=0;j<count;++j)if(j!=center) {
        const Real coefficient=-c[j];
        if(!std::isfinite(coefficient))throw std::runtime_error("nonfinite angular incidence");
        center_coefficient-=coefficient;one_norm+=std::abs(coefficient);
        terms_.push_back({neighbors[j],coefficient});++incoming[neighbors[j]];
      }
      one_norm+=std::abs(center_coefficient);
      if(one_norm>std::sqrt(Real(count))*(1+1e-10))throw std::runtime_error("angular residual coefficient bound failed");
      statistics_.max_coefficient_l1=std::max(statistics_.max_coefficient_l1,one_norm);
      Real error=0;
      for(int f=0;f<moments;++f) {
        Real residual=0;for(int j=0;j<count;++j)if(j!=center)
          residual-=c[j]*(v[j*moments+f]-v[center*moments+f]);
        error=std::max(error,std::abs(residual));
      }
      statistics_.max_moment_error=std::max(statistics_.max_moment_error,error);
      if(error>1e-11)throw std::runtime_error("angular polynomial reproduction failed");
      if(terms_.size()>size_t(INT_MAX))throw std::length_error("angular incidence plan too large");
      offsets_.push_back(terms_.size());
    }
    statistics_.max_incoming=*std::max_element(incoming.begin(),incoming.end());
    statistics_.incidences=terms_.size();
    std::vector<std::vector<Term>> transpose(points_);
    std::vector<long double> bound(points_,0);
    for(int row=0;row<points_;++row) {
      long double row_sum=0;
      for(int k=offsets_[row];k<offsets_[row+1];++k) {
        const auto&t=terms_[k];
        transpose[t.point].push_back({row,t.coefficient});
        transpose[row].push_back({row,-t.coefficient});
        // Absolute incidence sum also bounds the absolute effective anchor
        // diagonal. This is deliberately more conservative than the prototype
        // bound that combines the anchor coefficients before taking abs().
        row_sum+=std::abs(static_cast<long double>(t.coefficient))
          *(1+std::sqrt(static_cast<long double>(areas_[row])/areas_[t.point]));
      }
      for(int k=offsets_[row];k<offsets_[row+1];++k) {
        const auto&t=terms_[k];const auto magnitude=std::abs(static_cast<long double>(t.coefficient));
        bound[row]+=magnitude*row_sum;
        bound[t.point]+=magnitude*std::sqrt(static_cast<long double>(areas_[row])/areas_[t.point])*row_sum;
      }
    }
    statistics_.spectral_upper_bound=static_cast<Real>(*std::max_element(bound.begin(),bound.end()))*(1+1e-12);
    transpose_offsets_.push_back(0);
    for(const auto&row:transpose) {
      transpose_terms_.insert(transpose_terms_.end(),row.begin(),row.end());
      if(transpose_terms_.size()>size_t(INT_MAX))throw std::length_error("angular transpose plan too large");
      transpose_offsets_.push_back(transpose_terms_.size());
    }
  }

  std::vector<Real> Residual(const std::vector<Real>&values)const {
    CheckShape(values,native_fields);std::vector<Real> out(size_t(points_)*free_fields,0);
    for(int row=0;row<points_;++row)for(int f=0;f<free_fields;++f) {
      const int field=FreeField(f);const Real anchor=values[native_fields*row+field];
      if(!std::isfinite(anchor))throw std::invalid_argument("nonfinite angular input");
      Real value=0;for(int k=offsets_[row];k<offsets_[row+1];++k) {
        const auto&t=terms_[k];const Real donor=values[native_fields*t.point+field];
        if(!std::isfinite(donor))throw std::invalid_argument("nonfinite angular input");
        value+=t.coefficient*(donor-anchor);
      }
      if(!std::isfinite(value))throw std::runtime_error("nonfinite angular residual");
      out[free_fields*row+f]=value;
    }
    return out;
  }
  std::vector<Real> ApplyResidual(const std::vector<Real>&residual,Real epsilon)const {
    CheckShape(residual,free_fields);CheckEpsilon(epsilon);
    std::vector<Real> out(size_t(points_)*native_fields,0);
    for(int row=0;row<points_;++row)for(int f=0;f<free_fields;++f) {
      Real value=0;for(int k=transpose_offsets_[row];k<transpose_offsets_[row+1];++k) {
        const auto&t=transpose_terms_[k];const Real donor=residual[free_fields*t.point+f];
        if(!std::isfinite(donor))throw std::invalid_argument("nonfinite angular residual input");
        value+=t.coefficient*areas_[t.point]*donor;
      }
      value*=-epsilon*n_/areas_[row];
      if(!std::isfinite(value))throw std::runtime_error("nonfinite angular dissipation output");
      out[native_fields*row+FreeField(f)]=value;
    }
    return out;
  }
  std::vector<Real> Apply(const std::vector<Real>&values,Real epsilon)const {
    CheckEpsilon(epsilon);return ApplyResidual(Residual(values),epsilon);
  }
  int N()const{return n_;}
  int Degree()const{return degree_;}
  int Points()const{return points_;}
  Real SpectralUpperBound()const{return statistics_.spectral_upper_bound;}
  const Statistics&Stats()const{return statistics_;}
  const std::vector<std::array<Real,3>>&Coordinates()const{return coordinates_;}
  const std::vector<Real>&Areas()const{return areas_;}
  const std::vector<int>&Offsets()const{return offsets_;}
  const std::vector<Term>&Terms()const{return terms_;}
  const std::vector<int>&TransposeOffsets()const{return transpose_offsets_;}
  const std::vector<Term>&TransposeTerms()const{return transpose_terms_;}
};
}  // namespace cubed_sphere
#endif  // MESH_CUBED_SPHERE_ANGULAR_DISSIPATION_HPP_
