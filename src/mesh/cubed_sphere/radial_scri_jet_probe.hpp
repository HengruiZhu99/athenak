// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_RADIAL_SCRI_JET_PROBE_HPP_
#define MESH_CUBED_SPHERE_RADIAL_SCRI_JET_PROBE_HPP_
#include "mesh/cubed_sphere/hermite_boundary_probe.hpp"
namespace cubed_sphere {
// DIAGNOSTIC ONLY. Unconstrained polynomial interpolation in a fixed smooth
// radial coordinate. All coefficients (including constant, linear, quadratic)
// remain free. Uses actual active cells across aligned shell layers, no ghosts.
// Angular refinement changing the angular ray lattice is explicitly unsupported.
class RadialScriJetProbe {
 public:
 virtual ~RadialScriJetProbe()=default;
 enum class Basis {Radius=0, Omega=1, LogRadius=2};
 struct Node {Real r;int point;};
 struct Row {std::vector<Node> nodes;std::vector<Real> first,second;
  Real condition=0,reproduction=0,first_l1=0,second_l1=0,first_ratio=0,second_ratio=0;};
 const ScriSurface &surface;int requested;Basis basis;
 std::vector<Row> rows;Real condition=0,reproduction=0,first_ratio=0,second_ratio=0;
 static const char *Name(Basis b){return b==Basis::Radius?"r":b==Basis::Omega?"Omega":"minus_log_r";}
 static long double Coordinate(Real r,Real R,Basis b){
  if(b==Basis::Radius)return (long double)r-R;
  if(b==Basis::Omega)return ((long double)R*R-(long double)r*r)/(2*R);
  if(b==Basis::LogRadius)return -std::log((long double)r/R);
  throw std::invalid_argument("radial probe unknown coordinate basis");
 }
 static Row Weights(std::vector<Node> nodes,Real R,Basis basis){
  const int n=nodes.size();if(n<3||n>16)throw std::invalid_argument("radial probe exact node count must be3..16");
  Row out;out.nodes=std::move(nodes);out.first.resize(n);out.second.resize(n);
  long double scale=0;std::vector<long double> z(n);
  for(int i=0;i<n;++i){if(!(out.nodes[i].r>0&&out.nodes[i].r<=R*(1+1e-13)))throw std::runtime_error("radial probe invalid physical radius");z[i]=Coordinate(out.nodes[i].r,R,basis);scale=std::max(scale,std::abs(z[i]));}
  if(!(scale>0&&std::isfinite(scale)))throw std::runtime_error("radial probe zero/nonfinite coordinate span");
  for(auto &v:z)v/=scale;
  std::vector<std::vector<long double>> matrix(n,std::vector<long double>(n)),inverse=matrix;
  long double norm=0;
  for(int i=0;i<n;++i){long double total=0;inverse[i][i]=1;for(int j=0;j<n;++j){matrix[i][j]=std::pow(z[i],j);total+=std::abs(matrix[i][j]);}norm=std::max(norm,total);}
  for(int k=0;k<n;++k){int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(matrix[i][k])>std::abs(matrix[pivot][k]))pivot=i;
   if(std::abs(matrix[pivot][k])<1e-14L)throw std::runtime_error("radial probe rank guard");std::swap(matrix[k],matrix[pivot]);std::swap(inverse[k],inverse[pivot]);long double s=matrix[k][k];for(int j=0;j<n;++j){matrix[k][j]/=s;inverse[k][j]/=s;}for(int i=0;i<n;++i)if(i!=k){s=matrix[i][k];for(int j=0;j<n;++j){matrix[i][j]-=s*matrix[k][j];inverse[i][j]-=s*inverse[k][j];}}}
  long double invnorm=0;for(auto &row:inverse){long double total=0;for(auto v:row)total+=std::abs(v);invnorm=std::max(invnorm,total);}out.condition=norm*invnorm;
  if(!(out.condition<1e8))throw std::runtime_error("radial probe condition guard: "+std::to_string(out.condition));
  long double a=1,b=0;if(basis==Basis::Omega){a=-1;b=-1/(long double)R;}if(basis==Basis::LogRadius){a=-1/(long double)R;b=1/((long double)R*R);}
  // Lagrange polynomials, evaluated with extended intermediates then stored in
  // host Real. Anchored differences in Apply preserve constants exactly.
  std::vector<long double> w1(n),w2(n);
  for(int i=0;i<n;++i){std::vector<long double> p(n);p[0]=1;int degree=0;for(int j=0;j<n;++j)if(i!=j){long double d=z[i]-z[j];if(std::abs(d)<1e-13L)throw std::runtime_error("radial probe duplicate radius");std::vector<long double> next(n);for(int k=0;k<=degree;++k){next[k]-=p[k]*z[j]/d;next[k+1]+=p[k]/d;}p.swap(next);++degree;}w1[i]=p[1];w2[i]=2*p[2];out.first[i]=a/scale*w1[i];out.second[i]=a*a/(scale*scale)*w2[i]+b/scale*w1[i];out.first_l1+=std::abs(out.first[i]);out.second_l1+=std::abs(out.second[i]);}
  for(int i=0;i<n;++i){w1[i]=(long double)out.first[i]*scale/a;w2[i]=(long double)out.second[i]*scale*scale/(a*a)-b*scale/(a*a)*w1[i];}
  for(int k=0;k<n;++k){long double d=0,dd=0;for(int i=0;i<n;++i){d+=w1[i]*std::pow(z[i],k);dd+=w2[i]*std::pow(z[i],k);}out.reproduction=std::max({out.reproduction,Real(std::abs(d-(k==1?1:0))),Real(std::abs(dd-(k==2?2:0)))});}
  if(!(out.reproduction<1e-10))throw std::runtime_error("radial probe polynomial reproduction guard");return out;
 }
 RadialScriJetProbe(const ScriSurface&s,int count,Basis mode):surface(s),requested(count),basis(mode){
  const auto &g=s.grid;const Real R=g.mesh.radii.back();if(count<3||count>16)throw std::invalid_argument("radial probe exact requested nodes3..16");
  for(const auto &row:s.rows){const auto &outer=g.mesh.blocks[row.block];std::vector<Node> candidates;
   for(size_t m=0;m<g.mesh.blocks.size();++m){const auto &block=g.mesh.blocks[m];if(block.face!=outer.face)continue;
    bool aligned=true;for(int a=1;a<3;++a)aligned=aligned&&std::abs(block.lo[a]-outer.lo[a])<1e-14&&std::abs(block.hi[a]-outer.hi[a])<1e-14;if(!aligned)continue;
    for(int i=0;i<g.n;++i){Real q[3]={(i+.5)/g.n,(row.j+.5)/g.n,(row.k+.5)/g.n},x[3];g.mesh.Forward(m,q,x);Real radius=0;for(int a=0;a<3;++a)radius+=x[a]*x[a];radius=std::sqrt(radius);Real err=0;for(int a=0;a<3;++a)err=std::max(err,std::abs(x[a]/radius-row.geometry.x[a]/R));if(err>1e-12)throw std::runtime_error("radial probe nonaligned physical ray");candidates.push_back({radius,g.Index(m,i,row.j,row.k)});}}
   std::sort(candidates.begin(),candidates.end(),[](const Node&a,const Node&b){return a.r>b.r;});
   if(candidates.size()<size_t(count-1))throw std::runtime_error("radial probe insufficient active points; no node-count fallback");
   std::vector<Node> selected(candidates.begin(),candidates.begin()+count-1);std::reverse(selected.begin(),selected.end());selected.push_back({R,-1});auto plan=Weights(selected,R,mode);
   // Predeclared field-independent amplification bound relative to the existing
   // last-layer seven-point (or available n+1) physical-r derivative stencil.
   const int base_count=std::min(g.n+1,7);std::vector<Node> base;for(int i=g.n-(base_count-1);i<g.n;++i){Real q[3]={(i+.5)/g.n,(row.j+.5)/g.n,(row.k+.5)/g.n},x[3];g.mesh.Forward(row.block,q,x);Real rr=0;for(Real v:x)rr+=v*v;base.push_back({std::sqrt(rr),g.Index(row.block,i,row.j,row.k)});}base.push_back({R,-1});const auto baseline=Weights(base,R,Basis::Radius);
   plan.first_ratio=plan.first_l1/baseline.first_l1;plan.second_ratio=plan.second_l1/baseline.second_l1;
   if(!(plan.first_ratio<=4&&plan.second_ratio<=4))throw std::runtime_error("radial probe derivative amplification guard: "+std::to_string(plan.first_ratio)+","+std::to_string(plan.second_ratio));
   condition=std::max(condition,plan.condition);reproduction=std::max(reproduction,plan.reproduction);first_ratio=std::max(first_ratio,plan.first_ratio);second_ratio=std::max(second_ratio,plan.second_ratio);rows.push_back(std::move(plan));
  }
 }
 void Apply(HermiteBoundaryProbe&target)const{
  const auto &g=surface.grid;if(&target.surface!=&surface||target.fields<1)throw std::invalid_argument("radial probe foreign Hermite target");
  for(size_t p=0;p<rows.size();++p)for(int f=0;f<target.fields;++f){const Real anchor=target.boundary[p*target.fields+f];long double d=0,dd=0;for(size_t i=0;i<rows[p].nodes.size();++i){const auto &node=rows[p].nodes[i];if(node.point<0)continue;const Real delta=target.volume[f*g.count+node.point]-anchor;if(!std::isfinite(delta))throw std::runtime_error("radial probe nonfinite current data");d+=(long double)rows[p].first[i]*delta;dd+=(long double)rows[p].second[i]*delta;}target.normal_first[p*target.fields+f]=d;target.normal_second[p*target.fields+f]=dd;}
 }
};
} // namespace cubed_sphere
#endif
