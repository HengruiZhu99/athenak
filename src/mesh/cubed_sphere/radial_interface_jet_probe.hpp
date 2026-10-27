// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_RADIAL_INTERFACE_JET_PROBE_HPP_
#define MESH_CUBED_SPHERE_RADIAL_INTERFACE_JET_PROBE_HPP_
#include <map>
#include "mesh/cubed_sphere/composite_difference.hpp"
namespace cubed_sphere {
// HOST DIAGNOSTIC ONLY: full Cartesian AD jets of the existing six-point
// global radial-chain interpolant. No field projection/production replacement.
// Angular donor radii are fixed geometry constants, never query-angle jets.
class RadialInterfaceJetProbe {
 public:
 struct Term {int point;Jet weight;};
 struct Plan {std::vector<Term>terms;Real value_l1=0,gradient_l1_scaled=0,hessian_l1_scaled=0,reproduction=0,radial_condition=0;};
 struct Node{Real radius;int point;};
 const CompositeDifference &g;unsigned long revision;int n,count;size_t blocks;
 using Key=std::array<Real,5>;
 std::map<Key,std::vector<std::vector<Node>>> lines;
 mutable std::map<std::array<Real,6>,Real> condition_cache;
 mutable Real maximum_value_l1=0,maximum_gradient_l1_scaled=0,maximum_hessian_l1_scaled=0,maximum_reproduction=0,maximum_radial_condition=0;
 explicit RadialInterfaceJetProbe(const CompositeDifference&grid):g(grid),revision(g.mesh.revision),n(g.n),count(g.count),blocks(g.mesh.blocks.size()){
  if(n<6)throw std::invalid_argument("radial jet probe requires exact six-point shape");
  for(size_t m=0;m<blocks;++m){const auto &b=g.mesh.blocks[m];if(b.face<0)continue;const auto key=AngularKey(b);if(lines.count(key))continue;auto &all=lines[key];all.resize(n*n);
   for(int k=0;k<n;++k)for(int j=0;j<n;++j){const Real u=b.lo[1]+(j+.5)/n*(b.hi[1]-b.lo[1]),v=b.lo[2]+(k+.5)/n*(b.hi[2]-b.lo[2]);auto &line=all[j+n*k];for(size_t c=0;c<blocks;++c){const auto &a=g.mesh.blocks[c];if(a.face<0||AngularKey(a)!=key)continue;const Real lo=a.layer==0?g.mesh.half_width*std::sqrt(1+u*u+v*v):g.mesh.radii[a.layer-1],hi=g.mesh.radii[a.layer];for(int i=0;i<n;++i){const Real s=a.lo[0]+(i+.5)/n*(a.hi[0]-a.lo[0]);line.push_back({lo+s*(hi-lo),g.Index(c,i,j,k)});}}
    std::sort(line.begin(),line.end(),[](const Node&a,const Node&b){return a.radius<b.radius;});if(line.size()<6)throw std::runtime_error("radial chain insufficient exact nodes");for(size_t i=1;i<line.size();++i)if(!(line[i].radius-line[i-1].radius>1e-14*g.mesh.radii.back()))throw std::runtime_error("radial chain duplicate/unresolved physical nodes");}
  }
 }
 static Key AngularKey(const CompositeBlock&b){return {Real(b.face),b.lo[1],b.hi[1],b.lo[2],b.hi[2]};}
 static Jet Divide(const Jet&a,const Jet&b){if(b.f==0)throw std::runtime_error("radial jet zero denominator");Jet c(a.f/b.f);for(int i=0;i<3;++i)c.d[i]=(a.d[i]-c.f*b.d[i])/b.f;for(int i=0;i<3;++i)for(int j=0;j<3;++j)c.h[i][j]=(a.h[i][j]-c.f*b.h[i][j]-c.d[i]*b.d[j]-c.d[j]*b.d[i])/b.f;return c;}
 static Real Condition(const Node*node){Real a[6][6]{},inverse[6][6]{},norm=0;const Real span=node[5].radius-node[0].radius;if(!(span>0))throw std::runtime_error("radial jet span guard");for(int i=0;i<6;++i){Real total=0,z=(node[i].radius-node[0].radius)/span;inverse[i][i]=1;for(int j=0;j<6;++j){a[i][j]=std::pow(z,j);total+=std::abs(a[i][j]);}norm=std::max(norm,total);}for(int k=0;k<6;++k){int pivot=k;for(int i=k+1;i<6;++i)if(std::abs(a[i][k])>std::abs(a[pivot][k]))pivot=i;if(std::abs(a[pivot][k])<1e-13)throw std::runtime_error("radial jet rank guard");for(int j=0;j<6;++j){std::swap(a[k][j],a[pivot][j]);std::swap(inverse[k][j],inverse[pivot][j]);}Real scale=a[k][k];for(int j=0;j<6;++j){a[k][j]/=scale;inverse[k][j]/=scale;}for(int i=0;i<6;++i)if(i!=k){scale=a[i][k];for(int j=0;j<6;++j){a[i][j]-=scale*a[k][j];inverse[i][j]-=scale*inverse[k][j];}}}Real invnorm=0;for(int i=0;i<6;++i){Real total=0;for(int j=0;j<6;++j)total+=std::abs(inverse[i][j]);invnorm=std::max(invnorm,total);}return norm*invnorm;}
 Plan Build(const Real*x,int forced_owner=-1)const{
  if(g.mesh.revision!=revision||g.n!=n||g.count!=count||g.mesh.blocks.size()!=blocks)throw std::runtime_error("radial jet stale geometry");const int owner=forced_owner<0?g.mesh.Owner(x):forced_owner;if(owner<0||owner>=int(blocks))throw std::invalid_argument("radial jet query outside grid");const auto &block=g.mesh.blocks[owner];if(block.face<0)throw std::invalid_argument("radial jet Build requires shell; SampleJets uses unchanged Cartesian sampler");
  const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);if(!(r>0&&r<=g.mesh.radii.back()))throw std::invalid_argument("radial jet finite radius guard");Jet X[3];for(int i=0;i<3;++i)X[i]=Jet::Variable(x[i],i);Jet radius(r);for(int i=0;i<3;++i){radius.d[i]=x[i]/r;for(int j=0;j<3;++j)radius.h[i][j]=(i==j?1/r:0)-x[i]*x[j]/(r*r*r);}const int axis=block.face/2;const Jet scale=Jet(block.face%2?-1:1)*X[axis];if(!(scale.f>0))throw std::invalid_argument("radial jet forced face orientation");Jet local[2]={Divide(Divide(X[(axis+1)%3],scale)-Jet(block.lo[1]),Jet(block.hi[1]-block.lo[1])),Divide(Divide(X[(axis+2)%3],scale)-Jet(block.lo[2]),Jet(block.hi[2]-block.lo[2]))};
  Jet angular[2][6];int first[2];for(int a=0;a<2;++a){if(local[a].f< -1e-12||local[a].f>1+1e-12)throw std::invalid_argument("radial jet forced angular patch mismatch");Jet t=Jet(n)*local[a]-Jet(.5);first[a]=std::clamp(int(std::floor(t.f))-2,0,n-6);for(int i=0;i<6;++i){angular[a][i]=Jet(1);for(int j=0;j<6;++j)if(j!=i)angular[a][i]=Divide(angular[a][i]*(t-Jet(first[a]+j)),Jet(i-j));}}
  const auto &all=lines.at(AngularKey(block));Plan out;out.terms.reserve(216);Jet total(0);Real smallest_dr=1e300;for(int k=0;k<6;++k)for(int j=0;j<6;++j){const auto &line=all[first[0]+j+n*(first[1]+k)];auto pos=std::lower_bound(line.begin(),line.end(),r,[](const Node&a,Real radius){return a.radius<radius;});const int start=std::clamp(int(pos-line.begin())-3,0,int(line.size())-6);const Node*nodes=line.data()+start;std::array<Real,6> condition_key;for(int c=0;c<6;++c){condition_key[c]=nodes[c].radius;if(c)smallest_dr=std::min(smallest_dr,nodes[c].radius-nodes[c-1].radius);}auto found=condition_cache.find(condition_key);if(found==condition_cache.end())found=condition_cache.emplace(condition_key,Condition(nodes)).first;out.radial_condition=std::max(out.radial_condition,found->second);for(int i=0;i<6;++i){Jet weight=angular[0][j]*angular[1][k];for(int b=0;b<6;++b)if(i!=b)weight=Divide(weight*(radius-Jet(nodes[b].radius)),Jet(nodes[i].radius-nodes[b].radius));out.terms.push_back({nodes[i].point,weight});total=total+weight;}}
  const Real u=block.lo[1]+local[0].f*(block.hi[1]-block.lo[1]),v=block.lo[2]+local[1].f*(block.hi[2]-block.lo[2]);const Real h=std::min(smallest_dr,r*std::min(block.hi[1]-block.lo[1],block.hi[2]-block.lo[2])/(n*(1+u*u+v*v)));out.reproduction=std::abs(total.f-1);for(int i=0;i<3;++i){out.reproduction=std::max(out.reproduction,std::abs(h*total.d[i]));for(int j=0;j<3;++j)out.reproduction=std::max(out.reproduction,std::abs(h*h*total.h[i][j]));}for(const auto&t:out.terms){out.value_l1+=std::abs(t.weight.f);for(int i=0;i<3;++i){out.gradient_l1_scaled+=h*std::abs(t.weight.d[i]);for(int j=0;j<3;++j)out.hessian_l1_scaled+=h*h*std::abs(t.weight.h[i][j]);}}
  // New value-interpolant guards, not a change to the qualified Hermite sampler:
  // six-point one-sided tensor extrapolation can already have L1≈18.25³.
  if(!(out.radial_condition<1e8&&out.value_l1<=8192&&out.gradient_l1_scaled<=1e5&&out.hessian_l1_scaled<=1e6&&out.reproduction<1e-10))throw std::runtime_error("radial jet condition/amplification/reproduction guard");maximum_value_l1=std::max(maximum_value_l1,out.value_l1);maximum_gradient_l1_scaled=std::max(maximum_gradient_l1_scaled,out.gradient_l1_scaled);maximum_hessian_l1_scaled=std::max(maximum_hessian_l1_scaled,out.hessian_l1_scaled);maximum_reproduction=std::max(maximum_reproduction,out.reproduction);maximum_radial_condition=std::max(maximum_radial_condition,out.radial_condition);return out;
 }
 void SamplePlan(const Plan&plan,const std::vector<Real>&field,int components,std::vector<Real>&values,std::vector<std::array<Real,3>>&first,std::vector<std::array<std::array<Real,3>,3>>&second)const{
  if(components<1||components>64||field.size()!=size_t(count)*components||plan.terms.empty())throw std::invalid_argument("radial jet field layout");values.resize(components);first.assign(components,{});second.assign(components,{});for(int f=0;f<components;++f){const Real anchor=field[f*count+plan.terms[0].point];Jet value(anchor);for(const auto&t:plan.terms){const Real delta=field[f*count+t.point]-anchor;if(!std::isfinite(delta))throw std::runtime_error("radial jet nonfinite active field");value=value+t.weight*Jet(delta);}if(!std::isfinite(value.f))throw std::runtime_error("radial jet nonfinite sample");values[f]=value.f;for(int i=0;i<3;++i){first[f][i]=value.d[i];for(int j=0;j<3;++j)second[f][i][j]=value.h[i][j];}}
 }
 void SampleJets(const std::vector<Real>&field,int components,const Real*x,std::vector<Real>&values,std::vector<std::array<Real,3>>&first,std::vector<std::array<std::array<Real,3>,3>>&second)const{
  if(components<1||components>64||field.size()!=size_t(count)*components)throw std::invalid_argument("radial jet field layout");const int owner=g.mesh.Owner(x);if(owner<0)throw std::invalid_argument("radial jet query outside domain");if(g.mesh.blocks[owner].face<0){g.SampleJets(field,components,x,values,first,second);return;}SamplePlan(Build(x,owner),field,components,values,first,second);
 }
};
}
#endif
