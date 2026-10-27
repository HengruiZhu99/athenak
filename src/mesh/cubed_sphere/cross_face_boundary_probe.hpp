// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_CROSS_FACE_BOUNDARY_PROBE_HPP_
#define MESH_CUBED_SPHERE_CROSS_FACE_BOUNDARY_PROBE_HPP_
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "mesh/cubed_sphere/centered_surface_atlas.hpp"
namespace cubed_sphere {
// STANDALONE DIAGNOSTIC ONLY. A single stored-field pullback supplies ALL
// q first/second derivatives, including mixed normal/angular ones. Radial
// one-sided Lagrange nodes and coefficients are exactly the original surface
// fit. Cross-face physical samples retain global Cartesian components.
class CrossFaceBoundaryProbe {
 public:
 using Jets=CenteredSurfaceAtlas::Jets;
 const CompositeDifference &g;const ScriSurface &surface;
 const std::vector<Real> &volume,&boundary;int fields;
 CenteredSurfaceAtlas atlas;
 // Independent analytic MMS oracle only; NEVER enabled by a PDE caller.
 std::function<std::vector<Real>(const Real*)> exact_mms_query;
 // Optional geometry-only same-field transfer probe; native nodes remain exact.
 std::function<std::vector<Real>(int,const Real*,bool)> off_node_query;
 CrossFaceBoundaryProbe(const ScriSurface&s,const std::vector<Real>&v,const std::vector<Real>&b,int components):g(s.grid),surface(s),volume(v),boundary(b),fields(components),atlas(g.mesh,components){if(v.size()!=size_t(g.count*fields)||b.size()!=atlas.values.size())throw std::invalid_argument("boundary probe layout");atlas.values=b;}
 std::vector<Real> Sample(int block,const Real*q,bool is_boundary)const{if(exact_mms_query){Real x[3];g.mesh.Forward(block,q,x);return exact_mms_query(x);}int logical[3];bool active=true;for(int a=0;a<3;++a){Real t=g.n*q[a]-.5;logical[a]=std::lround(t);if(std::abs(t-logical[a])>1e-10||logical[a]<0||logical[a]>=g.n)active=false;}if(is_boundary){active=true;for(int a=1;a<3;++a){Real t=g.n*q[a]-.5;logical[a]=std::lround(t);if(std::abs(t-logical[a])>1e-10||logical[a]<0||logical[a]>=g.n)active=false;}if(active){std::vector<Real> out(fields);const int p=surface.base[block]+logical[1]+g.n*logical[2];for(int f=0;f<fields;++f)out[f]=boundary[p*fields+f];return out;}}
  else if(active){std::vector<Real> out(fields);const int p=g.Index(block,logical[0],logical[1],logical[2]);for(int f=0;f<fields;++f)out[f]=volume[f*g.count+p];return out;}
  if(off_node_query)return off_node_query(block,q,is_boundary);Real x[3];g.mesh.Forward(block,q,x);if(is_boundary)return atlas.Sample(x);const auto entries=g.Interpolate(x,6);if(entries.empty())throw std::runtime_error("boundary probe missing active donor");std::vector<Real> out(fields);for(int f=0;f<fields;++f){const Real anchor=volume[f*g.count+entries[0].point];out[f]=anchor;for(const auto&e:entries)out[f]+=e.weight*(volume[f*g.count+e.point]-anchor);}return out;
 }
 Jets At(int block,const Real*q)const{if(q[0]!=1||surface.base.at(block)<0)throw std::invalid_argument("boundary probe exact scri coordinate");const int nodes=surface.radial_nodes,n=g.n;const Real h=1./n,w[5]={1./12,-2./3,0,2./3,-1./12},w2[5]={-1./12,4./3,-2.5,4./3,-1./12};std::vector<Real> sample[8][5][5];Real radial[8],dr[8],ddr[8];for(int r=0;r<nodes;++r){radial[r]=r==nodes-1?1:Real(n-(nodes-1)+r+.5)/n;Jet basis(1);for(int k=0;k<nodes;++k)if(k!=r){Real other=k==nodes-1?1:Real(n-(nodes-1)+k+.5)/n;basis=basis*(Jet::Variable(1,0)-Jet(other))/Jet(radial[r]-other);}dr[r]=basis.d[0];ddr[r]=basis.h[0][0];for(int j=0;j<5;++j)for(int k=0;k<5;++k){const Real point[3]={radial[r],q[1]+(j-2)*h,q[2]+(k-2)*h};sample[r][j][k]=Sample(block,point,r==nodes-1);}}
  Jets out;out.value=sample[nodes-1][2][2];out.d.resize(fields);out.dd.resize(fields);const auto geometry=g.mesh.Geometry(block,q);for(int f=0;f<fields;++f){Real dq[3]{},ddq[3][3]{};const Real anchor=out.value[f];for(int r=0;r<nodes;++r){dq[0]+=dr[r]*(sample[r][2][2][f]-anchor);ddq[0][0]+=ddr[r]*(sample[r][2][2][f]-anchor);for(int j=0;j<5;++j){ddq[0][1]+=dr[r]*w[j]*(sample[r][j][2][f]-anchor)/h;ddq[0][2]+=dr[r]*w[j]*(sample[r][2][j][f]-anchor)/h;}}
   for(int j=0;j<5;++j){dq[1]+=w[j]*(sample[nodes-1][j][2][f]-anchor)/h;dq[2]+=w[j]*(sample[nodes-1][2][j][f]-anchor)/h;ddq[1][1]+=w2[j]*(sample[nodes-1][j][2][f]-anchor)/(h*h);ddq[2][2]+=w2[j]*(sample[nodes-1][2][j][f]-anchor)/(h*h);for(int k=0;k<5;++k)ddq[1][2]+=w[j]*w[k]*(sample[nodes-1][j][k][f]-anchor)/(h*h);}ddq[1][0]=ddq[0][1];ddq[2][0]=ddq[0][2];ddq[2][1]=ddq[1][2];for(int i=0;i<3;++i)for(int A=0;A<3;++A){out.d[f][i]+=geometry.J[A][i]*dq[A];for(int j=0;j<3;++j){out.dd[f][i][j]+=geometry.H[A][i][j]*dq[A];for(int B=0;B<3;++B)out.dd[f][i][j]+=geometry.J[A][i]*geometry.J[B][j]*ddq[A][B];}}}return out;
 }
};
} // namespace cubed_sphere
#endif
