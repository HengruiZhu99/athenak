// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_CENTERED_SURFACE_ATLAS_HPP_
#define MESH_CUBED_SPHERE_CENTERED_SURFACE_ATLAS_HPP_
#include <functional>
#include "mesh/cubed_sphere/composite.hpp"
namespace cubed_sphere {
// CPU diagnostic prototype ONLY. No evolution/operator caller uses this class.
// Global Cartesian components are interpolated from the physical owning face.
// q=(1,u,v), centered fourth-order angular differences may cross face boundaries.
// Returned Cartesian jets are the constant-on-radial-rays extension; retaining
// original radial derivatives in a PDE comparison is a separate caller action.
class CenteredSurfaceAtlas {
 public:
 struct Jets {std::vector<Real> value;std::vector<std::array<Real,3>> d;std::vector<std::array<std::array<Real,3>,3>> dd;};
 const CompositeMesh &mesh;int fields;std::vector<int> base;std::vector<Real> values;
 // Test-only alternative off-node interpolation; no production caller.
 std::function<std::vector<Real>(const Real*)> sample_override;
 CenteredSurfaceAtlas(const CompositeMesh&m,int components):mesh(m),fields(components),base(m.blocks.size(),-1){if(m.cells<6||components<1)throw std::invalid_argument("surface atlas N>=6,fields>0");for(size_t b=0;b<m.blocks.size();++b)if(m.blocks[b].layer==int(m.radii.size())-1){base[b]=values.size()/fields;values.resize(values.size()+fields*m.cells*m.cells);}}
 std::vector<Real> Sample(const Real*x)const{if(sample_override)return sample_override(x);const int owner=mesh.Owner(x),n=mesh.cells;if(owner<0||base[owner]<0)throw std::runtime_error("surface atlas physical owner");Real q[3],w[2][6];mesh.Inverse(owner,x,q);int first[2];for(int a=0;a<2;++a){const Real t=n*q[a+1]-.5;first[a]=std::clamp(int(std::floor(t))-2,0,n-6);for(int i=0;i<6;++i){w[a][i]=1;for(int j=0;j<6;++j)if(i!=j)w[a][i]*=(t-first[a]-j)/Real(i-j);}}std::vector<Real> out(fields);for(int f=0;f<fields;++f){const Real anchor=values[(base[owner]+first[0]+n*first[1])*fields+f];out[f]=anchor;for(int k=0;k<6;++k)for(int j=0;j<6;++j)out[f]+=w[0][j]*w[1][k]*(values[(base[owner]+first[0]+j+n*(first[1]+k))*fields+f]-anchor);}return out;}
 Jets At(int block,const Real*q)const{if(base.at(block)<0||q[0]!=1)throw std::invalid_argument("surface atlas requires exact outer radial coordinate");const auto geometry=mesh.Geometry(block,q);const Real h=1./mesh.cells;std::vector<Real> sample[5][5];for(int k=-2;k<=2;++k)for(int j=-2;j<=2;++j){Real p[3]={1,q[1]+j*h,q[2]+k*h},x[3];mesh.Forward(block,p,x);sample[j+2][k+2]=Sample(x);}Jets out;out.value=sample[2][2];out.d.resize(fields);out.dd.resize(fields);const Real w[5]={1./12,-2./3,0,2./3,-1./12},w2[5]={-1./12,4./3,-2.5,4./3,-1./12};
 for(int f=0;f<fields;++f){Real dq[3]{},ddq[3][3]{};const Real anchor=out.value[f];for(int j=0;j<5;++j){dq[1]+=w[j]*(sample[j][2][f]-anchor)/h;dq[2]+=w[j]*(sample[2][j][f]-anchor)/h;ddq[1][1]+=w2[j]*(sample[j][2][f]-anchor)/(h*h);ddq[2][2]+=w2[j]*(sample[2][j][f]-anchor)/(h*h);for(int k=0;k<5;++k)ddq[1][2]+=w[j]*w[k]*(sample[j][k][f]-anchor)/(h*h);}ddq[2][1]=ddq[1][2];for(int i=0;i<3;++i)for(int A=0;A<3;++A){out.d[f][i]+=geometry.J[A][i]*dq[A];for(int j=0;j<3;++j){out.dd[f][i][j]+=geometry.H[A][i][j]*dq[A];for(int B=0;B<3;++B)out.dd[f][i][j]+=geometry.J[A][i]*geometry.J[B][j]*ddq[A][B];}}}return out;}
};
} // namespace cubed_sphere
#endif
