// Isolated Goal 1 research operator. No production option or boundary theorem.
#ifndef GOAL1_RESEARCH_RADIAL_SBP_HPP_
#define GOAL1_RESEARCH_RADIAL_SBP_HPP_
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "athena.hpp"
#include "mesh/cubed_sphere/conformal_bridge.hpp"
#include "mesh/cubed_sphere/research_sbp_coefficients.hpp"
namespace cubed_sphere {
struct ResearchRadialSBP {
  DvceArray2D<Real> d1,d2;
  DvceArray1D<Real> norm,amplification;
  int first=0,last=-1;
  KOKKOS_INLINE_FUNCTION bool Enabled() const {return d1.extent(0)>0;}
};
inline ResearchRadialSBP MakeResearchRadialSBP(int nodes,int first,Real spacing) {
  if(nodes<19||!(spacing>0))throw std::invalid_argument("radial6/3SBP requires >=19active nodes and positive spacing");
  ResearchRadialSBP op;op.first=first;op.last=first+nodes-1;
  op.d1=DvceArray2D<Real>("research radial SBP D1",nodes,nodes);
  op.d2=DvceArray2D<Real>("research compatible radial D2",nodes,nodes);
  op.norm=DvceArray1D<Real>("research radial SBP norm",nodes);
  op.amplification=DvceArray1D<Real>("research radial CFL amplification",nodes);
  auto d=Kokkos::create_mirror_view(op.d1),d2=Kokkos::create_mirror_view(op.d2);
  auto norm=Kokkos::create_mirror_view(op.norm),amp=Kokkos::create_mirror_view(op.amplification);
  Kokkos::deep_copy(d,0.);Kokkos::deep_copy(d2,0.);
  for(int i=0;i<nodes;++i)norm(i)=1;
  for(int i=0;i<8;++i){norm(i)=research_sbp::weights[i];norm(nodes-1-i)=research_sbp::weights[i];
    for(int j=0;j<11;++j){d(i,j)=research_sbp::rows[i][j]/spacing;d(nodes-1-i,nodes-1-j)=-research_sbp::rows[i][j]/spacing;}}
  const int offsets[6]={-3,-2,-1,1,2,3};const Real coeffs[6]={-1./60,3./20,-3./4,3./4,-3./20,1./60};
  for(int i=8;i<nodes-8;++i)for(int j=0;j<6;++j)d(i,i+offsets[j])=coeffs[j]/spacing;
  // Explicit anchor equivalents preserve constants even when rounded numerical
  // closure coefficients have tiny row-sum residuals. D2 uses this actual D1.
  for(int i=0;i<nodes;++i){Real sum=0;for(int j=0;j<nodes;++j)sum+=d(i,j);d(i,i)-=sum;}
  for(int i=0;i<nodes;++i)for(int j=0;j<nodes;++j)for(int k=0;k<nodes;++k)d2(i,j)+=d(i,k)*d(k,j);
  for(int i=0;i<nodes;++i){Real sum=0,n1=0,n2=0;for(int j=0;j<nodes;++j){sum+=d2(i,j);n1+=fabs(d(i,j))*spacing;n2+=fabs(d2(i,j))*spacing*spacing;}d2(i,i)-=sum;
    amp(i)=std::max({Real(1),n1/(Real(11)/6),sqrt(n2/(Real(16)/3))});}
  Kokkos::deep_copy(op.d1,d);Kokkos::deep_copy(op.d2,d2);Kokkos::deep_copy(op.norm,norm);Kokkos::deep_copy(op.amplification,amp);return op;
}
// Sample(radial active index, angular-j offset, angular-k offset).
template<class Sample>KOKKOS_INLINE_FUNCTION
z4c::hyperboloidal::ScalarJet<Real> ResearchSBPScalar(const Sample&sample,int row,
    const ResearchRadialSBP&op,Real du,Real dv) {
  z4c::hyperboloidal::ScalarJet<Real> out{};out.value=sample(row,0,0);
  const int shift[4]={-2,-1,1,2};const Real c1[4]={1./12,-2./3,2./3,-1./12};
  auto angular=[&](int radial,int axis){Real value=0,anchor=sample(radial,0,0);
    for(int l=0;l<4;++l)value+=c1[l]*(sample(radial,axis==1?shift[l]:0,axis==2?shift[l]:0)-anchor);
    return value/(axis==1?du:dv);};
  out.d[1]=angular(row,1);out.d[2]=angular(row,2);
  for(int col=0;col<op.d1.extent_int(0);++col){Real delta=sample(col,0,0)-out.value;
    out.d[0]+=op.d1(row,col)*delta;out.dd[0][0]+=op.d2(row,col)*delta;
    if(op.d1(row,col)!=0)for(int axis=1;axis<3;++axis)out.dd[0][axis]+=op.d1(row,col)*(angular(col,axis)-out.d[axis]);}
  for(int axis=1;axis<3;++axis){out.dd[axis][0]=out.dd[0][axis];Real v=0;
    for(int l=0;l<4;++l){int off=shift[l];Real c=abs(off)==2?-1./12:4./3;
      v+=c*(sample(row,axis==1?off:0,axis==2?off:0)-out.value);}
    out.dd[axis][axis]=v/(axis==1?du*du:dv*dv);}
  for(int j=0;j<4;++j)for(int k=0;k<4;++k)
    out.dd[1][2]+=c1[j]*c1[k]*(sample(row,shift[j],shift[k])-out.value)/(du*dv);
  out.dd[2][1]=out.dd[1][2];return out;
}
KOKKOS_INLINE_FUNCTION Real ResearchLogicalVolume(const Point&p){
  const auto &j=p.J;return 1/fabs(j[0][0]*(j[1][1]*j[2][2]-j[1][2]*j[2][1])
    -j[0][1]*(j[1][0]*j[2][2]-j[1][2]*j[2][0])+j[0][2]*(j[1][0]*j[2][1]-j[1][1]*j[2][0]));
}
template<class Sample,class Geometry>KOKKOS_INLINE_FUNCTION
Real ResearchRadialDissipation(const Sample&q,const Geometry&geometry,int row,
    const ResearchRadialSBP&op,Real ds,Real epsilon){
  const Real c[4]={-1,3,-3,1};Real out=0;const Real volume=ResearchLogicalVolume(geometry(row));
  for(int l=0;l<4;++l){int start=row-l;if(start<0||start+3>=op.d1.extent_int(0))continue;
    Real third=0,line=0,anchor=q(start,0,0);
    for(int j=0;j<4;++j){third+=c[j]*(q(start+j,0,0)-anchor);const auto p=geometry(start+j);Real norm=0;
      for(int i=0;i<3;++i)norm+=p.J[0][i]*p.J[0][i];line+=ResearchLogicalVolume(p)*sqrt(norm)/4;}
    out-=epsilon*c[l]*line*third/(64*ds*op.norm(row)*volume);
  }
  return out;
}
} // namespace cubed_sphere
#endif
