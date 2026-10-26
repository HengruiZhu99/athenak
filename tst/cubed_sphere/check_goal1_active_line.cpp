// Actual candidate full-Pi line kernel: no evolution or donor-interface theorem.
#include <iostream>
#include <iomanip>
#include <cmath>
#include "mesh/cubed_sphere/research_canonical_full.hpp"
namespace cs=cubed_sphere;
struct Geo {
 cs::Map map{.25,.98,2.8592697712828787};int nodes,face,axis;Real radial;bool mapped;
 cs::Point operator()(int i)const {if(mapped){Real q[3]={radial,.4,.4};q[axis]=-1+2*(i+.5)/nodes;return map.Geometry(face,q);}
  cs::Point p{};p.x[axis]=-.25+.5*(i+.5)/nodes;for(int a=0;a<3;++a)p.J[a][a]=1;return p;}
 Real coordinate(int i)const{return mapped?-1+2*(i+.5)/nodes:-.25+.5*(i+.5)/nodes;}
};
struct Sample {Geo g;int kind;Real operator()(int i)const {return kind==0?1:(kind==1?exp(g.coordinate(i)):sin(.17*i)+.2*cos(.31*i));}};
void Audit(const Geo&g) {
 Real spacing=(g.mapped?2.:.5)/g.nodes,energy=0,absolute=0,expected=0,square=0,volume=0,constant=0;
 for(int i=0;i<g.nodes;++i){Real w=cs::ResearchLogicalVolume(g(i))*spacing;
  Real q=Sample{g,2}(i),pen=cs::full_canonical::ActiveLineTapered(Sample{g,2},g,i,g.nodes,g.axis,spacing,.1),work=2*w*q*pen;
  energy+=work;absolute+=fabs(work);Real s=cs::full_canonical::ActiveLineTapered(Sample{g,1},g,i,g.nodes,g.axis,spacing,.1);square+=w*s*s;volume+=w;
  constant=std::max(constant,fabs(cs::full_canonical::ActiveLineTapered(Sample{g,0},g,i,g.nodes,g.axis,spacing,.1)));
  if(i+3<g.nodes){Real third=0,line=0,coeff[4]={-1,3,-3,1},anchor=Sample{g,2}(i);
   for(int j=0;j<4;++j){third+=coeff[j]*(Sample{g,2}(i+j)-anchor);auto p=g(i+j);Real norm=0;for(int a=0;a<3;++a)norm+=p.J[g.axis][a]*p.J[g.axis][a];line+=cs::ResearchLogicalVolume(p)*sqrt(norm)/4;}
   const Real z=(i+Real(1.5))/(g.nodes-1),taper=16*z*z*(1-z)*(1-z);expected-=.1*line*taper*third*third/32;
  }
 }
 Real scale=std::max({fabs(energy),fabs(expected),absolute,Real(1e-30)});
 std::cout<<"{\"mapped\":"<<(g.mapped?"true":"false")<<",\"nodes\":"<<g.nodes<<",\"face\":"<<g.face<<",\"axis\":"<<g.axis<<",\"radial\":"<<g.radial
  <<",\"constant_penalty\":"<<constant<<",\"energy_rate\":"<<energy<<",\"negative_line_rate\":"<<expected<<",\"identity_relative_error\":"<<fabs(energy-expected)/scale
  <<",\"signed_energy_relative\":"<<energy/scale<<",\"smooth_penalty_rms\":"<<sqrt(square/volume)<<"}\n";
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 for(int n:{8,24,48,96}){for(int axis=0;axis<3;++axis)Audit(Geo{{.25,.98,2.8592697712828787},n,-1,axis,0,false});
  for(int face=0;face<6;++face)for(int axis:{1,2})for(Real s:{.2,.8,.98})Audit(Geo{{.25,.98,2.8592697712828787},n,face,axis,s,true});}
 Kokkos::finalize();return 0;
}
