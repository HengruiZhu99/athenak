// Continuous independent quadrature of the analytic native TT configuration
// bandwidth. No grid evolution, state transform or acceptance-budget mutation.
#include "goal1_tensor_runtime.hpp"
#include "goal1_operator_oracle.hpp"
#include <iostream>
#include <iomanip>
int main(){std::cout<<std::setprecision(17);const Real panels[]={0,.05,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7};for(int rule:{32,48}){cs_unused_label:;cubed_sphere::clock_oracle::Height rad(rule),angular(rule==32?8:12),height(rule==32?48:64);int nphi=rule==32?24:32;double val[2]{},grad[2]{};
for(int panel=0;panel<11;++panel)for(int ir=0;ir<rule;++ir){double r=(panels[panel]+panels[panel+1])/2+(panels[panel+1]-panels[panel])*rad.nodes[ir]/2;auto geometry=cubed_sphere::tensor_oracle::Reference(cubed_sphere::tensor_oracle::S::R(r));double volume=r*r*geometry.L.value()/geometry.A.value()*(panels[panel+1]-panels[panel])*rad.weights[ir]/2;auto co=cubed_sphere::tensor_oracle::Evaluate(r,0,height);
for(int iu=0;iu<angular.order;++iu)for(int ip=0;ip<nphi;++ip){double u=angular.nodes[iu],phi=2*3.14159265358979323846*ip/nphi;Real n[3]={std::sqrt(1-u*u)*std::cos(phi),std::sqrt(1-u*u)*std::sin(phi),u};double weight=volume*angular.weights[iu]*2*3.14159265358979323846/nphi;
for(int sector=0;sector<2;++sector){auto state=cubed_sphere::tensor_oracle::NativeJet(co,r,n,sector);for(int field:{0,1,2,3,4,5,6,18,19,20,21}){auto q=cubed_sphere::operator_oracle::Component(state,field);double tensor_weight=(field==2||field==3||field==5)?2:1;val[sector]+=weight*tensor_weight*q.value*q.value;for(int d=0;d<3;++d)grad[sector]+=weight*tensor_weight*q.d[d]*q.d[d];}}}}
for(int sector=0;sector<2;++sector)std::cout<<"{\"radial_rule\":"<<rule<<",\"sector\":"<<sector<<",\"configuration_integral\":"<<val[sector]<<",\"gradient_integral\":"<<grad[sector]<<",\"spectral_length\":"<<std::sqrt(val[sector]/grad[sector])<<"}\n";
}return 0;}
