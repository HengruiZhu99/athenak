// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include "z4c_hyperboloidal/scri_quadrature.hpp"
using namespace z4c_hyperboloidal;
using C=std::complex<double>;
void Require(bool b,const char *m) {if(!b)throw std::runtime_error(m);}
int main() {
  const double pi=acos(-1.);
  const Real north[3]={0,0,1},east[3]={1,0,0};
  Require(std::abs(ScriScalarHarmonic(1,0,north)[0]-sqrt(3/(4*pi)))<1e-15,"Y10 normalization");
  Require(std::abs(ScriScalarHarmonic(1,1,east)[0]+sqrt(3/(8*pi)))<1e-15,"Y11 phase");
  for(bool angular:{false,true}) {
    double previous=1e9;
    for(int n:{12,24,48}) {
      double area=0,worst=0;
      // Every m at l=3 and l=8, projected onto every l<=8.
      const int nm=81;std::vector<C> gram(nm*nm);
      for(int face=0;face<6;++face)for(int k=0;k<n;++k)for(int j=0;j<n;++j) {
        chart_atlas::BlockMap map;map.face=face;map.outer_radius=1;map.outer_surface=true;
        map.lo[1]=map.lo[2]=-1;map.hi[0]=map.hi[1]=map.hi[2]=1;map.equiangular=angular;
        const Real q[3]={1,(j+.5)/n,(k+.5)/n};Real x[3];map.Forward(q,x);
        const double w=ScriSolidAngleWeight(map,n,j,k);area+=w;
        C y[nm];for(int l=0;l<=8;++l)for(int m=-l;m<=l;++m) {auto a=ScriScalarHarmonic(l,m,x);y[l*l+l+m]={a[0],a[1]};}
        for(int l:{3,8})for(int m=-l;m<=l;++m) {
          int input=l*l+l+m;
          for(int output=0;output<nm;++output)gram[input*nm+output]+=w*y[input]*std::conj(y[output]);
        }
      }
      for(int l:{3,8})for(int m=-l;m<=l;++m) {
        int input=l*l+l+m;
        for(int output=0;output<nm;++output)worst=std::max(worst,std::abs(gram[input*nm+output]-C(input==output?1:0)));
      }
      std::cout<<"equiangular="<<angular<<" N="<<n<<" area_error="<<fabs(area-4*pi)<<" gram_error="<<worst<<'\n';
      Require(worst<.02,"analytic harmonic recovery/leakage");
      Require(worst<previous*.3,"quadrature errors must converge at second order");previous=worst;
    }
    // Subdividing a face conserves nodes/weights exactly to roundoff.
    chart_atlas::BlockMap whole;whole.equiangular=angular;
    whole.lo[1]=whole.lo[2]=-1;whole.hi[1]=whole.hi[2]=1;
    for(int b=0;b<2;++b)for(int a=0;a<2;++a)for(int k=0;k<12;++k)for(int j=0;j<12;++j) {
      auto tile=whole;tile.lo[1]=a-1;tile.hi[1]=a;tile.lo[2]=b-1;tile.hi[2]=b;
      Require(std::abs(ScriSolidAngleWeight(tile,12,j,k)-ScriSolidAngleWeight(whole,24,12*a+j,12*b+k))<1e-15,"patch subdivision");
    }
  }
}
