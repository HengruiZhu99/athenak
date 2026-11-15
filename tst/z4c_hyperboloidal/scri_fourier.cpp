// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Independent full frozen Fourier generator; no principal-term selection.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
namespace nh=z4c_hyperboloidal;
bool Position(int f) {return f<=6 || f==18 || (f>=22 && f<25);}
int main(int argc,char **argv) {
  if(argc!=9)throw std::runtime_error("usage: symbol radius epsilon theta phi background fdstep collar_begin frequency");
  const double r=std::stod(argv[1]),eps=std::stod(argv[2]),theta=std::stod(argv[3]),phi=std::stod(argv[4]);
  const double delta=std::stod(argv[6]);
  nh::CentralZ4cConfiguration config;config.compact_slicing_end=.97;
  auto p=config.Resolve().gauge;p.outer_lapse_damping=.25;
  p.scri_treatment=nh::ScriTreatment::clock_source;p.scri_source_epsilon=eps;
  p.scri_collar_begin=std::stod(argv[7]);p.scri_collar_end=(1+p.scri_collar_begin)/2;
  p.Validate();nh::mapped::Point map{};map.x[0]=r;
  for(int a=0;a<3;++a)map.J[a][a]=1;
  nh::CartesianChartJets base{};base[0].f=1;
  nh::StationaryGaugeSource<Real> source{};source.alpha=(1+r*r)/2;
  if(std::string(argv[5])=="wormhole") {
    nh::SphericalReference3D reference(p);base=reference.At(map.x);source=reference.source.At(r);
  } else if(std::string(argv[5])!="flat")throw std::runtime_error("unknown background");
  const double omega=p.profile.At(map.x).omega;
  if(!(omega>0))throw std::runtime_error("principal symbol requires an interior point");
  const auto u=nh::SphericalPolicy::Physical(map.x,base);const auto D=nh::SphericalPolicy::Vector(base,25);
  source.scri_obstruction_reference=nh::EvaluateScriGeometry(map.x,u,D.value,p.profile.At(map.x),p).obstruction;
  const double direction[3]={std::cos(theta),std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi)};
  std::vector<int> fields;for(int f=0;f<nh::Fields::count;++f)if(nh::Fields::Independent(f))fields.push_back(f);
  auto scale=[&](int f){return f==13 || f==14 ? omega : 1.;};
  const double frequency=std::stod(argv[8]);
  auto fourier_scale=[&](int f){return Position(f) ? 1/frequency : scale(f);};
  double matrix[26][26]{},imaginary[26][26]{};
  for(int col=0;col<26;++col)for(int part=0;part<2;++part) {
    const int f=fields[col];nh::BulkPointResult rates[2];
    for(int sign=0;sign<2;++sign) {
      auto j=base;const double step=(sign==0?-delta:delta)*fourier_scale(f);
      if(part==0) {
        j[f].f+=step;
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)j[f].h[a][b]-=step*frequency*frequency*direction[a]*direction[b];
      } else for(int a=0;a<3;++a)j[f].d[a]+=step*frequency*direction[a];
      if(!nh::TryExpandCartesianChart(j))throw std::runtime_error("invalid Fourier chart");
      rates[sign]=nh::EvaluateBulkPoint(map,j,p,source);
      if(rates[sign].status!=nh::PointStatus::success)throw std::runtime_error("invalid Fourier probe");
    }
    for(int row=0;row<26;++row) {
      const double value=(rates[1].rate[fields[row]]-rates[0].rate[fields[row]])/(2*delta*fourier_scale(fields[row])*frequency);
      if(part==0)matrix[row][col]=value;else imaginary[row][col]=value;
    }
  }
  std::cout<<std::setprecision(17)<<"{\"radius\":"<<r<<",\"epsilon\":"<<eps
      <<",\"theta\":"<<theta<<",\"phi\":"<<phi<<",\"omega\":"<<omega
      <<",\"restoration_end\":"<<p.restoration_end<<",\"fields\":[";
  for(int i=0;i<26;++i)std::cout<<(i?",":"")<<fields[i];
  std::cout<<"],\"matrix\":[";
  for(int i=0;i<26;++i){std::cout<<(i?",[":"[");for(int j=0;j<26;++j)std::cout<<(j?",":"")<<matrix[i][j];std::cout<<"]";}
  std::cout<<"],\"imaginary\":[";
  for(int i=0;i<26;++i){std::cout<<(i?",[":"[");for(int j=0;j<26;++j)std::cout<<(j?",":"")<<imaginary[i][j];std::cout<<"]";}
  std::cout<<"]}\n";
}
