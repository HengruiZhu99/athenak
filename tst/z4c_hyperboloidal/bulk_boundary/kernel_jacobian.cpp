// Diagnostic only: linearize the actual 26-field bulk and scri point kernels.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
using Jets=h::CartesianChartJets;
using Rates=std::array<double,30>;
const int aa[6]={0,0,0,1,1,2},bb[6]={0,1,2,1,2,2};
void perturb(Jets &j,int field,int kind,double delta) {
  if(kind==0)j[field].f+=delta;
  else if(kind<4)j[field].d[kind-1]+=delta;
  else {int a=aa[kind-4],b=bb[kind-4];j[field].h[a][b]+=delta;
    if(a!=b)j[field].h[b][a]+=delta;}
}
int main(int argc,char **argv) {
  if(argc!=8)throw std::runtime_error("usage: probe radii_csv damping fdstep background spacing penalty output_prefix");
  std::vector<double> radii;std::stringstream in(argv[1]);std::string item;
  while(std::getline(in,item,','))radii.push_back(std::stod(item));
  const double kappa=std::stod(argv[2]),epsilon=std::stod(argv[3]);
  const std::string background=argv[4],prefix=argv[7];
  const double spacing=std::stod(argv[5]),penalty=std::stod(argv[6]);
  h::CentralZ4cConfiguration config;config.inner_z4c_radius_M=20;config.compact_slicing_end=.97;
  auto p=config.Resolve().gauge;p.outer_lapse_damping=kappa;
  p.Validate();
  h::SphericalReference3D reference(p);
  std::vector<int> fields;for(int f=0;f<31;++f)if(h::Fields::Independent(f))fields.push_back(f);
  std::ofstream data(prefix+".bin",std::ios::binary),meta(prefix+".json");
  meta<<std::setprecision(17)<<"{\"shape\":["<<radii.size()<<",30,260],\"fields\":[";
  for(int q=0;q<26;++q)meta<<(q?",":"")<<fields[q];
  meta<<"],\"jet_order\":[\"value\",\"x\",\"y\",\"z\",\"xx\",\"xy\",\"xz\",\"yy\",\"yz\",\"zz\"],\"kappa\":"<<kappa<<",\"fdstep\":"<<epsilon<<",\"spacing\":"<<spacing<<",\"penalty\":"<<penalty<<",\"background\":\""<<background<<"\",\"points\":[";
  for(size_t point=0;point<radii.size();++point) {
    const double r=radii[point];h::mapped::Point map{};map.x[0]=r;
    for(int a=0;a<3;++a)map.J[a][a]=1;
    Jets base{};h::StationaryGaugeSource<double> source{};
    if(background=="flat") {base[0].f=1;source.alpha=(1+r*r)/2;}
    else if(background=="wormhole") {base=reference.At(map.x,r==1);source=reference.source.At(r);}
    else throw std::runtime_error("unknown background");
    const auto eval=[&](Jets j) {
      if(!h::TryExpandCartesianChart(j))throw std::runtime_error("invalid chart");
      Rates result{};
      if(r==1) {
        auto out=h::EvaluateScriPoint(map,spacing,j,p,source,penalty);
        if(!out.valid)throw std::runtime_error("invalid boundary");
        for(int a=0;a<26;++a)result[a]=out.rate[fields[a]];
        for(int a=0;a<3;++a)result[26+a]=out.defect[a];result[29]=out.clock;
      } else {
        auto out=h::EvaluateBulkPoint(map,j,p,source);
        if(out.status!=h::PointStatus::success)throw std::runtime_error("invalid bulk");
        for(int a=0;a<26;++a)result[a]=out.rate[fields[a]];
      }
      return result;
    };
    double jac[30][260]{};
    for(int f=0;f<26;++f)for(int kind=0;kind<10;++kind) {
      Jets plus=base,minus=base;perturb(plus,fields[f],kind,epsilon);perturb(minus,fields[f],kind,-epsilon);
      const auto a=eval(plus),b=eval(minus);
      for(int row=0;row<30;++row)jac[row][10*f+kind]=(a[row]-b[row])/(2*epsilon);
    }
    data.write(reinterpret_cast<const char*>(jac),sizeof(jac));
    const auto rhs=eval(base);
    meta<<(point?",":"")<<"{\"r\":"<<r<<",\"omega\":"<<p.profile.At(map.x).omega<<",\"values\":[";
    for(int q=0;q<26;++q)meta<<(q?",":"")<<base[fields[q]].f;
    meta<<"],\"rate\":[";for(int q=0;q<30;++q)meta<<(q?",":"")<<rhs[q];meta<<"]}";
  }
  meta<<"]}\n";
}
