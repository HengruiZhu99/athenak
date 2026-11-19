// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Exact smooth hyperboloidal Minkowski jets in the unblended outer region.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/scri_reconstruction.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main() {
  h::CentralZ4cConfiguration config;config.compact_slicing_end=.97;
  auto p=config.Resolve().gauge;p.outer_lapse_damping=.25;
  double bulk=0,boundary=0,correction=0;
  p.Validate();
  for(double theta:{0.,.4,1.1})for(double phi:{0.,.7}) {
    const double n[3]={cos(theta),sin(theta)*cos(phi),sin(theta)*sin(phi)};
    for(double r:{.97,.98,.985,.995,.999,.9999}) {
      h::mapped::Point map{};for(int a=0;a<3;++a){map.x[a]=r*n[a];map.J[a][a]=1;}
      h::CartesianChartJets j{};j[0].f=1;
      h::StationaryGaugeSource<Real> source{};source.alpha=(1+r*r)/2;
      const auto rhs=h::EvaluateBulkPoint(map,j,p,source);
      if(rhs.status!=h::PointStatus::success)throw std::runtime_error("invalid smooth bulk reference");
      for(double v:rhs.rate)bulk=fmax(bulk,fabs(v));
    }
    h::mapped::Point map{};for(int a=0;a<3;++a)map.x[a]=n[a];
    h::CartesianChartJets j{};j[0].f=1;
    h::StationaryGaugeSource<Real> source{};source.alpha=1;
    const auto rhs=h::SphericalPolicy::Boundary(map,.005,j,p,source,.5);
    if(!rhs.valid)throw std::runtime_error("invalid smooth boundary reference");
    const auto u=h::SphericalPolicy::Physical(map.x,j);
    const auto fix=h::ReconstructRegularScri(map.x,u,h::Geometry(u.metric),rhs,p);
    if(!fix.valid)throw std::runtime_error("invalid smooth reference reconstruction");
    for(double v:rhs.rate)boundary=fmax(boundary,fabs(v));
    boundary=fmax(boundary,fabs(rhs.clock));
    correction=fmax(correction,fabs(fix.lapse));
    for(double v:fix.shift)correction=fmax(correction,fabs(v));
  }
  std::cout<<std::setprecision(17)<<"smooth outer reference: bulk="<<bulk
    <<" boundary="<<boundary<<" correction="<<correction<<'\n';
  if(!(bulk<1e-9 && boundary<1e-11 && correction<1e-11))
    throw std::runtime_error("smooth reference preservation failure");
}
