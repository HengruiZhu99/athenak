// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <cmath>
#include <iostream>
#include "z4c_hyperboloidal/scri_rate_diagnostic.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
namespace h=z4c_hyperboloidal;
int main(){
 h::CentralZ4cConfiguration cfg;cfg.compact_slicing_end=.97;
 auto p=cfg.Resolve().gauge;p.outer_lapse_damping=.25;
 h::StationaryGaugeSource<Real> source{};source.alpha=1;
 double identity=0,tangent_error=0;
 for(double angle:{.2,.7,1.3}) {
  h::mapped::Point map{};map.x[0]=cos(angle);map.x[1]=sin(angle);
  h::CartesianChartJets j{};j[0].f=1.002;j[1].f=.003;j[2].f=.002;j[4].f=-.001;
  j[13].f=.002;j[14].f=-.001;
  for(int i=0;i<3;++i){j[18].d[i]=.07*map.x[i];j[13].d[i]=.03*map.x[i];}
  h::ExpandCartesianChart(j);
  auto physical=h::SphericalPolicy::Physical(map.x,j);auto g=h::Geometry(physical.metric);
  double Q=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)Q+=g.inverse[a][b]*map.x[a]*map.x[b];
  for(int a=0;a<3;++a){j[22+a].f=(1-sqrt(j[0].f*Q))*map.x[a];j[22+a].d[a]=.03;}
  auto d=h::DiagnoseScriRate(map,.005,j,p,source,.5);
  identity=fmax(identity,fabs(d.lapse[10]));
  double values[2][11]{};const double step=1e-7;
  for(int side=0;side<2;++side){auto v=j;double eps=(side?1:-1)*step;
   for(int f=0;f<31;++f)if(h::CartesianIndependent(f))v[f].f+=eps*d.unprojected[f];
   h::ExpandCartesianChart(v);v[18].f=0;
   const auto u=h::SphericalPolicy::Physical(map.x,v);const auto geometry=h::Geometry(u.metric);
   double q=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)q+=geometry.inverse[a][b]*map.x[a]*map.x[b];
   for(int f=0;f<7;++f)values[side][f]=v[f].f;
   for(int a=0;a<3;++a)values[side][8+a]=-sqrt(v[0].f*q)*map.x[a];
  }
  for(int f=0;f<11;++f)tangent_error=fmax(tangent_error,fabs((values[1][f]-values[0][f])/(2*step)-d.metric_rate[f]));
 }
 std::cout<<"lapse identity="<<identity<<" projection tangent finite difference="<<tangent_error<<'\n';
 if(identity>1e-11||tangent_error>2e-7)return 1;
}
