// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Bounded representation/assembly/topology contracts; no hierarchy changes.
#include <iostream>
#include <iomanip>
#include "multigrid/composite_radial_line.hpp"
using Grid=cubed_sphere::CompositeDifference;
using V=std::vector<Real>;
void Apply(const Grid &grid,const V &u,V &out){V field(grid.count,0);for(size_t p=0;p<u.size();++p)field[grid.active[p]]=u[p];grid.Fill(field,1);out.resize(u.size());for(size_t p=0;p<u.size();++p){Real d[3],dd[3][3];grid.Derivatives(field,0,p,d,dd);out[p]=-(dd[0][0]+dd[1][1]+dd[2][2]);}}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  const std::vector<Real> radii{.45,.55,.60,.65,.70,.75,1};
  Grid angular(cubed_sphere::CompositeMesh(.25,radii,2,4,2,2),2,true);multigrid::CompositeRadialLine lines(angular);
  const auto &diag=lines.GetDiagnostics();const Real factor=lines.MaximumFactorResidual(),assembly=lines.MaximumAssemblyResidual(12);
  if(diag.lines!=384||diag.maximum_length!=28||diag.shell_points!=10752||factor>1e-12||assembly>1e-12)throw std::runtime_error("subdivided-face global-line assembly contract failed");
  V zero(angular.active.size(),0),u=zero;const auto apply=[&](const V &x,V &y){Apply(angular,x,y);};
  lines.Smooth(zero,u,1,apply);Real zero_error=0;for(Real x:u)zero_error=std::max(zero_error,std::abs(x));
  if(zero_error!=0)throw std::runtime_error("zero correction not preserved");
  std::vector<std::string> rejected;
  const auto expect=[&](const std::string &name,const std::function<void()> &action){bool caught=false;try{action();}catch(const std::exception &){caught=true;}if(!caught)throw std::runtime_error("missing explicit contract rejection: "+name);rejected.push_back(name);};
  expect("invalid_damping",[&](){lines.Smooth(zero,u,0,apply,0.);});
  expect("shape_mismatch",[&](){V bad;lines.Smooth(bad,u,0,apply);});
  expect("line_cap",[&](){multigrid::CompositeRadialLine too_short(angular,27);});
  angular.SetRadialInterfaceStencils(false);expect("stale_ghost_stencils",[&](){lines.Smooth(zero,u,0,apply);});
  Grid simple(cubed_sphere::CompositeMesh(.25,radii,1,2,2,2),2,true);multigrid::CompositeRadialLine tracked(simple);
  V simple_zero(simple.active.size(),0);simple.mesh.revision++;
  expect("stale_topology_revision",[&](){tracked.Smooth(simple_zero,simple_zero,0,[&](const V &x,V &y){Apply(simple,x,y);});});
  Grid fourth(cubed_sphere::CompositeMesh(.25,{.65,1},1,4),4,true);
  expect("unsupported_fourth_order_preconditioner",[&](){multigrid::CompositeRadialLine wrong_order(fourth);});
  // Altered angular metadata intentionally breaks one layer's matching face.
  // This checks explicit rejection rather than silently splitting a global line.
  Grid broken(cubed_sphere::CompositeMesh(.25,radii,1,2,2,2),2,true);
  for(auto &b:broken.mesh.blocks)if(b.face==0&&b.layer==6){b.lo[1]=-.9;break;}
  expect("unmatched_angular_layers",[&](){multigrid::CompositeRadialLine incomplete(broken);});
  std::cout<<std::setprecision(17)<<"{\"cells\":"<<angular.active.size()<<",\"shell_layers\":7,\"face_subdivisions\":2,\"lines\":"<<diag.lines<<",\"line_length\":"<<diag.maximum_length<<",\"factor_residual\":"<<factor<<",\"assembly_residual\":"<<assembly<<",\"zero_correction_error\":"<<zero_error<<",\"expected_rejections\":[";
  for(size_t i=0;i<rejected.size();++i){if(i)std::cout<<',';std::cout<<'\"'<<rejected[i]<<'\"';}std::cout<<"],\"scope\":\"CPU helper contracts only; no scalable/native/GPU claim\"}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';status=1;}Kokkos::finalize();return status;}
