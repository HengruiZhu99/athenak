// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "crossface_ray_interface_value.hpp"
#include "z4c/hyperboloidal/puncture_refinement.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
 namespace c=cubed_sphere; namespace h=z4c::hyperboloidal;
 auto mesh=c::CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,6,2,2);
 mesh=h::RefinePunctureNeighborhood(std::move(mesh),h::PuncturePosition{},3,50000).mesh;
 c::CompositeDifference g(std::move(mesh),4,true);
 Real x[3]={.2402147115485436,-.1201073557742718,-.1201073557742718};
 interface_probe::CartesianPlaneAtlas atlas(g);
 interface_probe::CrossFaceRayValue row;row.Build(g,x,16,argc>1?&atlas:nullptr);
 const auto&d=row.diagnostic;
 std::cout<<std::setprecision(17)<<"{\"status\":\""<<d.status<<"\",\"value_L1\":"<<d.value_L1<<",\"maximum_plane_L1\":"<<d.maximum_plane_L1<<",\"maximum_radial_L1\":"<<d.maximum_radial_L1<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
