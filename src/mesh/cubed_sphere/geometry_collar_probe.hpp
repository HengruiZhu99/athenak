// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_GEOMETRY_COLLAR_PROBE_HPP_
#define MESH_CUBED_SPHERE_GEOMETRY_COLLAR_PROBE_HPP_
#include <memory>
#include "mesh/cubed_sphere/balanced_volume_mls_probe.hpp"
#include "mesh/cubed_sphere/ray_collar_probe.hpp"
namespace cubed_sphere {
// Explicit geometry specialization, diagnostic only. Constant-radius angular
// surfaces cut the Cartesian core in a one-layer transition. Use the tested
// Cartesian polynomial construction there; spherical outer layers admit a
// physical-radius tensor interpolation. No row- or field-dependent fallback.
class GeometryCollarProbe {
 public:
 int angular_donors=48;Real maximum_l1=0,maximum_condition=0,maximum_reproduction=0;size_t rows_built=0;
 std::unique_ptr<BalancedVolumeMLSProbe> transition;
 std::unique_ptr<RayCollarProbe> spherical;
 GeometryCollarProbe(const ScriSurface&s,const std::vector<Real>&v,const std::vector<Real>&b,int fields){if(s.grid.mesh.radii.size()==1)transition=std::make_unique<BalancedVolumeMLSProbe>(s,v,b,fields,48);else spherical=std::make_unique<RayCollarProbe>(s,v,b,fields,48);}
 const char*Method()const{return transition?"fixed_ring_Cartesian_P5_MLS56":"physical_ray_Pn_times_angular_chart_P5";}
 size_t PlanCount()const{return transition?transition->plans.size():spherical->angular_cache.size();}
 std::vector<Real> Sample(int block,const Real*q){std::vector<Real> out;if(transition){out=transition->Sample(block,q);rows_built=transition->rows_built;maximum_l1=transition->maximum_l1;maximum_condition=transition->maximum_condition;maximum_reproduction=transition->maximum_reproduction;}else{out=spherical->Sample(block,q);rows_built=spherical->rows_built;maximum_l1=spherical->maximum_l1;maximum_condition=spherical->maximum_condition;maximum_reproduction=spherical->maximum_reproduction;}return out;}
};
} // namespace cubed_sphere
#endif
