// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PUNCTURE_NEIGHBORHOOD_HPP_
#define Z4C_HYPERBOLOIDAL_PUNCTURE_NEIGHBORHOOD_HPP_

#include <vector>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/puncture_position.hpp"

namespace z4c { namespace hyperboloidal {

// Position-stamped metadata for source/chart/AMR/halo consumers. Selections do
// not replace the grid's active set, mutate a halo plan, or impose field data.
// The owner must rebuild this object if the referenced grid's lifetime ends.
class PunctureNeighborhood {
 public:
  explicit PunctureNeighborhood(const cubed_sphere::CompositeDifference &grid,
      Real radius):grid_(grid){SetRadius(radius);}
  Real Radius() const{return radius_;}
  std::uint64_t RebuildCount()const{return rebuilds_;}
  void SetRadius(Real radius){
    if(!(radius>0)||!std::isfinite(radius)||!std::isfinite(radius*radius))
      throw std::invalid_argument("invalid tracked puncture neighborhood radius");
    if(radius!=radius_){radius_=radius;valid_=false;}
  }
  bool Matches(const PuncturePosition &puncture)const{
    puncture.Validate();
    return valid_&&puncture.Matches(stamp_)&&mesh_revision_==grid_.mesh.revision
      &&active_count_==grid_.active.size()&&halo_count_==grid_.halos.size();
  }
  bool Refresh(const PuncturePosition &puncture){
    if(Matches(puncture))return false;
    if(rebuilds_==std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("puncture neighborhood revision exhausted");
    valid_=false;active_.clear();halo_.clear();
    for(size_t p=0;p<grid_.active.size();++p)
      if(puncture.InNeighborhood(grid_.geometry[p].x,radius_))active_.push_back(p);
    for(const auto &row:grid_.halos){const auto x=HaloCoordinate(row.target);
      if(puncture.InNeighborhood(x.data(),radius_))halo_.push_back(row.target);}
    stamp_=puncture.Capture();mesh_revision_=grid_.mesh.revision;
    active_count_=grid_.active.size();halo_count_=grid_.halos.size();valid_=true;
    ++rebuilds_;return true;
  }
  const std::vector<size_t> &ActivePoints(const PuncturePosition &puncture){
    Refresh(puncture);return active_;
  }
  const std::vector<int> &HaloTargets(const PuncturePosition &puncture){
    Refresh(puncture);return halo_;
  }
  std::vector<Real> BoundedEquilibration(const PuncturePosition &puncture){
    Refresh(puncture);
    Real center2=0;for(Real x:puncture.position)center2+=x*x;
    if(!std::isfinite(center2)||!(std::sqrt(center2)+radius_<grid_.mesh.radii.back()))
      throw std::invalid_argument("puncture equilibration support must lie strictly inside scri");
    std::vector<Real> scale(grid_.active.size(),1);
    const Real radius2=radius_*radius_;
    for(size_t p:active_){const Real q=puncture.RadiusSquared(grid_.geometry[p].x)/radius2;
      const Real taper=1-q;scale[p]=1-Real(.5)*taper*taper*taper*taper;
      if(!(scale[p]>=Real(.5)&&scale[p]<=1)||!std::isfinite(scale[p]))
        throw std::runtime_error("bounded puncture equilibration range violation");}
    return scale;
  }
 private:
  const cubed_sphere::CompositeDifference &grid_;
  Real radius_=0;
  bool valid_=false;
  PuncturePosition::Stamp stamp_{};
  unsigned long mesh_revision_=0;
  size_t active_count_=0,halo_count_=0;
  std::uint64_t rebuilds_=0;
  std::vector<size_t> active_;
  std::vector<int> halo_;
  std::array<Real,3> HaloCoordinate(int point)const{
    if(point<0||point>=grid_.count)throw std::out_of_range("tracked halo index outside grid");
    const int padded=grid_.padded;
    const int block=point/(padded*padded*padded);
    const int i=point%padded-grid_.ng,j=point/padded%padded-grid_.ng;
    const int k=point/(padded*padded)%padded-grid_.ng;
    const Real q[3]={(i+.5)/grid_.n,(j+.5)/grid_.n,(k+.5)/grid_.n};
    std::array<Real,3>x;grid_.mesh.Forward(block,q,x.data());return x;
  }
};

}}  // namespace z4c::hyperboloidal
#endif
