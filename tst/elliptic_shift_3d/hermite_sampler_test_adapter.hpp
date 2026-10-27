// Test-only adapter for regression against the qualified prototype interface.
#ifndef TEST_HERMITE_SAMPLER_ADAPTER_HPP_
#define TEST_HERMITE_SAMPLER_ADAPTER_HPP_
#include "mesh/cubed_sphere/hermite_boundary_sampler.hpp"
#include "mesh/cubed_sphere/centered_surface_atlas.hpp"
class HermiteSamplerTestAdapter {
 public:
  HermiteSamplerTestAdapter(const cubed_sphere::ScriSurface&s,const std::vector<Real>&v,const std::vector<Real>&b,int fields):sampler_(s),prepared_(sampler_.Prepare(v,b,fields,1)){}
  cubed_sphere::CenteredSurfaceAtlas::Jets At(int block,const Real*q)const{auto value=sampler_.SampleAt(block,q,prepared_);return {std::move(value.value),std::move(value.d),std::move(value.dd)};}
  auto diagnostics()const{return sampler_.diagnostics();}
 private:
  cubed_sphere::HermiteBoundarySampler sampler_;
  cubed_sphere::HermiteBoundarySampler::Prepared prepared_;
};
#endif
