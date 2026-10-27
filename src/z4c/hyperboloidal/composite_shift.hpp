// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_

#include <array>
#include <chrono>
#include <limits>
#include "multigrid/composite_poisson.hpp"
#include "multigrid/flexible_gmres.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"
#include "z4c/hyperboloidal/cartesian_affine_derivative.hpp"
#include "z4c/hyperboloidal/log_volume_norm.hpp"

namespace z4c {
namespace hyperboloidal {

// Optional algebraic controls, never changes to the physical operator or gauge.
// A bounded column transform beta=S y is applied by reconstruction BEFORE the
// original discrete stencil. This avoids separately differencing singular
// continuum conjugation terms. Physical residuals are always recomputed.
struct ShiftSolveControl {
  std::vector<Real> column_scale;
  Real conformal_norm_fraction=.5;
  Real maximum_residual_tolerance=1e-7;
  int krylov_restart=24;
};
struct ShiftSolveReport : multigrid::KrylovReport {
  Real physical_residual_rms=0,conformal_residual_rms=0,residual_max=0;
  Real physical_residual_l2=0,physical_volume=0;
  Real physical_log_volume=0,physical_log_residual_l2=0;
  bool physical_integrals_representable=true;
  Real original_residual_target=0,auxiliary_residual_target=0;
};

// Stage-owned frozen geometry: setting this does not modify evolved fields or
// project algebraic/differential constraints. Caller rebuilds after any regrid.
class CompositeShiftStage {
 public:
  using Boundary = std::function<std::array<Real, 3>(const std::array<Real, 3> &)>;
  cubed_sphere::CompositeDifference grid;
  std::vector<MetricJet<Real>> metric;
  std::vector<SpatialGeometry<Real>> geometry;
  // Normalized quadrature weights; the physical measure is retained in logs.
  std::vector<Real> norm_volume,log_norm_volume;
  Real log_physical_volume=0;
  std::vector<Real> conformal_volume;
  multigrid::CompositePoisson preconditioner;
  std::unique_ptr<CartesianAffineDerivative> affine_derivative;

  explicit CompositeShiftStage(cubed_sphere::CompositeMesh mesh, int order = 4,bool radial_stencils=false,bool radial_lines=false,bool topology=false,bool affine=false)
      : grid(mesh, order,radial_stencils), preconditioner(std::move(mesh),radial_stencils,radial_lines,topology) {
    if(affine)affine_derivative=std::make_unique<CartesianAffineDerivative>(grid);
  }

  void SetGeometry(const std::vector<MetricJet<Real>> &g, const std::vector<Real> &chi = {}) {
    if (g.size() != grid.active.size()) throw std::invalid_argument("stage metric shape");
    if (!chi.empty() && chi.size() != g.size()) throw std::invalid_argument("stage chi shape");
    metric = g; geometry.resize(g.size()); norm_volume.resize(g.size());conformal_volume.resize(g.size());
    log_norm_volume.clear();
    for (size_t p = 0; p < g.size(); ++p) {
      geometry[p] = Geometry(g[p]);
      if (!geometry[p].valid) throw std::runtime_error("invalid frozen shift geometry");
      const Real conformal_chi = chi.empty() ? 1 : chi[p];
      if (!(conformal_chi > 0) || !std::isfinite(conformal_chi))
        throw std::runtime_error("invalid conformal-volume coefficient");
      conformal_volume[p]=grid.volumes[p]*std::sqrt(geometry[p].determinant);
      log_norm_volume.push_back(std::log(conformal_volume[p])-1.5*std::log(conformal_chi));
    }
    log_physical_volume=LogSumPositive(log_norm_volume);
    for(size_t p=0;p<g.size();++p)norm_volume[p]=std::exp(log_norm_volume[p]-log_physical_volume);
  }
  void Apply(const multigrid::Vector &beta, multigrid::Vector &out,
              const Boundary &boundary = {}) const {
    const size_t size = grid.active.size();
    if (geometry.size() != size || beta.size() != 3*size)
      throw std::invalid_argument("shift stage geometry/vector shape");
    if(affine_derivative){
      const auto affine=affine_derivative->Fit(beta);const auto remainder=affine_derivative->PaddedRemainder(beta,affine,boundary);
      out.resize(beta.size());for(size_t p=0;p<size;++p){const auto jet=affine_derivative->Sample(beta,remainder,affine,p);
        Real result[3];ApplyEllipticShift(metric[p],geometry[p],jet,result);for(int v=0;v<3;++v)out[3*p+v]=-result[v];}
      return;
    }
    std::vector<Real> field(3*grid.count, 0);
    for (size_t p = 0; p < size; ++p)
      for (int v = 0; v < 3; ++v) field[v*grid.count+grid.active[p]] = beta[3*p+v];
    grid.Fill(field, 3, boundary ? std::function<Real(int, const std::array<Real, 3> &)>(
        [&](int v, const std::array<Real, 3> &x) {return boundary(x)[v];}) : nullptr);
    out.resize(beta.size());
    for (size_t p = 0; p < size; ++p) {
      VectorJet<Real> jet{};
      for (int v = 0; v < 3; ++v) {
        jet.value[v] = beta[3*p+v];
        Real d[3], dd[3][3]; grid.Derivatives(field, v, p, d, dd);
        for (int a = 0; a < 3; ++a) {
          jet.d[a][v] = d[a];
          for (int b = 0; b < 3; ++b) jet.dd[a][b][v] = dd[a][b];
        }
      }
      Real result[3]; ApplyEllipticShift(metric[p], geometry[p], jet, result);
      for (int v = 0; v < 3; ++v) out[3*p+v] = -result[v];
    }
  }
  double Dot(const multigrid::Vector &a, const multigrid::Vector &b) const {
    if (a.size() != 3*grid.active.size() || b.size() != a.size())
      throw std::invalid_argument("vector norm shape");
    return LogWeightedDot(a,b,log_norm_volume,log_physical_volume);
  }
  // Source means the ORIGINAL equation L_g beta = source. Inhomogeneous vector
  // boundary data enter the affine ghost lift; no potential boundary is assumed.
  ShiftSolveReport Solve(const multigrid::Vector &source,
      const Boundary &boundary, multigrid::Vector &beta, double atol = 1e-10,
      double rtol = 1e-10, int max_iterations = 160,const ShiftSolveControl &control={}) const {
    const Real mix=control.conformal_norm_fraction;
    if(beta.size()!=source.size())throw std::invalid_argument("shift solution/source vector shape");
    if(control.krylov_restart<1||control.krylov_restart>128||!(mix>=0&&mix<=1)||!std::isfinite(mix)||control.maximum_residual_tolerance<0
        ||!std::isfinite(control.maximum_residual_tolerance))throw std::invalid_argument("shift norm controls");
    const bool scaled=!control.column_scale.empty();
    if(scaled&&control.column_scale.size()!=grid.active.size())throw std::invalid_argument("shift column scale shape");
    for(Real s:control.column_scale)if(!std::isfinite(s)||s<Real(.25)||s>Real(4))
      throw std::invalid_argument("shift equilibration must remain in [1/4,4]");
    auto conformal_dot=[&](const auto &a,const auto &b){Real sum=0,volume=0;
      for(size_t p=0;p<grid.active.size();++p){for(int v=0;v<3;++v)sum+=conformal_volume[p]*a[3*p+v]*b[3*p+v];volume+=conformal_volume[p];}return sum/volume;};
    auto solver_dot=[&](const auto &a,const auto &b){return mix==0?Dot(a,b):(1-mix)*Dot(a,b)+mix*conformal_dot(a,b);};
    multigrid::Vector affine, zero(source.size(), 0), rhs(source.size());
    Apply(zero, affine, boundary);
    for (size_t p = 0; p < rhs.size(); ++p) rhs[p] = -source[p]-affine[p];
    auto y=beta;if(scaled)for(size_t p=0;p<grid.active.size();++p)for(int v=0;v<3;++v)y[3*p+v]/=control.column_scale[p];
    const Real target=std::max(atol,rtol*std::sqrt(Dot(source,source)));
    Real lower_bound=1;
    if(mix>0){Real vc=0;for(Real w:conformal_volume)vc+=w;
      for(size_t p=0;p<norm_volume.size();++p){
        const Real ratio=std::exp(std::min(Real(0),std::log(conformal_volume[p])-std::log(vc)
            -log_norm_volume[p]+log_physical_volume));
        lower_bound=std::min(lower_bound,(1-mix)+mix*ratio);}}
    if(!(lower_bound>0))throw std::runtime_error("shift solver norm loses physical residual control");
    int verification_applications=0;
    auto original_accept=[&](const auto &candidate,const auto &){
      auto physical=candidate;
      if(scaled)for(size_t p=0;p<grid.active.size();++p)for(int v=0;v<3;++v)physical[3*p+v]*=control.column_scale[p];
      multigrid::Vector applied,residual(source.size());Apply(physical,applied,boundary);++verification_applications;
      Real maximum=0;for(size_t i=0;i<source.size();++i){residual[i]=-source[i]-applied[i];maximum=std::max(maximum,std::abs(residual[i]));}
      return std::sqrt(Dot(residual,residual))<=target
          &&(mix==0||std::sqrt(conformal_dot(residual,residual))<=target)
          &&(!(control.maximum_residual_tolerance>0)||maximum<=control.maximum_residual_tolerance);
    };
    const auto krylov = multigrid::FlexibleGMRES(
        [&](const auto &a, auto &b) {if(!scaled){Apply(a,b);return;}auto physical=a;
          for(size_t p=0;p<grid.active.size();++p)for(int v=0;v<3;++v)physical[3*p+v]*=control.column_scale[p];Apply(physical,b);},
        [&](const auto &a, auto &b) {
          b.resize(a.size());
          for (int v = 0; v < 3; ++v) {
            multigrid::Vector scalar(grid.active.size());
            for (size_t p = 0; p < scalar.size(); ++p) scalar[p] = a[3*p+v];
            const auto correction = preconditioner.Cycle(scalar);
            for (size_t p = 0; p < scalar.size(); ++p) b[3*p+v] = correction[p]/(scaled?control.column_scale[p]:1);
          }
        }, solver_dot,rhs,y,target*std::sqrt(lower_bound),0.,max_iterations,control.krylov_restart,original_accept);
    beta=y;if(scaled)for(size_t p=0;p<grid.active.size();++p)for(int v=0;v<3;++v)beta[3*p+v]*=control.column_scale[p];
    ShiftSolveReport report;static_cast<multigrid::KrylovReport&>(report)=krylov;
    report.applications+=verification_applications;
    report.original_residual_target=target;report.auxiliary_residual_target=target*std::sqrt(lower_bound);
    multigrid::Vector applied,residual(source.size());Apply(beta,applied,boundary);++report.applications;
    for(size_t p=0;p<source.size();++p){residual[p]=-source[p]-applied[p];report.residual_max=std::max(report.residual_max,std::abs(residual[p]));}
    report.physical_residual_rms=std::sqrt(Dot(residual,residual));report.conformal_residual_rms=std::sqrt(conformal_dot(residual,residual));
    report.physical_log_volume=log_physical_volume;
    report.physical_log_residual_l2=report.physical_residual_rms>0?
        std::log(report.physical_residual_rms)+.5*log_physical_volume:-std::numeric_limits<Real>::infinity();
    const Real log_max=std::log(std::numeric_limits<Real>::max());
    report.physical_integrals_representable=log_physical_volume<=log_max&&report.physical_log_residual_l2<=log_max;
    report.physical_volume=log_physical_volume<=log_max?std::exp(log_physical_volume):std::numeric_limits<Real>::infinity();
    report.physical_residual_l2=report.physical_log_residual_l2<=log_max?
        std::exp(report.physical_log_residual_l2):std::numeric_limits<Real>::infinity();
    report.converged=report.converged&&report.physical_residual_rms<=target
      &&(mix==0||report.conformal_residual_rms<=target)
      &&(!(control.maximum_residual_tolerance>0)||report.residual_max<=control.maximum_residual_tolerance);
    return report;
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_COMPOSITE_SHIFT_HPP_
