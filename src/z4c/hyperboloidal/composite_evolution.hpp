// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_EVOLUTION_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_EVOLUTION_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "mesh/cubed_sphere/hermite_boundary_sampler.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"
#include "z4c/hyperboloidal/composite_state.hpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/supported_puncture_log_plans.hpp"
#include "z4c/hyperboloidal/puncture_log_value_plans.hpp"

namespace z4c {
namespace hyperboloidal {

// Existing host stage engine, mechanically extracted from the bounded research
// pilot. Physics, boundary treatment, solver controls and classical RK4 remain
// unchanged. This reusable production-location class is NOT native task wiring;
// the caller owns field adaptation, native RK stages and mesh/restart lifecycle.
inline OmegaJet<Real> CompositeOmegaAt(const Real x[3], const Z4cJet<Real> &u, bool boundary = false) {
  OmegaJet<Real> o{};
  o.omega = boundary ? 0 : (1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;
  for (int a = 0; a < 3; ++a) {o.gradient[a] = -x[a]; o.hessian[a][a] = -1;}
  SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
  return o;
}

class CompositeTensorEvolution {
 public:
  using V = multigrid::Vector;
  static constexpr int fields = 22;
  inline static constexpr int tensor_i[6] = {0,0,0,1,1,2};
  inline static constexpr int tensor_j[6] = {0,1,2,1,2,2};
  CompositeShiftStage shift;
  V state, beta_delta;
  std::vector<Z4cJet<Real>> jets;
  double max_shift_residual = 0;
  Real max_shift_point_residual=0;
  int elliptic_iterations = 0;
  double elliptic_atol;
  bool telegraph, wormhole=false;
  int isotropic_reference=0;
  int positive_storage=0;
  std::unique_ptr<cubed_sphere::ScriSurface> scri;
  std::unique_ptr<cubed_sphere::HermiteBoundarySampler> hermite_boundary;
  int boundary_derivative_scheme=0;
  std::uint64_t boundary_epoch=0;
  V boundary_rhs, beta_padded;
  Real pole_sat=4,regularity_sat=8;
  Real telegraph_width=.24;
  std::unique_ptr<PunctureLogPlans> log_plans;
  std::unique_ptr<SupportedPunctureLogPlans> supported_log_plans;
  int puncture_derivative_scheme=0;
  std::unique_ptr<PunctureLogValuePlans> log_halo_plans;
  double log_halo_setup_seconds=0;
  Real log_halo_radius=.12;
  Real boundary_pole=0,raw_clock_rate=0,boundary_regularity=0;
  CompositeTensorEvolution(int n, int shells, bool flux, Real tolerance, int order = 4,bool regular_scri=false)
      : CompositeTensorEvolution(cubed_sphere::CompositeMesh(.25, shells == 1 ? V{1} : V{.65, 1}, 1, n), flux, tolerance, order,regular_scri) {}
  CompositeTensorEvolution(cubed_sphere::CompositeMesh mesh, bool flux, Real tolerance, int order = 4,bool regular_scri=false,bool puncture=false,int matched=0,int surface_points=6,bool radial_stencils=false,int positive_mode=0,Real taper_width=.24,bool enriched=false,bool radial_lines=false,bool topology=false,bool log_halos=false,Real halo_radius=.12,int puncture_scheme=0,int boundary_derivatives=0)
      : shift(std::move(mesh), order,radial_stencils,radial_lines,topology),
        state(fields*shift.grid.active.size(), 0), beta_delta(3*shift.grid.active.size(), 0),
        elliptic_atol(tolerance), telegraph(flux),wormhole(puncture),isotropic_reference(matched),positive_storage(positive_mode),telegraph_width(taper_width),log_halo_radius(halo_radius) {
    if(puncture_scheme<0||puncture_scheme>1||(puncture_scheme&&(!enriched||!regular_scri)))
      throw std::invalid_argument("unsupported experimental puncture derivative scheme");
    puncture_derivative_scheme=puncture_scheme;
    if(boundary_derivatives<0||boundary_derivatives>1||(boundary_derivatives&&!regular_scri))
      throw std::invalid_argument("unsupported boundary derivative scheme");
    boundary_derivative_scheme=boundary_derivatives;
    if(enriched){if(!wormhole||positive_storage!=2)throw std::invalid_argument("enrichment requires exponent-independent log storage");
      if(puncture_derivative_scheme) supported_log_plans=std::make_unique<SupportedPunctureLogPlans>(shift.grid);
      else log_plans=std::make_unique<PunctureLogPlans>(shift.grid);}
    if(log_halos){
      if((!log_plans&&!supported_log_plans)||!regular_scri)throw std::invalid_argument("log halo enrichment requires enriched log-storage wormhole with tensor scri");
      if(!(log_halo_radius>0&&log_halo_radius<=shift.grid.mesh.half_width)||!std::isfinite(log_halo_radius))
        throw std::invalid_argument("log halo radius must lie inside the Cartesian half width");
      const auto start=std::chrono::steady_clock::now();
      log_halo_plans=std::make_unique<PunctureLogValuePlans>(shift.grid,PositivePlanPosition(),log_halo_radius,20000,8,175);
      log_halo_setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    if(regular_scri)scri=std::make_unique<cubed_sphere::ScriSurface>(shift.grid,fields,surface_points);
    if(boundary_derivative_scheme)hermite_boundary=std::make_unique<cubed_sphere::HermiteBoundarySampler>(*scri);
  }
  const PuncturePosition &PositivePlanPosition()const{
    if(supported_log_plans)return supported_log_plans->position;
    if(log_plans)return log_plans->position;
    throw std::logic_error("no positive-field derivative plans");
  }
  size_t EnrichedCells()const{return supported_log_plans?supported_log_plans->Rows().size():(log_plans?log_plans->rows.size():0);}
  PositiveFieldChart<Real> PositiveChart(int f,const Real x[3])const{
    PositiveFieldChart<Real> chart;chart.mode=positive_storage==1?PositiveFieldMode::direct:
      (positive_storage==2?PositiveFieldMode::log_reference:PositiveFieldMode::square);
    chart.reference.value=1;
    if(f==18){chart.reference.value=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
      for(int a=0;a<3;++a){chart.reference.d[a]=x[a];chart.reference.dd[a][a]=1;}}
    return chart;
  }
  Real EncodePositive(int f,Real physical,const Real x[3])const{
    ScalarJet<Real> p{},q{};p.value=physical;const auto s=PositiveChart(f,x).Encode(p,q);
    if(s!=PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart encode: ")+PositiveFieldStatusName(s));return q.value;
  }
  Real DecodePositive(int f,Real stored,const Real x[3])const{
    ScalarJet<Real> p{},q{};q.value=stored;const auto s=PositiveChart(f,x).Reconstruct(q,p);
    if(s!=PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart reconstruct: ")+PositiveFieldStatusName(s));return p.value;
  }
  Real PositiveRate(int f,Real value,Real rate,const Real x[3])const{
    Real result;const auto s=PositiveChart(f,x).EncodeRate(value,rate,result);
    if(s!=PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart rate: ")+PositiveFieldStatusName(s));return result;
  }

  void Initialize(Real amplitude, bool angular) {
    for (size_t p = 0; p < shift.grid.active.size(); ++p) {
      const auto &x = shift.grid.geometry[p].x;
      const Real r2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
      state[p*fields+18] = amplitude*std::pow(1-r2, 4)*std::exp(-r2/.09)
          *(angular ? 1+.8*x[0]*x[1]+.3*x[2] : 1);
    }
    // The analytic pulse vanishes at the actual endpoint. Geometric reference
    // deviations and flux start at zero; no evolved constraint is reset later.
    if(scri)std::fill(scri->values.begin(),scri->values.end(),0.);
    if(scri&&positive_storage)for(size_t p=0;p<scri->rows.size();++p){const auto &x=scri->rows[p].geometry.x;
      scri->values[p*fields]=EncodePositive(0,1,x);scri->values[p*fields+18]=EncodePositive(18,1,x);}
  }
  void InitializeWormhole(const std::string &path,Real amplitude,bool angular){
    if(!wormhole||!scri)throw std::invalid_argument("wormhole requires positive chart and tensor scri");
    std::ifstream file(path);size_t count;std::string magic;
    file>>magic>>count;
    if(magic!="ISOTROPIC_M1_3"||count!=shift.grid.active.size())throw std::runtime_error("wormhole initializer layout");
    for(size_t p=0;p<count;++p){const auto &x=shift.grid.geometry[p].x;
      Real r,chi,alpha,arr,khat;file>>r>>chi>>alpha>>arr>>khat;
      const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      if(!file||std::abs(r-radius)>1e-13||!(r>0&&chi>0&&alpha>0))throw std::runtime_error("wormhole initial data/radius mismatch");
      const auto chart=WormholeChart::At(x,isotropic_reference);
      const Real pulse=amplitude*std::pow(1-r*r,4)*std::exp(-r*r/.09)*(angular?1+.8*x[0]*x[1]+.3*x[2]:1);
      state[p*fields]=positive_storage?EncodePositive(0,chi,x):std::log(chi/chart.chi.f);
      state[p*fields+18]=positive_storage?EncodePositive(18,alpha*std::exp(pulse),x):std::log(alpha/chart.alpha.f)+pulse;
      state[p*fields+13]=3+khat/(positive_storage?1:chart.weight.f);
      for(int t=0;t<6;++t){const int i=tensor_i[t],j=tensor_j[t];
        state[p*fields+7+t]=1.5*arr*(x[i]*x[j]/(r*r)-(i==j?1./3:0))/(positive_storage?1:chart.weight.f);}
    }
    std::string extra;if(file>>extra)throw std::runtime_error("wormhole initializer trailing payload");
  }
  V Padded(const V &input) const {
    const auto &grid = shift.grid;
    V padded(fields*grid.count, 0);
    for (size_t p = 0; p < grid.active.size(); ++p)
    for (int v = 0; v < fields; ++v) padded[v*grid.count+grid.active[p]] = input[p*fields+v];
    if(scri){
      std::map<std::array<Real,3>,V> cache;
      grid.Fill(padded,fields,[&](int f,const auto &x){auto found=cache.find(x);
        if(found==cache.end())found=cache.emplace(x,scri->Interpolate(scri->values,fields,x.data())).first;
        return found->second[f];});
      if(log_halo_plans){
        V chi(padded.begin(),padded.begin()+grid.count);
        V lapse(padded.begin()+18*grid.count,padded.begin()+19*grid.count);
        log_halo_plans->FillPositiveLogs(chi,lapse,PositivePlanPosition());
        std::copy(chi.begin(),chi.end(),padded.begin());
        std::copy(lapse.begin(),lapse.end(),padded.begin()+18*grid.count);
      }
      return padded;
    }
    grid.FillExtrapolated(padded, fields);
    // The prescribed lapse clock supplies one gauge boundary datum. All geometric
    // fields use the disclosed candidate outflow extrapolation, without projection.
    for (const auto &row : grid.halos) if (row.boundary_weight) {
      const int base = 18*grid.count;
      const Real anchor = padded[base+row.entries[0].point];
      Real value = anchor;
      for (const auto &entry : row.entries) value += entry.weight*(padded[base+entry.point]-anchor);
      padded[base+row.target] = value-row.boundary_weight*anchor;
    }
    return padded;
  }
  void GeometryJets(const V &input, const V &padded) {
    const auto &grid = shift.grid;
    jets.resize(grid.active.size());
    V values(fields);
    std::vector<std::array<Real, 3>> d(fields);
    std::vector<std::array<std::array<Real, 3>, 3>> dd(fields);
    std::vector<MetricJet<Real>> metrics;
    std::vector<Real> chi;
    for (size_t p = 0; p < jets.size(); ++p) {
      for (int v = 0; v < fields; ++v) {
        values[v] = input[p*fields+v];
        Real first[3], second[3][3]; grid.Derivatives(padded, v, p, first, second);
        for (int a = 0; a < 3; ++a) {
          d[v][a] = first[a];
          for (int b = 0; b < 3; ++b) dd[v][a][b] = second[a][b];
        }
      }
      jets[p] = UnpackComposite(grid.geometry[p].x, values, d, dd,wormhole,isotropic_reference,positive_storage);
      if (!(jets[p].alpha.value > 0 && jets[p].chi.value > 0))
        throw std::runtime_error("coupled stage lost lapse/chi positivity");
      metrics.push_back(jets[p].metric);
      chi.push_back(jets[p].chi.value);
    }
    if(log_plans)for(const auto &row:log_plans->rows){
      for(int field:{0,18}){auto stored=log_plans->Sample(row,input,fields,field);
        auto &physical=field==0?jets[row.active].chi:jets[row.active].alpha;
        const auto status=PositiveChart(field,grid.geometry[row.active].x).Reconstruct(stored,physical);
        if(status!=PositiveFieldStatus::valid)throw std::runtime_error("enriched positive-field reconstruction failed");}}
    if(supported_log_plans)for(const auto &row:supported_log_plans->Rows()){
      for(int field:{0,18}){auto stored=supported_log_plans->Sample(row,input,fields,field);
        auto &physical=field==0?jets[row.active].chi:jets[row.active].alpha;
        if(PositiveChart(field,grid.geometry[row.active].x).Reconstruct(stored,physical)!=PositiveFieldStatus::valid)
          throw std::runtime_error("supported positive-field reconstruction failed");}}
    shift.SetGeometry(metrics, chi);
  }
  CompositeShiftStage::Boundary Boundary(const V &padded) const {
    return [this, &padded, cache = std::map<std::array<Real, 3>, std::array<Real, 3>>{}]
        (const std::array<Real, 3> &x) mutable {
      const auto found = cache.find(x);
      if (found != cache.end()) return found->second;
      const auto &grid = shift.grid;
      Real chi = 1;
      MetricJet<Real> metric{};
      for (int i = 0; i < 3; ++i) metric.g[i][i] = 1;
      if(scri){const auto value=scri->Interpolate(scri->values,fields,x.data());chi=positive_storage?DecodePositive(0,value[0],x.data()):(wormhole?std::exp(value[0]):chi+value[0]);
        for(int t=0;t<6;++t){int i=tensor_i[t],j=tensor_j[t];metric.g[i][j]+=value[1+t];if(i!=j)metric.g[j][i]=metric.g[i][j];}}
      else for (const auto &entry : grid.Interpolate(x.data(), std::min(grid.n, 6))) {
        chi += entry.weight*padded[entry.point];
        for (int t = 0; t < 6; ++t) {
          const int i = tensor_i[t], j = tensor_j[t];
          metric.g[i][j] += entry.weight*padded[(1+t)*grid.count+entry.point];
          if (i != j) metric.g[j][i] = metric.g[i][j];
        }
      }
      Real beta[3];
      if (!FixedAngleScriShift(Geometry(metric), 1., chi, x.data(), beta))
        throw std::runtime_error("invalid null scri shift data at coupled stage");
      std::array<Real, 3> delta{beta[0]+x[0], beta[1]+x[1], beta[2]+x[2]};
      cache.emplace(x, delta);
      return delta;
    };
  }
  void SolveShift(const V &padded) {
    const auto &grid = shift.grid;
    V source(3*jets.size());
    for (size_t p = 0; p < jets.size(); ++p) {
      Real rhs[3], reference_rate[3];
      EllipticShiftSource(jets[p], shift.geometry[p], 0., rhs);
      ApplyEllipticShift(jets[p].metric, shift.geometry[p], jets[p].beta, reference_rate);
      for (int v = 0; v < 3; ++v) source[3*p+v] = rhs[v]-reference_rate[v];
    }
    const auto boundary = Boundary(padded);
    ShiftSolveControl controls;if(positive_storage){controls.conformal_norm_fraction=.5;controls.maximum_residual_tolerance=1e-7;}
    controls.krylov_restart=48;
    const auto solve_start=std::chrono::steady_clock::now();
    const auto report = shift.Solve(source, boundary, beta_delta, elliptic_atol, 0., 240,controls);
    if(wormhole)std::cerr<<std::setprecision(17)<<"shift_solve iterations="<<report.iterations
      <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-solve_start).count()
      <<" physical_rms="<<report.physical_residual_rms<<" point_max="<<report.residual_max<<'\n';
    if (!report.converged) {
      std::cerr<<std::setprecision(17)<<"shift_failure iterations="<<report.iterations
        <<" physical_rms="<<report.physical_residual_rms
        <<" conformal_rms="<<report.conformal_residual_rms
        <<" point_max="<<report.residual_max
        <<" last_krylov="<<(report.residual.empty()?0:report.residual.back())<<'\n';
      throw std::runtime_error("coupled stage original shift equation did not converge");
    }
    elliptic_iterations += report.iterations;
    max_shift_residual = std::max(max_shift_residual, report.physical_residual_rms);
    max_shift_point_residual=std::max(max_shift_point_residual,report.residual_max);
    V delta(3*grid.count, 0);
    for (size_t p = 0; p < jets.size(); ++p)
    for (int v = 0; v < 3; ++v) delta[v*grid.count+grid.active[p]] = beta_delta[3*p+v];
    grid.Fill(delta, 3, [&](int v, const auto &x) {return boundary(x)[v];});
    if(scri){beta_padded=delta;
      for(size_t p=0;p<scri->rows.size();++p){const auto &x=scri->rows[p].geometry.x;
        const auto datum=boundary({x[0],x[1],x[2]});for(int v=0;v<3;++v)scri->beta[3*p+v]=datum[v];}}
    for (size_t p = 0; p < jets.size(); ++p)
    for (int v = 0; v < 3; ++v) {
      jets[p].beta.value[v] += beta_delta[3*p+v];
      Real d[3], dd[3][3]; grid.Derivatives(delta, v, p, d, dd);
      for (int a = 0; a < 3; ++a) {
        jets[p].beta.d[a][v] += d[a];
        for (int b = 0; b < 3; ++b) jets[p].beta.dd[a][b][v] += dd[a][b];
      }
    }
  }
  void BoundaryRates(const V &padded,Real dt){
    if(!scri)return;
    const CMCReference<Real> reference{1,1};boundary_rhs.assign(scri->values.size(),0);
    boundary_pole=raw_clock_rate=boundary_regularity=0;
    using Prepared=cubed_sphere::HermiteBoundarySampler::Prepared;
    std::optional<Prepared> prepared_fields,prepared_beta;
    if(hermite_boundary){
      if(boundary_epoch==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("boundary epoch exhausted");
      const auto epoch=++boundary_epoch;
      prepared_fields.emplace(hermite_boundary->Prepare(padded,scri->values,fields,epoch));
      prepared_beta.emplace(hermite_boundary->Prepare(beta_padded,scri->beta,3,epoch));
    }
    for(size_t p=0;p<scri->rows.size();++p){
      const auto &row=scri->rows[p];const auto &x=row.geometry.x;
      const auto sampled=hermite_boundary?hermite_boundary->Sample(p,*prepared_fields):scri->Sample(p,padded,scri->values,fields);
      auto u=UnpackComposite(x,sampled.value,sampled.d,sampled.dd,wormhole,isotropic_reference,positive_storage);
      const auto shift_jet=hermite_boundary?hermite_boundary->Sample(p,*prepared_beta):scri->Sample(p,beta_padded,scri->beta,3);
      for(int v=0;v<3;++v){u.beta.value[v]+=shift_jet.value[v];
        for(int a=0;a<3;++a){u.beta.d[a][v]+=shift_jet.d[v][a];for(int b=0;b<3;++b)u.beta.dd[a][b][v]+=shift_jet.dd[v][a][b];}}
      const auto geometry=Geometry(u.metric);
      if(!geometry.valid||!(u.alpha.value>0&&u.chi.value>0))throw std::runtime_error("invalid evolved scri state");
      const auto omega=CompositeOmegaAt(x,u,true);
      auto parts=ConformalRHS(u,omega,4.,-.99,true);
      const Real normal[3]={x[0],x[1],x[2]};
      const auto pole=TensorPoleNormalJet(u,omega,x,normal,4.,-.99,1.,1.,true,wormhole);
      auto rate=TensorScriLimit(parts.regular,pole,-1.);
      const auto gauge=UnfactoredReferenceGauge(reference,reference.At(x[0],x[1],x[2]),
          u,GaugeParameters<Real>{1,0,wormhole?2.:1.,0},wormhole,true);
      if(!parts.valid||!gauge.valid)throw std::runtime_error("invalid complete scri source");
      raw_clock_rate=std::max(raw_clock_rate,std::abs(gauge.regular.alpha-pole.lapse.derivative));
      const Real h=row.radial_spacing;
      if(dt*(pole_sat+regularity_sat)/h>.5)throw std::runtime_error("scri weak-boundary explicit-step guard");
      rate.chi+=pole_sat/h*parts.pole.chi;rate.trace+=pole_sat/h*parts.pole.trace;
      rate.theta+=pole_sat/h*parts.pole.theta;
      Real inverse_normal=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)inverse_normal+=geometry.inverse[i][j]*normal[i]*normal[j];
      // Necessary regularity traces follow from S_chi=S_Theta=S_alpha=0,
      // future nullness and alpha_scri=1. No conformal-flatness or Lambda=V
      // assumption enters. These are weak rate penalties, never field resets.
      const Real c=u.chi.value-1/inverse_normal,pk=u.trace.value+3;
      rate.chi-=regularity_sat/h*c;rate.trace-=regularity_sat/h*pk;rate.theta-=regularity_sat/h*u.theta.value;
      boundary_regularity=std::max({boundary_regularity,std::abs(c),std::abs(pk),std::abs(u.theta.value)});
      boundary_pole=std::max({boundary_pole,std::abs(parts.pole.chi),std::abs(parts.pole.trace),std::abs(parts.pole.theta),std::abs(gauge.pole.alpha)});
      boundary_rhs[p*fields]=rate.chi;boundary_rhs[p*fields+13]=rate.trace;boundary_rhs[p*fields+14]=rate.theta;
      for(int t=0;t<6;++t){const int i=tensor_i[t],j=tensor_j[t];
        rate.a[i][j]+=pole_sat/h*parts.pole.a[i][j];
        boundary_rhs[p*fields+1+t]=rate.metric[i][j];boundary_rhs[p*fields+7+t]=rate.a[i][j];
        boundary_pole=std::max(boundary_pole,std::abs(parts.pole.a[i][j]));}
      for(int v=0;v<3;++v){rate.lambda[v]+=pole_sat/h*parts.pole.lambda[v];
        boundary_rhs[p*fields+15+v]=rate.lambda[v];boundary_pole=std::max(boundary_pole,std::abs(parts.pole.lambda[v]));
        if(telegraph){Real flux=-sampled.value[19+v]/.02;
          for(int a=0;a<3;++a)flux+=u.beta.value[a]*sampled.d[19+v][a];boundary_rhs[p*fields+19+v]=flux;}}
      boundary_rhs[p*fields+18]=0; // prescribed incoming lapse clock, not a geometry projection
      if(positive_storage)boundary_rhs[p*fields]=PositiveRate(0,u.chi.value,boundary_rhs[p*fields],x);
      else if(wormhole)boundary_rhs[p*fields]/=u.chi.value;
    }
    for(Real value:boundary_rhs)if(!std::isfinite(value))throw std::runtime_error("nonfinite regular tensor scri rate");
  }
  V RHS(const V &input, Real dt) {
    const auto &grid = shift.grid;
    const V padded = Padded(input);
    GeometryJets(input, padded); SolveShift(padded);
    V rhs(input.size(), 0);
    const CMCReference<Real> reference{1, 1};
    for (size_t p = 0; p < jets.size(); ++p) {
      const auto &u = jets[p]; const auto &x = grid.geometry[p].x;
      const auto omega = CompositeOmegaAt(x, u);
      if (!(omega.omega > 0)) throw std::runtime_error("active coupled cell not inside scri");
      auto parts = ConformalRHS(u, omega, 4., -.99,true);
      // Same disclosed Appendix-C Theta source as the verified 1D adapter.
      Z4cRHS<Real> rate{};
      if (!AssembleInterior(parts, omega.omega, rate)) throw std::runtime_error("invalid tensor evolution rate");
      const auto gauge = UnfactoredReferenceGauge(reference, reference.At(x[0], x[1], x[2]),
          u, GaugeParameters<Real>{1, 0, 1, 0}, false, true);
      GaugeRHS<Real> lapse{};
      if (!AssembleGaugeInterior(gauge, omega.omega, lapse)) throw std::runtime_error("invalid lapse rate");
      if(wormhole)lapse.alpha=WormholeLapseRate(x,u);
      rhs[p*fields] = rate.chi; rhs[p*fields+13] = rate.trace; rhs[p*fields+14] = rate.theta;
      rhs[p*fields+18] = lapse.alpha;
      for (int t = 0; t < 6; ++t) {
        rhs[p*fields+1+t] = rate.metric[tensor_i[t]][tensor_j[t]];
        rhs[p*fields+7+t] = rate.a[tensor_i[t]][tensor_j[t]];
      }
      for (int i = 0; i < 3; ++i) rhs[p*fields+15+i] = rate.lambda[i];
      const Real radius = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const auto support=ReferenceTelegraphSupport(radius,.8,telegraph_width);
      if(!support.valid)throw std::runtime_error("invalid selected telegraph support");
      const Real window=support.value,window_r=support.radial_derivative;
      if (telegraph) {
        VectorJet<Real> flux{};
        for (int v = 0; v < 3; ++v) {
          flux.value[v] = input[p*fields+19+v];
          Real d[3], dd[3][3]; grid.Derivatives(padded, 19+v, p, d, dd);
          for (int a = 0; a < 3; ++a) flux.d[a][v] = d[a];
        }
        const Real reference_gradient[3] = {x[0], x[1], x[2]};
        const Real radial_gradient = radius > 0 ? window_r/radius : 0;
        const Real window_gradient[3] = {radial_gradient*x[0], radial_gradient*x[1], radial_gradient*x[2]};
        const auto addition = ReferenceTelegraph(u, flux, reference_gradient, window, window_gradient, .02, .005);
        rhs[p*fields+18] += addition.lapse;
        for (int i = 0; i < 3; ++i) rhs[p*fields+19+i] = addition.flux[i];
      }
      if(positive_storage){rhs[p*fields]=PositiveRate(0,u.chi.value,rhs[p*fields],x);
        rhs[p*fields+18]=PositiveRate(18,u.alpha.value,rhs[p*fields+18],x);}
      else if(wormhole){const WormholeChart chart(x);
        rhs[p*fields]/=u.chi.value;rhs[p*fields+18]/=u.alpha.value;
        rhs[p*fields+13]/=chart.weight.f;rhs[p*fields+14]/=chart.weight.f;
        for(int t=0;t<6;++t)rhs[p*fields+7+t]/=chart.weight.f;}
      // Conservative stage guard for transport/lapse/telegrapher speeds.
      Real speed = 0;
      const Real F = u.alpha.value*u.alpha.value+4*omega.omega*omega.omega*u.alpha.value;
      for (int A = 0; A < 3; ++A) {
        Real transport = 0, norm = 0;
        for (int i = 0; i < 3; ++i) {
          transport += grid.geometry[p].J[A][i]*u.beta.value[i];
          for (int j = 0; j < 3; ++j)
            norm += u.chi.value*shift.geometry[p].inverse[i][j]
                *grid.geometry[p].J[A][i]*grid.geometry[p].J[A][j];
        }
        speed += std::abs(transport)+std::sqrt(norm*(F+(telegraph ? .25*window*window : 0)));
      }
      if (dt*grid.n*speed > .5 || (telegraph && dt/.02 > .25))
        throw std::runtime_error("bounded coupled stage CFL/relaxation guard");
    }
    for (Real value : rhs) if (!std::isfinite(value)) throw std::runtime_error("nonfinite coupled stage");
    BoundaryRates(padded,dt);
    return rhs;
  }
  void Step(Real dt) {
    V temporary(state.size());
    V start_boundary=scri?scri->values:V{};
    const auto k1 = RHS(state, dt);
    const auto b1=boundary_rhs;
    if(scri)for(size_t i=0;i<b1.size();++i)scri->values[i]=start_boundary[i]+dt*b1[i]/2;
    for (size_t i = 0; i < state.size(); ++i) temporary[i] = state[i]+dt*k1[i]/2;
    const auto k2 = RHS(temporary, dt);
    const auto b2=boundary_rhs;
    if(scri)for(size_t i=0;i<b2.size();++i)scri->values[i]=start_boundary[i]+dt*b2[i]/2;
    for (size_t i = 0; i < state.size(); ++i) temporary[i] = state[i]+dt*k2[i]/2;
    const auto k3 = RHS(temporary, dt);
    const auto b3=boundary_rhs;
    if(scri)for(size_t i=0;i<b3.size();++i)scri->values[i]=start_boundary[i]+dt*b3[i];
    for (size_t i = 0; i < state.size(); ++i) temporary[i] = state[i]+dt*k3[i];
    const auto k4 = RHS(temporary, dt);
    const auto b4=boundary_rhs;
    if(scri)for(size_t i=0;i<b4.size();++i)scri->values[i]=start_boundary[i]+dt*(b1[i]+2*b2[i]+2*b3[i]+b4[i])/6;
    for (size_t i = 0; i < state.size(); ++i) state[i] += dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_COMPOSITE_EVOLUTION_HPP_
