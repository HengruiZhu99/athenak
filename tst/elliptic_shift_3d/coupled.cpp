// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Experimental bounded CPU composite evolution. No production guards relaxed.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "mesh/cubed_sphere/composite_checkpoint.hpp"
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"
#include "z4c/hyperboloidal/composite_state.hpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/puncture_refinement.hpp"

namespace h = z4c::hyperboloidal;
using V = multigrid::Vector;
constexpr int fields = 22;
const int tensor_i[6] = {0,0,0,1,1,2}, tensor_j[6] = {0,1,2,1,2,2};

cubed_sphere::CompositeMesh PilotMesh(int n,const V &radii,int bulk,int refinement=0){
  auto mesh=cubed_sphere::CompositeMesh(.25,radii,1,n,2,bulk);
  return h::RefinePunctureNeighborhood(std::move(mesh),h::PuncturePosition{},refinement,50000).mesh;
}

h::OmegaJet<Real> Omega(const Real x[3], const h::Z4cJet<Real> &u, bool boundary = false) {
  h::OmegaJet<Real> o{};
  o.omega = boundary ? 0 : (1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;
  for (int a = 0; a < 3; ++a) {o.gradient[a] = -x[a]; o.hessian[a][a] = -1;}
  h::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
  return o;
}

class Pilot {
 public:
  h::CompositeShiftStage shift;
  V state, beta_delta;
  std::vector<h::Z4cJet<Real>> jets;
  double max_shift_residual = 0;
  Real max_shift_point_residual=0;
  int elliptic_iterations = 0;
  double elliptic_atol;
  bool telegraph, wormhole=false;
  int isotropic_reference=0;
  int positive_storage=0;
  std::unique_ptr<cubed_sphere::ScriSurface> scri;
  V boundary_rhs, beta_padded;
  Real pole_sat=4,regularity_sat=8;
  Real telegraph_width=.24;
  std::unique_ptr<h::PunctureLogPlans> log_plans;
  Real boundary_pole=0,raw_clock_rate=0,boundary_regularity=0;
  Pilot(int n, int shells, bool flux, Real tolerance, int order = 4,bool regular_scri=false)
      : Pilot(cubed_sphere::CompositeMesh(.25, shells == 1 ? V{1} : V{.65, 1}, 1, n), flux, tolerance, order,regular_scri) {}
  Pilot(cubed_sphere::CompositeMesh mesh, bool flux, Real tolerance, int order = 4,bool regular_scri=false,bool puncture=false,int matched=0,int surface_points=6,bool radial_stencils=false,int positive_mode=0,Real taper_width=.24,bool enriched=false,bool radial_lines=false,bool topology=false)
      : shift(std::move(mesh), order,radial_stencils,radial_lines,topology),
        state(fields*shift.grid.active.size(), 0), beta_delta(3*shift.grid.active.size(), 0),
        elliptic_atol(tolerance), telegraph(flux),wormhole(puncture),isotropic_reference(matched),positive_storage(positive_mode),telegraph_width(taper_width) {
    if(enriched){if(!wormhole||positive_storage!=2)throw std::invalid_argument("enrichment requires exponent-independent log storage");
      log_plans=std::make_unique<h::PunctureLogPlans>(shift.grid);}
    if(regular_scri)scri=std::make_unique<cubed_sphere::ScriSurface>(shift.grid,fields,surface_points);
  }
  h::PositiveFieldChart<Real> PositiveChart(int f,const Real x[3])const{
    h::PositiveFieldChart<Real> chart;chart.mode=positive_storage==1?h::PositiveFieldMode::direct:
      (positive_storage==2?h::PositiveFieldMode::log_reference:h::PositiveFieldMode::square);
    chart.reference.value=1;
    if(f==18){chart.reference.value=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
      for(int a=0;a<3;++a){chart.reference.d[a]=x[a];chart.reference.dd[a][a]=1;}}
    return chart;
  }
  Real EncodePositive(int f,Real physical,const Real x[3])const{
    h::ScalarJet<Real> p{},q{};p.value=physical;const auto s=PositiveChart(f,x).Encode(p,q);
    if(s!=h::PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart encode: ")+h::PositiveFieldStatusName(s));return q.value;
  }
  Real DecodePositive(int f,Real stored,const Real x[3])const{
    h::ScalarJet<Real> p{},q{};q.value=stored;const auto s=PositiveChart(f,x).Reconstruct(q,p);
    if(s!=h::PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart reconstruct: ")+h::PositiveFieldStatusName(s));return p.value;
  }
  Real PositiveRate(int f,Real value,Real rate,const Real x[3])const{
    Real result;const auto s=PositiveChart(f,x).EncodeRate(value,rate,result);
    if(s!=h::PositiveFieldStatus::valid)throw std::runtime_error(std::string("positive chart rate: ")+h::PositiveFieldStatusName(s));return result;
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
      const auto chart=h::WormholeChart::At(x,isotropic_reference);
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
    std::vector<h::MetricJet<Real>> metrics;
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
      jets[p] = h::UnpackComposite(grid.geometry[p].x, values, d, dd,wormhole,isotropic_reference,positive_storage);
      if (!(jets[p].alpha.value > 0 && jets[p].chi.value > 0))
        throw std::runtime_error("coupled stage lost lapse/chi positivity");
      metrics.push_back(jets[p].metric);
      chi.push_back(jets[p].chi.value);
    }
    if(log_plans)for(const auto &row:log_plans->rows){
      for(int field:{0,18}){auto stored=log_plans->Sample(row,input,fields,field);
        auto &physical=field==0?jets[row.active].chi:jets[row.active].alpha;
        const auto status=PositiveChart(field,grid.geometry[row.active].x).Reconstruct(stored,physical);
        if(status!=h::PositiveFieldStatus::valid)throw std::runtime_error("enriched positive-field reconstruction failed");}}
    shift.SetGeometry(metrics, chi);
  }
  h::CompositeShiftStage::Boundary Boundary(const V &padded) const {
    return [this, &padded, cache = std::map<std::array<Real, 3>, std::array<Real, 3>>{}]
        (const std::array<Real, 3> &x) mutable {
      const auto found = cache.find(x);
      if (found != cache.end()) return found->second;
      const auto &grid = shift.grid;
      Real chi = 1;
      h::MetricJet<Real> metric{};
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
      if (!h::FixedAngleScriShift(h::Geometry(metric), 1., chi, x.data(), beta))
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
      h::EllipticShiftSource(jets[p], shift.geometry[p], 0., rhs);
      h::ApplyEllipticShift(jets[p].metric, shift.geometry[p], jets[p].beta, reference_rate);
      for (int v = 0; v < 3; ++v) source[3*p+v] = rhs[v]-reference_rate[v];
    }
    const auto boundary = Boundary(padded);
    h::ShiftSolveControl controls;if(positive_storage){controls.conformal_norm_fraction=.5;controls.maximum_residual_tolerance=1e-7;}
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
    const h::CMCReference<Real> reference{1,1};boundary_rhs.assign(scri->values.size(),0);
    boundary_pole=raw_clock_rate=boundary_regularity=0;
    for(size_t p=0;p<scri->rows.size();++p){
      const auto &row=scri->rows[p];const auto &x=row.geometry.x;
      const auto sampled=scri->Sample(p,padded,scri->values,fields);
      auto u=h::UnpackComposite(x,sampled.value,sampled.d,sampled.dd,wormhole,isotropic_reference,positive_storage);
      const auto shift_jet=scri->Sample(p,beta_padded,scri->beta,3);
      for(int v=0;v<3;++v){u.beta.value[v]+=shift_jet.value[v];
        for(int a=0;a<3;++a){u.beta.d[a][v]+=shift_jet.d[v][a];for(int b=0;b<3;++b)u.beta.dd[a][b][v]+=shift_jet.dd[v][a][b];}}
      const auto geometry=h::Geometry(u.metric);
      if(!geometry.valid||!(u.alpha.value>0&&u.chi.value>0))throw std::runtime_error("invalid evolved scri state");
      const auto omega=Omega(x,u,true);
      auto parts=h::ConformalRHS(u,omega,4.,-.99,true);
      const Real normal[3]={x[0],x[1],x[2]};
      const auto pole=h::TensorPoleNormalJet(u,omega,x,normal,4.,-.99,1.,1.,true,wormhole);
      auto rate=h::TensorScriLimit(parts.regular,pole,-1.);
      const auto gauge=h::UnfactoredReferenceGauge(reference,reference.At(x[0],x[1],x[2]),
          u,h::GaugeParameters<Real>{1,0,wormhole?2.:1.,0},wormhole,true);
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
    const h::CMCReference<Real> reference{1, 1};
    for (size_t p = 0; p < jets.size(); ++p) {
      const auto &u = jets[p]; const auto &x = grid.geometry[p].x;
      const auto omega = Omega(x, u);
      if (!(omega.omega > 0)) throw std::runtime_error("active coupled cell not inside scri");
      auto parts = h::ConformalRHS(u, omega, 4., -.99,true);
      // Same disclosed Appendix-C Theta source as the verified 1D adapter.
      h::Z4cRHS<Real> rate{};
      if (!h::AssembleInterior(parts, omega.omega, rate)) throw std::runtime_error("invalid tensor evolution rate");
      const auto gauge = h::UnfactoredReferenceGauge(reference, reference.At(x[0], x[1], x[2]),
          u, h::GaugeParameters<Real>{1, 0, 1, 0}, false, true);
      h::GaugeRHS<Real> lapse{};
      if (!h::AssembleGaugeInterior(gauge, omega.omega, lapse)) throw std::runtime_error("invalid lapse rate");
      if(wormhole)lapse.alpha=h::WormholeLapseRate(x,u);
      rhs[p*fields] = rate.chi; rhs[p*fields+13] = rate.trace; rhs[p*fields+14] = rate.theta;
      rhs[p*fields+18] = lapse.alpha;
      for (int t = 0; t < 6; ++t) {
        rhs[p*fields+1+t] = rate.metric[tensor_i[t]][tensor_j[t]];
        rhs[p*fields+7+t] = rate.a[tensor_i[t]][tensor_j[t]];
      }
      for (int i = 0; i < 3; ++i) rhs[p*fields+15+i] = rate.lambda[i];
      const Real radius = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const auto support=h::ReferenceTelegraphSupport(radius,.8,telegraph_width);
      if(!support.valid)throw std::runtime_error("invalid selected telegraph support");
      const Real window=support.value,window_r=support.radial_derivative;
      if (telegraph) {
        h::VectorJet<Real> flux{};
        for (int v = 0; v < 3; ++v) {
          flux.value[v] = input[p*fields+19+v];
          Real d[3], dd[3][3]; grid.Derivatives(padded, 19+v, p, d, dd);
          for (int a = 0; a < 3; ++a) flux.d[a][v] = d[a];
        }
        const Real reference_gradient[3] = {x[0], x[1], x[2]};
        const Real radial_gradient = radius > 0 ? window_r/radius : 0;
        const Real window_gradient[3] = {radial_gradient*x[0], radial_gradient*x[1], radial_gradient*x[2]};
        const auto addition = h::ReferenceTelegraph(u, flux, reference_gradient, window, window_gradient, .02, .005);
        rhs[p*fields+18] += addition.lapse;
        for (int i = 0; i < 3; ++i) rhs[p*fields+19+i] = addition.flux[i];
      }
      if(positive_storage){rhs[p*fields]=PositiveRate(0,u.chi.value,rhs[p*fields],x);
        rhs[p*fields+18]=PositiveRate(18,u.alpha.value,rhs[p*fields+18],x);}
      else if(wormhole){const h::WormholeChart chart(x);
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

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if(argc>1&&std::string(argv[1])=="--coordinates"){
      if(argc!=7)throw std::invalid_argument("coordinates: N radii bulk refinement output.csv");
      const int n=std::stoi(argv[2]),bulk=std::stoi(argv[4]),refinement=std::stoi(argv[5]);
      V radii;std::stringstream list(argv[3]);std::string part;
      while(std::getline(list,part,','))radii.push_back(std::stod(part));
      if(n<6||n>16||bulk<1||bulk>4||radii.empty()||radii.size()>16||radii.back()!=1)
        throw std::invalid_argument("bounded coordinate descriptor");
      const auto mesh=PilotMesh(n,radii,bulk,refinement);
      std::ofstream out(argv[6]);out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";size_t id=0;
      for(size_t m=0;m<mesh.blocks.size();++m)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
        const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(m,q,x);
        out<<0<<','<<id++<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<'\n';}
      if(!out)throw std::runtime_error("coordinate output failed");
      std::cout<<"{\"cells\":"<<id<<",\"revision\":"<<mesh.revision<<"}\n";
      Kokkos::finalize();return 0;
    }
    const int n = argc > 1 ? std::stoi(argv[1]) : 6;
    const Real amplitude = argc > 2 ? std::stod(argv[2]) : 0.;
    const Real dt = argc > 3 ? std::stod(argv[3]) : .0005;
    const int steps = argc > 4 ? std::stoi(argv[4]) : 2;
    const bool angular = argc > 5 ? std::stoi(argv[5]) : false;
    const int shells = argc > 6 ? std::stoi(argv[6]) : 1;
    const Real tolerance = argc > 7 ? std::stod(argv[7]) : 1e-11;
    const std::string profile = argc > 8 ? argv[8] : "";
    const std::string checkpoint = argc > 9 ? argv[9] : "";
    const std::string source_tag = argc > 10 ? argv[10] : "";
    const int spatial_order = argc > 11 ? std::stoi(argv[11]) : 4;
    const bool regular_scri = argc > 12 ? std::stoi(argv[12]) : false;
    const bool wormhole=argc>15?std::stoi(argv[15]):false;
    const std::string initial=argc>16?argv[16]:"";
    const int bulk=argc>17?std::stoi(argv[17]):1;
    const int isotropic_reference=argc>18?std::stoi(argv[18]):0;
    V radii=shells==1?V{1}:V{.65,1};
    if(argc>19&&std::string(argv[19]).size()){radii.clear();std::stringstream list(argv[19]);std::string value;
      while(std::getline(list,value,','))radii.push_back(std::stod(value));}
    const int surface_points=argc>20?std::stoi(argv[20]):6;
    const bool radial_stencils=argc>21?std::stoi(argv[21]):false;
    const int positive_storage=argc>22?std::stoi(argv[22]):0;
    const Real telegraph_width=argc>23?std::stod(argv[23]):.24;
    const bool enriched=argc>24?std::stoi(argv[24]):false;
    const bool radial_lines=argc>25?std::stoi(argv[25]):false;
    const bool topology=argc>26?std::stoi(argv[26]):false;
    const int refinement=argc>27?std::stoi(argv[27]):0;
    if (n < 6 || n > 16 || steps < 0 || steps > 200 || dt <= 0 || dt*steps > .1
        || std::abs(amplitude) > 1e-3 || shells<1||shells>16||radii.size()!=size_t(shells)||radii.back()!=1||bulk<1||bulk>4
        ||size_t(n)*n*n*(bulk*bulk*bulk+6*shells)>50000
        ||surface_points<6||surface_points>8||surface_points>n+1
        ||refinement<0||refinement>30|| (refinement>0&&!topology)
        ||positive_storage<0||positive_storage>3||(!wormhole&&positive_storage)
        ||!(telegraph_width>0&&telegraph_width<.395)|| (enriched&&positive_storage!=2)
        ||isotropic_reference<0||isotropic_reference>2||(!wormhole&&isotropic_reference)
        ||(wormhole&&(!regular_scri||n%2||initial.empty())))
      throw std::invalid_argument("bounded coupled pilot budget");
    const auto start = std::chrono::steady_clock::now();
    Pilot pilot(PilotMesh(n,radii,bulk,refinement),true,tolerance,spatial_order,regular_scri,wormhole,isotropic_reference,surface_points,radial_stencils,positive_storage,telegraph_width,enriched,radial_lines,topology);
    if(argc>13)pilot.pole_sat=std::stod(argv[13]);if(argc>14)pilot.regularity_sat=std::stod(argv[14]);
    if(pilot.pole_sat<0||pilot.regularity_sat<0)throw std::invalid_argument("negative scri weak-boundary coefficient");
    pilot.Initialize(amplitude, angular);
    if(wormhole)pilot.InitializeWormhole(initial,amplitude,angular);
    const int saved_step = steps/2;
    for (int i = 0; i < steps; ++i) {
      pilot.Step(dt);
      if (!checkpoint.empty() && i+1 == saved_step) {
        cubed_sphere::CompositeSnapshot snapshot{pilot.shift.grid.mesh, source_tag, dt, tolerance,
            static_cast<std::uint64_t>(i+1), fields, pilot.state, pilot.beta_delta};
        snapshot.spatial_order = spatial_order;
        if(pilot.scri){snapshot.boundary_scheme=1;snapshot.boundary_fields=pilot.scri->values;}
        snapshot.boundary_pole_sat=pilot.pole_sat;snapshot.boundary_regularity_sat=pilot.regularity_sat;
        snapshot.wormhole_chart=wormhole;
        snapshot.isotropic_reference=isotropic_reference;
        snapshot.scri_radial_points=surface_points;snapshot.radial_interface_stencils=radial_stencils;
        snapshot.positive_storage=positive_storage;snapshot.telegraph_width=telegraph_width;snapshot.enriched_log_derivatives=enriched;snapshot.radial_line_smoothing=radial_lines;snapshot.topology_coarsening=topology;
        cubed_sphere::CompositeCheckpoint::Write(checkpoint, snapshot);
      }
    }
    bool restart_equal = checkpoint.empty();
    if (!checkpoint.empty()) {
      const auto snapshot = cubed_sphere::CompositeCheckpoint::Read(checkpoint, source_tag);
      Pilot restarted(snapshot.mesh, snapshot.telegraph, snapshot.tolerance, snapshot.spatial_order,snapshot.boundary_scheme==1,snapshot.wormhole_chart,snapshot.isotropic_reference,snapshot.scri_radial_points,snapshot.radial_interface_stencils,snapshot.positive_storage,snapshot.telegraph_width,snapshot.enriched_log_derivatives,snapshot.radial_line_smoothing,snapshot.topology_coarsening);
      restarted.state = snapshot.fields; restarted.beta_delta = snapshot.shift_warm_start;
      restarted.pole_sat=snapshot.boundary_pole_sat;restarted.regularity_sat=snapshot.boundary_regularity_sat;
      if(restarted.scri)restarted.scri->values=snapshot.boundary_fields;
      for (std::uint64_t i = snapshot.step; i < static_cast<std::uint64_t>(steps); ++i)
        restarted.Step(snapshot.dt);
      restart_equal = restarted.state == pilot.state && restarted.beta_delta == pilot.beta_delta;
      if(pilot.scri)restart_equal=restart_equal&&restarted.scri->values==pilot.scri->values;
      if (!restart_equal) throw std::runtime_error("coupled restart is not bitwise reproducible");
    }
    pilot.RHS(pilot.state, dt);  // accepted-state diagnostics, with a fresh stage solve
    const auto &grid = pilot.shift.grid;
    Real H2 = 0, M2 = 0, theta2 = 0, Z2 = 0, volume = 0;
    Real maxH = 0, maxM = 0, drift = 0, min_alpha = 1, min_chi = 1;
    std::ofstream profiles;
    if (!profile.empty()) {
      profiles.open(profile); profiles << std::setprecision(17)
          << "x,y,z,r,weight,alpha,chi,g_rr,A_rr,Khat,Theta,Lambda_r,beta_r,H,M_norm,det_error";
      for (int v = 0; v < fields; ++v) profiles << ",state_" << v;
      profiles << ",delta_beta_x,delta_beta_y,delta_beta_z\n";
    }
    for (size_t p = 0; p < grid.active.size(); ++p) {
      const auto &u = pilot.jets[p];
      const auto c = h::EvolvedConstraints(u, Omega(grid.geometry[p].x, u));
      if (!c.valid) throw std::runtime_error("invalid coupled constraint diagnostic");
      const Real weight = grid.volumes[p]*std::sqrt(pilot.shift.geometry[p].determinant)/std::pow(u.chi.value, 1.5);
      volume += weight; H2 += weight*c.hamiltonian*c.hamiltonian;
      M2 += weight*c.momentum_conformal_norm2; theta2 += weight*u.theta.value*u.theta.value;
      Z2 += weight*c.z4.z_conformal_norm2;
      maxH = std::max(maxH, std::abs(c.hamiltonian));
      maxM = std::max(maxM, std::sqrt(c.momentum_conformal_norm2));
      min_alpha = std::min(min_alpha, u.alpha.value); min_chi = std::min(min_chi, u.chi.value);
      for (int v = 0; v < fields; ++v) drift = std::max(drift, std::abs(pilot.state[p*fields+v]));
      if (profiles) {
        const auto &x = grid.geometry[p].x;
        const Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        Real grr = 0, arr = 0, lambda = 0, beta_r = 0;
        if (r > 0) for (int a = 0; a < 3; ++a) {
          lambda += u.lambda.value[a]*x[a]/r; beta_r += u.beta.value[a]*x[a]/r;
          for (int b = 0; b < 3; ++b) {
            grr += u.metric.g[a][b]*x[a]*x[b]/(r*r);
            arr += u.a.k[a][b]*x[a]*x[b]/(r*r);
          }
        }
        profiles << x[0] << ',' << x[1] << ',' << x[2] << ',' << r << ',' << weight << ','
                 << u.alpha.value << ',' << u.chi.value << ',' << grr << ',' << arr << ','
                 << u.trace.value << ',' << u.theta.value << ',' << lambda << ',' << beta_r << ','
                 << c.hamiltonian << ',' << std::sqrt(c.momentum_conformal_norm2) << ','
                 << c.z4.determinant_residual;
        for (int v = 0; v < fields; ++v) profiles << ',' << pilot.state[p*fields+v];
        for (int v = 0; v < 3; ++v) profiles << ',' << pilot.beta_delta[3*p+v];
        profiles << '\n';
      }
    }
    const V padded = pilot.Padded(pilot.state);
    Real scri_clock=0,scri_null=0,scri_tangent=0,ghost_clock=0,ghost_null=0,ghost_tangent=0;
    if(pilot.scri)for(size_t p=0;p<pilot.scri->rows.size();++p){
      const auto &x=pilot.scri->rows[p].geometry.x;
      const auto sample=pilot.scri->Sample(p,padded,pilot.scri->values,fields);
      const auto u=h::UnpackComposite(x,sample.value,sample.d,sample.dd,wormhole,isotropic_reference,positive_storage);const auto geometry=h::Geometry(u.metric);
      if(!geometry.valid)throw std::runtime_error("invalid accepted scri metric");
      Real bn=0,bghost=0,norm=0,ghost[3];
      for(int i=0;i<3;++i){bn+=(-x[i]+pilot.scri->beta[3*p+i])*x[i];
        ghost[i]=-x[i]+pilot.scri->ReconstructFromGhost(p,pilot.beta_padded,i);bghost+=ghost[i]*x[i];
        for(int j=0;j<3;++j)norm+=u.chi.value*geometry.inverse[i][j]*x[i]*x[j];}
      const Real ag=pilot.scri->ReconstructFromGhost(p,padded,18);
      const Real alpha_ghost=positive_storage?pilot.DecodePositive(18,ag,x):(wormhole?std::exp(ag):(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2+ag);
      scri_clock=std::max(scri_clock,std::abs(u.alpha.value-1));ghost_clock=std::max(ghost_clock,std::abs(alpha_ghost-1));
      scri_null=std::max(scri_null,std::abs(bn+u.alpha.value*std::sqrt(norm)));
      ghost_null=std::max(ghost_null,std::abs(bghost+alpha_ghost*std::sqrt(norm)));
      for(int i=0;i<3;++i){scri_tangent=std::max(scri_tangent,std::abs(-x[i]+pilot.scri->beta[3*p+i]-bn*x[i]));
        ghost_tangent=std::max(ghost_tangent,std::abs(ghost[i]-bghost*x[i]));}
    }
    if (!profile.empty()) {
      std::ofstream spheres(profile+".spheres.csv");
      spheres << std::setprecision(17) << "r,direction,alpha,chi,g_rr,A_rr,Khat,Theta,Lambda_r,beta_r\n";
      constexpr int directions = 64;
      for (int radial = 1; radial < 20; ++radial)
      for (int direction = 0; direction < directions; ++direction) {
        const Real radius = radial/20.;
        const Real z = 1-2*(direction+.5)/directions;
        const Real phi = direction*2.39996322972865332;
        const Real normal[3] = {std::sqrt(1-z*z)*std::cos(phi), std::sqrt(1-z*z)*std::sin(phi), z};
        const Real x[3] = {radius*normal[0], radius*normal[1], radius*normal[2]};
        V value(fields, 0); Real delta[3]{};
        for (const auto &entry : grid.Interpolate(x, std::min(n, 6))) {
          for (int v = 0; v < fields; ++v) value[v] += entry.weight*padded[v*grid.count+entry.point];
          // Convert the padded donor index back to its active lexicographic ID.
          const int i = entry.point%grid.padded-grid.ng;
          const int j = entry.point/grid.padded%grid.padded-grid.ng;
          const int k = entry.point/(grid.padded*grid.padded)%grid.padded-grid.ng;
          const int m = entry.point/(grid.padded*grid.padded*grid.padded);
          const size_t active = ((m*n+k)*n+j)*n+i;
          for (int v = 0; v < 3; ++v) delta[v] += entry.weight*pilot.beta_delta[3*active+v];
        }
        Real grr = 1, arr = 0, lambda = 0, beta_r = -radius;
        for (int t = 0; t < 6; ++t) {
          const int i = tensor_i[t], j = tensor_j[t];
          const Real weight = normal[i]*normal[j]*(i == j ? 1 : 2);
          grr += weight*value[1+t]; arr += weight*value[7+t];
        }
        for (int v = 0; v < 3; ++v) {lambda += value[15+v]*normal[v]; beta_r += delta[v]*normal[v];}
        std::vector<std::array<Real,3>> zeros(fields);std::vector<std::array<std::array<Real,3>,3>> zz(fields);
        const auto extracted=h::UnpackComposite(x,value,zeros,zz,wormhole,isotropic_reference,positive_storage);
        arr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)arr+=extracted.a.k[i][j]*normal[i]*normal[j];
        spheres << radius << ',' << direction << ',' << extracted.alpha.value << ',' << extracted.chi.value << ',' << grr << ',' << arr << ','
                << extracted.trace.value << ',' << extracted.theta.value << ',' << lambda << ',' << beta_r << '\n';
      }
    }
    V beta(3*grid.count, 0);
    for (size_t p = 0; p < grid.active.size(); ++p)
      for (int v = 0; v < 3; ++v) beta[v*grid.count+grid.active[p]] = pilot.beta_delta[3*p+v];
    Real pole = 0, clock = 0, null = 0, tangential = 0;
    const h::CMCReference<Real> reference{1, 1};
    for (size_t m = 0; m < grid.mesh.blocks.size(); ++m) {
      if (grid.mesh.blocks[m].layer != shells-1) continue;
      for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) {
        const Real q[3] = {1, (j+.5)/n, (k+.5)/n};
        Real x[3]; grid.mesh.Forward(m, q, x);
        V values; std::vector<std::array<Real, 3>> d;
        std::vector<std::array<std::array<Real, 3>, 3>> dd;
        grid.SampleJets(padded, fields, x, values, d, dd);
        auto u = h::UnpackComposite(x, values, d, dd,wormhole,isotropic_reference,positive_storage);
        grid.SampleJets(beta, 3, x, values, d, dd);
        for (int v = 0; v < 3; ++v) {
          u.beta.value[v] += values[v];
          for (int a = 0; a < 3; ++a) {
            u.beta.d[a][v] += d[v][a];
            for (int b = 0; b < 3; ++b) u.beta.dd[a][b][v] += dd[v][a][b];
          }
        }
        const auto omega = Omega(x, u, true);
        auto parts = h::ConformalRHS(u, omega, 4., -.99,true);
          const auto gauge = h::UnfactoredReferenceGauge(reference, reference.At(x[0], x[1], x[2]),
            u, h::GaugeParameters<Real>{1, 0, wormhole?2.:1., 0}, wormhole, true);
        pole = std::max({pole, std::abs(parts.pole.chi), std::abs(parts.pole.trace),
                        std::abs(parts.pole.theta), std::abs(gauge.pole.alpha)});
        for (int a = 0; a < 3; ++a) {
          pole = std::max(pole, std::abs(parts.pole.lambda[a]));
          for (int b = 0; b < 3; ++b) pole = std::max(pole, std::abs(parts.pole.a[a][b]));
        }
        const auto geometry = h::Geometry(u.metric);
        Real bn = 0, norm = 0;
        for (int a = 0; a < 3; ++a) {
          bn += u.beta.value[a]*x[a];
          for (int b = 0; b < 3; ++b) norm += u.chi.value*geometry.inverse[a][b]*x[a]*x[b];
        }
        null = std::max(null, std::abs(bn+u.alpha.value*std::sqrt(norm)));
        clock = std::max(clock, std::abs(u.alpha.value-1));
        for (int a = 0; a < 3; ++a) tangential = std::max(tangential, std::abs(u.beta.value[a]-bn*x[a]));
      }
    }
    std::cout << std::setprecision(17) << "{\"n\":" << n << ",\"shells\":" << shells
              << ",\"cells\":" << grid.active.size() << ",\"amplitude\":" << amplitude
              << ",\"angular\":" << angular << ",\"dt\":" << dt << ",\"steps\":" << steps
              << ",\"time\":" << dt*steps << ",\"seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << ",\"elliptic_iterations\":" << pilot.elliptic_iterations
              << ",\"shift_residual_max\":" << pilot.max_shift_residual
              << ",\"shift_residual_point_max\":"<<pilot.max_shift_point_residual
              << ",\"H_rms\":" << std::sqrt(H2/volume) << ",\"M_rms\":" << std::sqrt(M2/volume)
              << ",\"Theta_rms\":" << std::sqrt(theta2/volume) << ",\"Z_rms\":" << std::sqrt(Z2/volume)
              << ",\"H_max\":" << maxH << ",\"M_max\":" << maxM << ",\"max_deviation\":" << drift
              << ",\"min_alpha\":" << min_alpha << ",\"min_chi\":" << min_chi
              << ",\"scri_pole\":" << pole << ",\"scri_clock_extrapolation\":" << clock
              << ",\"scri_null_extrapolation\":" << null << ",\"scri_tangent_extrapolation\":" << tangential
              << ",\"restart_checked\":" << !checkpoint.empty() << ",\"restart_equal\":" << restart_equal
              << ",\"spatial_order\":" << spatial_order
              << ",\"wormhole_chart\":"<<wormhole<<",\"bulk_subdivisions\":"<<bulk
              << ",\"isotropic_reference\":"<<isotropic_reference
              << ",\"scri_radial_points\":"<<surface_points<<",\"radial_interface_stencils\":"<<radial_stencils
              << ",\"topology_coarsening\":"<<topology
              << ",\"mesh_revision\":"<<grid.mesh.revision
              << ",\"radial_line_smoothing\":"<<radial_lines
              << ",\"enriched_log_derivatives\":"<<enriched
              << ",\"enriched_cells\":"<<(pilot.log_plans?pilot.log_plans->rows.size():0)
              << ",\"telegraph_width\":"<<telegraph_width
              << ",\"positive_storage\":"<<positive_storage
              << ",\"regular_scri\":"<<regular_scri<<",\"scri_nodal_clock\":"<<scri_clock<<",\"scri_nodal_null\":"<<scri_null
              << ",\"scri_nodal_tangent\":"<<scri_tangent<<",\"scri_ghost_clock\":"<<ghost_clock<<",\"scri_ghost_null\":"<<ghost_null
              << ",\"scri_ghost_tangent\":"<<ghost_tangent<<",\"scri_tensor_pole\":"<<pilot.boundary_pole
              << ",\"scri_raw_clock_rate\":"<<pilot.raw_clock_rate<<",\"scri_regularity\":"<<pilot.boundary_regularity
              << ",\"closure\":\""<<(regular_scri?"evolved_tensor_limit":"experimental_active_only_extrapolation")<<"\"}\n";
    // Report original scientific gates without weakening them for a coarse grid.
    if(regular_scri){if(pilot.boundary_pole>1e-7||ghost_clock>1e-13||ghost_null>1e-12||ghost_tangent>1e-12
        ||(!wormhole&&amplitude==0&&drift>1e-9))status=1;}
    else if (pole > 1e-7 || clock > 1e-13 || null > 1e-12 || (amplitude == 0 && drift > 1e-9)) status = 1;
  } catch (const std::exception &error) {std::cerr << error.what() << "\n"; status = 2;}
  Kokkos::finalize();
  return status;
}
