// Constant-expansion continuation and independent numerical sandwich diagnostics.
#include "z4c/cartoon_m0_fastflow.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace z4c {
namespace {
constexpr Real pi = 3.141592653589793238462643383279502884;
std::array<Real,3> Shape(const M0CandidateSummary& s, Real theta) {
  std::array<Real,3> h{};
  for (std::size_t l=0;l<s.coefficients.size();++l) {
    const auto y=M0Harmonic(l,theta);
    for(int d=0;d<3;++d) h[d]+=s.coefficients[l]*y[d];
  }
  return h;
}
// Uniform-grid positivity with an analytic first-derivative bound between nodes.
// Floating-point evaluation still prevents interpreting this as a rigorous proof.
Real PositiveLowerBound(const std::vector<Real>& coefficients, int count) {
  Real derivative=0, minimum=std::numeric_limits<Real>::infinity();
  for(std::size_t l=1;l<coefficients.size();++l)
    derivative+=std::abs(coefficients[l])*std::sqrt((2*l+1)/(4*pi))*l*(l+1)/2;
  M0CandidateSummary s;s.coefficients=coefficients;
  for(int n=0;n<count;++n) minimum=std::min(minimum,Shape(s,pi*n/(count-1))[0]);
  return minimum-derivative*pi/(2*(count-1));
}
std::vector<Real> Difference(const M0CandidateSummary& a,const M0CandidateSummary& b) {
  std::vector<Real> c(std::max(a.coefficients.size(),b.coefficients.size()),0);
  for(std::size_t l=0;l<c.size();++l)
    c[l]=(l<a.coefficients.size()?a.coefficients[l]:0)-
         (l<b.coefficients.size()?b.coefficients[l]:0);
  return c;
}
bool Profile(const M0GeometrySampler& sample,const M0CandidateSummary& s,int count,
             std::vector<M0BracketPoint>& out) {
  std::vector<std::array<Real,2>> positions;
  std::vector<std::array<Real,3>> shapes;
  for(int n=0;n<count;++n) {
    const Real t=pi*n/(count-1);auto h=Shape(s,t);
    if(n==0 || n==count-1)h[1]=0;
    if(!(h[0]>0) || !std::isfinite(h[0])) return false;
    positions.push_back({(n==0 || n==count-1)?0:h[0]*std::sin(t),s.center_z+h[0]*std::cos(t)});
    shapes.push_back(h);
  }
  const auto geometry=sample(positions);
  if(geometry.size()!=positions.size()) return false;
  out.clear();
  for(int n=0;n<count;++n) {
    if(!geometry[n].valid) return false;
    const Real t=pi*n/(count-1);const auto& h=shapes[n];
    const auto p=EvaluateM0SurfacePoint(t,h[0],h[1],h[2],geometry[n]);
    if(!p.valid || !std::isfinite(p.ingoing_expansion)) return false;
    out.push_back({t,positions[n][0],positions[n][1],h[0],p.expansion,
                   p.ingoing_expansion,geometry[n].spacing});
  }
  return true;
}
// Proper lengths along the same-theta radial connectors, not minimum geodesic
// or closest-normal distances. Composite midpoint refinement estimates error.
bool Separation(const M0GeometrySampler& sample,const M0BracketResult& b,
                int angular,int radial,std::vector<std::array<Real,2>>& lengths) {
  std::vector<std::array<Real,2>> positions;
  std::vector<Real> increments;
  for(int n=0;n<angular;++n) {
    const Real t=pi*n/(angular-1);
    const Real r[3]={Shape(b.inner,t)[0],Shape(b.central,t)[0],Shape(b.outer,t)[0]};
    for(int k=0;k<2;++k) {
      const Real dr=(r[k+1]-r[k])/radial;
      if(!(dr>0)) return false;
      for(int j=0;j<radial;++j) {
        const Real rr=r[k]+(j+0.5)*dr;
        positions.push_back({(n==0 || n==angular-1)?0:rr*std::sin(t),b.central.center_z+rr*std::cos(t)});
        increments.push_back(dr);
      }
    }
  }
  const auto g=sample(positions);
  if(g.size()!=positions.size()) return false;
  lengths.assign(angular,{0,0});std::size_t i=0;
  for(int n=0;n<angular;++n) {
    const Real t=pi*n/(angular-1),st=std::sin(t),ct=std::cos(t);
    for(int k=0;k<2;++k) for(int j=0;j<radial;++j,++i) {
      const auto& m=g[i].metric;
      const Real norm=m[0]*st*st+2*m[2]*st*ct+m[5]*ct*ct;
      if(!g[i].valid || !(norm>0) || !std::isfinite(norm)) return false;
      lengths[n][k]+=increments[i]*std::sqrt(norm);
    }
  }
  return true;
}
} // namespace

M0BracketResult VerifyM0Bracket(const M0GeometrySampler& sample,
                                const M0BracketOptions& opt,const M0BracketResult& input) {
  auto b=input;b.supported=b.nested=b.signs_resolved=b.narrow=b.valid_geometry=false;
  b.stability_operator_verified=b.spatially_validated=false;
  if(opt.dense_points<133 || opt.separation_points<3 || opt.radial_points<2 ||
     !(opt.max_width_fraction>0) || !std::isfinite(opt.max_width_fraction))
    throw std::runtime_error("invalid CE verification controls");
  if(!(b.reference_radius>0) || !std::isfinite(b.reference_radius) || b.q!=0.05 ||
     b.central.expansion_target!=0 ||
     b.inner.expansion_target!=-b.q/b.reference_radius ||
     b.outer.expansion_target!=b.q/b.reference_radius) {
    b.failure="final_target_missing";return b;
  }
  if(b.inner.center_z!=b.central.center_z || b.outer.center_z!=b.central.center_z) {
    b.failure="different_centers";return b;
  }
  const auto inner_gap=Difference(b.central,b.inner),outer_gap=Difference(b.outer,b.central);
  for(int count=2*opt.dense_points+1;count<=65537;count=2*count-1) {
    b.geometry_points=count;
    b.radial_gap_inner=PositiveLowerBound(inner_gap,count);
    b.radial_gap_outer=PositiveLowerBound(outer_gap,count);
    b.nested=b.radial_gap_inner>0 && b.radial_gap_outer>0 &&
             PositiveLowerBound(b.inner.coefficients,count)>0;
    if(b.nested || count>=32769) break;
  }
  // A positive smooth radial graph is embedded and closed; strict radial
  // ordering at a shared center establishes geometric nesting, not area ordering.
  const auto coarse_inner=AssessM0Surface(sample,opt.dense_points,b.inner);
  const auto coarse_outer=AssessM0Surface(sample,opt.dense_points,b.outer);
  const int dense=2*opt.dense_points+1;
  b.inner=AssessM0Surface(sample,dense,b.inner);
  b.outer=AssessM0Surface(sample,dense,b.outer);
  // Assessments reset solve flags; retain the strict CE solve result separately.
  b.inner.ce_converged=input.inner.ce_converged;
  b.outer.ce_converged=input.outer.ce_converged;
  if(!(b.inner.area>0) || !(b.outer.area>0) || !(coarse_inner.area>0) || !(coarse_outer.area>0) ||
     !Profile(sample,b.inner,dense,b.profiles[0]) ||
     !Profile(sample,b.central,dense,b.profiles[1]) ||
     !Profile(sample,b.outer,dense,b.profiles[2])) {
    b.failure="unavailable_verification_geometry";return b;
  }
  b.valid_geometry=true;
  // Combine independent Gauss and uniform angular evaluations, including poles.
  auto extrema=[](const std::vector<M0BracketPoint>& p, Real& lo,Real& hi) {
    for(const auto& x:p) {lo=std::min(lo,x.outgoing);hi=std::max(hi,x.outgoing);}
  };
  extrema(b.profiles[0],b.inner.outgoing_min,b.inner.outgoing_max);
  extrema(b.profiles[2],b.outer.outgoing_min,b.outer.outgoing_max);
  b.angular_uncertainty_inner=std::max(std::abs(b.inner.outgoing_min-coarse_inner.outgoing_min),
                                     std::abs(b.inner.outgoing_max-coarse_inner.outgoing_max));
  b.angular_uncertainty_outer=std::max(std::abs(b.outer.outgoing_min-coarse_outer.outgoing_min),
                                     std::abs(b.outer.outgoing_max-coarse_outer.outgoing_max));
  auto include_profile=[](const std::vector<M0BracketPoint>& p,M0CandidateSummary& surface) {
    const Real ra=std::sqrt(surface.area/(4*pi));
    const Real scale=surface.reference_radius>0?surface.reference_radius:ra;
    for(const auto& x:p) {
      surface.outgoing_min=std::min(surface.outgoing_min,x.outgoing);
      surface.outgoing_max=std::max(surface.outgoing_max,x.outgoing);
      surface.ingoing_min=std::min(surface.ingoing_min,x.ingoing);
      surface.ingoing_max=std::max(surface.ingoing_max,x.ingoing);
      surface.epsilon_inf=std::max(surface.epsilon_inf,scale*std::abs(x.outgoing-surface.expansion_target));
      surface.physical_epsilon_inf=std::max(surface.physical_epsilon_inf,ra*std::abs(x.outgoing));
      surface.spacing=std::max(surface.spacing,x.spacing);
    }
  };
  include_profile(b.profiles[0],b.inner);include_profile(b.profiles[1],b.central);
  include_profile(b.profiles[2],b.outer);
  b.inner.ce_converged=b.inner.ce_converged &&
      b.inner.direct_residual<=b.inner.solve_epsilon2_tolerance &&
      b.inner.epsilon_inf<=b.inner.solve_epsilon_inf_tolerance;
  b.outer.ce_converged=b.outer.ce_converged &&
      b.outer.direct_residual<=b.outer.solve_epsilon2_tolerance &&
      b.outer.epsilon_inf<=b.outer.solve_epsilon_inf_tolerance;
  b.inner_margin=-b.inner.outgoing_max;b.outer_margin=b.outer.outgoing_min;
  const Real roundoff=128*std::numeric_limits<Real>::epsilon()/b.reference_radius;
  b.signs_resolved=b.inner_margin>2*b.angular_uncertainty_inner+roundoff &&
                   b.outer_margin>2*b.angular_uncertainty_outer+roundoff;
  if(!b.nested) {b.failure="nesting_unresolved";return b;}
  std::vector<std::array<Real,2>> lengths,coarse_lengths;
  if(!Separation(sample,b,opt.separation_points,opt.radial_points,coarse_lengths) ||
     !Separation(sample,b,opt.separation_points,2*opt.radial_points,lengths)) {
    b.failure="unavailable_separation_geometry";return b;
  }
  b.proper_inner_min=b.proper_outer_min=std::numeric_limits<Real>::infinity();
  b.proper_inner_max=b.proper_outer_max=b.proper_width_max=b.separation_quadrature_change=0;
  for(std::size_t n=0;n<lengths.size();++n) {
    b.proper_inner_min=std::min(b.proper_inner_min,lengths[n][0]);
    b.proper_outer_min=std::min(b.proper_outer_min,lengths[n][1]);
    b.proper_inner_max=std::max(b.proper_inner_max,lengths[n][0]);
    b.proper_outer_max=std::max(b.proper_outer_max,lengths[n][1]);
    b.proper_width_max=std::max(b.proper_width_max,lengths[n][0]+lengths[n][1]);
    for(int k=0;k<2;++k) b.separation_quadrature_change=std::max(b.separation_quadrature_change,
                                                    std::abs(lengths[n][k]-coarse_lengths[n][k]));
  }
  b.narrow=(b.proper_width_max+2*b.separation_quadrature_change)/b.reference_radius<=opt.max_width_fraction;
  // Sign/nesting evidence and localization width answer different questions.
  // A broad sandwich is still reported, with an explicit localization warning.
  b.supported=b.signs_resolved;
  b.failure=!b.signs_resolved?"expansion_signs_unresolved":"none";
  return b;
}

M0BracketResult FindM0Bracket(const M0GeometrySampler& sample,const M0SolveOptions& solver,
                              const M0BracketOptions& opt,const M0CandidateSummary& central) {
  M0BracketResult b;
  if(opt.continuation_steps<1 || solver.lmax>64 || central.expansion_target!=0)
    throw std::runtime_error("invalid CE continuation configuration");
  b.central=AssessM0Surface(sample,opt.dense_points,central);
  b.central.verified=central.verified;b.central.angular_candidate=central.angular_candidate;
  b.central.converged=central.converged;
  if(!(b.central.area>0)) {b.failure="invalid_central_geometry";return b;}
  b.reference_radius=std::sqrt(b.central.area/(4*pi));
  for(int sign:{-1,1}) {
    auto previous=b.central;
    for(int step=1;step<=opt.continuation_steps;++step) {
      auto ce=solver;ce.candidate_policy=ce.candidate_l32_window=false;
      ce.reference_radius=b.reference_radius;
      ce.expansion_target=(step==opt.continuation_steps?sign*b.q:
                          sign*b.q*step/opt.continuation_steps)/b.reference_radius;
      previous=SolveM0Refined(sample,ce,std::min(8,ce.lmax),sign<0?"ce_inner":"ce_outer",
                              central.center_z,central.mean_radius,previous.coefficients);
      b.continuation.push_back(previous);
      if(!(previous.area>0) || !std::isfinite(previous.direct_residual)) break;
    }
    if(sign<0)b.inner=previous;else b.outer=previous;
  }
  return VerifyM0Bracket(sample,opt,b);
}
namespace {
void Number(std::ostream& out,Real x) {
  if(std::isfinite(x))out<<x;else out<<"null";
}
void SurfaceJson(std::ostream& out,const M0CandidateSummary& s) {
  out<<"{\"target_c\":";Number(out,s.expansion_target);
  out<<",\"reference_radius\":";Number(out,s.reference_radius);
  out<<",\"target_q\":";Number(out,s.expansion_target*s.reference_radius);
  out<<",\"ce_converged\":"<<(s.ce_converged?"true":"false")
     <<",\"ordinary_candidate\":"<<((s.verified||s.angular_candidate)?"true":"false");
  const std::pair<const char*,Real> values[]={
    {"target_error_rms",s.direct_residual},{"target_error_max",s.epsilon_inf},
    {"physical_epsilon2",s.physical_epsilon2},{"physical_epsilon_inf",s.physical_epsilon_inf},
    {"outgoing_min",s.outgoing_min},{"outgoing_max",s.outgoing_max},
    {"ingoing_min",s.ingoing_min},{"ingoing_max",s.ingoing_max},
    {"area",s.area},{"area_radius",std::sqrt(s.area/(4*pi))},
    {"center_z",s.center_z},{"minimum_radius",s.minimum_radius},{"max_spacing",s.spacing},
    {"solve_epsilon2_tolerance",s.solve_epsilon2_tolerance},
    {"solve_epsilon_inf_tolerance",s.solve_epsilon_inf_tolerance}};
  for(auto x:values){out<<",\""<<x.first<<"\":";Number(out,x.second);}
  out<<",\"failure\":\""<<s.failure<<"\",\"coefficients\":[";
  for(std::size_t l=0;l<s.coefficients.size();++l) {if(l)out<<',';Number(out,s.coefficients[l]);}
  out<<"]}";
}
} // namespace
void WriteM0Bracket(const std::string& basename,int cycle,Real time,int candidate,
                    const M0BracketResult& b) {
  std::ofstream out(basename+".ce_brackets.jsonl",std::ios::app);
  out<<std::setprecision(17)<<"{\"cycle\":"<<cycle<<",\"time\":"<<time<<",\"candidate\":"<<candidate
     <<",\"supported\":"<<(b.supported?"true":"false")
     <<",\"nested\":"<<(b.nested?"true":"false")
     <<",\"signs_resolved\":"<<(b.signs_resolved?"true":"false")
     <<",\"narrow\":"<<(b.narrow?"true":"false")
     <<",\"localization_warning\":"<<((b.supported&&!b.narrow)?"true":"false")
     <<",\"valid_geometry\":"<<(b.valid_geometry?"true":"false")
     <<",\"stability_operator_verified\":false,\"spatially_validated\":false"
     <<",\"uncertainty_kind\":\"angular_estimate_not_rigorous_or_spatial_bound\""
     <<",\"separation_kind\":\"proper_length_along_radial_connectors\""
     <<",\"failure\":\""<<b.failure<<"\",\"geometry_points\":"<<b.geometry_points;
  const std::pair<const char*,Real> values[]={
    {"q",b.q},{"reference_radius",b.reference_radius},
    {"inner_sign_margin",b.inner_margin},{"outer_sign_margin",b.outer_margin},
    {"angular_uncertainty_inner",b.angular_uncertainty_inner},
    {"angular_uncertainty_outer",b.angular_uncertainty_outer},
    {"radial_gap_inner_lower_bound",b.radial_gap_inner},
    {"radial_gap_outer_lower_bound",b.radial_gap_outer},
    {"proper_inner_min",b.proper_inner_min},{"proper_inner_max",b.proper_inner_max},
    {"proper_outer_min",b.proper_outer_min},{"proper_outer_max",b.proper_outer_max},
    {"proper_width_max",b.proper_width_max},
    {"proper_width_fraction",b.proper_width_max/b.reference_radius},
    {"separation_quadrature_change",b.separation_quadrature_change}};
  for(auto x:values){out<<",\""<<x.first<<"\":";Number(out,x.second);}
  out<<",\"central\":";SurfaceJson(out,b.central);
  out<<",\"inner\":";SurfaceJson(out,b.inner);
  out<<",\"outer\":";SurfaceJson(out,b.outer);
  out<<",\"continuation\":[";
  for(std::size_t i=0;i<b.continuation.size();++i) {if(i)out<<',';SurfaceJson(out,b.continuation[i]);}
  out<<"]}\n";
  const char* labels[]={"inner","central","outer"};
  const M0CandidateSummary* surfaces[]={&b.inner,&b.central,&b.outer};
  for(int k=0;k<3;++k) {
    std::ofstream profile(basename+".ce_surface_"+std::to_string(cycle)+"_"+
                          std::to_string(candidate)+"_"+labels[k]+".csv");
    profile<<"theta,rho,z,radius,theta_plus,theta_minus,target_error,spacing\n"<<std::setprecision(17);
    for(const auto& p:b.profiles[k])
      profile<<p.theta<<','<<p.rho<<','<<p.z<<','<<p.radius<<','<<p.outgoing<<','<<p.ingoing<<','
             <<p.outgoing-surfaces[k]->expansion_target<<','<<p.spacing<<'\n';
  }
}
} // namespace z4c
