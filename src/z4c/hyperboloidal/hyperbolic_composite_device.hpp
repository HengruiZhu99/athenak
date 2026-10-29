// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_DEVICE_HPP_

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"

namespace z4c::hyperboloidal {

// Native point equations and the CPU prototype's mapped finite differences on
// an owned, contiguous active-cell range. No spherical reduction is performed.
// All views/plans are persistent. The caller supplies the completed stage's
// component-major padded field, including its scri-dependent D ghost values.
// Only owned cells are evaluated; the point-major result can be passed directly
// to a GPU-aware MPI gather or explicitly copied with DownloadRHS().
class HyperbolicCompositeDevice {
 public:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using Buffer = Kokkos::View<Real *, MemorySpace>;
  using ConstBuffer = Kokkos::View<const Real *, MemorySpace>;
  using IntBuffer = Kokkos::View<int *, MemorySpace>;
  static constexpr int nf = 31;

  enum Status : int {
    success = 0, invalid_input = 1, invalid_geometry = 2,
    invalid_assembly = 3, invalid_gauge = 4, invalid_auxiliary = 5,
    cfl_exceeded = 6, nonfinite_rhs = 7
  };
  struct Summary {
    Real min_gap = std::numeric_limits<Real>::max();
    Real max_cfl = 0;
    int first_bad = -1;  // Global active index, not local or padded index.
    int reason = success;
    int bad_cells = 0;
  };

  // These are weighted sums, not RMS values. Divide each by the corresponding
  // coordinate volume and take its square root after the MPI sum reduction.
  // m2/z2 match the existing Penrose-conformal invariant diagnostics; physical
  // norms are provided separately so their Omega factors are never implicit.
  struct DiagnosticSums {
    Real volume = 0, h2 = 0, mr2 = 0, m2 = 0, m2_physical = 0;
    Real theta2 = 0, zr2 = 0, z2 = 0, z2_physical = 0;
  };
  struct DiagnosticSummary {
    DiagnosticSums total{}, exterior{};
    Real min_chi = std::numeric_limits<Real>::max();
    Real min_lapse = std::numeric_limits<Real>::max();
    Real max_determinant_residual = 0, max_tracefree_residual = 0;
    Real max_state_deviation = 0;
    int invalid_cells = 0, first_bad = -1, reason = success;
  };

 private:
  // The optional sampled-normal control reuses the exact bulk jet arithmetic
  // on its surface-owned donor points; the default bulk path is unchanged.
  friend class HyperbolicSampledNormalDevice;
  struct KOTerm {
    int point[5];
    int target_coefficient;
    Real weight;
  };
  using GeometryBuffer = Kokkos::View<cubed_sphere::Point *, MemorySpace>;
  using KOTermBuffer = Kokkos::View<KOTerm *, MemorySpace>;
  using HostBuffer = Kokkos::View<Real *, Kokkos::HostSpace>;

  int first_, local_, n_, width_, padded_count_;
  IntBuffer active_, ko_offsets_, status_;
  GeometryBuffer geometry_;
  Buffer volumes_, rhs_, gap_, cfl_, uploaded_padded_;
  KOTermBuffer ko_terms_;
  HostBuffer host_rhs_, host_padded_;

  struct PointKernel {
    ConstBuffer padded;
    IntBuffer active, offsets, status;
    GeometryBuffer geometry;
    KOTermBuffer terms;
    Buffer volume, rhs, gap, cfl;
    int n, width, padded_count;
    Real dt, ko;
    bool adaptive;

    KOKKOS_INLINE_FUNCTION int Offset(int axis, int amount) const {
      return amount*(axis == 0 ? 1 : (axis == 1 ? width : width*width));
    }
    KOKKOS_INLINE_FUNCTION Real First(int point, int axis) const {
      return n*(padded(point+Offset(axis,-2))-8*padded(point+Offset(axis,-1))
          +8*padded(point+Offset(axis,1))-padded(point+Offset(axis,2)))/12;
    }
    KOKKOS_INLINE_FUNCTION Real Second(int point, int a, int b) const {
      if (a == b) return n*n*(-padded(point+Offset(a,-2))
          +16*padded(point+Offset(a,-1))-30*padded(point)
          +16*padded(point+Offset(a,1))-padded(point+Offset(a,2)))/12;
      Real value = 0;
      for (int i=-2;i<=2;++i) if (i)
      for (int j=-2;j<=2;++j) if (j) {
        const Real wi = (i==1 || i==-1) ? Real(2*i)/3 : Real(-i)/24;
        const Real wj = (j==1 || j==-1) ? Real(2*j)/3 : Real(-j)/24;
        value += wi*wj*padded(point+Offset(a,i)+Offset(b,j));
      }
      return value*n*n;
    }
    // Return both the Cartesian jet and the correction from centered to biased
    // transport. Analytic reference alpha/beta derivatives are added separately.
    KOKKOS_INLINE_FUNCTION ScalarJet<Real> Sample(int local, int f,
        const cubed_sphere::Point &map, const Real beta[3], bool need_second,
        Real &transport_correction) const {
      ScalarJet<Real> result{};
      const int point = f*padded_count+active(local);
      result.value = padded(point);
      Real first[3]{}, second[3][3]{};
      transport_correction = 0;
      for (int a=0;a<3;++a) {
        first[a] = First(point,a);
        if (need_second) for (int b=0;b<3;++b) second[a][b] = Second(point,a,b);
        Real b = 0;
        for (int i=0;i<3;++i) b += map.J[a][i]*beta[i];
        const int sign = b>=0 ? 1 : -1;
        const Real vminus = padded(point+Offset(a,-sign))-result.value;
        const Real v1 = padded(point+Offset(a,sign))-result.value;
        const Real v2 = padded(point+Offset(a,2*sign))-result.value;
        const Real v3 = padded(point+Offset(a,3*sign))-result.value;
        const Real biased = sign*n*(-3*vminus+18*v1-6*v2+v3)/12;
        transport_correction += b*(biased-first[a]);
      }
      for (int a=0;a<3;++a) for (int A=0;A<3;++A) {
        result.d[a] += map.J[A][a]*first[A];
        if (need_second) for (int b=0;b<3;++b) {
          result.dd[a][b] += map.H[A][a][b]*first[A];
          for (int B=0;B<3;++B)
            result.dd[a][b] += map.J[A][a]*map.J[B][b]*second[A][B];
        }
      }
      return result;
    }
    KOKKOS_INLINE_FUNCTION Real Dissipation(int local,int f) const {
      if (ko==0) return 0;
      constexpr int b[5] = {1,-4,6,-4,1};
      Real result = 0;
      const int base = f*padded_count;
      for (int i=offsets(local);i<offsets(local+1);++i) {
        const auto &term=terms(i);
        const Real anchor=padded(base+term.point[0]);
        Real fourth=0;
        for (int a=0;a<5;++a)
          fourth += b[a]*(padded(base+term.point[a])-anchor);
        result -= ko*n*term.target_coefficient*term.weight*fourth/(256*volume(local));
      }
      return result;
    }
    KOKKOS_INLINE_FUNCTION int BuildJet(int local,const cubed_sphere::Point &map,
        Real reference_alpha,Z4cJet<Real> &u,VectorJet<Real> &B,VectorJet<Real> &D,
        Real correction[nf]) const {
      const Real *x=map.x;
      Real beta[3]{};
      for (int i=0;i<3;++i) beta[i]=padded((22+i)*padded_count+active(local))-x[i];
      for (int f=0;f<nf;++f) if (!Kokkos::isfinite(padded(f*padded_count+active(local)))) {
        return invalid_input;
      }
      if (!(padded(active(local))>0 && padded(18*padded_count+active(local))+reference_alpha>0)) {
        return invalid_input;
      }
      u.chi=Sample(local,0,map,beta,true,correction[0]);
      u.trace=Sample(local,13,map,beta,false,correction[13]);u.trace.value-=3;
      u.theta=Sample(local,14,map,beta,false,correction[14]);
      u.alpha=Sample(local,18,map,beta,true,correction[18]);u.alpha.value+=reference_alpha;
      for (int i=0;i<3;++i) {u.alpha.d[i]+=x[i];u.alpha.dd[i][i]+=1;}
      constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for (int t=0;t<6;++t) {
        const int i=ti[t],j=tj[t];
        const auto metric=Sample(local,1+t,map,beta,true,correction[1+t]);
        const auto curvature=Sample(local,7+t,map,beta,false,correction[7+t]);
        u.metric.g[i][j]=u.metric.g[j][i]=metric.value+(i==j);
        u.a.k[i][j]=u.a.k[j][i]=curvature.value;
        for (int a=0;a<3;++a) {
          u.metric.dg[a][i][j]=u.metric.dg[a][j][i]=metric.d[a];
          u.a.dk[a][i][j]=u.a.dk[a][j][i]=curvature.d[a];
          for (int b=0;b<3;++b)
            u.metric.ddg[a][b][i][j]=u.metric.ddg[a][b][j][i]=metric.dd[a][b];
        }
      }
      for (int i=0;i<3;++i) {
        const auto shift=Sample(local,22+i,map,beta,true,correction[22+i]);
        const auto connection=Sample(local,15+i,map,beta,false,correction[15+i]);
        const auto flux=Sample(local,19+i,map,beta,false,correction[19+i]);
        const auto auxiliary=Sample(local,25+i,map,beta,false,correction[25+i]);
        u.beta.value[i]=shift.value-x[i];u.lambda.value[i]=connection.value;
        B.value[i]=flux.value;D.value[i]=auxiliary.value;
        for (int a=0;a<3;++a) {
          u.beta.d[a][i]=shift.d[a]-(a==i);u.lambda.d[a][i]=connection.d[a];
          B.d[a][i]=flux.d[a];D.d[a][i]=auxiliary.d[a];
          for (int b=0;b<3;++b) u.beta.dd[a][b][i]=shift.dd[a][b];
        }
      }
      return success;
    }
    KOKKOS_INLINE_FUNCTION void operator()(int local) const {
      status(local)=success;gap(local)=std::numeric_limits<Real>::max();cfl(local)=0;
      for (int f=0;f<nf;++f) rhs(nf*local+f)=0;
      const auto map=geometry(local);
      const Real *x=map.x;
      const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const Real reference_alpha=(1+r*r)/2;
      Z4cJet<Real> u{};VectorJet<Real> B{},D{};
      Real correction[nf]{},rate[nf]{};
      const int input_status=BuildJet(local,map,reference_alpha,u,B,D,correction);
      if (input_status!=success) {status(local)=input_status;return;}
      const auto g=Geometry(u.metric);
      if (!g.valid) {status(local)=invalid_geometry;return;}
      OmegaJet<Real> omega{};omega.omega=(1-r*r)/2;
      for (int i=0;i<3;++i) {omega.gradient[i]=-x[i];omega.hessian[i][i]=-1;}
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      auto parts=ConformalRHS(u,omega,Real(4),Real(-.99));
      if (!HyperbolicDamping(x,u,omega,parts)) {status(local)=invalid_geometry;return;}
      Z4cRHS<Real> geometry_rate{};
      if (!AssembleInterior(parts,omega.omega,geometry_rate)) {status(local)=invalid_assembly;return;}
      rate[0]=geometry_rate.chi;rate[13]=geometry_rate.trace;rate[14]=geometry_rate.theta;
      constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for (int t=0;t<6;++t) {
        rate[1+t]=geometry_rate.metric[ti[t]][tj[t]];
        rate[7+t]=geometry_rate.a[ti[t]][tj[t]];
      }
      for (int i=0;i<3;++i) rate[15+i]=geometry_rate.lambda[i];
      const auto gauge=HyperbolicGauge(x,u,B,D.value);
      if (!gauge.valid) {status(local)=invalid_gauge;return;}
      rate[18]=gauge.lapse_regular+gauge.lapse_pole/omega.omega;
      const auto coefficient=ShiftCoefficients(r,u.alpha.value,u.chi.value);
      const Real weight=adaptive ? 1-coefficient.collar : 0;
      Real driver[3]{};
      if (!HyperbolicAuxiliaryRate(u,D,geometry_rate.lambda,driver)) {
        status(local)=invalid_auxiliary;return;
      }
      for (int i=0;i<3;++i) {
        const Real source=padded((28+i)*padded_count+active(local));
        rate[19+i]=gauge.flux_rate[i];rate[25+i]=driver[i];
        rate[22+i]=gauge.shift_regular[i]+gauge.shift_pole[i]/omega.omega-weight*source;
        Real target=-x[i]-source;
        for (int j=0;j<3;++j) target+=u.beta.value[j]*u.beta.d[j][i];
        rate[28+i]=.1*weight*target;
      }
      for (int f=0;f<28;++f) rate[f]+=correction[f];
      if (adaptive) for (int i=0;i<3;++i) rate[28+i]+=.1*weight*correction[22+i];
      // Lambda's upwind correction cancels in D_t=Lambda_t-beta.grad Lambda
      // + beta.grad D-eta D. Its dissipation does not cancel and is added below.
      Real diss[nf]{};
      for (int f=0;f<nf;++f) {diss[f]=Dissipation(local,f);rate[f]+=diss[f];}
      for (int i=0;i<3;++i) rate[25+i]+=diss[15+i];
      gap(local)=AllDirectionShiftLapseGap(u.metric.g,u.chi.value,coefficient);
      Real speed=0;
      for (int A=0;A<3;++A) {
        Real transport=0,Q=0,E=0;
        for (int i=0;i<3;++i) {
          transport+=map.J[A][i]*u.beta.value[i];E+=map.J[A][i]*map.J[A][i];
          for (int j=0;j<3;++j) Q+=g.inverse[i][j]*map.J[A][i]*map.J[A][j];
        }
        speed+=fabs(transport)+sqrt(fmax(4*coefficient.mu*Q/3,
            u.chi.value*(coefficient.F*Q+coefficient.eta*.25*coefficient.w*coefficient.w*E)));
      }
      cfl(local)=dt*n*speed;
      if (cfl(local)>.4 || dt/.02>.25) status(local)=cfl_exceeded;
      for (int f=0;f<nf;++f) {
        rhs(nf*local+f)=rate[f];
        if (!Kokkos::isfinite(rate[f])) status(local)=nonfinite_rhs;
      }
    }
  };

  struct Summarize {
    using value_type=Summary;
    Buffer gap,cfl;
    IntBuffer status;
    int first;
    KOKKOS_INLINE_FUNCTION void init(Summary &result) const {
      result.min_gap=std::numeric_limits<Real>::max();result.max_cfl=0;
      result.first_bad=INT_MAX;result.reason=success;result.bad_cells=0;
    }
    KOKKOS_INLINE_FUNCTION void join(Summary &a,const Summary &b) const {
      if (b.min_gap<a.min_gap) a.min_gap=b.min_gap;
      if (b.max_cfl>a.max_cfl) a.max_cfl=b.max_cfl;
      if (b.first_bad<a.first_bad) {a.first_bad=b.first_bad;a.reason=b.reason;}
      a.bad_cells+=b.bad_cells;
    }
    KOKKOS_INLINE_FUNCTION void operator()(int local,Summary &result) const {
      if (gap(local)<result.min_gap) result.min_gap=gap(local);
      if (cfl(local)>result.max_cfl) result.max_cfl=cfl(local);
      if (status(local)!=success) {
        ++result.bad_cells;
        if (first+local<result.first_bad) {
          result.first_bad=first+local;result.reason=status(local);
        }
      }
    }
  };

  struct Diagnose {
    using value_type=DiagnosticSummary;
    PointKernel point;
    int first;
    Real exterior_radius;
    KOKKOS_INLINE_FUNCTION void init(DiagnosticSummary &result) const {
      result.total=DiagnosticSums{};result.exterior=DiagnosticSums{};
      result.min_chi=std::numeric_limits<Real>::max();result.min_lapse=result.min_chi;
      result.max_determinant_residual=0;result.max_tracefree_residual=0;
      result.max_state_deviation=0;result.invalid_cells=0;
      result.first_bad=INT_MAX;result.reason=success;
    }
    KOKKOS_INLINE_FUNCTION static void Add(DiagnosticSums &a,const DiagnosticSums &b) {
      a.volume+=b.volume;a.h2+=b.h2;a.mr2+=b.mr2;a.m2+=b.m2;
      a.m2_physical+=b.m2_physical;a.theta2+=b.theta2;a.zr2+=b.zr2;
      a.z2+=b.z2;a.z2_physical+=b.z2_physical;
    }
    KOKKOS_INLINE_FUNCTION void join(DiagnosticSummary &a,const DiagnosticSummary &b) const {
      Add(a.total,b.total);Add(a.exterior,b.exterior);
      if (b.min_chi<a.min_chi) a.min_chi=b.min_chi;
      if (b.min_lapse<a.min_lapse) a.min_lapse=b.min_lapse;
      if (b.max_determinant_residual>a.max_determinant_residual)
        a.max_determinant_residual=b.max_determinant_residual;
      if (b.max_tracefree_residual>a.max_tracefree_residual)
        a.max_tracefree_residual=b.max_tracefree_residual;
      if (b.max_state_deviation>a.max_state_deviation) a.max_state_deviation=b.max_state_deviation;
      a.invalid_cells+=b.invalid_cells;
      if (b.first_bad<a.first_bad) {a.first_bad=b.first_bad;a.reason=b.reason;}
    }
    KOKKOS_INLINE_FUNCTION void Fail(int local,int reason,DiagnosticSummary &result) const {
      ++result.invalid_cells;
      if (first+local<result.first_bad) {result.first_bad=first+local;result.reason=reason;}
    }
    KOKKOS_INLINE_FUNCTION void operator()(int local,DiagnosticSummary &result) const {
      const auto map=point.geometry(local);const Real *x=map.x;
      const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const Real reference_alpha=(1+r*r)/2,weight=point.volume(local);
      // Volumes include all owned cells. Invalid samples are counted explicitly,
      // never silently discarded from a reported RMS denominator.
      result.total.volume+=weight;
      if (r>exterior_radius) result.exterior.volume+=weight;
      const int active=point.active(local);
      const Real chi=point.padded(active);
      const Real lapse=point.padded(18*point.padded_count+active)+reference_alpha;
      if (chi<result.min_chi) result.min_chi=chi;
      if (lapse<result.min_lapse) result.min_lapse=lapse;
      for (int f=0;f<nf;++f) {
        const Real value=point.padded(f*point.padded_count+active)-(f==0 ? 1 : 0);
        if (!Kokkos::isfinite(value)) result.max_state_deviation=std::numeric_limits<Real>::max();
        else if (fabs(value)>result.max_state_deviation) result.max_state_deviation=fabs(value);
      }
      Z4cJet<Real> u{};VectorJet<Real> B{},D{};Real correction[nf]{};
      const int status=point.BuildJet(local,map,reference_alpha,u,B,D,correction);
      if (status!=success) {Fail(local,status,result);return;}
      OmegaJet<Real> omega{};omega.omega=(1-r*r)/2;
      for (int i=0;i<3;++i) {omega.gradient[i]=-x[i];omega.hessian[i][i]=-1;}
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      const auto constraints=EvolvedConstraints(u,omega);
      if (!constraints.valid) {Fail(local,invalid_geometry,result);return;}
      Real mr=0,zr=0;
      if (r>0) for (int i=0;i<3;++i) {
        mr+=x[i]*constraints.momentum[i]/r;
        zr+=x[i]*constraints.z4.z_covector[i]/r;
      }
      DiagnosticSums sums{};
      sums.h2=weight*constraints.hamiltonian*constraints.hamiltonian;
      sums.mr2=weight*mr*mr;sums.m2=weight*constraints.momentum_conformal_norm2;
      sums.m2_physical=weight*constraints.momentum_physical_norm2;
      sums.theta2=weight*u.theta.value*u.theta.value;sums.zr2=weight*zr*zr;
      sums.z2=weight*constraints.z4.z_conformal_norm2;
      sums.z2_physical=weight*constraints.z4.z_physical_norm2;
      Add(result.total,sums);if (r>exterior_radius) Add(result.exterior,sums);
      if (fabs(constraints.z4.determinant_residual)>result.max_determinant_residual)
        result.max_determinant_residual=fabs(constraints.z4.determinant_residual);
      if (fabs(constraints.z4.tracefree_residual)>result.max_tracefree_residual)
        result.max_tracefree_residual=fabs(constraints.z4.tracefree_residual);
    }
  };

 public:
  HyperbolicCompositeDevice(const cubed_sphere::CompositeDifference &grid,
                            int first_active,int local_count)
      :first_(first_active),local_(local_count),n_(grid.n),width_(grid.padded),
       padded_count_(grid.count),active_("owned active indices",local_count),
       ko_offsets_("owned KO offsets",local_count+1),status_("owned RHS status",local_count),
       geometry_("owned mapped geometry",local_count),volumes_("owned volumes",local_count),
       rhs_("owned hyperboloidal RHS",size_t(nf)*local_count),
       gap_("owned characteristic gap",local_count),cfl_("owned CFL",local_count),
       host_rhs_("host owned RHS",size_t(nf)*local_count) {
    if (grid.order!=4 || grid.ng<3 || first_active<0 || local_count<0
        || size_t(first_active)+size_t(local_count)>grid.active.size())
      throw std::invalid_argument("device bulk requires a valid owned range and fourth-order/three-halo grid");
    auto indices=Kokkos::create_mirror_view(active_);
    auto maps=Kokkos::create_mirror_view(geometry_);
    auto volumes=Kokkos::create_mirror_view(volumes_);
    auto offsets=Kokkos::create_mirror_view(ko_offsets_);
    std::vector<KOTerm> terms;
    constexpr int b[5]={1,-4,6,-4,1};
    struct Candidate {int start,axis,target;};
    for (int local=0;local<local_;++local) {
      const int p=first_+local;
      indices(local)=grid.active[p];maps(local)=grid.geometry[p];volumes(local)=grid.volumes[p];
      offsets(local)=terms.size();
      const int block_point=p%(n_*n_*n_);
      const int position[3]={block_point%n_,(block_point/n_)%n_,block_point/(n_*n_)};
      std::vector<Candidate> candidates;
      for (int A=0;A<3;++A) for (int a=0;a<5;++a) {
        if (position[A]-a<0 || position[A]-a+4>=n_) continue;
        const int stride=A==0?1:(A==1?n_:n_*n_);
        candidates.push_back({p-a*stride,A,a});
      }
      // Match the CPU scatter's global-start, then axis, accumulation order.
      std::sort(candidates.begin(),candidates.end(),[](const Candidate&a,const Candidate&b){
        return a.start<b.start || (a.start==b.start && a.axis<b.axis);
      });
      for (const auto &candidate:candidates) {
        const int stride=candidate.axis==0?1:(candidate.axis==1?n_:n_*n_);
        KOTerm term{};term.target_coefficient=b[candidate.target];
        for (int a=0;a<5;++a) {
          const int donor=candidate.start+a*stride;
          term.point[a]=grid.active[donor];
          const auto &map=grid.geometry[donor];Real inverse_length=0,r2=0;
          for (int d=0;d<3;++d) {
            inverse_length+=map.J[candidate.axis][d]*map.J[candidate.axis][d];
            r2+=map.x[d]*map.x[d];
          }
          // Preserve the prototype's sqrt(r2), then r*r expression exactly.
          const Real r=std::sqrt(r2),taper=std::pow(std::max(Real(0),1-r*r/.64),6);
          term.weight+=grid.volumes[donor]*std::sqrt(inverse_length)*taper/5;
        }
        terms.push_back(term);
      }
    }
    offsets(local_)=terms.size();ko_terms_=KOTermBuffer("owned KO terms",terms.size());
    auto host_terms=Kokkos::create_mirror_view(ko_terms_);
    for (size_t i=0;i<terms.size();++i) host_terms(i)=terms[i];
    Kokkos::deep_copy(active_,indices);Kokkos::deep_copy(geometry_,maps);
    Kokkos::deep_copy(volumes_,volumes);Kokkos::deep_copy(ko_offsets_,offsets);
    Kokkos::deep_copy(ko_terms_,host_terms);
  }

  Summary Evaluate(ConstBuffer padded,Real dt,bool adaptive=false,Real ko=.5) {
    if (padded.extent(0)!=size_t(nf)*padded_count_ || !(dt>0) || !std::isfinite(dt)
        || !(ko>=0) || !std::isfinite(ko))
      throw std::invalid_argument("invalid device RHS storage/timestep/dissipation");
    const PointKernel kernel{padded,active_,ko_offsets_,status_,geometry_,ko_terms_,
        volumes_,rhs_,gap_,cfl_,n_,width_,padded_count_,dt,ko,adaptive};
    Kokkos::parallel_for("hyperboloidal owned bulk RHS",
        Kokkos::RangePolicy<ExecutionSpace>(0,local_),kernel);
    Summary result;
    Kokkos::parallel_reduce("hyperboloidal owned bulk diagnostics",
        Kokkos::RangePolicy<ExecutionSpace>(0,local_),Summarize{gap_,cfl_,status_,first_},result);
    if (result.first_bad==INT_MAX) result.first_bad=-1;
    return result;
  }
  DiagnosticSummary Diagnostics(ConstBuffer padded,Real exterior_radius=0) {
    if (padded.extent(0)!=size_t(nf)*padded_count_ || !(exterior_radius>=0)
        || !std::isfinite(exterior_radius))
      throw std::invalid_argument("invalid device diagnostic storage/exterior cutoff");
    const PointKernel kernel{padded,active_,ko_offsets_,status_,geometry_,ko_terms_,
        volumes_,rhs_,gap_,cfl_,n_,width_,padded_count_,0,0,false};
    DiagnosticSummary result;
    Kokkos::parallel_reduce("hyperboloidal owned constraints",
        Kokkos::RangePolicy<ExecutionSpace>(0,local_),Diagnose{kernel,first_,exterior_radius},result);
    if (result.first_bad==INT_MAX) result.first_bad=-1;
    return result;
  }
  const Buffer &UploadPadded(const std::vector<Real>&values) {
    if (values.size()!=size_t(nf)*padded_count_)
      throw std::invalid_argument("invalid host padded storage");
    if (uploaded_padded_.extent(0)!=values.size()) {
      uploaded_padded_=Buffer("uploaded stage padding",values.size());
      host_padded_=HostBuffer("host stage padding",values.size());
    }
    std::copy(values.begin(),values.end(),host_padded_.data());
    Kokkos::deep_copy(uploaded_padded_,host_padded_);return uploaded_padded_;
  }
  std::vector<Real> DownloadRHS() {
    Kokkos::deep_copy(host_rhs_,rhs_);
    return std::vector<Real>(host_rhs_.data(),host_rhs_.data()+host_rhs_.extent(0));
  }
  const Buffer &LocalRHS() const {return rhs_;}
  const IntBuffer &CellStatus() const {return status_;}
  const Buffer &CellGap() const {return gap_;}
  const Buffer &CellCFL() const {return cfl_;}
  int FirstActive() const {return first_;}
  int LocalCount() const {return local_;}
  static const char *Reason(int code) {
    switch(code) {
      case success:return "success";
      case invalid_input:return "nonfinite or nonpositive input";
      case invalid_geometry:return "invalid native geometry/source";
      case invalid_assembly:return "invalid interior assembly";
      case invalid_gauge:return "invalid hyperbolic gauge";
      case invalid_auxiliary:return "invalid independent auxiliary rate";
      case cfl_exceeded:return "explicit CFL bound exceeded";
      case nonfinite_rhs:return "nonfinite bulk RHS";
      default:return "unknown device status";
    }
  }
};
}  // namespace z4c::hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_DEVICE_HPP_
