// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"  // NOLINT(build/include_subdir): root AthenaK types header
#include "utils/finite_diff.hpp"
#include "z4c/hyperboloidal/spherical_ghosts.hpp"
#include "z4c/hyperboloidal/interior_dissipation.hpp"

namespace hyp = z4c::hyperboloidal;
using View = Kokkos::View<double *>;
using Plans = Kokkos::View<hyp::SphericalGhostStencil *>;

struct FlatScalar {
  View q;
  int nx, ny, offset = 0;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int k, int j, int i) const { return q(offset+i+nx*(j+ny*k)); }
};
struct RadialShift {
  double origin, h;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int a, int k, int j, int i) const {
    const int index[3] = {i,j,k};
    return -(origin+index[a]*h);
  }
};

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

hyp::SphericalGhostGrid Grid(int n, bool anisotropic = false) {
  hyp::SphericalGhostGrid g{};
  g.radius = 1;
  for (int a = 0; a < 3; ++a) {
    g.h[a] = 2.*(anisotropic ? 0.8+0.2*a : 1)/n;
    g.n[a] = static_cast<int>(std::ceil(2/g.h[a]))+8;
    g.first[a] = -1-3.5*g.h[a];
  }
  return g;
}

Plans Upload(const std::vector<hyp::SphericalGhostStencil> &stencils) {
  Plans plans("ghost plans", stencils.size());
  auto host = Kokkos::create_mirror_view(plans);
  for (size_t p = 0; p < stencils.size(); ++p) host(p) = stencils[p];
  Kokkos::deep_copy(plans, host);
  return plans;
}


// Conformally invariant scalar wave on the CMC Minkowski reference, S=a=1.
// Pi=n_bar(phi), phi_t=beta.grad(phi)+alpha Pi,
// Pi_t=beta.grad(Pi)+alpha Lap(phi)+grad(alpha).grad(phi)-3 Pi-alpha R phi/6.
// alpha=(1+r^2)/2, beta=-x, R=6/alpha^3-6/alpha.
// Exact nonspherical solution: physical directional derivative of a regular spherical
// Minkowski wave, divided by Omega. The dipole axis is (0.3,0.4,sqrt(0.75)).
KOKKOS_INLINE_FUNCTION
void Pulse(double s, double &f, double &fp, double &fpp) {
  const double d = (s-0.6)/0.2;
  f = 0.01*Kokkos::exp(-d*d);
  fp = -10*d*f;
  fpp = (100*d*d-50)*f;
}
KOKKOS_INLINE_FUNCTION
void ExactWave(double x, double y, double z, double t, double &phi, double &pi,
               bool coordinate = false, double *gradient = nullptr) {
  const double r = Kokkos::sqrt(x*x+y*y+z*z), alpha = (1+r*r)/2;
  const double omega = (1-r*r)/2;
  double fu, pu, ppu, fv, pv, ppv;
  Pulse(t+(1-r)/(1+r),fu,pu,ppu);
  Pulse(t+(1+r)/(1-r),fv,pv,ppv);
  const double du = -2/((1+r)*(1+r)), dv = 2/((1-r)*(1-r));
  const double a = pu+pv, b = fu-fv;
  const double value = -(a/r+omega*b/(r*r));
  const double dt = -((ppu+ppv)/r+omega*(pu-pv)/(r*r));
  const double dr = -((ppu*du+ppv*dv)/r-a/(r*r)-b/r
      +omega*(pu*du-pv*dv)/(r*r)-2*omega*b/(r*r*r));
  const double angular = (0.3*x+0.4*y+Kokkos::sqrt(0.75)*z)/r;
  phi = angular*value;
  pi = coordinate ? angular*dt : angular*(dt+r*dr)/alpha;
  if (gradient != nullptr) {
    const double axis[3] = {0.3,0.4,Kokkos::sqrt(0.75)}, point[3] = {x,y,z};
    for (int d = 0; d < 3; ++d) {
      gradient[d] = axis[d]*value/r+angular*point[d]/r*(dr-value/r);
    }
  }
}

struct WaveEnergy { double numeric, exact; };
WaveEnergy Energy(const View &q, const Plans &plans, const Kokkos::View<int *> &active,
                  int cells, int nx, int ny, double h, double origin, double time,
                  bool coordinate) {
  auto phi_component = Kokkos::subview(q,std::make_pair(0,cells));
  hyp::FillSphericalGhosts(phi_component,plans);
  double energy = 0, exact_energy = 0;
  Kokkos::parallel_reduce("wave Killing energy",active.extent(0),
      KOKKOS_LAMBDA(const int p, double &numeric, double &reference) {
    const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    const double x[3] = {origin+i*h,origin+j*h,origin+k*h};
    const double alpha = (1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
    const double potential = 1/(alpha*alpha*alpha)-1/alpha;
    const Real idx[3] = {1/h,1/h,1/h};
    FlatScalar field{q,nx,ny,0};
    double grad2 = 0, dot = 0;
    for (int d = 0; d < 3; ++d) {
      const double grad = Dx<3>(d,idx,field,0,k,j,i);
      grad2 += grad*grad; dot += x[d]*grad;
    }
    const double pi = coordinate ? (q(cells+s)+dot)/alpha : q(cells+s);
    numeric += (alpha*(pi*pi+grad2+potential*q(s)*q(s))/2-pi*dot)*h*h*h;
    double phi_ref,pi_ref,grad_ref[3];
    ExactWave(x[0],x[1],x[2],time,phi_ref,pi_ref,false,grad_ref);
    grad2 = 0; dot = 0;
    for (int d = 0; d < 3; ++d) {
      grad2 += grad_ref[d]*grad_ref[d]; dot += x[d]*grad_ref[d];
    }
    reference += (alpha*(pi_ref*pi_ref+grad2+potential*phi_ref*phi_ref)/2
                  -pi_ref*dot)*h*h*h;
  },Kokkos::Sum<double>(energy),Kokkos::Sum<double>(exact_energy));
  return {energy,exact_energy};
}

struct WaveError { double l2, maximum, initial_energy, final_energy, peak_energy; };
WaveError EvolveWave(int n, int degree, double end, double dissipation = 0.1,
                     bool coordinate = false, const std::string &history = "",
                     bool interior_ko = false) {
  Check(n >= 8 && n%2 == 0 && std::isfinite(end) && end > 0
      && std::isfinite(dissipation) && dissipation >= 0, "invalid wave options");
  const auto g = Grid(n);
  const auto plans = Upload(hyp::PlanSphericalGhosts(g,3,degree));
  const int nx = g.n[0], ny = g.n[1], cells = nx*ny*g.n[2];
  const double h = g.h[0], origin = g.first[0];
  std::vector<int> ids;
  for (int k = 0; k < g.n[2]; ++k)
  for (int j = 0; j < ny; ++j)
  for (int i = 0; i < nx; ++i) {
    if (g.Interior(i,j,k)) ids.push_back(g.Index(i,j,k));
  }
  Kokkos::View<int *> active("wave interior",ids.size());
  Kokkos::View<unsigned char *> mask("wave mask",cells);
  auto mh = Kokkos::create_mirror_view(mask);
  for (int s = 0; s < cells; ++s) mh(s) = 0;
  for (int s : ids) mh(s) = 1;
  Kokkos::deep_copy(mask,mh);
  auto ah = Kokkos::create_mirror_view(active);
  for (size_t p = 0; p < ids.size(); ++p) ah(p) = ids[p];
  Kokkos::deep_copy(active,ah);
  View q("wave state",2*cells), stage("wave stage",2*cells);
  View rhs("wave rhs",2*cells), total("wave RK sum",2*cells);
  Kokkos::deep_copy(q,std::numeric_limits<double>::quiet_NaN());
  Kokkos::deep_copy(stage,std::numeric_limits<double>::quiet_NaN());
  Kokkos::parallel_for("wave initial data",active.extent(0),KOKKOS_LAMBDA(const int p) {
    const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    double phi,pi;
    ExactWave(origin+i*h,origin+j*h,origin+k*h,0,phi,pi,coordinate);
    q(s) = phi; q(cells+s) = pi;
  });
  std::ofstream log;
  if (!history.empty()) {
    log.open(history);
    Check(log.is_open(), "cannot open wave history");
    log << std::setprecision(17) << "t,max_error,energy,exact_energy\n";
  }
  const auto initial = Energy(q,plans,active,cells,nx,ny,h,origin,0,coordinate);
  Check(std::isfinite(initial.numeric) && initial.numeric > 0, "invalid initial energy");
  double peak_energy = initial.numeric;
  WaveEnergy last_energy = initial;
  if (log.is_open()) log << "0,0," << initial.numeric << ',' << initial.exact << '\n';
  double next_diagnostic = 0.1;
  double time = 0, maximum = 0;
  int steps = 0;
  while (time < end) {
    const double dt = std::min(0.075*h,end-time);
    Kokkos::parallel_for("wave RK start",active.extent(0),KOKKOS_LAMBDA(const int p) {
      for (int f = 0; f < 2; ++f) {
        const int s = f*cells+active(p);
        stage(s) = q(s); total(s) = 0;
      }
    });
    for (int rk = 0; rk < 4; ++rk) {
      for (int f = 0; f < 2; ++f) {
        auto component = Kokkos::subview(stage,std::make_pair(f*cells,(f+1)*cells));
        hyp::FillSphericalGhosts(component,plans);
      }
      const FlatScalar phi{stage,nx,ny,0}, pi{stage,nx,ny,cells};
      const RadialShift beta{origin,h};
      const auto phi_flat = Kokkos::subview(stage,std::make_pair(0,cells));
      const auto pi_flat = Kokkos::subview(stage,std::make_pair(cells,2*cells));
      Kokkos::parallel_for("conformal wave RHS",active.extent(0),
          KOKKOS_LAMBDA(const int p) {
        const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
        const double x[3] = {origin+i*h,origin+j*h,origin+k*h};
        const double alpha = (1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
        const Real idx[3] = {1/h,1/h,1/h};
        const int stride[3] = {1,nx,nx*ny};
        double f = alpha*stage(cells+s);
        double p_rhs = -3*stage(cells+s)-(1/(alpha*alpha)-1)*stage(s);
        const double r2 = 2*alpha-1;
        if (coordinate) {
          f = stage(cells+s);
          p_rhs = (-3+r2/alpha)*stage(cells+s)-(1/alpha-alpha)*stage(s);
        }
        for (int d = 0; d < 3; ++d) {
          if (coordinate) {
            p_rhs += 2*Lx<3>(d,idx,beta,pi,0,d,k,j,i)
                +(alpha*alpha-x[d]*x[d])*Dxx<3>(d,idx,phi,0,k,j,i)
                +(alpha-4+r2/alpha)*x[d]*Dx<3>(d,idx,phi,0,k,j,i);
            for (int e = d+1; e < 3; ++e) {
              p_rhs -= 2*x[d]*x[e]*Dxy<3>(d,e,idx,phi,0,k,j,i);
            }
          } else {
            f += Lx<3>(d,idx,beta,phi,0,d,k,j,i);
            p_rhs += Lx<3>(d,idx,beta,pi,0,d,k,j,i)
                +alpha*Dxx<3>(d,idx,phi,0,k,j,i)+x[d]*Dx<3>(d,idx,phi,0,k,j,i);
          }
          const int delta = stride[d];
          if (!interior_ko) {
            for (int v = 0; v < 2; ++v) {
              const int c = v*cells+s;
              const double ko = (stage(c-3*delta)-6*stage(c-2*delta)+15*stage(c-delta)
                  -20*stage(c)+15*stage(c+delta)-6*stage(c+2*delta)
                  +stage(c+3*delta))/(64*h);
              if (v == 0) f += dissipation*ko;
              else p_rhs += dissipation*ko;
            }
          }
        }
        if (interior_ko) {
          const double spacing[3] = {h,h,h};
          f += dissipation*hyp::InteriorKOSixth(phi_flat,mask,s,stride,spacing);
          p_rhs += dissipation*hyp::InteriorKOSixth(pi_flat,mask,s,stride,spacing);
        }
        rhs(s) = f; rhs(cells+s) = p_rhs;
      });
      const double weight = rk == 0 || rk == 3 ? 1./6 : 1./3;
      const double fraction = rk < 2 ? 0.5 : 1;
      Kokkos::parallel_for("wave RK update",active.extent(0),KOKKOS_LAMBDA(const int p) {
        for (int f = 0; f < 2; ++f) {
          const int s = f*cells+active(p);
          total(s) += weight*rhs(s);
          stage(s) = q(s)+fraction*dt*rhs(s);
          if (rk == 3) q(s) += dt*total(s);
        }
      });
    }
    time += dt; ++steps;
    double error = 0;
    Kokkos::parallel_reduce("wave maximum error",active.extent(0),
        KOKKOS_LAMBDA(const int p, double &maximum) {
      const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
      double phi,pi;
      ExactWave(origin+i*h,origin+j*h,origin+k*h,time,phi,pi,coordinate);
      const double diff = Kokkos::isfinite(q(s)) && Kokkos::isfinite(q(cells+s)) ?
          Kokkos::fmax(Kokkos::abs(q(s)-phi),Kokkos::abs(q(cells+s)-pi)) : 1e100;
      if (diff > maximum) maximum = diff;
    },Kokkos::Max<double>(error));
    maximum = std::max(maximum,error);
    if (error > 1e3) {
      std::cerr << "wave failure n=" << n << " degree=" << degree << " t=" << time
                << " error=" << error << '\n';
      throw std::runtime_error("conformal wave growth");
    }
    if (time >= next_diagnostic || time == end) {
      last_energy = Energy(q,plans,active,cells,nx,ny,h,origin,time,coordinate);
      Check(std::isfinite(last_energy.numeric), "nonfinite wave energy");
      peak_energy = std::max(peak_energy,last_energy.numeric);
      if (log.is_open()) log << time << ',' << error << ',' << last_energy.numeric << ','
                   << last_energy.exact << '\n';
      next_diagnostic += 0.1;
    }
  }
  double sum = 0;
  Kokkos::parallel_reduce("wave RMS error",active.extent(0),
      KOKKOS_LAMBDA(const int p, double &norm) {
    const int s = active(p), i = s%nx, j = s/nx%ny, k = s/(nx*ny);
    double phi,pi;
    ExactWave(origin+i*h,origin+j*h,origin+k*h,time,phi,pi,coordinate);
    norm += (q(s)-phi)*(q(s)-phi)+(q(cells+s)-pi)*(q(cells+s)-pi);
  },sum);
  const double rms = std::sqrt(sum/(2*ids.size()));
  std::cout << "wave n=" << n << " degree=" << degree << " t=" << time
            << " coordinate=" << coordinate << " dissipation=" << dissipation
            << " interior_ko=" << interior_ko
            << " steps=" << steps << " RMS=" << rms << " max_time_Linf="
            << maximum << " energy_initial=" << initial.numeric
            << " energy_final=" << last_energy.numeric << " energy_peak="
            << peak_energy << std::endl;
  return {rms,maximum,initial.numeric,last_energy.numeric,peak_energy};
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    if (argc >= 4 && argc <= 7) {
      const std::string mode = argc >= 6 ? argv[5] : "normal";
      Check(mode == "coordinate" || mode == "normal" || mode == "normal_ko"
          || mode == "coordinate_ko", "unknown wave form");
      EvolveWave(std::stoi(argv[1]),std::stoi(argv[2]),std::stod(argv[3]),
                 argc >= 5 ? std::stod(argv[4]) : 0.1,
                 mode == "coordinate" || mode == "coordinate_ko",
                 argc == 7 ? argv[6] : "",
                 mode == "normal_ko" || mode == "coordinate_ko");
    } else {
      Check(argc == 1, "expected N degree end [dissipation [form [history.csv]]]");
      const auto coarse = EvolveWave(24,3,1.2,0.1,false,"",true);
      const auto fine = EvolveWave(48,3,1.2,0.1,false,"",true);
      Check(coarse.l2/fine.l2 > 4 && coarse.maximum/fine.maximum > 4,
            "conformal wave does not converge");
      const auto longer = EvolveWave(48,3,4,0.1,false,"",true);
      Check(longer.l2 < 1e-3 && longer.maximum < 0.2
          && longer.peak_energy/longer.initial_energy < 1.1
          && longer.final_energy/longer.initial_energy < 1e-5,
          "conformal wave late-time field/energy bound");
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
