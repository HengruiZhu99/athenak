// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_ELLIPTIC_SHIFT_3D_MANUFACTURED_HPP_
#define TST_ELLIPTIC_SHIFT_3D_MANUFACTURED_HPP_
#include "z4c/hyperboloidal/elliptic_shift.hpp"
#include "mesh/cubed_sphere/map.hpp"
namespace manufactured {
using cubed_sphere::Jet;
inline Jet Exp(const Jet &x) {
  Jet out(std::exp(x.f));
  for (int a = 0; a < 3; ++a) {
    out.d[a] = out.f*x.d[a];
    for (int b = 0; b < 3; ++b) out.h[a][b] = out.f*(x.h[a][b]+x.d[a]*x.d[b]);
  }
  return out;
}
inline z4c::hyperboloidal::Z4cJet<Real> State(const Real position[3], bool curved) {
  namespace h = z4c::hyperboloidal;
  Jet x[3]; for (int a = 0; a < 3; ++a) x[a] = Jet::Variable(position[a], a);
  const Jet omega = Jet(1)-x[0]*x[0]-x[1]*x[1]-x[2]*x[2];
  Jet frame[3][3];
  const Jet f = Jet(curved ? .18 : 0)*x[0]*x[1]*omega;
  const Jet g = Jet(curved ? .12 : 0)*x[2]*x[0]*omega;
  frame[0][0] = Exp(f); frame[1][1] = Exp(g); frame[2][2] = Exp(Jet(0)-f-g);
  frame[0][1] = Jet(curved ? .14 : 0)*x[1]*x[2]*omega;
  frame[1][2] = Jet(curved ? .11 : 0)*x[0]*omega;
  h::Z4cJet<Real> out{};
  out.chi.value = out.alpha.value = 1;
  for (int i = 0; i < 3; ++i) {
    const Jet beta = Jet(0)-x[i]+Jet(.05*(i+1))*omega*omega*Exp(Jet(.2)*x[0]-Jet(.3)*x[1]+Jet(.1)*x[2]);
    out.beta.value[i] = beta.f;
    for (int a = 0; a < 3; ++a) {
      out.beta.d[a][i] = beta.d[a];
      for (int b = 0; b < 3; ++b) out.beta.dd[a][b][i] = beta.h[a][b];
    }
    for (int j = 0; j < 3; ++j) {
      Jet metric;
      for (int k = 0; k < 3; ++k) metric = metric+frame[k][i]*frame[k][j];
      out.metric.g[i][j] = metric.f;
      for (int a = 0; a < 3; ++a) {
        out.metric.dg[a][i][j] = metric.d[a];
        for (int b = 0; b < 3; ++b) out.metric.ddg[a][b][i][j] = metric.h[a][b];
      }
    }
  }
  // Manufactured trace-free curvature makes the exact metric rate vanish.
  // It is a gauge manufactured solution, not Einstein initial data.
  const auto rate = h::ShiftMetricRate(out.metric, out.beta);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.a.k[i][j] = rate.h[i][j]/2;
    for (int d = 0; d < 3; ++d) out.a.dk[d][i][j] = rate.dh[d][i][j]/2;
  }
  return out;
}
}  // namespace manufactured
#endif  // TST_ELLIPTIC_SHIFT_3D_MANUFACTURED_HPP_
