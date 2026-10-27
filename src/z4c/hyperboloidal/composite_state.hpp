// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_STATE_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_STATE_HPP_
#include <array>
#include <vector>
#include "z4c/hyperboloidal/wormhole_chart.hpp"
namespace z4c { namespace hyperboloidal {
inline constexpr int kCompositeTensorI[6]={0,0,0,1,1,2},kCompositeTensorJ[6]={0,1,2,1,2,2};
inline Z4cJet<Real> UnpackComposite(const Real x[3], const std::vector<Real> &values,
    const std::vector<std::array<Real, 3>> &d,
    const std::vector<std::array<std::array<Real, 3>, 3>> &dd, bool wormhole=false,int isotropic_reference=0) {
  Z4cJet<Real> u{};
  auto scalar = [&](int f) {
    ScalarJet<Real> out{}; out.value = values[f];
    for (int a = 0; a < 3; ++a) {
      out.d[a] = d[f][a];
      for (int b = 0; b < 3; ++b) out.dd[a][b] = dd[f][a][b];
    }
    return out;
  };
  u.chi = scalar(0); u.trace = scalar(13); u.theta = scalar(14); u.alpha = scalar(18);
  u.chi.value += 1; u.trace.value -= 3;
  u.alpha.value += (1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
  for (int a = 0; a < 3; ++a) {
    u.alpha.d[a] += x[a]; u.alpha.dd[a][a] += 1;
    u.lambda.value[a] = values[15+a]; u.beta.value[a] = -x[a]; u.beta.d[a][a] = -1;
    for (int b = 0; b < 3; ++b) {
      u.lambda.d[b][a] = d[15+a][b];
      for (int c = 0; c < 3; ++c) u.lambda.dd[b][c][a] = dd[15+a][b][c];
    }
  }
  for (int t = 0; t < 6; ++t) {
    const int i = kCompositeTensorI[t], j = kCompositeTensorJ[t];
    u.metric.g[i][j] = u.metric.g[j][i] = values[1+t]+(i == j ? 1 : 0);
    u.a.k[i][j] = u.a.k[j][i] = values[7+t];
    for (int a = 0; a < 3; ++a) {
      u.metric.dg[a][i][j] = u.metric.dg[a][j][i] = d[1+t][a];
      u.a.dk[a][i][j] = u.a.dk[a][j][i] = d[7+t][a];
      for (int b = 0; b < 3; ++b)
        u.metric.ddg[a][b][i][j] = u.metric.ddg[a][b][j][i] = dd[1+t][a][b];
    }
  }
  if(wormhole){
    const auto chart=WormholeChart::At(x,isotropic_reference);
    auto coefficient=[&](int f){cubed_sphere::Jet q(values[f]);
      for(int a=0;a<3;++a){q.d[a]=d[f][a];for(int b=0;b<3;++b)q.h[a][b]=dd[f][a][b];}return q;};
    auto assign=[](const cubed_sphere::Jet &j,ScalarJet<Real> &s){s.value=j.f;
      for(int a=0;a<3;++a){s.d[a]=j.d[a];for(int b=0;b<3;++b)s.dd[a][b]=j.h[a][b];}};
    assign(chart.chi*ChartExponential(coefficient(0)),u.chi);
    assign(chart.alpha*ChartExponential(coefficient(18)),u.alpha);
    assign(chart.weight*(coefficient(13)-cubed_sphere::Jet(3)),u.trace);
    assign(chart.weight*coefficient(14),u.theta);
    for(int t=0;t<6;++t){const auto j=chart.weight*coefficient(7+t);const int i=kCompositeTensorI[t],k=kCompositeTensorJ[t];
      u.a.k[i][k]=u.a.k[k][i]=j.f;
      for(int a=0;a<3;++a)u.a.dk[a][i][k]=u.a.dk[a][k][i]=j.d[a];}
  }
  return u;
}
}}  // namespace z4c::hyperboloidal
#endif
