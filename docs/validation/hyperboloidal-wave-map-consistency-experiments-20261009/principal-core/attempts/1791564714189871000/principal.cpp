// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
// Extract the constrained 20-field symbol from the actual tensor and gauge RHS.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "reference_wave_map.hpp"

namespace hyp = z4c::hyperboloidal;
using Jet = hyp::Z4cJet<double>;
using RHS = hyp::Z4cRHS<double>;

void Invert(const double e[3][3], double inv[3][3]) {
  const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                     e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                     e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      inv[i][j] = (e[(j + 1) % 3][(i + 1) % 3] * e[(j + 2) % 3][(i + 2) % 3] -
                   e[(j + 1) % 3][(i + 2) % 3] * e[(j + 2) % 3][(i + 1) % 3]) /
                  det;
    }
}

double TensorFrame(const double v[3][3], const double inv[3][3], int i, int j) {
  double out = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      out += inv[a][i] * inv[b][j] * v[a][b];
  return out;
}

void TensorInput(int column, double h[3][3], double a[3][3]) {
  if (column == 2) {
    h[0][0] = 1;
    h[1][1] = h[2][2] = -0.5;
  }
  if (column == 5) {
    a[0][0] = 1;
    a[1][1] = a[2][2] = -0.5;
  }
  for (int v = 0; v < 2; ++v) {
    if (column == 8 + 4 * v)
      h[0][1 + v] = h[1 + v][0] = 1;
    if (column == 9 + 4 * v)
      a[0][1 + v] = a[1 + v][0] = 1;
  }
  if (column == 16) {
    h[1][1] = 1;
    h[2][2] = -1;
  }
  if (column == 17) {
    a[1][1] = 1;
    a[2][2] = -1;
  }
  if (column == 18)
    h[1][2] = h[2][1] = 1;
  if (column == 19)
    a[1][2] = a[2][1] = 1;
}

void Extract(double alpha, double chi, double radius, bool oblique,
             double curvature_radius, double matrix[20][20], double &normal) {
  const double b[3][3] = {{1.3, 0.2, -0.1}, {0, 0.8, 0.12}, {0, 0, 1 / 1.04}};
  const double q[3][3] = {
      {0.36, -0.48, 0.8}, {0.8, 0.6, 0}, {-0.48, 0.64, 0.6}};
  double e[3][3]{}, inv[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k)
        e[i][j] += (oblique ? q[i][k] : (i == k ? 1 : 0)) *
                   (oblique ? b[k][j] : (k == j ? 1 : 0)) / std::sqrt(chi);
    }
  Invert(e, inv);
  Jet base{};
  base.alpha.value = alpha;
  base.chi.value = chi;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        base.metric.g[i][j] += chi * e[k][i] * e[k][j];
  const hyp::LayerReference<double> ref(1.,curvature_radius,{true,.05,.95});
  const double xyz[3]={radius,0,0};const auto p=ref.At(radius,0,0);
  const auto cref=rwm::ReferenceConnection(p,xyz);
  auto make_omega=[&](const Jet &u){hyp::OmegaJet<double> o{};o.omega=p.omega;
    for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
    hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;};
  const auto omega=make_omega(base);
  const auto baseparts=hyp::ConformalRHS(base,omega,0.,0.);
  auto difference=[&](const Jet &u,RHS &r){auto f=hyp::ConformalRHS(u,make_omega(u),0.,0.);
    auto subtract=[](RHS &v,const RHS &b){v.chi-=b.chi;v.trace-=b.trace;v.theta-=b.theta;for(int i=0;i<3;++i){v.lambda[i]-=b.lambda[i];for(int j=0;j<3;++j){v.metric[i][j]-=b.metric[i][j];v.a[i][j]-=b.a[i][j];}}};
    subtract(f.regular,baseparts.regular);subtract(f.pole,baseparts.pole);return hyp::AssembleInterior(f,omega.omega,r);};
  const auto gbasegeo=hyp::Geometry(base.metric);
  // Separate derivative orders to exclude lower-order physical-trace/Z4 terms.
  // The reduction enforces det(gtilde)=1 and trace(A)=0 before extraction.
  for (int column = 0; column < 20; ++column) {
    double h[3][3]{}, a[3][3]{}, lambda[3]{}, beta[3]{};
    TensorInput(column, h, a);
    if (column == 6)
      lambda[0] = 1;
    if (column == 7)
      beta[0] = 1;
    for (int v = 0; v < 2; ++v) {
      if (column == 10 + 4 * v)
        lambda[1 + v] = 1;
      if (column == 11 + 4 * v)
        beta[1 + v] = 1;
    }
    double hcoord[3][3]{}, acoord[3][3]{}, lcoord[3]{}, bcoord[3]{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        lcoord[i] += inv[i][j] * lambda[j] / chi;
        bcoord[i] += alpha * inv[i][j] * beta[j];
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l) {
            hcoord[i][j] += chi * e[k][i] * e[l][j] * h[k][l];
            acoord[i][j] += chi * e[k][i] * e[l][j] * a[k][l];
          }
      }
    }
    Jet first = base, second = base, connection = base, slicing = base,
        shift = base;
    first.trace.value = column == 3 ? omega.omega : 0;
    first.theta.value = column == 4 ? omega.omega : 0;
    slicing.trace.value = first.trace.value;
    for (int i = 0; i < 3; ++i) {
      connection.trace.d[i] = first.trace.value * e[0][i];
      connection.theta.d[i] = first.theta.value * e[0][i];
      shift.lambda.value[i] = lcoord[i];
      shift.alpha.d[i] = column == 0 ? alpha * e[0][i] : 0;
      shift.chi.d[i] = column == 1 ? chi * e[0][i] : 0;
      for (int j = 0; j < 3; ++j) {
        first.a.k[i][j] = acoord[i][j];
        first.beta.d[j][i] = bcoord[i] * e[0][j];
        second.lambda.d[j][i] = lcoord[i] * e[0][j];
        second.alpha.dd[i][j] = column == 0 ? alpha * e[0][i] * e[0][j] : 0;
        second.chi.dd[i][j] = column == 1 ? chi * e[0][i] * e[0][j] : 0;
        for (int k = 0; k < 3; ++k) {
          connection.beta.dd[j][k][i] = bcoord[i] * e[0][j] * e[0][k];
          for (int l = 0; l < 3; ++l) {
            second.metric.ddg[k][l][i][j] = hcoord[i][j] * e[0][k] * e[0][l];
          }
        }
      }
    }
    RHS r1{}, r2{}, r3{};
    if(!difference(first,r1)||!difference(second,r2)||!difference(connection,r3))throw std::runtime_error("kernel principal difference invalid");
    for(const auto&pair:{std::pair<const double(*)[3],const double(*)[3]>{hcoord,acoord},std::pair<const double(*)[3],const double(*)[3]>{r1.metric,r2.a}}){
      double trh=0,tra=0,nh=0,na=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){trh+=gbasegeo.inverse[i][j]*pair.first[i][j];tra+=gbasegeo.inverse[i][j]*pair.second[i][j];nh+=std::abs(pair.first[i][j]);na+=std::abs(pair.second[i][j]);}
      normal=std::max({normal,std::abs(trh)/std::max(1.,nh),std::abs(tra)/std::max(1.,na)});
    }
    hyp::GaugeRHS<double> gl{}, gs{}, gbase{};
    if (!rwm::Assemble(rwm::Gauge(p, slicing, cref),
                                    omega.omega, gl) ||
        !rwm::Assemble(rwm::Gauge(p, shift, cref),
                                    omega.omega, gs) ||
        !rwm::Assemble(rwm::Gauge(p, base, cref),
                                    omega.omega, gbase)) {
      throw std::runtime_error("gauge symbol extraction failed");
    }
    matrix[0][column] = (gl.alpha - gbase.alpha) / (alpha * alpha);
    matrix[1][column] = r1.chi / (alpha * chi);
    matrix[2][column] = TensorFrame(r1.metric, inv, 0, 0) / (alpha * chi);
    matrix[3][column] = r2.trace / (omega.omega * alpha);
    matrix[4][column] = r2.theta / (omega.omega * alpha);
    matrix[5][column] = TensorFrame(r2.a, inv, 0, 0) / (alpha * chi);
    double lframe[3]{}, bframe[3]{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        lframe[i] += chi * e[i][j] * r3.lambda[j] / alpha;
        bframe[i] += e[i][j] * (gs.beta[j] - gbase.beta[j]) / (alpha * alpha);
      }
    matrix[6][column] = lframe[0];
    matrix[7][column] = bframe[0];
    for (int v = 0; v < 2; ++v) {
      matrix[8 + 4 * v][column] =
          TensorFrame(r1.metric, inv, 0, 1 + v) / (alpha * chi);
      matrix[9 + 4 * v][column] =
          TensorFrame(r2.a, inv, 0, 1 + v) / (alpha * chi);
      matrix[10 + 4 * v][column] = lframe[1 + v];
      matrix[11 + 4 * v][column] = bframe[1 + v];
    }
    matrix[16][column] = TensorFrame(r1.metric, inv, 1, 1) / (alpha * chi);
    matrix[17][column] = TensorFrame(r2.a, inv, 1, 1) / (alpha * chi);
    matrix[18][column] = TensorFrame(r1.metric, inv, 1, 2) / (alpha * chi);
    matrix[19][column] = TensorFrame(r2.a, inv, 1, 2) / (alpha * chi);
    // Plus tensor polarization is the transverse difference, removing scalar.
    matrix[16][column] -= TensorFrame(r1.metric, inv, 2, 2) / (alpha * chi);
    matrix[17][column] -= TensorFrame(r2.a, inv, 2, 2) / (alpha * chi);
    matrix[16][column] /= 2;
    matrix[17][column] /= 2;
  }
}

int main() {
  std::cout<<std::setprecision(17)<<'[';bool first=true;
  for(double alpha:{.2,1.,3.})for(double chi:{.4,1.,2.})for(bool oblique:{false,true})
   for(double a:{.5,.75,1.,2.})for(double r:{0.,.45,.5,.65,.8,.83,.84,.849,.85,.95,.98}){
    if(!first)std::cout<<',';first=false;double M[20][20]{},normal=0;Extract(alpha,chi,r,oblique,a,M,normal);
    std::cout<<"{\"alpha\":"<<alpha<<",\"chi\":"<<chi<<",\"a\":"<<a<<",\"r\":"<<r<<",\"oblique\":"<<oblique<<",\"normal_scaled\":"<<normal<<",\"M\":[";
    for(int i=0;i<20;++i){std::cout<<(i?",":"")<<'[';for(int j=0;j<20;++j){if(!std::isfinite(M[i][j]))throw std::runtime_error("nonfinite principal entry");std::cout<<(j?",":"")<<M[i][j];}std::cout<<']';}std::cout<<"]}";
   }
  std::cout<<"]\n";
}
