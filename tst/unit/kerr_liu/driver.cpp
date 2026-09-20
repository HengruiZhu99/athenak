// Standalone test ABI: no Athena/Kokkos runtime or evolution is linked.
#include "coordinates/kerr_liu.hpp"

namespace {
void Pack(const kerr_liu::Jet<double> &x, double *&p) {
  *p++=x.value;
  for (int i=0; i<3; ++i) *p++=x.d[i];
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) *p++=x.dd[i][j];
}
}
extern "C" int liu_eval(double mass, double spin, const double xyz[3], int gauge,
                         double output[571]) {
  kerr_liu::Geometry<double> g;
  const auto status=kerr_liu::Evaluate(mass,spin,xyz,
      static_cast<kerr_liu::Gauge>(gauge),g);
  if (status!=kerr_liu::Status::success) return static_cast<int>(status);
  double *p=output;
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Pack(g.gamma[i][j],p);
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Pack(g.K[i][j],p);
  Pack(g.alpha,p);
  for (int i=0; i<3; ++i) Pack(g.beta[i],p);
  Pack(g.chi,p);
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Pack(g.conformal_metric[i][j],p);
  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) Pack(g.conformal_A[i][j],p);
  Pack(g.trace_K,p); Pack(g.signed_stationary_lapse,p);
  for (int i=0; i<3; ++i) {
    *p++=g.conformal_Gamma[i].value;
    for (int j=0; j<3; ++j) *p++=g.conformal_Gamma[i].d[j];
  }
  return p-output==571 ? 0 : -1;
}
