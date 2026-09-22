// Standalone geometry test ABI, independent of AthenaK runtime.
#include "coordinates/kerr_trumpet.hpp"
namespace {
void Pack(const kerr_trumpet::Jet<double> &x,double *&p) {
  *p++=x.value;
  for (int i=0;i<3;++i) *p++=x.d[i];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) *p++=x.dd[i][j];
}
void Pack(const kerr_trumpet::FirstDerivative<double> &x,double *&p) {
  *p++=x.value;
  for (int i=0;i<3;++i) *p++=x.d[i];
}
}
extern "C" int trumpet_eval(double mass,double spin,const double xyz[3],double output[387]) {
  kerr_trumpet::Geometry<double> g;
  const auto status=kerr_trumpet::Evaluate(mass,spin,xyz,g);
  if (status!=kerr_trumpet::Status::success) return static_cast<int>(status);
  double *p=output;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) Pack(g.gamma[i][j],p);
  Pack(g.alpha,p);
  for (int i=0;i<3;++i) Pack(g.beta[i],p);
  Pack(g.chi,p);
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) Pack(g.conformal_metric[i][j],p);
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) Pack(g.K[i][j],p);
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) Pack(g.conformal_A[i][j],p);
  Pack(g.trace_K,p);
  for (int i=0;i<3;++i) Pack(g.conformal_Gamma[i],p);
  return p-output==387 ? 0 : -1;
}
