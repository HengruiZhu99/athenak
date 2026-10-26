// Full nonlinear spherical restriction of the repository tensor kernel.
// Experimental C interface; flat reference connection, C_Z4c=0. No constraints reset.
#include <cmath>
#include <limits>
#include "z4c/hyperboloidal/spherical_tensor.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
namespace h = z4c::hyperboloidal;
extern "C" int spherical_parts(int n, const double *r, const double *q,
    const double *d, const double *dd, const double *beta, const double *dbeta,
    const double *ddbeta, double kappa1, double kappa2, int appendix_c,
    double slicing, double *regular, double *pole, double *diagnostic) {
  const h::CMCReference<double> ref{1, 1};
  for (int i=0; i<n; ++i) {
    if (!(r[i]>0 && r[i]<=1)) return i+1;
    auto u=h::SphericalJet(r[i], q+8*i, d+8*i, dd+8*i, ref);
    u.beta=h::RadialVector(beta[i], dbeta[i], ddbeta[i], r[i]);
    h::OmegaJet<double> o{};
    o.omega=(1-r[i]*r[i])/2; o.gradient[0]=-r[i];
    for(int j=0;j<3;++j)o.hessian[j][j]=-1;
    h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
    auto parts=h::ConformalRHS(u,o,kappa1,kappa2);
    if(!parts.valid)return i+1;
    // The single disclosed off-constraint scalar source change. Other equations
    // remain the repository C_Z4c=0 equations; this does not claim full paper equivalence.
    if(appendix_c)parts.pole.theta+=3*u.alpha.value*o.normal*u.theta.value;
    const auto gauge=h::UnfactoredReferenceGauge(ref,ref.At(r[i],0.,0.),u,
        h::GaugeParameters<double>{slicing,0,0,0},false,true);
    if(!gauge.valid)return i+1;
    auto pack=[](const h::Z4cRHS<double>& x,double* y) {
      y[0]=x.chi; y[1]=x.metric[0][0]; y[2]=x.a[0][0];
      y[3]=x.trace; y[4]=x.theta; y[5]=x.lambda[0]; y[6]=0; y[7]=0;
    };
    pack(parts.regular,regular+8*i);pack(parts.pole,pole+8*i);
    regular[8*i+6]=gauge.regular.alpha;pole[8*i+6]=gauge.pole.alpha;
    if(diagnostic) {
      const auto c=h::EvolvedConstraints(u,o);if(!c.valid)return i+1;
      auto out=diagnostic+12*i;
      out[0]=c.hamiltonian;out[1]=c.momentum[0];out[2]=u.theta.value;
      out[3]=c.z4.z_covector[0];out[4]=c.z4.determinant_residual;
      out[5]=c.z4.tracefree_residual;out[6]=c.null_residual;
      out[7]=u.alpha.value;out[8]=u.chi.value;out[9]=u.metric.g[0][0];
      out[10]=std::numeric_limits<double>::quiet_NaN();out[11]=out[10];
      if(o.omega>0){const auto s=h::SphereDiagnostics(r[i],u,o);
        out[10]=s.mass;out[11]=s.expansion_out;}
    }
  }
  return 0;
}
