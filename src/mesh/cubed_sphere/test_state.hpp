#ifndef MESH_CUBED_SPHERE_TEST_STATE_HPP_
#define MESH_CUBED_SPHERE_TEST_STATE_HPP_
#include "athena.hpp"
#include "z4c/z4c.hpp"
namespace cubed_sphere {
// Exact finite-amplitude harmonic gauge wave (flat spacetime), or a localized
// angular lapse/shift disturbance on a Cartesian Minkowski slice.
KOKKOS_INLINE_FUNCTION void TestState(const Real x[3], Real time, int kind,
                                     Real amp, Real wave_number, Real width, Real u[25]) {
  using Z=z4c::Z4c;
  for (int n=0;n<Z::nz4c;++n) u[n]=0;
  u[Z::I_Z4C_CHI]=u[Z::I_Z4C_ALPHA]=1;
  u[Z::I_Z4C_GXX]=u[Z::I_Z4C_GYY]=u[Z::I_Z4C_GZZ]=1;
  if (kind==1) {
    Real phase=wave_number*(x[0]-time);
    Real h=1-amp*sin(phase), hx=-amp*wave_number*cos(phase);
    Real chi=pow(h,-1.0/3), alpha=sqrt(h);
    Real K=hx/(2*h*alpha);
    u[Z::I_Z4C_CHI]=chi; u[Z::I_Z4C_ALPHA]=alpha;
    u[Z::I_Z4C_GXX]=chi*h;
    u[Z::I_Z4C_GYY]=u[Z::I_Z4C_GZZ]=chi;
    u[Z::I_Z4C_KHAT]=K;
    u[Z::I_Z4C_AXX]=chi*(hx/(2*alpha)-h*K/3);
    u[Z::I_Z4C_AYY]=u[Z::I_Z4C_AZZ]=-chi*K/3;
    u[Z::I_Z4C_GAMX]=(2.0/3)*pow(h,-5.0/3)*hx;
  } else if (kind==2) {
    Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
    Real f=amp*exp(-r2/(width*width));
    u[Z::I_Z4C_ALPHA]+=f*(1+0.2*x[0]+0.3*x[1]*x[2]);
    u[Z::I_Z4C_BETAX]=0.5*f*(1+0.3*x[1]*x[2]);
    u[Z::I_Z4C_BETAY]=0.1*f*x[0];
    u[Z::I_Z4C_BETAZ]=0.05*f*x[0]*x[1];
  }
}
}  // namespace cubed_sphere
#endif
