#ifndef COORDINATES_AFFINE_HORIZON_HPP_
#define COORDINATES_AFFINE_HORIZON_HPP_
#include <array>
#include <cmath>
#include <stdexcept>

// Spatial chart only: x=c+J(y-c), with J contracting the velocity direction.
// The laboratory slice, seed geometry and physical rotation axes are unchanged.
struct AffineHorizonChart {
  std::array<double,9> J{}, inverse{};
  std::array<double,27> rotations{};  // [physical axis][chart vector][chart position]
  double minimum_scale=1;
  explicit AffineHorizonChart(const double *velocity) {
    double v2=0;for (int i=0;i<3;++i) v2+=velocity[i]*velocity[i];
    if (!std::isfinite(v2) || v2>=1) throw std::runtime_error("Invalid affine horizon velocity");
    minimum_scale=std::sqrt(1-v2);
    // Stable at zero velocity; (sqrt(1-v²)-1)/v²=-1/(1+sqrt(1-v²)).
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      J[3*i+j]=(i==j)-velocity[i]*velocity[j]/(1+minimum_scale);
      inverse[3*i+j]=(i==j)+velocity[i]*velocity[j]/(minimum_scale*(1+minimum_scale));
    }
    for (int axis=0;axis<3;++axis) for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
      const int p=(axis+1)%3,q=(axis+2)%3;
      rotations[9*axis+3*a+b]=inverse[3*a+q]*J[3*p+b]-inverse[3*a+p]*J[3*q+b];
    }
  }
  void Position(const double *y,const double *center,double *x) const {
    for (int i=0;i<3;++i) {
      x[i]=center[i];for (int a=0;a<3;++a) x[i]+=J[3*i+a]*(y[a]-center[a]);
    }
  }
  void Pullback(const double *g,const double *K,const double *dg,
                double *gy,double *Ky,double *dgy) const {
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
      gy[3*a+b]=Ky[3*a+b]=0;
      for (int c=0;c<3;++c) dgy[9*c+3*a+b]=0;
      for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        const double weight=J[3*i+a]*J[3*j+b];
        gy[3*a+b]+=weight*g[3*i+j];Ky[3*a+b]+=weight*K[3*i+j];
        for (int c=0;c<3;++c) for (int k=0;k<3;++k)
          dgy[9*c+3*a+b]+=weight*J[3*k+c]*dg[9*k+3*i+j];
      }
    }
  }
};
#endif
