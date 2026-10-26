#include <cmath>
#include <iostream>
#include <stdexcept>
#include "mesh/cubed_sphere/conformal_bridge.hpp"

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    using namespace cubed_sphere;
    using U=z4c::hyperboloidal::Z4cJet<Real>;
    Real worst=0;
    // Forward-map jets are built separately from Geometry's inverse-map jets.
    // A Cartesian quadratic supplies exact mixed derivatives in every slot.
    for (int face=0;face<6;++face) for (Real radial:{0.0,0.43,1.0}) {
      Real q[3]={radial,0.31,-0.27}; Map map{0.25,1}; auto p=map.Geometry(face,q);
      Jet s=Jet::Variable(q[0],0),v=Jet::Variable(q[1],1),w=Jet::Variable(q[2],2);
      Jet d=(Jet(1)-s)*Jet(map.L)+s*Jet(map.R)*Power(Jet(1)+v*v+w*w,-0.5);
      Jet x[3]; int a=face/2,b=(a+1)%3,c=(a+2)%3;
      x[a]=Jet(face%2 ? -1 : 1)*d; x[b]=d*v; x[c]=d*w;
      Jet f=x[0]*x[1]+Jet(0.3)*x[2]*x[2]+Jet(0.2)*x[0];
      Real grad[3]={p.x[1]+0.2,p.x[0],0.6*p.x[2]};
      Real hess[3][3]={{0,1,0},{1,0,0},{0,0,0.6}};
      U u{};
      auto scalar=[&](auto &out,Real scale) {
        out.value=scale*f.f;
        for (int A=0;A<3;++A) {
          out.d[A]=scale*f.d[A];
          for (int B=0;B<3;++B) out.dd[A][B]=scale*f.h[A][B];
        }
      };
      scalar(u.chi,1); scalar(u.alpha,2); scalar(u.trace,3); scalar(u.theta,4);
      for (int i=0;i<3;++i) for (int A=0;A<3;++A) {
        u.beta.d[A][i]=(5+i)*f.d[A]; u.lambda.d[A][i]=(8+i)*f.d[A];
        for (int B=0;B<3;++B) {
          u.beta.dd[A][B][i]=(5+i)*f.h[A][B]; u.lambda.dd[A][B][i]=(8+i)*f.h[A][B];
          for (int j=0;j<3;++j) u.metric.ddg[A][B][i][j]=(11+3*i+j)*f.h[A][B];
        }
        for (int j=0;j<3;++j) {
          u.metric.dg[A][i][j]=(11+3*i+j)*f.d[A]; u.a.dk[A][i][j]=(20+3*i+j)*f.d[A];
        }
      }
      CartesianJet(u,p);
      auto check=[&](Real got,Real want){worst=std::max(worst,fabs(got-want));};
      for (int A=0;A<3;++A) {
        int scale=1;
        for (const auto *out:{&u.chi,&u.alpha,&u.trace,&u.theta}) {
          check(out->d[A],scale*grad[A]);
          for (int B=0;B<3;++B) check(out->dd[A][B],scale*hess[A][B]);
          ++scale;
        }
        for (int i=0;i<3;++i) {
          check(u.beta.d[A][i],(5+i)*grad[A]); check(u.lambda.d[A][i],(8+i)*grad[A]);
          for (int B=0;B<3;++B) {
            check(u.beta.dd[A][B][i],(5+i)*hess[A][B]);
            check(u.lambda.dd[A][B][i],(8+i)*hess[A][B]);
            for (int j=0;j<3;++j) check(u.metric.ddg[A][B][i][j],(11+3*i+j)*hess[A][B]);
          }
          for (int j=0;j<3;++j) {
            check(u.metric.dg[A][i][j],(11+3*i+j)*grad[A]);
            check(u.a.dk[A][i][j],(20+3*i+j)*grad[A]);
          }
        }
      }
    }
    std::cout<<"All conformal derivative slots, six faces and three radii: max error "<<worst<<"\n";
    if (worst>2e-12) throw std::runtime_error("mapped conformal jet chain rule");
  }
  Kokkos::finalize();
}
