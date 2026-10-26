#include <cmath>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"

struct Velocity {
  KOKKOS_INLINE_FUNCTION Real operator()(int,int a,int,int,int) const {
    return a==0 ? 0.3 : (a==1 ? -0.2 : 0.5);
  }
};
void Check(bool value,const char *why) {if (!value) throw std::runtime_error(why);}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    Kokkos::print_configuration(std::cout,true);
    using namespace cubed_sphere;
    for (Real stretch:{0.0,2.4,2.8592697712828787}) {
    Map map{1,4,stretch};
    std::cout<<"radial_stretch="<<stretch<<"\n";
    for (int p=0;p<6;++p) {
      Real q[3]={0.43,0.31,-0.27},x[3],back[3];
      map.Forward(p,q,x); map.Inverse(p,x,back);
      for (int a=0;a<3;++a) Check(fabs(q[a]-back[a])<2e-15,"map inverse");
      for (Real radial:{-0.04,1.04}) for (Real angular:{-1.2,1.2}) {
        Real ghost[3]={radial,angular,-0.7};map.Forward(p,ghost,x);map.Inverse(p,x,back);
        for (int a=0;a<3;++a) Check(fabs(ghost[a]-back[a])<2e-13,"ghost map inverse");
      }
      for (Real u:{-1.0,-0.1,1.0}) for (Real v:{-1.0,0.4,1.0}) {
        Real edge[3]={1,u,v}; map.Forward(p,edge,x);
        Check(fabs(sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])-4)<2e-15,"outer sphere");
        edge[0]=0; map.Forward(p,edge,x);
        Check(fabs(fabs(x[p/2])-1)<2e-15,"inner cube");
      }
      auto g=map.Geometry(p,q); const Real delta=2e-5;
      for (int a=0;a<3;++a) {
        Real xp[3],xm[3],qp[3],qm[3];
        for (int b=0;b<3;++b) xp[b]=xm[b]=g.x[b];
        xp[a]+=delta; xm[a]-=delta;
        map.Inverse(p,xp,qp); map.Inverse(p,xm,qm);
        for (int A=0;A<3;++A) {
          Check(fabs((qp[A]-qm[A])/(2*delta)-g.J[A][a])<2e-9,"inverse Jacobian");
          Check(fabs((qp[A]-2*q[A]+qm[A])/(delta*delta)-g.H[A][a][a])<4e-6,
                "inverse Hessian");
        }
      }
    }
    Real previous=0;
    for (Real h:{0.04,0.02,0.01}) {
      DvceArray4D<Real> field("analytic field",6,7,7,7);
      DvceArray1D<Real> errors("errors",6);
      par_for("mapped analytic field",DevExeSpace(),0,5,0,6,0,6,0,6,
          KOKKOS_LAMBDA(int p,int k,int j,int i) {
        Real q[3]={0.43+(i-3)*h,0.31+(j-3)*h,-0.27+(k-3)*h},x[3];
        map.Forward(p,q,x); field(p,k,j,i)=exp(0.2*x[0]+0.3*x[1]-0.1*x[2]);
      });
      Kokkos::parallel_for("mapped derivative check",Kokkos::RangePolicy<>(DevExeSpace(),0,6),
          KOKKOS_LAMBDA(int p) {
        Real q[3]={0.43,0.31,-0.27},v[3]={0.2,0.3,-0.1};
        Stencil s; s.mapped=true; s.p=map.Geometry(p,q);
        for (int A=0;A<3;++A) s.idx[A]=1/h;
        Real f=field(p,3,3,3),error=0;
        for (int a=0;a<3;++a) {
          error=fmax(error,fabs(::Dx<3>(a,s,field,p,3,3,3)-v[a]*f));
          for (int b=0;b<3;++b)
            error=fmax(error,fabs(::Dxy<3>(a,b,s,field,p,3,3,3)-v[a]*v[b]*f));
        }
        Velocity velocity;
        Real adv=0,exact=0;
        for (int a=0;a<3;++a) {
          adv+=::Lx<3>(a,s,velocity,field,p,a,3,3,3);
          exact+=velocity(p,a,3,3,3)*v[a]*f;
        }
        error=fmax(error,fabs(adv-exact));
        errors(p)=error;
      });
      auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),errors);
      Real error=0; for (int p=0;p<6;++p) error=fmax(error,host(p));
      std::cout<<"h="<<h<<" derivative error="<<error<<"\n";
      if (previous) Check(previous/error>13,"fourth order mapped derivative convergence");
      previous=error;
    }
    }
  }
  Kokkos::finalize();
}
