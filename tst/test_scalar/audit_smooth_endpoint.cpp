// Probe the actual physical geometry close to both smooth-layer endpoints.
#include "../../src/test_scalar/geometry.hpp"
#include <fstream>
#include <iomanip>
int main(int argc,char **argv) {
  if(argc!=2)return 1;std::ofstream f(argv[1]);f<<std::setprecision(17)
    <<"mass,spin,x,y,z,omega,density,potential";
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)f<<",g"<<i<<j;
  for(int i=0;i<4;++i)f<<",b"<<i;
  for(int i=0;i<3;++i){f<<",alpha"<<i;for(int j=0;j<3;++j)f<<",beta"<<i<<j;}
  f<<"\n";
  for(double M:{0.,1.})for(double spin:{0.,.5,-.5,.9})
    for(double rho:{8.000001,8.001,8.1,15.9,15.99,15.999,15.999872713026816,15.999999,16.000001})for(int direction=0;direction<2;++direction) {
      test_scalar::Parameters p;p.mass=M;p.spin=spin;p.smooth_layer=true;
      double x[3]={.3*rho,.4*rho,std::sqrt(.75)*rho};
      if(direction==1){x[0]=.6*rho;x[1]=.8*rho;x[2]=0;}
      if(direction==1 && rho==15.999872713026816) {
        x[0]=11.938042950856074;x[1]=2.0891575163998133;x[2]=-10.445787581999065;
      }
      auto c=test_scalar::coefficients(x,p);
      f<<M<<","<<spin;for(double y:x)f<<","<<y;
      f<<","<<c.omega<<","<<c.density<<","<<c.potential;
      for(auto &row:c.g)for(double y:row)f<<","<<y;for(double y:c.b)f<<","<<y;
      for(int i=0;i<3;++i){f<<","<<c.alpha_gradient[i];for(int j=0;j<3;++j)f<<","<<c.beta_gradient[i][j];}
      f<<"\n";
    }
  return f?0:1;
}
