// Standalone algebra control: c++ -std=c++17 -I src this_file -o /tmp/check_affine
#include "coordinates/affine_horizon.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
void require(bool ok) {if (!ok) throw std::runtime_error("affine chart control failed");}
void fields(const double *x,double *g,double *K,double *dg) {
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    g[3*i+j]=(i==j)+x[i]*x[j];K[3*i+j]=2*x[i]*x[j]+(i==j)*.3;
    for (int k=0;k<3;++k) dg[9*k+3*i+j]=(i==k)*x[j]+(j==k)*x[i];
  }
}
double bilinear(const double *g,const double *u,const double *v) {
  double s=0;for (int i=0;i<3;++i) for (int j=0;j<3;++j) s+=u[i]*g[3*i+j]*v[j];return s;
}
int main() {
  const double velocities[3][3]={{0,0,0},{.3,-.4,.2},{std::sqrt(.99),0,0}};
  double worst_derivative=0,worst_rotation=0,worst_area=0;
  for (const auto &v:velocities) {
    AffineHorizonChart c(v);const double center[3]={2,-3,4},y[3]={2.2,-2.7,3.6};
    double x[3],g[9],K[9],dg[27],gy[9],Ky[9],dgy[27];c.Position(y,center,x);
    fields(x,g,K,dg);c.Pullback(g,K,dg,gy,Ky,dgy);
    for (int k=0;k<3;++k) {
      double yp[3],ym[3];std::copy(y,y+3,yp);std::copy(y,y+3,ym);yp[k]+=1e-5;ym[k]-=1e-5;
      double gp[9],gm[9],unusedK[9],unusedD[27],a[9],b[9];
      c.Position(yp,center,x);fields(x,g,K,dg);c.Pullback(g,K,dg,gp,a,unusedD);
      c.Position(ym,center,x);fields(x,g,K,dg);c.Pullback(g,K,dg,gm,b,unusedD);
      for (int i=0;i<9;++i) worst_derivative=std::max(worst_derivative,std::abs((gp[i]-gm[i])/2e-5-dgy[9*k+i]));
    }
    c.Position(y,center,x);fields(x,g,K,dg);
    const double u[3]={.4,-.1,.7},w[3]={.2,.9,-.3};double Ju[3]={},Jw[3]={};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {Ju[i]+=c.J[3*i+j]*u[j];Jw[i]+=c.J[3*i+j]*w[j];}
    double area_y=bilinear(gy,u,u)*bilinear(gy,w,w)-std::pow(bilinear(gy,u,w),2);
    double area_x=bilinear(g,Ju,Ju)*bilinear(g,Jw,Jw)-std::pow(bilinear(g,Ju,Jw),2);
    worst_area=std::max(worst_area,std::abs(area_x-area_y));
    for (int axis=0;axis<3;++axis) {
      double phi_y[3]={},phi_x[3]={};const int p=(axis+1)%3,q=(axis+2)%3;
      phi_x[p]=-(x[q]-center[q]);phi_x[q]=x[p]-center[p];
      for (int i=0;i<3;++i) for (int j=0;j<3;++j) phi_y[i]+=c.rotations[9*axis+3*i+j]*(y[j]-center[j]);
      worst_rotation=std::max(worst_rotation,std::abs(bilinear(Ky,phi_y,u)-bilinear(K,phi_x,Ju)));
    }
  }
  require(worst_derivative<1e-8 && worst_rotation<1e-12 && worst_area<1e-12);
  bool rejected=false;try {double v[3]={1,0,0};AffineHorizonChart bad(v);} catch(const std::runtime_error&) {rejected=true;}require(rejected);
  std::cout << "derivative " << worst_derivative << " spin_integrand " << worst_rotation << " area " << worst_area << '\n';
}
