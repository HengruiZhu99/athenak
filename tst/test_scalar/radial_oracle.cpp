// Independent conservative 1+1 Schwarzschild scalar, two evolved fields.
// Background is supplied by a value-only high-precision physical metric pullback.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sbp63.hpp"
using Vec=std::vector<double>;
using State=std::array<Vec,2>;
int main(int argc,char **argv) {
  try {
    if(argc!=4)throw std::invalid_argument("table output duration");
    std::ifstream in(argv[1]);int n;double h;in>>n>>h;
    Vec x(n),a(n),b(n),c(n),D(n),V(n),angular(n);State y{Vec(n),Vec(n)},k=y,k2=y,k3=y,k4=y,tmp=y;
    auto number=[&]() {std::string token;in>>token;char *end=nullptr;double v=std::strtod(token.c_str(),&end);
      if(token.empty() || *end || !std::isfinite(v))throw std::runtime_error("invalid finite table value");return v;};
    double dt=1e99;
    for(int i=0;i<n;++i) {
      x[i]=number();a[i]=number();b[i]=number();c[i]=number();D[i]=number();V[i]=number();angular[i]=number();y[0][i]=number();y[1][i]=number();
      double v=b[i]/a[i],speed=std::abs(v)+std::sqrt(v*v-c[i]/a[i]);dt=std::min(dt,.16*h/speed);
    }
    if(!in)throw std::runtime_error("invalid coefficient table");
    Vec norm(n,1);for(int i=0;i<6;++i)norm[i]=norm[n-1-i]=radial_sbp::H[i];
    auto derivative=[&](const Vec &v,Vec &d) {
      for(int i=6;i<n-6;++i)d[i]=(-v[i-3]/60+3*v[i-2]/20-3*v[i-1]/4+3*v[i+1]/4-3*v[i+2]/20+v[i+3]/60)/h;
      for(int i=0;i<6;++i){d[i]=d[n-1-i]=0;for(int j=0;j<9;++j){d[i]+=radial_sbp::left[i][j]*v[j]/h;d[n-1-i]-=radial_sbp::left[i][j]*v[n-1-j]/h;}}
    };
    Vec q(n),velocity(n),flux(n),df(n);
    auto rhs=[&](const State &u,State &r) {
      derivative(u[0],q);
      for(int i=0;i<n;++i) {velocity[i]=(u[1][i]/D[i]-b[i]*q[i])/a[i];flux[i]=D[i]*(b[i]*velocity[i]+c[i]*q[i]);r[0][i]=velocity[i];}
      derivative(flux,df);
      for(int i=0;i<n;++i)r[1][i]=-df[i]-2*flux[i]/x[i]+D[i]*(6*angular[i]+V[i])*u[0][i];
      const double d3[4]={-1,3,-3,1};
      for(int i=0;i<n-3;++i)for(int v=0;v<2;++v) {double delta=0;for(int j=0;j<4;++j)delta+=d3[j]*u[v][i+j];for(int j=0;j<4;++j)r[v][i+j]-=.15/(64*h*norm[i+j])*d3[j]*delta;}
    };
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"time,radial_psi\n";
    auto report=[&](double t) {
      int j=std::clamp(int((10-x[0])/h)-2,0,n-6);double value=0;
      for(int k=0;k<6;++k){double w=1;for(int l=0;l<6;++l)if(k!=l)w*=(10-x[j+l])/(x[j+k]-x[j+l]);value+=w*y[0][j+k];}
      out<<t<<","<<value<<"\n";out.flush();
      if(std::abs(t-2)<1e-9||std::abs(t-6)<1e-9||std::abs(t-12)<1e-9||std::abs(t-35)<1e-9||std::abs(t-65)<1e-9||std::abs(t-100)<1e-9) {
        std::ofstream profile(std::string(argv[2])+"-profile-"+std::to_string(int(t))+".csv");profile<<std::setprecision(17)<<"rho,psi,v,Dpsi\n";derivative(y[0],q);
        for(int i=0;i<n;++i)profile<<x[i]<<","<<y[0][i]<<","<<(y[1][i]/D[i]-b[i]*q[i])/a[i]<<","<<q[i]<<"\n";
      }if(!std::isfinite(value)||std::abs(value)>10)throw std::runtime_error("radial oracle amplitude guard");
    };
    double t=0,end=std::stod(argv[3]),next=.5;report(t);int steps=0;
    while(t<end-1e-12) {
      double target=std::min(next,end),step=std::min(dt,target-t);rhs(y,k);
      for(int v=0;v<2;++v)for(int i=0;i<n;++i)tmp[v][i]=y[v][i]+step*k[v][i]/2;
      rhs(tmp,k2);for(int v=0;v<2;++v)for(int i=0;i<n;++i)tmp[v][i]=y[v][i]+step*k2[v][i]/2;
      rhs(tmp,k3);for(int v=0;v<2;++v)for(int i=0;i<n;++i)tmp[v][i]=y[v][i]+step*k3[v][i];
      rhs(tmp,k4);for(int v=0;v<2;++v)for(int i=0;i<n;++i)y[v][i]+=step*(k[v][i]+2*k2[v][i]+2*k3[v][i]+k4[v][i])/6;
      t+=step;++steps;if(t>=target-1e-12){report(t);next+=.5;}
    }
    std::cout<<"n="<<n<<" steps="<<steps<<" dt="<<dt<<" end="<<t<<"\n";
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
