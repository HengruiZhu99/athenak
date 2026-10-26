#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
using V=std::vector<double>;using State=std::array<V,3>;
int main(int argc,char **argv) {
 try {
  if(argc<4)throw std::runtime_error("table output duration [dt factor]");std::ifstream input(argv[1]);int n;double h;input>>n>>h;
  std::vector<std::array<double,13>> g(n);State u{V(n),V(n),V(n)},tmp=u,k1=u,k2=u,k3=u,k4=u;
  auto number=[&](){std::string z;input>>z;char *end;double v=std::strtod(z.c_str(),&end);if(z.empty()||*end||!std::isfinite(v))throw std::runtime_error("invalid table");return v;};double dt=1e99;
  for(int i=0;i<n;++i){for(double &x:g[i])x=number();for(int a=0;a<3;++a)u[a][i]=g[i][10+a];double speed=std::abs(g[i][2])+g[i][1]*std::sqrt(g[i][3]);dt=std::min(dt,.16*h/speed);}
  if(argc==5)dt*=std::stod(argv[4]);
  auto diff=[&](const V &f,int i,double speed) {
   if(speed>=0) {
    if(i==0)return (f[1]-f[0])/h;
    if(i<3)return (f[i]-f[i-1])/h;
    if(i==n-1)return (25*f[i]/12-4*f[i-1]+3*f[i-2]-4*f[i-3]/3+f[i-4]/4)/h;
    return (-f[i-3]/12+f[i-2]/2-3*f[i-1]/2+5*f[i]/6+f[i+1]/4)/h;
   }
   if(i==n-1)return (f[i]-f[i-1])/h;
   if(i>n-4)return (f[i+1]-f[i])/h;
   if(i==0)return (-25*f[i]/12+4*f[i+1]-3*f[i+2]+4*f[i+3]/3-f[i+4]/4)/h;
   return (-f[i-1]/4-5*f[i]/6+3*f[i+1]/2-f[i+2]/2+f[i+3]/12)/h;
  };
  auto centered=[&](const V &f,int i){if(i<2)return diff(f,i,-1);if(i>n-3)return diff(f,i,1);return (f[i-2]-8*f[i-1]+8*f[i+1]-f[i+2])/(12*h);};
  auto rhs=[&](const State &y,State &r) {
   for(int i=0;i<n;++i){auto a=g[i];double alpha=a[1],beta=a[2],s=std::sqrt(a[3]),cp=-beta+alpha*s,cm=-beta-alpha*s;
    double plus=diff(y[1],i,cp)+s*diff(y[2],i,cp),minus=diff(y[1],i,cm)-s*diff(y[2],i,cm);
    r[0][i]=beta*diff(y[0],i,-beta)-alpha*y[1][i];
    r[1][i]=-.5*(cp*plus+cm*minus)+a[6]*y[1][i]+a[7]*y[2][i]+alpha*(6*a[9]+a[8])*y[0][i];
    r[2][i]=-.5*(cp*plus-cm*minus)/s+a[5]*y[2][i]-a[4]*y[1][i]-2*(y[2][i]-centered(y[0],i));
   }
  };
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"time,radial_psi\n";
  auto report=[&](double t){int j=std::clamp(int((10-g[0][0])/h)-2,0,n-6);double f=0;for(int a=0;a<6;++a){double w=1;for(int b=0;b<6;++b)if(a!=b)w*=(10-g[j+b][0])/(g[j+a][0]-g[j+b][0]);f+=w*u[0][j+a];}out<<t<<","<<f<<"\n";out.flush();
   if(std::abs(t-2)<1e-9||std::abs(t-6)<1e-9||std::abs(t-12)<1e-9||std::abs(t-35)<1e-9||std::abs(t-65)<1e-9||std::abs(t-100)<1e-9){std::ofstream p(std::string(argv[2])+"-profile-"+std::to_string(int(t))+".csv");p<<std::setprecision(17)<<"rho,psi,v,Dpsi\n";for(int i=0;i<n;++i)p<<g[i][0]<<","<<u[0][i]<<","<<g[i][2]*diff(u[0],i,-g[i][2])-g[i][1]*u[1][i]<<","<<centered(u[0],i)<<"\n";}
   if(!std::isfinite(f)||std::abs(f)>10)throw std::runtime_error("normal oracle amplitude guard");};
  double t=0,end=std::stod(argv[3]),next=.5;int steps=0;report(t);
  while(t<end-1e-12){double target=std::min(next,end),q=std::min(dt,target-t);rhs(u,k1);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k1[a][i]/2;rhs(tmp,k2);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k2[a][i]/2;rhs(tmp,k3);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k3[a][i];rhs(tmp,k4);for(int a=0;a<3;++a)for(int i=0;i<n;++i)u[a][i]+=q*(k1[a][i]+2*k2[a][i]+2*k3[a][i]+k4[a][i])/6;t+=q;++steps;if(t>=target-1e-12){report(t);next+=.5;}}
  std::ofstream checkpoint(std::string(argv[2])+"-checkpoint.f64",std::ios::binary);
  for(const auto &field:u)checkpoint.write(reinterpret_cast<const char*>(field.data()),field.size()*sizeof(double));
  std::ofstream metadata(std::string(argv[2])+"-checkpoint.json");metadata<<std::setprecision(17)<<"{\"time\":"<<t<<",\"n\":"<<n<<",\"dtype\":\"float64\",\"layout\":\"field,rho\",\"fields\":[\"psi\",\"Pi\",\"q_rho\"]}\n";
  if(!checkpoint||!metadata)throw std::runtime_error("Independent scalar checkpoint write failed");
  std::cout<<"n="<<n<<" steps="<<steps<<" dt="<<dt<<" end="<<t<<"\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
