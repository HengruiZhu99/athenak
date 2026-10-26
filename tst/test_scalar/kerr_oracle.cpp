// Independent 2+1 m=2 normal-momentum massless scalar. No primary RHS/provider.
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using C=std::complex<double>;using V=std::vector<C>;using State=std::array<V,3>;
int main(int argc,char **argv){try{
 if(argc<4)throw std::runtime_error("table output duration [dt factor]");std::ifstream input(argv[1]);int nr,na;double h;input>>nr>>na>>h;
 auto number=[&](){std::string z;input>>z;char *end;double v=std::strtod(z.c_str(),&end);if(z.empty()||*end||!std::isfinite(v))throw std::runtime_error("invalid table");return v;};
 std::vector<double> z(na),weight(na),L(na*na);for(double &x:z)x=number();for(double &x:weight)x=number();for(double &x:L)x=number();int n=nr*na;
 std::vector<std::array<double,17>> g(n);State u{V(n),V(n),V(n)},tmp=u,k1=u,k2=u,k3=u,k4=u;double dt=1e99;C im(0,2);
 for(int i=0;i<n;++i){for(double &x:g[i])x=number();u[0][i]=C(0,-.25*g[i][15]);u[2][i]=C(0,-.25*g[i][16]);u[1][i]=(g[i][2]*u[2][i]+im*g[i][3]*u[0][i])/g[i][1];double speed=std::abs(g[i][2])+g[i][1]*std::sqrt(g[i][4]);dt=std::min(dt,.16*h/speed);}
 if(argc==5)dt*=std::stod(argv[4]);
 auto diff=[&](const V &f,int i,int j,double speed){int p=i*na+j;
  if(speed>=0){if(i==0)return (f[p+na]-f[p])/h;if(i<3)return (f[p]-f[p-na])/h;if(i==nr-1)return (25.*f[p]/12.-4.*f[p-na]+3.*f[p-2*na]-4.*f[p-3*na]/3.+f[p-4*na]/4.)/h;return (-f[p-3*na]/12.+f[p-2*na]/2.-3.*f[p-na]/2.+5.*f[p]/6.+f[p+na]/4.)/h;}
  if(i==nr-1)return (f[p]-f[p-na])/h;if(i>nr-4)return (f[p+na]-f[p])/h;if(i==0)return (-25.*f[p]/12.+4.*f[p+na]-3.*f[p+2*na]+4.*f[p+3*na]/3.-f[p+4*na]/4.)/h;return (-f[p-na]/4.-5.*f[p]/6.+3.*f[p+na]/2.-f[p+2*na]/2.+f[p+3*na]/12.)/h;};
 auto centered=[&](const V &f,int i,int j){int p=i*na+j;if(i<2)return diff(f,i,j,-1);if(i>nr-3)return diff(f,i,j,1);return (f[p-2*na]-8.*f[p-na]+8.*f[p+na]-f[p+2*na])/(12*h);};
 auto rhs=[&](const State &y,State &r){
  for(int i=0;i<nr;++i)for(int j=0;j<na;++j){int p=i*na+j;const auto &a=g[p];double alpha=a[1],br=a[2],bp=a[3],s=std::sqrt(a[4]),cp=-br+alpha*s,cm=-br-alpha*s;
   C plus=diff(y[1],i,j,cp)+s*diff(y[2],i,j,cp),minus=diff(y[1],i,j,cm)-s*diff(y[2],i,j,cm),angular=0;
   for(int k=0;k<na;++k)angular+=L[j*na+k]*y[0][i*na+k];
   r[0][p]=br*diff(y[0],i,j,-br)+im*bp*y[0][p]-alpha*y[1][p];
   r[1][p]=-.5*(cp*plus+cm*minus)+im*bp*y[1][p]+a[8]*y[1][p]+a[9]*y[2][p]-2.*im*alpha*a[10]*y[2][p]+im*a[11]*y[0][p]-alpha*a[13]*angular+alpha*(4*a[14]+a[12])*y[0][p];
   r[2][p]=-.5*(cp*plus-cm*minus)/s+im*bp*y[2][p]+a[6]*y[2][p]+im*a[7]*y[0][p]-a[5]*y[1][p]-2.*(y[2][p]-centered(y[0],i,j));
  }
 };
 std::ofstream out(argv[2]);out<<std::setprecision(17)<<"time,real22,imag22\n";
 auto report=[&](double t){int start=std::clamp(int((10-g[0][0])/h)-2,0,nr-6);C value=0;double normalization=2*acos(-1.)*std::sqrt(15/(32*acos(-1.)));
  for(int j=0;j<na;++j){C f=0;for(int a=0;a<6;++a){double w=1;for(int b=0;b<6;++b)if(a!=b)w*=(10-g[(start+b)*na+j][0])/(g[(start+a)*na+j][0]-g[(start+b)*na+j][0]);f+=w*u[0][(start+a)*na+j];}value+=normalization*weight[j]*f;}
  out<<t<<","<<value.real()<<","<<value.imag()<<"\n";out.flush();
  if(std::abs(t-2)<1e-9||std::abs(t-6)<1e-9||std::abs(t-12)<1e-9||std::abs(t-35)<1e-9||std::abs(t-65)<1e-9||std::abs(t-100)<1e-9){std::ofstream p(std::string(argv[2])+"-profile-"+std::to_string(int(t))+".csv");p<<std::setprecision(17)<<"rho,z,W_real,W_imag,v_real,v_imag,Dr_real,Dr_imag\n";for(int i=0;i<nr;++i)for(int j=0;j<na;++j){int k=i*na+j;C v=g[k][2]*diff(u[0],i,j,-g[k][2])+im*g[k][3]*u[0][k]-g[k][1]*u[1][k],d=centered(u[0],i,j);p<<g[k][0]<<","<<z[j]<<","<<u[0][k].real()<<","<<u[0][k].imag()<<","<<v.real()<<","<<v.imag()<<","<<d.real()<<","<<d.imag()<<"\n";}}
  if(!std::isfinite(std::abs(value))||std::abs(value)>10)throw std::runtime_error("Kerr oracle amplitude guard");};
 double t=0,end=std::stod(argv[3]),next=.5;int steps=0;report(t);
 while(t<end-1e-12){double target=std::min(next,end),q=std::min(dt,target-t);rhs(u,k1);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k1[a][i]/2.;rhs(tmp,k2);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k2[a][i]/2.;rhs(tmp,k3);for(int a=0;a<3;++a)for(int i=0;i<n;++i)tmp[a][i]=u[a][i]+q*k3[a][i];rhs(tmp,k4);for(int a=0;a<3;++a)for(int i=0;i<n;++i)u[a][i]+=q*(k1[a][i]+2.*k2[a][i]+2.*k3[a][i]+k4[a][i])/6.;t+=q;++steps;if(t>=target-1e-12){report(t);next+=.5;}}
 // Preserve all complex scalar auxiliaries, rather than only derived profiles.
 std::ofstream checkpoint(std::string(argv[2])+"-checkpoint.c128",std::ios::binary);
 for(const auto &field:u)checkpoint.write(reinterpret_cast<const char*>(field.data()),field.size()*sizeof(C));
 std::ofstream metadata(std::string(argv[2])+"-checkpoint.json");metadata<<std::setprecision(17)<<"{\"time\":"<<t<<",\"nr\":"<<nr<<",\"na\":"<<na<<",\"dtype\":\"native_complex128\",\"layout\":\"field,rho,z\",\"fields\":[\"W\",\"Pi\",\"q_rho\"]}\n";
 if(!checkpoint||!metadata)throw std::runtime_error("Kerr oracle checkpoint write failed");
 std::cout<<"nr="<<nr<<" na="<<na<<" steps="<<steps<<" dt="<<dt<<" end="<<t<<"\n";
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
