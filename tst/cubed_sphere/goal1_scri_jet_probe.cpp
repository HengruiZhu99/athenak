// Diagnostic Taylor algebra for the ACTUAL reference-wave-map endpoint poles.
// No evolution integrator or numerical endpoint closure is supplied here.
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
struct T {
  static constexpr int N=6; double c[N+1]{};
  T(double x=0){c[0]=x;}
  T&operator+=(const T&);T&operator-=(const T&);
};
T operator+(T a,const T&b){for(int i=0;i<=T::N;++i)a.c[i]+=b.c[i];return a;}
T operator-(T a){for(auto&v:a.c)v=-v;return a;}
T operator-(const T&a,const T&b){return a+(-b);}
T operator*(const T&a,const T&b){T r;for(int i=0;i<=T::N;++i)for(int j=0;j<=i;++j)r.c[i]+=a.c[j]*b.c[i-j];return r;}
T operator/(const T&a,const T&b){T r;for(int i=0;i<=T::N;++i){double x=a.c[i];for(int j=1;j<=i;++j)x-=b.c[j]*r.c[i-j];r.c[i]=x/b.c[0];}return r;}
T&T::operator+=(const T&b){return *this=*this+b;}T&T::operator-=(const T&b){return *this=*this-b;}
bool operator>(const T&a,const T&b){return a.c[0]>b.c[0];}
bool operator==(const T&a,const T&b){return a.c[0]==b.c[0];}
T D(const T&a){T b;for(int i=1;i<=T::N;++i)b.c[i-1]=i*a.c[i];return b;}
T powi(T a,int n){T r(1);for(int i=0;i<n;++i)r=r*a;return r;}
namespace Kokkos {inline bool isfinite(const T&a){for(auto x:a.c)if(!std::isfinite(x))return false;return true;}}
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace h=z4c::hyperboloidal;
void Scalar(h::ScalarJet<T>&s,T value){s.value=value;s.d[0]=D(value);s.dd[0][0]=D(D(value));}
void Vector(h::VectorJet<T>&s,T value){s.value[0]=value;s.d[0][0]=D(value);s.dd[0][0][0]=D(D(value));}
h::LayerPoint<T> Reference(T r,double a){
  h::LayerPoint<T> p{};p.radius=r;p.omega=(1-r*r)/(2*a);p.alpha=(1+r*r)/(2*a);
  p.L=p.alpha;p.b=r/a;p.k_physical=-3/a;p.k_bar=-3/(a*p.alpha);
  p.beta[0]=-r/a;p.dalpha[0]=r/a;p.domega[0]=-r/a;
  h::Z4cJet<T> axis{};Scalar(axis.chi,1);Scalar(axis.alpha,p.alpha);Scalar(axis.trace,p.k_physical);Vector(axis.beta,p.beta[0]);
  for(int i=0;i<3;++i){axis.metric.g[i][i]=1;p.omega_hessian[i][i]=-1/a;}
  T n[3]={1,0,0};p.state=h::CartesianRadialJet(axis,r,n);return p;
}
// Independently simplified outer-CMC expression, checked against the existing
// ReferenceConnection at positive Omega. It has a regular value at Omega=0.
rwm::Connection<T> Connection(T r,double a){
  const T alpha=(1+r*r)/(2*a),omega=(1-r*r)/(2*a);T x[3]={r,0,0};
  rwm::Connection<T> c{};c.valid=true;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){c.scaled[0][i][j]=(i==j?T(1)/(a*alpha):T(0));
    for(int k=0;k<3;++k)c.scaled[k+1][i][j]=(k==i?x[j]/a:T(0))+(k==j?x[i]/a:T(0))+(i==j?x[k]*omega/(a*alpha):T(0));}
  return c;
}
h::OmegaJet<T> Omega(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p){h::OmegaJet<T> o{};o.omega=p.omega;
  for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;
}
T Quotient(T pole,T omega,double &residue){residue=std::max(residue,std::abs(pole.c[0]));
  T s,o;for(int i=0;i<T::N;++i){s.c[i]=pole.c[i+1];o.c[i]=omega.c[i+1];}return s/o;
}
h::Z4cJet<T> Rates(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,T r,double a,double&residue){
  auto f=h::ConformalRHS(u,Omega(u,p),10/u.alpha.value,T(0));auto g=rwm::Gauge(p,u,Connection(r,a));
  if(!f.valid||!g.valid)throw std::runtime_error("invalid Taylor RHS parts");
  h::Z4cJet<T> axis{};auto F=[&](T regular,T pole){return regular+Quotient(pole,p.omega,residue);};
  Scalar(axis.chi,F(f.regular.chi,f.pole.chi));Scalar(axis.alpha,F(g.regular.alpha,g.pole.alpha));
  Scalar(axis.trace,F(f.regular.trace,f.pole.trace));Scalar(axis.theta,F(f.regular.theta,f.pole.theta));
  Vector(axis.beta,F(g.regular.beta[0],g.pole.beta[0]));Vector(axis.lambda,F(f.regular.lambda[0],f.pole.lambda[0]));
  for(int i=0;i<3;++i){T v=F(f.regular.metric[i][i],f.pole.metric[i][i]);axis.metric.g[i][i]=v;axis.metric.dg[0][i][i]=D(v);axis.metric.ddg[0][0][i][i]=D(D(v));
    v=F(f.regular.a[i][i],f.pole.a[i][i]);axis.a.k[i][i]=v;axis.a.dk[0][i][i]=D(v);}
  T n[3]={1,0,0};return h::CartesianRadialJet(axis,r,n);
}
h::Z4cJet<T> Shift(h::Z4cJet<T> u,const h::Z4cJet<T>&v,double step){
  for(auto pair:{std::make_pair(&u.chi,&v.chi),{&u.alpha,&v.alpha},{&u.trace,&v.trace},{&u.theta,&v.theta}}){
    pair.first->value+=step*pair.second->value;for(int i=0;i<3;++i){pair.first->d[i]+=step*pair.second->d[i];for(int j=0;j<3;++j)pair.first->dd[i][j]+=step*pair.second->dd[i][j];}}
  for(auto pair:{std::make_pair(&u.beta,&v.beta),{&u.lambda,&v.lambda}})for(int i=0;i<3;++i){pair.first->value[i]+=step*pair.second->value[i];for(int j=0;j<3;++j){pair.first->d[j][i]+=step*pair.second->d[j][i];for(int k=0;k<3;++k)pair.first->dd[j][k][i]+=step*pair.second->dd[j][k][i];}}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.metric.g[i][j]+=step*v.metric.g[i][j];u.a.k[i][j]+=step*v.a.k[i][j];for(int k=0;k<3;++k){u.metric.dg[k][i][j]+=step*v.metric.dg[k][i][j];u.a.dk[k][i][j]+=step*v.a.dk[k][i][j];for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j]+=step*v.metric.ddg[k][l][i][j];}}
  return u;
}
T Null(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p){auto inv=h::Geometry(u.metric);auto o=Omega(u,p);T n=-o.normal*o.normal;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)n+=u.chi.value*inv.inverse[i][j]*p.domega[i]*p.domega[j];return n;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 {T z;z.c[1]=1;T geometric=1/(1-z),identity=(1+z)/(1+z),quadratic=D(D((1+z)*(1+z)));double error=0;
  for(int i=0;i<=4;++i){error=std::max(error,std::abs(geometric.c[i]-1));error=std::max(error,std::abs(identity.c[i]-(i==0?1:0)));}
  error=std::max(error,std::abs(quadratic.c[0]-2));T r=1+z,omega=(1-r*r)/2,expected=1+r+r*r;double residue=0;T limit=Quotient(omega*expected,omega,residue);
  for(int i=0;i<=3;++i)error=std::max(error,std::abs(limit.c[i]-expected.c[i]));
  if(error>1e-12||residue>1e-12)throw std::runtime_error("Taylor primitive or removable quotient failure");
  std::cerr<<"Taylor primitive/quotient error "<<error<<"\n";}
 if(argc==2 && std::string(argv[1])=="--finite"){
  std::cout<<"a,sector,amplitude,delta,Omega,null_t,normalized_radial_rate,expected_radial_rate,H,Mmax\n";
  for(double a:{.5,1.,2.})for(int sector:{0,1,2})for(double amp:{.001,-.001})for(double delta:{.01,.005,.0025,.00125}){
   const double r=1-delta;h::LayerReference<double> ref(1.,a,{true,.05,.95});auto p=ref.At(r,0.,0.);auto u=p.state;
   const double op=-r/a,opp=-1/a,om=p.omega,pulse=amp*om*om,dp=2*amp*om*op,ddp=2*amp*(op*op+om*opp);
   if(sector==1){u.alpha.value+=pulse;u.alpha.d[0]+=dp;u.alpha.dd[0][0]+=ddp;u.alpha.dd[1][1]+=dp/r;u.alpha.dd[2][2]+=dp/r;}
   if(sector==2){h::Z4cJet<double> axis=u;axis.beta.value[0]+=pulse;axis.beta.d[0][0]+=dp;axis.beta.dd[0][0][0]+=ddp;double n[3]={1,0,0};u=h::CartesianRadialJet(axis,r,n);}
   h::OmegaJet<double> o{};o.omega=om;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
   h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);h::Z4cRHS<double> f{};h::GaugeRHS<double> g{};double x[3]={r,0,0};
   if(!h::AssembleInterior(h::ConformalRHS(u,o,10/u.alpha.value,0.),om,f)||!rwm::Assemble(rwm::Gauge(p,u,rwm::ReferenceConnection(p,x)),om,g))throw std::runtime_error("invalid native finite control");
   double wt=-op*g.beta[0]/u.alpha.value-o.normal*g.alpha/u.alpha.value;
   double nt=(f.chi-f.metric[0][0])*op*op-2*o.normal*wt;auto con=h::EvolvedConstraints(u,o);double mm=0;for(auto v:con.momentum)mm=std::max(mm,std::abs(v));
   std::cout<<a<<","<<sector<<","<<amp<<","<<delta<<","<<om<<","<<nt<<","<<nt/(-delta)<<","<<(sector?4*amp/std::pow(a,4):0)<<","<<con.hamiltonian<<","<<mm<<"\n";
  }
  Kokkos::finalize();return 0;
 }
 std::cout<<"a,sector,power,amplitude,pole0_max,H0,M0max,Z0max,Theta0,null0,null_dr,rate_poleA_nn,rate_poleLambda_n,null_t0,null_t_dr,fd_step,connection_error\n";
 for(double a:{.5,1.,2.})for(int sector:{0,1,2,3})for(int power:{2,3,4})for(double amp:{.001,-.001}){
  T r(1);r.c[1]=1;auto p=Reference(r,a);auto u=p.state;T pulse=amp*powi(p.omega,power);
  if(sector==1)Scalar(u.alpha,p.alpha+pulse);
  if(sector==2)Vector(u.beta,-r/a+pulse);
  if(sector==3)Scalar(u.trace,p.k_physical+pulse);
  T n[3]={1,0,0};u=h::CartesianRadialJet(u,r,n);double residue=0;auto v=Rates(u,p,r,a,residue);auto c=h::EvolvedConstraints(u,Omega(u,p));
  T rr(.98);auto pp=Reference(rr,a);T xx[3]={rr,0,0};auto native=rwm::ReferenceConnection(pp,xx),simplified=Connection(rr,a);double ce=0;
  for(int i=0;i<4;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)ce=std::max(ce,std::abs(native.scaled[i][j][k].c[0]-simplified.scaled[i][j][k].c[0]));
  for(double step:{1e-4,5e-5,2.5e-5}){
   auto plus=Shift(u,v,step),minus=Shift(u,v,-step);auto fp=h::ConformalRHS(plus,Omega(plus,p),10/plus.alpha.value,T(0)),fm=h::ConformalRHS(minus,Omega(minus,p),10/minus.alpha.value,T(0));
   T nt=(Null(plus,p)-Null(minus,p))/(2*step),nu=Null(u,p);double mm=0,zz=0;for(int i=0;i<3;++i){mm=std::max(mm,std::abs(c.momentum[i].c[0]));zz=std::max(zz,std::abs(c.z4.z_covector[i].c[0]));}
   std::cout<<a<<","<<sector<<","<<power<<","<<amp<<","<<residue<<","<<c.hamiltonian.c[0]<<","<<mm<<","<<zz<<","<<u.theta.value.c[0]<<","<<nu.c[0]<<","<<nu.c[1]<<","<<(fp.pole.a[0][0].c[0]-fm.pole.a[0][0].c[0])/(2*step)<<","<<(fp.pole.lambda[0].c[0]-fm.pole.lambda[0].c[0])/(2*step)<<","<<nt.c[0]<<","<<nt.c[1]<<","<<step<<","<<ce<<"\n";
  }
 }
 Kokkos::finalize();
}
