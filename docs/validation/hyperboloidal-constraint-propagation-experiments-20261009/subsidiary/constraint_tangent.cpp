// Exploratory continuum subsidiary audit. The dual derivative is exact;
// fourth-order finite differences differentiate only smooth operator coefficients.
#include <Kokkos_Core.hpp>
#include <array>
#include <complex>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
struct D {double v=0,d=0;D()=default;D(double x):v(x){}D(double x,double y):v(x),d(y){};
 D&operator+=(D x){v+=x.v;d+=x.d;return *this;}D&operator-=(D x){v-=x.v;d-=x.d;return *this;}
 D&operator*=(D x){d=d*x.v+v*x.d;v*=x.v;return *this;}D&operator/=(D x){d=(d*x.v-v*x.d)/(x.v*x.v);v/=x.v;return *this;}};
D operator+(D a,D b){return a+=b;}D operator-(D a,D b){return a-=b;}D operator*(D a,D b){return a*=b;}D operator/(D a,D b){return a/=b;}D operator-(D a){return {-a.v,-a.d};}
bool operator>(D a,D b){return a.v>b.v;}bool operator<(D a,D b){return a.v<b.v;}
namespace Kokkos {inline bool isfinite(D x){return std::isfinite(x.v)&&std::isfinite(x.d);}}
#include "z4c/hyperboloidal/layer_reference.hpp"
namespace hyp=z4c::hyperboloidal;
using Jet=hyp::Z4cJet<D>;using C=std::complex<double>;const C I(0,1);
using M=std::array<std::array<C,20>,20>;
struct Sample{M f{},q{};};
struct J {D v{},d[3]{},dd[3][3]{};J()=default;J(D x):v(x){}};
J operator+(const J&a,const J&b){J c(a.v+b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]+b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]+b.dd[i][j];}return c;}
J operator-(const J&a,const J&b){J c(a.v-b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]-b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]-b.dd[i][j];}return c;}
J operator*(const J&a,const J&b){J c(a.v*b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]*b.v+a.v*b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]*b.v+a.v*b.dd[i][j]+a.d[i]*b.d[j]+a.d[j]*b.d[i];}return c;}
J inv(const J&a){J b(D(1)/a.v);for(int i=0;i<3;++i){b.d[i]=-a.d[i]/(a.v*a.v);for(int j=0;j<3;++j)b.dd[i][j]=D(2)*a.d[i]*a.d[j]/(a.v*a.v*a.v)-a.dd[i][j]/(a.v*a.v);}return b;}
J operator/(const J&a,const J&b){return a*inv(b);}
J Metric(const Jet&u,int i,int j){J x(u.metric.g[i][j]);for(int d=0;d<3;++d){x.d[d]=u.metric.dg[d][i][j];for(int e=0;e<3;++e)x.dd[d][e]=u.metric.ddg[d][e][i][j];}return x;}
J A(const Jet&u,int i,int j){J x(u.a.k[i][j]);for(int d=0;d<3;++d)x.d[d]=u.a.dk[d][i][j];return x;}
void SetMetric(Jet&u,int i,int j,const J&x){u.metric.g[i][j]=u.metric.g[j][i]=x.v;for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=x.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=x.dd[d][e];}}
void SetA(Jet&u,int i,int j,const J&x){u.a.k[i][j]=u.a.k[j][i]=x.v;for(int d=0;d<3;++d)u.a.dk[d][i][j]=u.a.dk[d][j][i]=x.d[d];}
void Consistent(Jet&u){const J x=Metric(u,0,0),y=Metric(u,0,1),z=Metric(u,0,2),w=Metric(u,1,1),v=Metric(u,1,2);
 SetMetric(u,2,2,(J(D(1))-J(D(2))*y*z*v+x*v*v+w*z*z)/(x*w-y*y));const J t=Metric(u,2,2);
 const J ix=w*t-v*v,iy=z*v-y*t,iz=y*v-z*w,iw=x*t-z*z,iv=y*z-x*v,it=x*w-y*y;
 SetA(u,2,2,(J(D(0))-ix*A(u,0,0)-J(D(2))*iy*A(u,0,1)-J(D(2))*iz*A(u,0,2)-iw*A(u,1,1)-J(D(2))*iv*A(u,1,2))/it);}
void Scalar(hyp::ScalarJet<D>&u,const J&x){u.value+=x.v;for(int d=0;d<3;++d){u.d[d]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e]+=x.dd[d][e];}}
void Vector(hyp::VectorJet<D>&u,int i,const J&x){u.value[i]+=x.v;for(int d=0;d<3;++d){u.d[d][i]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e][i]+=x.dd[d][e];}}
void Seed(Jet&u,int c,const J&x){const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 if(c==0)Scalar(u.alpha,x);if(c==1)Scalar(u.chi,x);if(c==2)Scalar(u.trace,x);if(c==3)Scalar(u.theta,x);
 if(c>=4&&c<7)Vector(u.beta,c-4,x);if(c>=7&&c<12){const int i=ti[c-7],j=tj[c-7];SetMetric(u,i,j,Metric(u,i,j)+x);}
 if(c>=12&&c<17){const int i=ti[c-12],j=tj[c-12];SetA(u,i,j,A(u,i,j)+x);}if(c>=17)Vector(u.lambda,c-17,x);}
Jet Lift(const hyp::Z4cJet<double>&s){Jet u{};auto scalar=[](auto&a,const auto&b){a.value=b.value;for(int i=0;i<3;++i){a.d[i]=b.d[i];for(int j=0;j<3;++j)a.dd[i][j]=b.dd[i][j];}};
 scalar(u.alpha,s.alpha);scalar(u.chi,s.chi);scalar(u.trace,s.trace);scalar(u.theta,s.theta);
 for(int i=0;i<3;++i){u.beta.value[i]=s.beta.value[i];u.lambda.value[i]=s.lambda.value[i];for(int j=0;j<3;++j){u.beta.d[j][i]=s.beta.d[j][i];u.lambda.d[j][i]=s.lambda.d[j][i];u.metric.g[i][j]=s.metric.g[i][j];u.a.k[i][j]=s.a.k[i][j];for(int k=0;k<3;++k){u.beta.dd[k][j][i]=s.beta.dd[k][j][i];u.lambda.dd[k][j][i]=s.lambda.dd[k][j][i];u.metric.dg[k][i][j]=s.metric.dg[k][i][j];u.a.dk[k][i][j]=s.a.dk[k][i][j];for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j]=s.metric.ddg[k][l][i][j];}}}return u;}
hyp::OmegaJet<D> Omega(const Jet&u,const hyp::LayerPoint<double>&p){hyp::OmegaJet<D> o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}hyp::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;}
std::array<double,8> Constraints(const Jet&u,const hyp::LayerPoint<double>&p){auto c=hyp::EvolvedConstraints(u,Omega(u,p));if(!c.valid)throw std::runtime_error("constraints invalid");return {c.hamiltonian.d,c.momentum[0].d,c.momentum[1].d,c.momentum[2].d,c.z4.z_covector[0].d,c.z4.z_covector[1].d,c.z4.z_covector[2].d,c.z4.theta_physical.d};}
Sample Point(const hyp::LayerReference<double>&ref,const std::array<double,3>&x,double k,const std::array<double,3>&n,double kappa){const auto p=ref.At(x[0],x[1],x[2]);Sample s;
 for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase){auto u=Lift(p.state);J seed(D(0,phase?0:1));for(int i=0;i<3;++i){seed.d[i]=D(0,phase?k*n[i]:0);for(int j=0;j<3;++j)seed.dd[i][j]=D(0,phase?0:-k*k*n[i]*n[j]);}Seed(u,col,seed);Consistent(u);
  auto o=Omega(u,p);hyp::Z4cRHS<D> f;if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(kappa)/u.alpha.value,D(0)),o.omega,f))throw std::runtime_error("rhs invalid");
  double v[20]{};v[1]=f.chi.d;v[2]=f.trace.d;v[3]=f.theta.d;const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};for(int j=0;j<3;++j)v[17+j]=f.lambda[j].d;for(int j=0;j<5;++j){v[7+j]=f.metric[ti[j]][tj[j]].d;v[12+j]=f.a[ti[j]][tj[j]].d;}
  const C factor=phase?I:C(1,0);for(int row=0;row<20;++row)s.f[row][col]+=factor*v[row];auto q=Constraints(u,p);for(int row=0;row<8;++row)s.q[row][col]+=factor*q[row];
 }return s;}
#include "subsidiary.hpp"
void Print(const M&m,int rows){std::cout<<'[';for(int i=0;i<rows;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<'['<<m[i][j].real()<<','<<m[i][j].imag()<<']';}std::cout<<']';}std::cout<<']';}
int main(int argc,char**argv){std::cout<<std::setprecision(17)<<'[';bool first=true;const hyp::LayerReference<double> ref(1.,.5,{true,.05,.95});
 if(argc>2){const double kap=std::stod(argv[1]);for(double r:{.75,.85,.9,.95,.98})for(double k:{32.,64.,128.,256.,512.,1024.,2048.,4096.,8192.,16384.})for(bool oblique:{false,true}){const std::array<double,3> n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};auto p=ref.At(r,0.,0.);M g{};for(int col=0;col<8;++col){ConstraintJet q{};q.q[col]=1.;for(int i=0;i<3;++i){q.d[i][col]=I*k*n[i];for(int j=0;j<3;++j)q.dd[i][j][col]=-k*k*n[i]*n[j];}auto v=Subsidiary(p,kap,q);for(int row=0;row<8;++row)g[row][col]=v[row];}const auto gt=hyp::Geometry(p.state.metric);double bn=0,gnn=0;for(int i=0;i<3;++i){bn+=p.beta[i]*n[i];for(int j=0;j<3;++j)gnn+=gt.inverse[i][j]*n[i]*n[j];}if(!first)std::cout<<',';first=false;std::cout<<"{\"kappa\":"<<kap<<",\"r\":"<<r<<",\"k\":"<<k<<",\"oblique\":"<<oblique<<",\"beta_n\":"<<bn<<",\"light_speed\":"<<p.alpha*std::sqrt(p.state.chi.value*gnn)<<",\"G\":";Print(g,8);std::cout<<'}';}std::cout<<"]\n";return 0;}
 for(double r:{.3,.5,.75,.85,.9,.95,.98})for(double k:{0.,1.,2.,4.,8.,16.,32.,64.,128.,256.})for(bool oblique:{false,true})for(double h:{.002,.001,.0005,.00025,.000125}){
 const std::array<double,3>x={r,0,0},n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};const double kappa=argc>1?std::stod(argv[1]):10;const auto p=ref.At(x[0],x[1],x[2]);const auto base=Point(ref,x,k,n,kappa);M df[3]{},ddf[3][3]{},dq[3]{},ddq[3][3]{};
 const int offsets[4]={-2,-1,1,2};const double dc[4]={1.,-8.,8.,-1.};
 for(int axis=0;axis<3;++axis){Sample side[4];for(int a=0;a<4;++a){auto y=x;y[axis]+=offsets[a]*h;side[a]=Point(ref,y,k,n,kappa);}
  for(int row=0;row<20;++row)for(int col=0;col<20;++col){for(int a=0;a<4;++a){df[axis][row][col]+=dc[a]*side[a].f[row][col]/(12*h);dq[axis][row][col]+=dc[a]*side[a].q[row][col]/(12*h);}
   ddq[axis][axis][row][col]=(-side[3].q[row][col]+16.*side[2].q[row][col]-30.*base.q[row][col]+16.*side[1].q[row][col]-side[0].q[row][col])/(12*h*h);
   ddf[axis][axis][row][col]=(-side[3].f[row][col]+16.*side[2].f[row][col]-30.*base.f[row][col]+16.*side[1].f[row][col]-side[0].f[row][col])/(12*h*h);}}
 for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)for(int a=0;a<4;++a)for(int b=0;b<4;++b){auto y=x;y[i]+=offsets[a]*h;y[j]+=offsets[b]*h;auto v=Point(ref,y,k,n,kappa);for(int row=0;row<20;++row)for(int col=0;col<20;++col){ddf[i][j][row][col]+=dc[a]*dc[b]*v.f[row][col]/(144*h*h);ddq[i][j][row][col]+=dc[a]*dc[b]*v.q[row][col]/(144*h*h);}}
 M exact{},frozen{},prediction{},subsidiary{},constraint_generator{};const auto gt=hyp::Geometry(p.state.metric);hyp::OmegaJet<D> op=Omega(Lift(p.state),p);
 for(int col=0;col<20;++col){for(int phase=0;phase<2;++phase){Jet ue=Lift(p.state),uf=Lift(p.state);auto part=[phase](C z){return phase?z.imag():z.real();};
  for(int row=0;row<20;++row){J e(D(0,part(base.f[row][col]))),f=e;for(int i=0;i<3;++i){e.d[i]=D(0,part(df[i][row][col]+I*k*n[i]*base.f[row][col]));f.d[i]=D(0,part(I*k*n[i]*base.f[row][col]));for(int j=0;j<3;++j){const C dd=i<=j?ddf[i][j][row][col]:ddf[j][i][row][col];e.dd[i][j]=D(0,part(dd+I*k*(n[i]*df[j][row][col]+n[j]*df[i][row][col])-k*k*n[i]*n[j]*base.f[row][col]));f.dd[i][j]=D(0,part(-k*k*n[i]*n[j]*base.f[row][col]));}}Seed(ue,row,e);Seed(uf,row,f);}Consistent(ue);Consistent(uf);auto ce=Constraints(ue,p),cf=Constraints(uf,p);for(int row=0;row<8;++row){exact[row][col]+=(phase?I:C(1))*ce[row];frozen[row][col]+=(phase?I:C(1))*cf[row];}}
  ConstraintJet cj{};for(int row=0;row<8;++row){cj.q[row]=base.q[row][col];for(int a=0;a<3;++a){cj.d[a][row]=dq[a][row][col]+I*k*n[a]*base.q[row][col];for(int b=0;b<3;++b){const C dd=a<=b?ddq[a][b][row][col]:ddq[b][a][row][col];cj.dd[a][b][row]=dd+I*k*(n[a]*dq[b][row][col]+n[b]*dq[a][row][col])-k*k*n[a]*n[b]*base.q[row][col];}}}auto pred=Subsidiary(p,kappa,cj);for(int row=0;row<8;++row)subsidiary[row][col]=pred[row];
  C grad[3][8]{};for(int i=0;i<3;++i)for(int row=0;row<8;++row)grad[i][row]=dq[i][row][col]+I*k*n[i]*base.q[row][col];
  C th=p.alpha*base.q[0][col]/(2*p.omega)-(3*p.alpha*op.normal.v+2*kappa)*base.q[7][col]/p.omega;
  for(int i=0;i<3;++i){th+=p.beta[i]*grad[i][7];for(int j=0;j<3;++j){C cov=grad[i][4+j];for(int l=0;l<3;++l)cov-=gt.connection[l][i][j]*base.q[4+l][col];th+=p.alpha*p.omega*p.state.chi.value*gt.inverse[i][j]*cov;}}
  prediction[7][col]=th;
  for(int i=0;i<3;++i){C z=p.alpha*(base.q[1+i][col]+grad[i][7])/p.omega-p.alpha*(2*p.state.trace.value/3+kappa/p.alpha)*base.q[4+i][col]/p.omega;
   for(int j=0;j<3;++j){z+=p.beta[j]*grad[j][4+i]+p.state.beta.d[i][j]*base.q[4+j][col];for(int l=0;l<3;++l){for(int m=0;m<3;++m)z+=(p.state.metric.g[i][j]*p.state.beta.d[l][j]-2*p.alpha*p.state.a.k[i][l]*(j==l))*gt.inverse[l][m]*base.q[4+m][col];}}
   prediction[4+i][col]=z;}
 }
 for(int col=0;col<8;++col){ConstraintJet cj{};cj.q[col]=1.;for(int a=0;a<3;++a){cj.d[a][col]=I*k*n[a];for(int b=0;b<3;++b)cj.dd[a][b][col]=-k*k*n[a]*n[b];}auto v=Subsidiary(p,kappa,cj);for(int row=0;row<8;++row)constraint_generator[row][col]=v[row];}
 if(!first)std::cout<<',';first=false;std::cout<<"{\"kappa\":"<<kappa<<",\"r\":"<<r<<",\"k\":"<<k<<",\"h\":"<<h<<",\"oblique\":"<<oblique<<",\"Q\":";Print(base.q,8);std::cout<<",\"D\":";Print(exact,8);std::cout<<",\"frozen\":";Print(frozen,8);std::cout<<",\"predicted_Theta_Z\":";Print(prediction,8);std::cout<<",\"subsidiary\":";Print(subsidiary,8);std::cout<<",\"constraint_generator\":";Print(constraint_generator,8);std::cout<<'}';
 }std::cout<<"]\n";
}
