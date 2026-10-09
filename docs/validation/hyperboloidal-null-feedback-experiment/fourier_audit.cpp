#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "null_feedback.hpp"
namespace hyp=z4c::hyperboloidal;
using Jet=hyp::Z4cJet<double>;
struct J { double v=0,d[3]{},dd[3][3]{}; J()=default; J(double x):v(x){} };
J operator+(const J&a,const J&b){J c(a.v+b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]+b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]+b.dd[i][j];}return c;}
J operator-(const J&a,const J&b){J c(a.v-b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]-b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]-b.dd[i][j];}return c;}
J operator*(const J&a,const J&b){J c(a.v*b.v);for(int i=0;i<3;++i){c.d[i]=a.d[i]*b.v+a.v*b.d[i];for(int j=0;j<3;++j)c.dd[i][j]=a.dd[i][j]*b.v+a.v*b.dd[i][j]+a.d[i]*b.d[j]+a.d[j]*b.d[i];}return c;}
J inverse(const J&a){J b(1/a.v);for(int i=0;i<3;++i){b.d[i]=-a.d[i]/(a.v*a.v);for(int j=0;j<3;++j)b.dd[i][j]=2*a.d[i]*a.d[j]/(a.v*a.v*a.v)-a.dd[i][j]/(a.v*a.v);}return b;}
J operator/(const J&a,const J&b){return a*inverse(b);}
J Metric(const Jet&u,int i,int j){J x(u.metric.g[i][j]);for(int d=0;d<3;++d){x.d[d]=u.metric.dg[d][i][j];for(int e=0;e<3;++e)x.dd[d][e]=u.metric.ddg[d][e][i][j];}return x;}
J A(const Jet&u,int i,int j){J x(u.a.k[i][j]);for(int d=0;d<3;++d)x.d[d]=u.a.dk[d][i][j];return x;}
void SetMetric(Jet&u,int i,int j,const J&x){u.metric.g[i][j]=u.metric.g[j][i]=x.v;for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=x.d[d];for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=x.dd[d][e];}}
void SetA(Jet&u,int i,int j,const J&x){u.a.k[i][j]=u.a.k[j][i]=x.v;for(int d=0;d<3;++d)u.a.dk[d][i][j]=u.a.dk[d][j][i]=x.d[d];}
// Five free metric and A components; reconstruct both algebraic constraints
// including their derivative jets before evaluating the actual tensor kernel.
void Consistent(Jet&u){
  const J x=Metric(u,0,0),y=Metric(u,0,1),z=Metric(u,0,2),w=Metric(u,1,1),v=Metric(u,1,2);
  SetMetric(u,2,2,(J(1)-J(2)*y*z*v+x*v*v+w*z*z)/(x*w-y*y));
  const J t=Metric(u,2,2);
  const J ix=w*t-v*v,iy=z*v-y*t,iz=y*v-z*w,iw=x*t-z*z,iv=y*z-x*v,it=x*w-y*y;
  SetA(u,2,2,(J(0)-ix*A(u,0,0)-J(2)*iy*A(u,0,1)-J(2)*iz*A(u,0,2)-iw*A(u,1,1)-J(2)*iv*A(u,1,2))/it);
}
J Phase(double amplitude,double k,const std::array<double,3>&n,bool sine){J x(sine?0:amplitude);for(int i=0;i<3;++i){x.d[i]=sine?amplitude*k*n[i]:0;for(int j=0;j<3;++j)x.dd[i][j]=sine?0:-amplitude*k*k*n[i]*n[j];}return x;}
void Scalar(hyp::ScalarJet<double>&u,const J&x){u.value+=x.v;for(int d=0;d<3;++d){u.d[d]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e]+=x.dd[d][e];}}
void Vector(hyp::VectorJet<double>&u,int i,const J&x){u.value[i]+=x.v;for(int d=0;d<3;++d){u.d[d][i]+=x.d[d];for(int e=0;e<3;++e)u.dd[d][e][i]+=x.dd[d][e];}}
void Perturb(Jet&u,int c,double amplitude,double k,const std::array<double,3>&n,bool sine){
  const J x=Phase(amplitude,k,n,sine);const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
  if(c==0)Scalar(u.alpha,x);if(c==1)Scalar(u.chi,x);if(c==2)Scalar(u.trace,x);if(c==3)Scalar(u.theta,x);
  if(c>=4&&c<7)Vector(u.beta,c-4,x);
  if(c>=7&&c<12){const int i=ti[c-7],j=tj[c-7];SetMetric(u,i,j,Metric(u,i,j)+x);}
  if(c>=12&&c<17){const int i=ti[c-12],j=tj[c-12];SetA(u,i,j,A(u,i,j)+x);}
  if(c>=17)Vector(u.lambda,c-17,x);
  Consistent(u);
}
void Evaluate(const hyp::LayerPoint<double>&p,const Jet&u,double kappa,bool candidate,double out[20]){
  const auto o=hyp::CartesianOmega(u,p);hyp::Z4cRHS<double> rhs{};
  if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,kappa/u.alpha.value,0.),o.omega,rhs))throw std::runtime_error("invalid Fourier geometry");
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
  hyp::GaugeRHS<double> gauge{};
  if(candidate)gauge=research::Assemble(research::Gauge(p,u,g),p.omega);
  else if(!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,g),p.omega,gauge))throw std::runtime_error("invalid Fourier gauge");
  out[0]=gauge.alpha;out[1]=rhs.chi;out[2]=rhs.trace;out[3]=rhs.theta;
  for(int i=0;i<3;++i){out[4+i]=gauge.beta[i];out[17+i]=rhs.lambda[i];}
  const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
  for(int i=0;i<5;++i){out[7+i]=rhs.metric[ti[i]][tj[i]];out[12+i]=rhs.a[ti[i]][tj[i]];}
}
void Matrix(const hyp::LayerPoint<double>&p,double k,double kappa,bool candidate,const std::array<double,3>&n,double eps,double b[20][20],double c[20][20],double (*hb)[20]=nullptr,double (*hc)[20]=nullptr){
  for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase){double v[2][20],constraint[2][9];for(int sign=0;sign<2;++sign){auto u=p.state;Perturb(u,col,sign?eps:-eps,k,n,phase);Evaluate(p,u,kappa,candidate,v[sign]);
      if(hb){const auto e=hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,p));if(!e.valid)throw std::runtime_error("invalid Fourier constraints");constraint[sign][0]=e.hamiltonian;constraint[sign][7]=e.z4.theta_physical;constraint[sign][8]=e.null_residual;for(int i=0;i<3;++i){constraint[sign][1+i]=e.momentum[i];constraint[sign][4+i]=e.z4.z_covector[i];}}
    }for(int row=0;row<20;++row)(phase?c:b)[row][col]=(v[1][row]-v[0][row])/(2*eps);if(hb)for(int row=0;row<9;++row)(phase?hc:hb)[row][col]=(constraint[1][row]-constraint[0][row])/(2*eps);
  }
}
void Print(const double m[][20],int rows=20){std::cout<<'[';for(int i=0;i<rows;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<m[i][j];}std::cout<<']';}std::cout<<']';}
int main(int argc,char**argv){
  if(argc>1&&std::string(argv[1])=="--modes"){
    std::cout<<std::setprecision(17)<<'[';bool first=true;
    for(bool wide:{false,true}){const hyp::LayerReference<double> ref(1.,.5,{true,wide?.05:.2,wide?.95:.8});
    for(double r:{.75,.85,.9,.95,.98})for(double kappa:{5.,10.})for(bool candidate:{false,true})for(bool oblique:{false,true})
    for(double k:{32.,64.,128.,256.})for(double eps:{1e-6,5e-7}){
      const auto p=ref.At(r,0.,0.);const std::array<double,3> n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};
      double b[20][20]{},c[20][20]{},hb[9][20]{},hc[9][20]{};Matrix(p,k,kappa,candidate,n,eps,b,c,hb,hc);
      const auto g=hyp::Geometry(p.state.metric);double bn=0,gnn=0;for(int i=0;i<3;++i){bn+=p.beta[i]*n[i];for(int j=0;j<3;++j)gnn+=g.inverse[i][j]*n[i]*n[j];}
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"wide\":"<<wide<<",\"r\":"<<r<<",\"omega\":"<<p.omega<<",\"kappa\":"<<kappa<<",\"candidate\":"<<candidate<<",\"oblique\":"<<oblique<<",\"k\":"<<k<<",\"eps\":"<<eps<<",\"beta_n\":"<<bn<<",\"light_speed\":"<<p.alpha*std::sqrt(p.state.chi.value*gnn)<<",\"B\":";Print(b);std::cout<<",\"C\":";Print(c);std::cout<<",\"HB\":";Print(hb,9);std::cout<<",\"HC\":";Print(hc,9);std::cout<<'}';
    }}std::cout<<"]\n";return 0;
  }
  if(argc>1&&std::string(argv[1])=="--k0-pole"){
    const hyp::LayerReference<double> ref(1.,.5,{true,.2,.8});std::cout<<std::setprecision(17)<<'[';bool first=true;
    for(double r:{.999,.9999,.99999,.999999,.9999999})for(double kappa:{5.,10.})for(bool candidate:{false,true}){
      const auto p=ref.At(r,0.,0.);double b[20][20]{},c[20][20]{};Matrix(p,0.,kappa,candidate,{1,0,0},1e-6,b,c);
      if(!first)std::cout<<',';first=false;std::cout<<"{\"r\":"<<r<<",\"omega\":"<<p.omega<<",\"kappa\":"<<kappa<<",\"candidate\":"<<candidate<<",\"B\":";Print(b);std::cout<<'}';
    }std::cout<<"]\n";return 0;
  }
  std::cout<<std::setprecision(17)<<'[';bool first=true;
  for(bool wide:{false,true}){const hyp::LayerReference<double> ref(1.,.5,{true,wide?.05:.2,wide?.95:.8});
  for(double r:{.75,.85,.9,.95,.98})for(double kappa:{5.,10.})for(bool candidate:{false,true})
  for(bool oblique:{false,true})for(double k:{0.,1.,2.,4.,8.,16.,32.,64.})for(double eps:{2e-6,1e-6,5e-7}){
    const auto p=ref.At(r,0.,0.);const std::array<double,3> n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};double b[20][20]{},c[20][20]{};Matrix(p,k,kappa,candidate,n,eps,b,c);
    if(!first)std::cout<<',';first=false;std::cout<<"{\"wide\":"<<wide<<",\"r\":"<<r<<",\"omega\":"<<p.omega<<",\"kappa\":"<<kappa<<",\"candidate\":"<<candidate<<",\"oblique\":"<<oblique<<",\"k\":"<<k<<",\"eps\":"<<eps<<",\"B\":";Print(b);std::cout<<",\"C\":";Print(c);std::cout<<'}';
  }}std::cout<<"]\n";
}
