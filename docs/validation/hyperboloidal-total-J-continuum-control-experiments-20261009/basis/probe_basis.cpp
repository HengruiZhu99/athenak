#include "reference_conversion.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
using J=totalj::Jet<double>; using M=totalj::MatrixJet<double>;
J X(int i,const std::array<double,3>&x) { J v; v.value=x[i];v.d[i]=1;return v; }
J Pow(const J&a,double exponent) { J b; b.value=std::pow(a.value,exponent);
  const double p=exponent*std::pow(a.value,exponent-1),q=exponent*(exponent-1)*std::pow(a.value,exponent-2);
  for(int i=0;i<3;++i) { b.d[i]=p*a.d[i]; for(int k=0;k<3;++k) b.dd[i][k]=q*a.d[i]*a.d[k]+p*a.dd[i][k]; }return b; }
void Print(const J&j) { std::cout<<'['<<j.value;for(auto v:j.d)std::cout<<','<<v;for(const auto&r:j.dd)for(auto v:r)std::cout<<','<<v;std::cout<<']'; }
void Print(const M&m) { std::cout<<'[';for(int i=0;i<3;++i)for(int j=0;j<3;++j) {if(i||j)std::cout<<',';Print(m[i][j]);}std::cout<<']'; }
int main() {
  const std::array<std::array<double,3>,4> points={{{0,0,0},{.125,-.25,.375},{.36,-.48,.8},{-.2,.3,-.4}}};
  std::cout<<std::setprecision(17)<<"{\"basis_values\":[";bool first=true;
  for(const auto&r:totalj::records) for(int p=0;p<4;++p) for(int q=0;q<4;++q) {
    const auto&x=points[p];const double rho=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];totalj::WJet<double>w;
    if(q==0)w={1,0,0};if(q==1)w={rho,1,0};if(q==2)w={rho*rho,2*rho,2};if(q==3)w={1+rho+rho*rho,1+2*rho,2};
    const auto v=totalj::EvaluateBasis(r.J,r.spin,r.L,x,w);
    if(!first)std::cout<<',';first=false;std::cout<<"{\"J\":"<<r.J<<",\"spin\":"<<r.spin<<",\"L\":"<<r.L<<",\"point\":"<<p<<",\"W\":"<<q<<",\"jets\":[";
    for(int c=0;c<v.components;++c) {if(c)std::cout<<',';Print(v.component[c]);}std::cout<<"]}";
  }
  std::cout<<"],\"conversions\":[";first=true;
  for(int p=0;p<4;++p) {
    const auto&x=points[p];const J xx=X(0,x),yy=X(1,x),zz=X(2,x),one=totalj::Constant(1.0);
    M bar{},delta{},rawA{},independent{};J det=one;
    const J cs[3]={xx,yy,zz};for(int i=0;i<3;++i) {bar[i][i]=totalj::Add(one,totalj::Multiply(cs[i],cs[i]));det=totalj::Multiply(det,bar[i][i]);}
    const J chi=Pow(det,-1.0/3);
    delta[0][0]=totalj::Add(xx,yy);delta[1][1]=totalj::Add(zz,totalj::Multiply(xx,yy));delta[2][2]=totalj::Add(one,totalj::Multiply(zz,zz));
    delta[0][1]=delta[1][0]=totalj::Multiply(xx,zz);delta[0][2]=delta[2][0]=totalj::Add(yy,zz);delta[1][2]=delta[2][1]=totalj::Multiply(yy,yy);
    independent[0][0]=xx;independent[1][1]=yy;independent[2][2]=totalj::Scale(totalj::Add(xx,yy),-1.0);
    independent[0][1]=independent[1][0]=zz;independent[0][2]=independent[2][0]=totalj::Multiply(xx,yy);independent[1][2]=independent[2][1]=totalj::Multiply(yy,zz);
    const double constants[3][3]={{1,.2,.3},{.2,-.4,.5},{.3,.5,-.6}};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)rawA[i][j]=totalj::Constant(constants[i][j]);
    const auto metric=totalj::ConvertMetric(bar,chi,delta);
    const auto tr=totalj::Contract(metric.reference_inverse,rawA);
    M Aref{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)Aref[i][j]=totalj::Sub(rawA[i][j],totalj::Scale(totalj::Multiply(metric.reference_metric[i][j],tr),1.0/3));
    const auto da=totalj::ConvertA(metric,Aref,independent);
    if(!first)std::cout<<',';first=false;std::cout<<"{\"point\":"<<p<<",\"delta_chi\":";Print(metric.chi);std::cout<<",\"delta_g\":";Print(metric.metric);std::cout<<",\"delta_A\":";Print(da);std::cout<<'}';
  }
  std::cout<<"]}\n";
}
