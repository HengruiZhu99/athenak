#include "athena.hpp"
#include "z4c/hyperboloidal/reference_variables.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
namespace h=z4c::hyperboloidal;
struct Term {double c;int n[3];};
using Poly=std::vector<Term>;
Poly product(const Poly&a,const Poly&b){Poly c;for(auto u:a)for(auto v:b)c.push_back({u.c*v.c,{u.n[0]+v.n[0],u.n[1]+v.n[1],u.n[2]+v.n[2]}});return c;}
double at(const Poly&p,const double*x,int i=-1,int j=-1){double sum=0;for(auto t:p){
 if(i>=0){t.c*=t.n[i];--t.n[i];}if(j>=0){t.c*=t.n[j];--t.n[j];}
 if(t.c==0)continue;double v=t.c;for(int a=0;a<3;++a)v*=std::pow(x[a],t.n[a]);sum+=v;}return sum;}
h::ScalarJet<double> jet(const Poly&p,const double*x){h::ScalarJet<double>u{};u.value=at(p,x);
 for(int i=0;i<3;++i){u.d[i]=at(p,x,i);for(int j=0;j<3;++j)u.dd[i][j]=at(p,x,i,j);}return u;}
double worst=0;
void near(double a,double b){const double err=std::abs(a-b)/(1+std::abs(b));worst=std::max(worst,err);if(err>3e-12)throw std::runtime_error("reference Cartesian polynomial mismatch");}
void compare(const h::ScalarJet<double>&a,const h::ScalarJet<double>&b){near(a.value,b.value);for(int i=0;i<3;++i){near(a.d[i],b.d[i]);for(int j=0;j<3;++j)near(a.dd[i][j],b.dd[i][j]);}}
int main(){try{
 Poly f2{{1,{2,0,0}},{1,{0,2,0}},{1,{0,0,2}}};
 Poly u{{.9,{0,0,0}},{.2,{1,0,0}},{-.1,{0,1,0}},{.04,{0,1,1}},{.03,{2,0,0}}};
 int cases=0;for(int power:{2,4})for(int j=1;j<=99;++j){const double x[3]={j/100.,-.19,.27};
  const auto f=power==2?f2:product(f2,f2);const auto expected=jet(product(f,u),x);
  const auto ref=h::PunctureReference(x,power);compare(ref,jet(f,x));
  const auto got=h::ReferenceProduct(jet(u,x),ref);compare(got,expected);
  compare(h::ReferenceQuotient(expected,ref),jet(u,x));++cases;
 }
 std::cout<<"{\"passed\":true,\"nonspherical_polynomial_cases\":"<<cases<<",\"max_scaled_error\":"<<worst<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
