// Held-out raw Cartesian polynomials; no CG seed or radial grid.
#define main FrozenAngularBridgeMain
#include "../total-j-local-angular-20261009/bridge.cpp"
#undef main
#include "inputs/flat_formula.hpp"
TJ Monomial(const X3&x,const std::array<int,3>&power) {
  auto evaluate=[&](std::array<int,3> order) {double value=1;
    for(int i=0;i<3;++i) {if(order[i]>power[i])return 0.;
      for(int k=0;k<order[i];++k)value*=power[i]-k;
      for(int k=0;k<power[i]-order[i];++k)value*=x[i];}
    return value;};
  TJ out{};out.value=evaluate({0,0,0});
  for(int i=0;i<3;++i) {std::array<int,3> d{};++d[i];out.d[i]=evaluate(d);
    for(int j=0;j<3;++j) {++d[j];out.dd[i][j]=evaluate(d);--d[j];}}
  return out;
}
Physical CartesianColumn(int column,const TJ&v) {
  Physical p{};
  if(column==0)p.alpha=v;
  if(column==1)for(int i=0;i<3;++i)p.h[i][i]=Times(v,1/std::sqrt(3.));
  if(column==2)p.P=v;
  if(column==3)p.theta=v;
  if(column>=4&&column<7)p.beta[column-4]=v;
  if(column>=7&&column<10)p.lambda[column-7]=v;
  if(column>=10) {
    const int c=(column-10)%5;TM& tensor=column<15?p.h:p.s;
    if(c==0) {tensor[0][0]=Times(v,1/std::sqrt(2.));tensor[1][1]=Times(v,-1/std::sqrt(2.));}
    if(c==1) {tensor[0][0]=tensor[1][1]=Times(v,1/std::sqrt(6.));tensor[2][2]=Times(v,-2/std::sqrt(6.));}
    if(c>=2) {const int i=c==4?1:0,j=c==2?1:2;tensor[i][j]=tensor[j][i]=Times(v,1/std::sqrt(2.));}
  }
  return p;
}
int main() {try {
  int cases=0;double rhs=0,rhs_absolute=0,constraints=0,constraints_absolute=0;
  for(const X3&x:{X3{0,0,0},X3{.01,-.02,.03},X3{-.017,.011,.025},X3{.021,.018,-.022}}) {
    const auto p=reference.At(x[0],x[1],x[2]);
    if(p.omega!=1||p.alpha!=1)throw std::runtime_error("outside exact core");
    for(int degree=0;degree<=4;++degree)for(int px=0;px<=degree;++px)for(int py=0;py<=degree-px;++py) {
      const int pz=degree-px-py;const auto v=Monomial(x,{px,py,pz});
      for(int column=0;column<20;++column) {
        const auto u=LiftPhysical(p,CartesianColumn(column,v));
        const auto actual=Derivative(ActualDual(p,u)),expected=FlatFormula(u),d=Difference(actual,expected);
        rhs=std::max(rhs,Norm(d)/std::max(1.,Norm(expected)));rhs_absolute=std::max(rhs_absolute,Max(d));
        const auto q=Constraints(u,p),qc=FlatConstraints(u);
        for(int k=0;k<8;++k) {constraints=std::max(constraints,std::abs(q[k]-qc[k])/std::max(1.,std::abs(qc[k])));constraints_absolute=std::max(constraints_absolute,std::abs(q[k]-qc[k]));}
        ++cases;
      }
    }
  }
  std::cout<<std::setprecision(17)<<"{\"cases\":"<<cases<<",\"full22_rhs_scaled\":"<<rhs<<",\"full22_rhs_absolute_max\":"<<rhs_absolute<<",\"physical8_constraints_scaled\":"<<constraints<<",\"physical8_constraints_absolute_max\":"<<constraints_absolute<<",\"origin_included\":true,\"CG_seed_used\":false,\"no_radial_operator\":true}\n";
  return cases==2800&&rhs<=5e-12&&constraints<=5e-12?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
