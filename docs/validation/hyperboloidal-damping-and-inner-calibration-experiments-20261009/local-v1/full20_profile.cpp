#include "profile_helpers.hpp"
M Pole(const hyp::LayerPoint<double>&p,double a,bool profile){
 M m{};for(int col=0;col<20;++col){auto u=Lift(p.state);Seed(u,col,J(D(0,1)));Consistent(u);
 auto o=Omega(u,p);auto k2=profile?hyp::ResearchKappa2Profile(o.omega,D(1),D(a),D(10)):D(0);
 const auto parts=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,k2);
 const auto gp=GaugeParts(Reference(p),u,a,true);hyp::GaugeRHS<D>g;g.alpha=gp.pole.alpha;
 for(int i=0;i<3;++i)g.beta[i]=gp.pole.beta[i];
 auto f=Fields(parts.pole,g);for(int row=0;row<20;++row)m[row][col]=f[row].d;}return m;
}
int main(int argc,char**){std::cout<<std::setprecision(17)<<'[';bool first=true;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(int mode:{0,1}){
 if(argc>1){const auto p=ref.At(1.,0.,0.);if(!first)std::cout<<',';first=false;
 std::cout<<"{\"a\":"<<a<<",\"profile\":"<<mode<<",\"P\":";Print(Pole(p,a,mode));std::cout<<'}';continue;}
 for(double pert:{0.,.01})for(double r:{.3,.65,.75,.85,.95,.98,std::sqrt(1-2*a*.0142578125),std::sqrt(1-2*a*.0038368055555556),std::sqrt(1-2*a*.003251953125),std::sqrt(1-2*a*1e-5)}){
 const auto p=ref.At(r,0.,0.);for(double k:{0.,32.,64.,128.,256.,std::acos(-1.)*24/2.2,std::acos(-1.)*36/2.2,std::acos(-1.)*48/2.2})for(bool oblique:{false,true}){
 if(a!=.5&&(k!=0||oblique))continue;const std::array<double,3>n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};
 auto m=Matrix(p,a,10,mode,true,pert,k,n);double fixed=0;for(auto v:Evaluate(p,Background(p,0),a,10,mode,true))fixed=std::max(fixed,std::abs(v.v));
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"profile\":"<<mode<<",\"perturb\":"<<pert<<",\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"k\":"<<k<<",\"oblique\":"<<oblique<<",\"reference_fixedpoint\":"<<fixed<<",\"L\":";Print(m);std::cout<<'}';}
 }}}
 std::cout<<"]\n";
}
