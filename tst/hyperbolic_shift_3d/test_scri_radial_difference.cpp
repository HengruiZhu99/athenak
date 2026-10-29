// Direct analytic derivatives and poisoned exterior-donor control. No evolution.
#include <iostream>
#include <iomanip>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"

using Model=z4c::hyperboloidal::HyperbolicComposite;
struct Errors{Real first=0,second=0,constant=0,polynomial=0;size_t points=0;};
Errors Check(int n){
  Model model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,2,
      Model::NormalDerivative::sampled_numerator,Model::RadialClosure::donor_seven);
  const auto &grid=model.grid;const auto &plan=*model.radial_plan;
  Model::V volume(31*grid.count,0),surface(model.scri.values.size(),0);
  const Real a[3]={.3,-.2,.4};
  auto field=[&](const Real*x){return std::exp(a[0]*x[0]+a[1]*x[1]+a[2]*x[2]);};
  for(size_t m=0;m<grid.mesh.blocks.size();++m)
  for(int k=-grid.ng;k<n+grid.ng;++k)for(int j=-grid.ng;j<n+grid.ng;++j)
  for(int i=-grid.ng;i<n+grid.ng;++i){
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];grid.mesh.Forward(m,q,x);
    volume[grid.Index(m,i,j,k)]=field(x);
    if(model.scri.base[m]>=0&&i>=n)volume[grid.Index(m,i,j,k)]=std::numeric_limits<Real>::quiet_NaN();
  }
  for(size_t p=0;p<model.scri.rows.size();++p)surface[31*p]=field(model.scri.rows[p].geometry.x);
  Errors e;
  for(size_t p=0;p<grid.active.size();++p)if(plan.Contains(p)){
    Real d[3],dd[3][3];plan.Derivatives(volume,surface,0,p,d,dd,true);const auto &g=grid.geometry[p];
    for(int i=0;i<3;++i){if(!std::isfinite(d[i]))throw std::runtime_error("read poisoned radial ghost");
      e.first+=std::pow(d[i]-a[i]*field(g.x),2);
      for(int j=0;j<3;++j){if(!std::isfinite(dd[i][j]))throw std::runtime_error("read poisoned radial ghost");
        e.second+=std::pow(dd[i][j]-a[i]*a[j]*field(g.x),2);}}
    ++e.points;
  }
  e.first=std::sqrt(e.first/(3*e.points));e.second=std::sqrt(e.second/(9*e.points));
  // Constants are annihilated exactly by anchored sums on every face and seam.
  std::fill(volume.begin(),volume.end(),3.25);std::fill(surface.begin(),surface.end(),3.25);
  for(size_t p=0;p<grid.active.size();++p)if(plan.Contains(p)){
    Real d[3],dd[3][3];plan.Derivatives(volume,surface,0,p,d,dd,true);
    for(int i=0;i<3;++i){e.constant=std::max(e.constant,std::abs(d[i]));
      for(int j=0;j<3;++j)e.constant=std::max(e.constant,std::abs(dd[i][j]));}}
  // Interior angular rows reproduce a degree-six radial polynomial and its
  // mixed derivatives, including all J/H terms of the nonlinear chart map.
  auto polynomial=[](const Real*q){return std::pow(q[0],6)*std::pow(q[1],3)*q[2]*q[2];};
  for(size_t m=0;m<grid.mesh.blocks.size();++m)
  for(int k=-grid.ng;k<n+grid.ng;++k)for(int j=-grid.ng;j<n+grid.ng;++j)
  for(int i=-grid.ng;i<n+grid.ng;++i){const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};
    volume[grid.Index(m,i,j,k)]=polynomial(q);}
  for(size_t p=0;p<model.scri.rows.size();++p){const auto&r=model.scri.rows[p];
    const Real q[3]={1,(r.j+.5)/n,(r.k+.5)/n};surface[31*p]=polynomial(q);}
  for(size_t p=0;p<grid.active.size();++p)if(plan.Contains(p)){
    const int i=p%n,j=(p/n)%n,k=(p/(n*n))%n;if(j<2||j>=n-2||k<2||k>=n-2)continue;
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n},powers[3]={6,3,2};const Real f=polynomial(q);
    Real dq[3],ddq[3][3];for(int A=0;A<3;++A){dq[A]=powers[A]*f/q[A];
      for(int B=0;B<3;++B)ddq[A][B]=powers[A]*(powers[B]-(A==B))*f/(q[A]*q[B]);}
    Real d[3],dd[3][3];plan.Derivatives(volume,surface,0,p,d,dd,true);const auto&g=grid.geometry[p];
    for(int x=0;x<3;++x){Real exact=0;for(int A=0;A<3;++A)exact+=g.J[A][x]*dq[A];
      e.polynomial=std::max(e.polynomial,std::abs(d[x]-exact)/(1+std::abs(exact)));
      for(int y=0;y<3;++y){exact=0;for(int A=0;A<3;++A){exact+=g.H[A][x][y]*dq[A];
        for(int B=0;B<3;++B)exact+=g.J[A][x]*g.J[B][y]*ddq[A][B];}
        e.polynomial=std::max(e.polynomial,std::abs(dd[x][y]-exact)/(1+std::abs(exact)));}}
  }
  if(e.constant!=0||e.polynomial>2e-9)throw std::runtime_error("radial closure polynomial/constant inconsistency");
  return e;
}
int main(){try{
  const auto a=Check(8),b=Check(16);
  const Real p1=std::log(a.first/b.first)/std::log(2.),p2=std::log(a.second/b.second)/std::log(2.);
  if(p1<3.5||p2<3.5)throw std::runtime_error("mapped analytic derivative convergence below 3.5");
  std::cout<<std::setprecision(17)<<"{\"passed\":true,\"n8_first\":"<<a.first<<",\"n16_first\":"<<b.first
    <<",\"n8_second\":"<<a.second<<",\"n16_second\":"<<b.second<<",\"first_order\":"<<p1
    <<",\"second_order\":"<<p2<<",\"constant_error\":"<<std::max(a.constant,b.constant)
    <<",\"polynomial_error\":"<<std::max(a.polynomial,b.polynomial)<<",\"poisoned_radial_ghosts_ignored\":true}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
