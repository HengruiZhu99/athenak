// Read-only actual C0 tensor normalization check; no alternative RHS injection.
#include "../discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp"
int main() {
  double error=0,unchanged=0;
  int rows=0;
  for(double a:{.5,.75,1.,2.})for(double rho:{-.2,0.,.3,1.}) {
    hyp::LayerReference<double>ref(1,a,{true,.05,.95});
    for(int ir=0;ir<101;++ir){
      double r=ir/101.;auto p=ref.At(.36*r,-.48*r,.8*r);auto u=Lift(p.state);
      for(int col=0;col<20;++col){J seed(D(.01*std::sin(col+1.)));
        for(int i=0;i<3;++i){seed.d[i]=.01*std::cos((col+1.)*(i+1.))/3;
          for(int j=0;j<3;++j)seed.dd[i][j]=.01*std::sin((col+1.)*(i+j+2.))/5;}
        Seed(u,col,seed);}
      Consistent(u);auto o=Omega(u,p);hyp::Z4cRHS<D>f{},f0{};
      D kap=D(10)/u.alpha.value;
      if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,kap,D(rho)),o.omega,f)
          ||!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(0),D(rho)),o.omega,f0))return 2;
      const D A=u.alpha.value/o.omega,theta=u.theta.value;
      auto compare=[&](D actual,D expected){error=std::max(error,
          std::abs(actual.v-expected.v)/(1+std::abs(expected.v)));};
      compare(f.trace-f0.trace,A*kap*(1-rho)*theta);
      compare(f.theta-f0.theta,-A*kap*(2+rho)*theta);
      auto gt=hyp::Geometry(u.metric);
      for(int i=0;i<3;++i){
        D zup=D(.5)*(u.lambda.value[i]-gt.contracted[i]);
        compare(f.lambda[i]-f0.lambda[i],-2*A*kap*zup);
        for(int j=0;j<3;++j){
          unchanged=std::max(unchanged,std::abs(f.metric[i][j].v-f0.metric[i][j].v));
          unchanged=std::max(unchanged,std::abs(f.a[i][j].v-f0.a[i][j].v));}}
      unchanged=std::max(unchanged,std::abs(f.chi.v-f0.chi.v));++rows;
    }
  }
  std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows
      <<",\"normal_damping_relative_error\":"<<error
      <<",\"metric_chi_A_difference\":"<<unchanged<<"}\n";
  return error<1e-11&&unchanged==0?0:1;
}
