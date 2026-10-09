// Analytic reference coefficients only, no matrix or propagation.
#include "projected_base.hpp"
int main(int argc,char**argv){Kokkos::ScopeGuard guard(argc,argv);try{
 Audit a(16,2.2,1e-4);std::cout<<std::setprecision(17)<<'[';
 for(size_t j=0;j<a.cells.size();++j){const auto&p=a.points[j];const auto&c=a.cells[j];
  const auto q=hyp::LayerCoefficients(p.radius,p.alpha,a.patch.layer_gauge);
  auto put=[](double x){std::cout<<','<<x;};std::cout<<(j?",":"")<<'['<<c.xyz[0];put(c.xyz[1]);put(c.xyz[2]);put(p.radius);put(p.omega);put(p.alpha);put(p.state.chi.value);put(p.k_bar);put(p.w_omega);put(q.weight);put(q.nu);put(q.eta);put(hyp::SmoothCutoff(p.radius,Real(.85),Real(.95)).value);
  for(int i=0;i<3;++i)put(p.beta[i]);for(int i=0;i<3;++i)put(p.domega[i]);for(int i=0;i<3;++i)put(p.dalpha[i]);for(int i=0;i<3;++i)put(p.state.chi.d[i]);for(int i=0;i<3;++i)put(p.state.lambda.value[i]);
  for(int i=0;i<3;++i)for(int k=0;k<3;++k)put(c.inv[i][k]);for(int i=0;i<3;++i)for(int k=0;k<3;++k)put(p.omega_hessian[i][k]);std::cout<<']';
 }std::cout<<"]\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
