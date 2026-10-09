// Scratch oracle: analytic reference beta/omega jets, no finite differencing.
#include "projected_base.hpp"
int main(int argc,char**argv){Kokkos::ScopeGuard guard(argc,argv);try{
 Audit a(16,2.2,1e-4);std::cout<<std::setprecision(17)<<'[';
 for(size_t j=0;j<a.cells.size();++j){const auto&p=a.points[j];const auto&c=a.cells[j];const auto o=hyp::CartesianOmega(p.state,p);
  std::cout<<(j?",":"")<<'['<<c.xyz[0]<<','<<c.xyz[1]<<','<<c.xyz[2]<<','<<o.omega<<','<<p.radius<<','<<p.alpha;
  for(int i=0;i<3;++i)std::cout<<','<<p.beta[i];
  for(int i=0;i<3;++i)std::cout<<','<<o.gradient[i];
  std::cout<<','<<hyp::SmoothCutoff(p.radius,Real(.15),Real(.3)).value<<','<<hyp::ResearchLiveKappa2Profile(p.state,o,p.radius,Real(10))<<']';}
 std::cout<<"]\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
