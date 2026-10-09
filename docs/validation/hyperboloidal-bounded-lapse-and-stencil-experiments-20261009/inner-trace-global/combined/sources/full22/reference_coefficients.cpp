// Actual reference jets and cutoff for independent alpha-row attribution.
#include "projected_base.hpp"
int main(int argc,char**argv){Kokkos::ScopeGuard guard(argc,argv);try{Audit a(16,2.2,1e-4);std::cout<<std::setprecision(17)<<'[';
for(size_t j=0;j<a.cells.size();++j){const auto&p=a.points[j];const auto&c=a.cells[j];std::cout<<(j?",":"")<<'['<<c.xyz[0]<<','<<c.xyz[1]<<','<<c.xyz[2]<<','<<p.omega<<','<<p.radius<<','<<p.alpha;
for(int i=0;i<3;++i)std::cout<<','<<p.beta[i];for(int i=0;i<3;++i)std::cout<<','<<p.domega[i];for(int i=0;i<3;++i)std::cout<<','<<p.dalpha[i];std::cout<<','<<1-hyp::SmoothCutoff(p.radius,Real(a.patch.layer_gauge.r0),Real(a.patch.layer_gauge.r1)).value<<']';}std::cout<<"]\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
