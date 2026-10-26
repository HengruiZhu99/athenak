#include "goal1_native_outer_gauge_identity.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;
Real Read(){std::string s;if(!(std::cin>>s))throw std::runtime_error("short outer gauge fixture");char*e;Real v=std::strtod(s.c_str(),&e);if(*e||!std::isfinite(v))throw std::runtime_error("invalid outer gauge token");return v;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;std::cout<<std::setprecision(17);try{while(std::cin>>std::ws,std::cin.peek()!=EOF){int id=Read();Real n[3];for(auto&v:n)v=Read();h::Z4cJet<Real>zero{};auto u=cs::typed_support::Map<Real>(zero,[](int,Real){return Read();});u.theta.value=.003;for(Real radius:{.951,.98,.999,.999999,1.}){Real x[3];for(int i=0;i<3;++i)x[i]=radius*n[i];h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});auto p=descriptor.At(x[0],x[1],x[2]);rwm::Connection<Real>connection;
 if(radius==1){p.omega=0;auto c=cs::native_endpoint::OuterConnection(x);connection.valid=true;for(int a=0;a<4;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)connection.scaled[a][i][j]=c.scaled[a][i][j].f;}else connection=rwm::ReferenceConnection(p,x);
 auto expanded=rwm::Gauge(p,u,connection),compact=cs::native_outer_gauge_identity::Compact(u,p,x);if(!expanded.valid||!compact.valid)throw std::runtime_error("invalid outer gauge identity state");Real decomp=0,assembled=0,scale=1;auto cmp=[&](Real a,Real b){decomp=std::max(decomp,std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)}));};Real rr=radius*radius,alpha=u.alpha.value;cmp(expanded.regular.alpha-compact.regular.alpha,4*alpha*rr/p.L);cmp(expanded.pole.alpha-compact.pole.alpha,-4*alpha*rr*p.omega/p.L);
 for(int i=0;i<3;++i){cmp(expanded.regular.beta[i]-compact.regular.beta[i],-2*p.omega*x[i]);cmp(expanded.pole.beta[i]-compact.pole.beta[i],2*p.omega*p.omega*x[i]);}
 if(p.omega>0){assembled=std::abs(expanded.regular.alpha+expanded.pole.alpha/p.omega-compact.regular.alpha-compact.pole.alpha/p.omega);scale=std::max(scale,std::abs(expanded.regular.alpha+expanded.pole.alpha/p.omega));for(int i=0;i<3;++i){Real a=expanded.regular.beta[i]+expanded.pole.beta[i]/p.omega,b=compact.regular.beta[i]+compact.pole.beta[i]/p.omega;assembled=std::max(assembled,std::abs(a-b));scale=std::max(scale,std::abs(a));}}
 auto wrong=u;wrong.trace.value+=2*u.theta.value;auto bad=cs::native_outer_gauge_identity::Compact(wrong,p,x);Real negative=std::abs(bad.pole.alpha-compact.pole.alpha);std::cout<<"{\"id\":"<<id<<",\"radius\":"<<radius<<",\"Omega\":"<<p.omega<<",\"decomposition_scaled\":"<<decomp<<",\"assembled_scaled\":"<<assembled/scale<<",\"negative_replace_P_by_K_pole\":"<<negative<<"}\n";
 }}}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
