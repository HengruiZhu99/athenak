// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Standalone read-only initial-data derivative/source localization; no evolution.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include <chrono>
using Engine=h::CompositeTensorEvolution;
using U=h::Z4cJet<Real>;
U Analytic(const Point &x,const Exact&e){U u{};for(int i=0;i<3;++i)u.metric.g[i][i]=1;u.chi=Physical(x.data(),e,0);u.alpha=Physical(x.data(),e,1);u.trace=Radial(x.data(),e.khat,e.khatp,0);
 const Real r=e.r;Real n[3];for(int a=0;a<3;++a)n[a]=x[a]/r;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){const Real tensor=n[i]*n[j]-(i==j?1./3:0);u.a.k[i][j]=1.5*e.arr*tensor;for(int a=0;a<3;++a)u.a.dk[a][i][j]=1.5*(e.arrp*n[a]*tensor+e.arr*((a==i?1.:0)-n[a]*n[i])*n[j]/r+e.arr*n[i]*((a==j?1.:0)-n[a]*n[j])/r);}
 const Real f=e.beta/r,fp=e.bp/r-e.beta/(r*r),fpp=e.bpp/r-2*e.bp/(r*r)+2*e.beta/(r*r*r);
 for(int i=0;i<3;++i){u.beta.value[i]=f*x[i];for(int a=0;a<3;++a){u.beta.d[a][i]=f*(a==i?1:0)+fp*n[a]*x[i];for(int b=0;b<3;++b)u.beta.dd[a][b][i]=fp*(n[b]*(a==i?1:0)+n[a]*(b==i?1:0))+(fpp*n[a]*n[b]+fp*((a==b?1.:0)-n[a]*n[b])/r)*x[i];}}
 return u;
}
std::array<Real,24> Quantities(Engine &engine,U u,const Point&x){const auto omega=h::CompositeOmegaAt(x.data(),u);const auto c=h::EvolvedConstraints(u,omega);const auto p=h::ConformalRHS(u,omega,Real(4),Real(-.99),true);h::Z4cRHS<Real> rhs{};if(!c.valid||!h::AssembleInterior(p,omega.omega,rhs))throw std::runtime_error("invalid independent initial source");std::array<Real,24> out{};out[0]=c.hamiltonian;out[1]=std::sqrt(c.momentum_conformal_norm2);out[2]=rhs.chi;out[3]=rhs.trace;out[4]=rhs.theta;out[5]=h::WormholeLapseRate(x.data(),u);for(int t=0;t<6;++t){out[6+t]=rhs.metric[Engine::tensor_i[t]][Engine::tensor_j[t]];out[12+t]=rhs.a[Engine::tensor_i[t]][Engine::tensor_j[t]];}for(int i=0;i<3;++i){out[18+i]=rhs.lambda[i];out[21+i]=.005/.02*h::ReferenceTelegraphSupport(Radius(x.data()),.8,.24).value*u.alpha.d[i];}for(Real v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite full initial source");return out;}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{if(argc<4)throw std::invalid_argument("catalog initial.dat catalog.csv OR diagnose initial.dat exact.dat output.csv");const auto start=std::chrono::steady_clock::now();auto grid=MakeGrid();const auto initial=ReadInitial(argv[2],grid);std::cout<<std::setprecision(17);
 if(std::string(argv[1])=="catalog"){
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";auto emit=[&](int kind,size_t id,const Real*x){out<<kind<<','<<id<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<Radius(x)<<'\n';};for(size_t p=0;p<grid.active.size();++p)emit(0,p,grid.geometry[p].x);size_t count=0;for(const auto &row:grid.halos){const auto x=PaddedPoint(grid,row.target);if(Radius(x.data())>0&&Radius(x.data())<.85){emit(2,row.target,x.data());++count;}}const Real sample[3]={.5,0,0},scri[3]={1,0,0};emit(1,1,sample);emit(1,0,scri);std::cout<<"{\"active\":"<<initial.size()<<",\"interior_halo_rows\":"<<count<<"}\n";
 }else if(std::string(argv[1])=="diagnose"){
  if(argc!=5)throw std::invalid_argument("diagnose output required");const auto exact=ReadExact(argv[3]);Engine engine(grid.mesh,true,1e-10,4,true,true,0,6,true,2,.24,true,true,true,true,.24,1);engine.Initialize(0,false);engine.InitializeWormhole(argv[2],0,false);const auto state=engine.state;const auto baseline=engine.Padded(state);auto oracle_halos=baseline;Real value_error=0;size_t changed=0;
  for(size_t p=0;p<initial.size();++p){const auto&e=exact.at({0,p});value_error=std::max({value_error,std::abs(initial[p].chi-e.chi),std::abs(initial[p].alpha-e.alpha)});}
  for(const auto &item:exact)if(item.first.first==2){const int index=item.first.second;const auto x=PaddedPoint(grid,index);const auto&e=item.second;oracle_halos[index]=StoredExact(x.data(),e,0).value;oracle_halos[18*grid.count+index]=StoredExact(x.data(),e,1).value;++changed;}
  engine.GeometryJets(state,baseline);const auto basejets=engine.jets;engine.GeometryJets(state,oracle_halos);const auto halojets=engine.jets;
  std::ofstream out(argv[4]);out<<std::setprecision(17)<<"point,block,face,i,j,k,x,y,z,r,weight,method,chi_d_error,chi_dd_error,alpha_d_error,alpha_dd_error";out<<",H,M";for(int f=0;f<24;++f)out<<",dq"<<f;out<<'\n';
  const char *names[]={"default_supported","exact_internal_halos","exact_chi_first","exact_chi_second","exact_chi_all","exact_lapse_all","exact_curvature_first","all_analytic"};Real exact_maxH=0,exact_maxM=0;
  for(size_t p=0;p<initial.size();++p){const auto &x=grid.geometry[p].x;const auto e=Analytic({x[0],x[1],x[2]},exact.at({0,p}));const auto truth=Quantities(engine,e,{x[0],x[1],x[2]});exact_maxH=std::max(exact_maxH,std::abs(truth[0]));exact_maxM=std::max(exact_maxM,std::abs(truth[1]));
   for(int mode=0;mode<8;++mode){auto u=mode==1?halojets[p]:basejets[p];u.beta=e.beta;if(mode==2||mode==4)for(int a=0;a<3;++a)u.chi.d[a]=e.chi.d[a];if(mode==3||mode==4)for(int a=0;a<3;++a)for(int b=0;b<3;++b)u.chi.dd[a][b]=e.chi.dd[a][b];if(mode==5)u.alpha=e.alpha;if(mode==6){for(int a=0;a<3;++a){u.trace.d[a]=e.trace.d[a];for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.dk[a][i][j]=e.a.dk[a][i][j];}}if(mode==7)u=e;
    Real dg=0,dh=0,ag=0,ah=0;for(int a=0;a<3;++a){dg=std::max(dg,std::abs(u.chi.d[a]-e.chi.d[a]));ag=std::max(ag,std::abs(u.alpha.d[a]-e.alpha.d[a]));for(int b=0;b<3;++b){dh=std::max(dh,std::abs(u.chi.dd[a][b]-e.chi.dd[a][b]));ah=std::max(ah,std::abs(u.alpha.dd[a][b]-e.alpha.dd[a][b]));}}
    const auto values=Quantities(engine,u,{x[0],x[1],x[2]});const int block=p/(grid.n*grid.n*grid.n),i=p%grid.n,j=p/grid.n%grid.n,k=p/(grid.n*grid.n)%grid.n;const Real weight=grid.volumes[p]/std::pow(e.chi.value,1.5);out<<p<<','<<block<<','<<grid.mesh.blocks[block].face<<','<<i<<','<<j<<','<<k<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<Radius(x)<<','<<weight<<','<<names[mode]<<','<<dg<<','<<dh<<','<<ag<<','<<ah;out<<','<<values[0]<<','<<values[1];for(int f=0;f<24;++f)out<<','<<values[f]-truth[f];out<<'\n';
    if(p==126)std::cout<<"{\"phase\":\"target\",\"method\":\""<<names[mode]<<"\",\"H\":"<<values[0]<<",\"M\":"<<values[1]<<",\"H_error\":"<<values[0]-truth[0]<<",\"chi_d_error\":"<<dg<<",\"chi_dd_error\":"<<dh<<",\"lapse_d_error\":"<<ag<<",\"lapse_dd_error\":"<<ah<<"}\n";
   }
  }
  std::cout<<"{\"phase\":\"summary\",\"active_cells\":"<<initial.size()<<",\"initial_value_match_max\":"<<value_error<<",\"oracle_internal_halos\":"<<changed<<",\"exact_H_max\":"<<exact_maxH<<",\"exact_M_max\":"<<exact_maxM<<",\"all_active_cells_retained\":true,\"shared_exact_radial_shift_in_source_comparisons\":true,\"no_solve_or_evolution\":true,\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
 }else throw std::runtime_error("unknown mode");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
