// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Actual initial-state falsifier, not an evolution operator selection.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/puncture_log_value_plans.hpp"
#include "z4c/hyperboloidal/rho_log_derivative.hpp"
#include "z4c/hyperboloidal/odd_radial_log_derivative.hpp"
#include <chrono>

using Single=h::RhoLogDerivative;using Odd=h::OddRadialLogDerivative;using Jet=cubed_sphere::Jet;
struct Norm {long double volume=0,square=0;size_t cells=0,peak=0;Real maximum=0;
  void Add(size_t p,Real weight,Real value){if(!std::isfinite(value))throw std::runtime_error("incomplete candidate cannot enter all-cell norm");++cells;volume+=weight;square+=static_cast<long double>(weight)*value*value;if(std::abs(value)>maximum){maximum=std::abs(value);peak=p;}}
  void Print()const{std::cout<<"{\"cells\":"<<cells<<",\"rms\":"<<std::sqrt(Real(square/volume))<<",\"maximum\":"<<maximum<<",\"maximum_point\":"<<peak<<'}';}
};
template<class Fit>WeightAudit AuditActual(const Fit &fit,Real dx,const h::PuncturePosition &p,size_t count){WeightAudit out;Real g[3]{},dd[3][3]{};
  for(size_t i=0;i<count;++i){V unit(count,0);unit[i]=1;typename Fit::Result r;if(!fit.Evaluate(unit,p,r))throw std::runtime_error("impulse rejected");out.value+=std::abs(r.fitted_value);out.enrichment+=std::abs(r.enrichment_coefficient);
    for(int a=0;a<3;++a){g[a]+=dx*std::abs(r.gradient[a]);for(int b=0;b<3;++b)dd[a][b]+=dx*dx*std::abs(r.hessian[a][b]);}}
  for(int a=0;a<3;++a){out.gradient=std::max(out.gradient,g[a]);for(int b=0;b<3;++b)out.hessian=std::max(out.hessian,dd[a][b]);}return out;}
template<class Fit>S EvaluateActual(const Fit &fit,const V &values,Real value,const h::PuncturePosition &p){typename Fit::Result r;if(!fit.Evaluate(values,p,r))throw std::runtime_error("actual evaluation rejected");S out{};out.value=value;
  for(int a=0;a<3;++a){out.d[a]=r.gradient[a];for(int b=0;b<3;++b)out.dd[a][b]=r.hessian[a][b];}return out;}
template<class Fit>Real NoiseActual(const Fit &fit,const V &values,const S &q,Real dx,const h::PuncturePosition &p,const WeightAudit &audit){V perturbed=values;for(size_t i=0;i<values.size();++i)perturbed[i]+=1e-12*(std::sin(1.7*i)+std::cos(.9*i))/2;const auto nq=EvaluateActual(fit,perturbed,q.value,p);Real error=0;
  for(int a=0;a<3;++a){error=std::max(error,dx*std::abs(q.d[a]-nq.d[a]));for(int b=0;b<3;++b)error=std::max(error,dx*dx*std::abs(q.dd[a][b]-nq.dd[a][b]));}
  if(error>std::max(audit.gradient,audit.hessian)*1e-12+2e-13)throw std::runtime_error("actual noise exceeds functional bound");return error;}
Jet ExactInner(const Real x[3],int field){Jet y[3];for(int a=0;a<3;++a)y[a]=Jet::Variable(x[a],a);const Jet r2=y[0]*y[0]+y[1]*y[1]+y[2]*y[2],r=cubed_sphere::Power(r2,.5),omega=Jet(.5)*(Jet(1)-r2);const Real k=.37754694621653295;
  if(field==0)return Jet(std::log(k*k))+Jet(2)*cubed_sphere::Log(r2)-Jet(2)*cubed_sphere::Log(omega)-Jet(4)*cubed_sphere::Log(r+Jet(k/6));
  return cubed_sphere::Log(omega)+cubed_sphere::Log(r2)-Jet(2)*cubed_sphere::Log(r+Jet(k/6))-cubed_sphere::Log(Jet(.5)*(Jet(1)+r2));}
Real JetError(const S&q,const Jet&e,Real dx){Real error=0;for(int a=0;a<3;++a){error=std::max(error,dx*std::abs(q.d[a]-e.d[a]));for(int b=0;b<3;++b)error=std::max(error,dx*dx*std::abs(q.dd[a][b]-e.h[a][b]));}return error;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc!=2)throw std::invalid_argument("usage: actual initial.dat");const auto start=std::chrono::steady_clock::now();auto grid=MakeGrid();const auto initial=ReadInitial(argv[1],grid);h::PuncturePosition point;std::cout<<std::setprecision(17);
  const auto cb=std::chrono::steady_clock::now();h::PunctureLogValuePlans halo(grid,point,.24,20000,8,175);const Real cache_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-cb).count();
  V chi(grid.count,0),alpha(grid.count,0),stored(2*initial.size());for(size_t p=0;p<initial.size();++p){const auto &x=grid.geometry[p].x;chi[grid.active[p]]=stored[2*p]=std::log(initial[p].chi);alpha[grid.active[p]]=stored[2*p+1]=std::log(initial[p].alpha)-std::log(Chart(x,1).reference.value);}grid.Fill(chi,1);grid.Fill(alpha,1);halo.FillPositiveLogs(chi,alpha,point);
  V padded(2*grid.count);for(int i=0;i<grid.count;++i){padded[i]=chi[i];padded[grid.count+i]=alpha[i];}h::PunctureLogPlans base(grid,point);std::cout<<"{\"phase\":\"setup\",\"halo_rows\":"<<halo.rows.size()<<",\"halo_assignments\":"<<halo.halo_assignments<<",\"halo_build_seconds\":"<<cache_seconds<<",\"selected_derivative_rows\":"<<base.rows.size()<<"}\n"<<std::flush;
  V baseH(initial.size()),strictH(initial.size()),rawH(initial.size()),oddH(initial.size());
  auto Hamiltonian=[&](size_t p,const S &q){const auto &x=grid.geometry[p].x;const Real omega=(1-initial[p].r*initial[p].r)/2;return HChi(x,q)+(2./3)*initial[p].khat*initial[p].khat-1.5*omega*omega*initial[p].arr*initial[p].arr;};
  for(size_t p=0;p<initial.size();++p)baseH[p]=strictH[p]=rawH[p]=oddH[p]=Hamiltonian(p,Sample(grid,padded,stored,p,0));
  size_t strict_ready=0,raw_ready=0,odd_ready=0;Real maximum_condition[3]{},max_g[3]{},max_dd[3]{},max_noise[3]{};std::map<std::string,size_t> strict_rejections,raw_rejections,odd_rejections;
  for(const auto &row:base.rows){const size_t p=row.active;const auto &x=grid.geometry[p].x;const Real dx=Spacing(grid,p);std::vector<Single::Point> nodes;for(size_t donor:row.donors){const auto &d=grid.geometry[donor].x;nodes.push_back({d[0],d[1],d[2]});}
    Single single;const bool ready=single.Build({x[0],x[1],x[2]},nodes,dx,point);strict_ready+=ready;if(!ready){++strict_rejections[single.diagnostics().status];strictH[p]=std::numeric_limits<Real>::quiet_NaN();std::cout<<"{\"phase\":\"strict_rejection\",\"point\":"<<p<<",\"rho_over_h\":"<<initial[p].r/dx<<",\"status\":\""<<single.diagnostics().status<<"\",\"gradient_l1\":"<<single.diagnostics().scaled_gradient_weight_l1<<",\"hessian_l1\":"<<single.diagnostics().scaled_hessian_weight_l1<<"}\n";}
    Single raw=single;bool raw_valid=ready;if(!raw_valid){Single::Options o;o.maximum_gradient_l1=o.maximum_hessian_l1=1e9;raw_valid=raw.Build({x[0],x[1],x[2]},nodes,dx,point,o);}raw_ready+=raw_valid;if(!raw_valid){++raw_rejections[raw.diagnostics().status];rawH[p]=std::numeric_limits<Real>::quiet_NaN();}
    Odd odd;const bool odd_valid=odd.Build({x[0],x[1],x[2]},nodes,dx,point);odd_ready+=odd_valid;if(!odd_valid){++odd_rejections[odd.diagnostics().status];oddH[p]=std::numeric_limits<Real>::quiet_NaN();}
    const auto ba=AuditActual(row.plan,dx,point,row.donors.size());maximum_condition[0]=std::max(maximum_condition[0],row.plan.diagnostics().condition);max_g[0]=std::max(max_g[0],ba.gradient);max_dd[0]=std::max(max_dd[0],ba.hessian);
    WeightAudit ra,oa;if(raw_valid){ra=AuditActual(raw,dx,point,row.donors.size());maximum_condition[1]=std::max(maximum_condition[1],raw.diagnostics().condition);max_g[1]=std::max(max_g[1],ra.gradient);max_dd[1]=std::max(max_dd[1],ra.hessian);}
    if(odd_valid){oa=AuditActual(odd,dx,point,row.donors.size());maximum_condition[2]=std::max(maximum_condition[2],odd.diagnostics().condition);max_g[2]=std::max(max_g[2],oa.gradient);max_dd[2]=std::max(max_dd[2],oa.hessian);}
    for(int f=0;f<2;++f){V values;for(size_t donor:row.donors)values.push_back(stored[2*donor+f]);const auto bq=base.Sample(row,stored,2,f);max_noise[0]=std::max(max_noise[0],NoiseActual(row.plan,values,bq,dx,point,ba));S rq{},oq{};
      if(raw_valid){rq=EvaluateActual(raw,values,stored[2*p+f],point);max_noise[1]=std::max(max_noise[1],NoiseActual(raw,values,rq,dx,point,ra));}
      if(odd_valid){oq=EvaluateActual(odd,values,stored[2*p+f],point);max_noise[2]=std::max(max_noise[2],NoiseActual(odd,values,oq,dx,point,oa));}
      if(f==0){baseH[p]=Hamiltonian(p,bq);if(ready)strictH[p]=Hamiltonian(p,rq);if(raw_valid)rawH[p]=Hamiltonian(p,rq);if(odd_valid)oddH[p]=Hamiltonian(p,oq);}
      if(p==34919){const auto analytic=ExactInner(x,f);if(std::abs(analytic.f-stored[2*p+f])>1e-12)throw std::runtime_error("exact initial scalar does not match actual input");
        std::cout<<"{\"phase\":\"target34919\",\"field\":\""<<(f==0?"chi":"alpha")<<"\",\"rho\":"<<initial[p].r<<",\"h\":"<<dx<<",\"strict_ready\":"<<ready<<",\"base_scaled_error\":"<<JetError(bq,analytic,dx)<<",\"rho_raw_scaled_error\":"<<(raw_valid?JetError(rq,analytic,dx):-1)<<",\"fullodd_scaled_error\":"<<(odd_valid?JetError(oq,analytic,dx):-1);
        if(f==0)std::cout<<",\"base_H\":"<<Hamiltonian(p,bq)<<",\"rho_raw_H\":"<<(raw_valid?Hamiltonian(p,rq):-1)<<",\"fullodd_H\":"<<(odd_valid?Hamiltonian(p,oq):-1);
        std::cout<<"}\n";}
    }
  }
  const bool complete[4]={true,strict_ready==base.rows.size(),raw_ready==base.rows.size(),odd_ready==base.rows.size()};const V *profiles[4]={&baseH,&strictH,&rawH,&oddH};const char *names[4]={"base","rho_strict","rho_raw","fullodd"};
  std::cout<<"{\"phase\":\"norms\",\"all_active_cells\":"<<initial.size()<<",\"profiles\":{";
  for(int b=0;b<4;++b){if(b)std::cout<<',';std::cout<<'"'<<names[b]<<"\":{\"complete\":"<<complete[b]<<",\"full_domain\":";if(!complete[b])std::cout<<"null,\"whole_core\":null}";else{Norm all,core;for(size_t p=0;p<initial.size();++p){const Real w=grid.volumes[p]/std::pow(initial[p].chi,1.5);all.Add(p,w,(*profiles[b])[p]);if(initial[p].r<.12)core.Add(p,w,(*profiles[b])[p]);}all.Print();std::cout<<",\"whole_core\":";core.Print();std::cout<<'}';}}
  std::cout<<"},\"strict_ready\":"<<strict_ready<<",\"raw_ready\":"<<raw_ready<<",\"odd_ready\":"<<odd_ready<<",\"selected_rows\":"<<base.rows.size()<<",\"condition_base_rho_odd\":["<<maximum_condition[0]<<','<<maximum_condition[1]<<','<<maximum_condition[2]<<"],\"gradient_l1_base_rho_odd\":["<<max_g[0]<<','<<max_g[1]<<','<<max_g[2]<<"],\"hessian_l1_base_rho_odd\":["<<max_dd[0]<<','<<max_dd[1]<<','<<max_dd[2]<<"],\"scaled_noise_base_rho_odd\":["<<max_noise[0]<<','<<max_noise[1]<<','<<max_noise[2]<<"],\"strict_rejections\":{";
  bool first=true;for(const auto &e:strict_rejections){if(!first)std::cout<<',';first=false;std::cout<<'"'<<e.first<<"\":"<<e.second;}std::cout<<"},\"raw_rejections\":{";first=true;for(const auto &e:raw_rejections){if(!first)std::cout<<',';first=false;std::cout<<'"'<<e.first<<"\":"<<e.second;}std::cout<<"},\"fullodd_rejections\":{";first=true;for(const auto &e:odd_rejections){if(!first)std::cout<<',';first=false;std::cout<<'"'<<e.first<<"\":"<<e.second;}
  std::cout<<"},\"total_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"active_values_changed\":false,\"raw_relaxes_only_functional_guards\":true,\"unsupported_rows_have_no_fallback\":true,\"production_adopted\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
