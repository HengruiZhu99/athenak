// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Standalone frozen-header experiment. The runner changes ONLY the frozen
// header's engineering row-cap validation from10000to20000; shared code is read-only.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/puncture_log_value_plans.hpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include <chrono>

using Clock=std::chrono::steady_clock;
double Seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct HNorm {size_t cells=0,maximum_point=0;long double volume=0,square=0;Real maximum=0;
  void Add(size_t p,Real w,Real h){++cells;volume+=w;square+=static_cast<long double>(w)*h*h;if(std::abs(h)>maximum){maximum=std::abs(h);maximum_point=p;}}
  void Print(const char *name)const{std::cout<<'"'<<name<<"\":{\"cells\":"<<cells<<",\"rms\":"<<std::sqrt(Real(square/volume))<<",\"maximum\":"<<maximum<<",\"maximum_point\":"<<maximum_point<<'}';}
};
Real InitialHamiltonian(const Real x[3],const Datum &data,const S &q,Real &formula_difference){
  h::Z4cJet<Real> u{};const Real r=Radius(x);u.alpha.value=data.alpha;u.trace.value=data.khat;
  if(Chart(x,0).Reconstruct(q,u.chi)!=h::PositiveFieldStatus::valid)throw std::runtime_error("positive chi reconstruction failed");
  for(int i=0;i<3;++i){u.metric.g[i][i]=1;for(int j=0;j<3;++j)u.a.k[i][j]=1.5*data.arr*(x[i]*x[j]/(r*r)-(i==j?1./3:0));}
  h::OmegaJet<Real> o{};o.omega=(1-r*r)/2;for(int i=0;i<3;++i){o.gradient[i]=-x[i];o.hessian[i][i]=-1;}
  // H cancels all normal/shift dependence in this actual tensor kernel. No
  // elliptic solve is needed to evaluate the initial Hamiltonian diagnostic.
  const auto result=h::EvolvedConstraints(u,o);if(!result.valid)throw std::runtime_error("invalid full initial constraint diagnostic");
  const Real explicit_flat=HChi(x,q)+(2./3)*data.khat*data.khat-o.omega*o.omega*1.5*data.arr*data.arr;
  formula_difference=std::max(formula_difference,std::abs(result.hamiltonian-explicit_flat));return result.hamiltonian;
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc!=6)throw std::invalid_argument("usage: initial.dat radius maximum_rows maximum_donors count|diagnose");
  const auto start=Clock::now();auto grid=MakeGrid();const auto initial=ReadInitial(argv[1],grid);const Real radius=std::stod(argv[2]);const size_t maximum_rows=std::stoul(argv[3]);const int maximum_donors=std::stoi(argv[4]);const bool count_only=std::string(argv[5])=="count";h::PuncturePosition point;
  std::cout<<std::setprecision(17);const auto targets=h::PunctureLogValuePlans::SelectTargets(grid,point,radius);Real min_spacing=1e99;
  for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0)min_spacing=std::min(min_spacing,Spacing(grid,p));
  std::map<std::array<long long,4>,size_t> groups;
  for(int index:targets){const auto x=PaddedPoint(grid,index);const auto &b=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];std::array<long long,4> key;
    for(int a=0;a<3;++a)key[a]=std::llround((x[a]+grid.mesh.half_width)*2/min_spacing);key[3]=std::llround((b.hi[0]-b.lo[0])/(grid.n*min_spacing));++groups[key];}
  std::cout<<"{\"phase\":\"count\",\"radius\":"<<radius<<",\"receiving_assignments\":"<<targets.size()<<",\"unique_query_spacing_plans\":"<<groups.size()<<",\"maximum_rows\":"<<maximum_rows<<",\"maximum_donors\":"<<maximum_donors<<",\"value_l1_cap\":8,\"actual_active_cells\":"<<grid.active.size()<<",\"seconds\":"<<Seconds(start)<<"}\n"<<std::flush;
  if(count_only){Kokkos::finalize();return 0;}
  const auto before=Clock::now();h::PunctureLogValuePlans cache(grid,point,radius,maximum_rows,8,maximum_donors);const Real build_seconds=Seconds(before);
  Real actual_reproduction=0;
  for(const auto &row:cache.rows)for(int index:row.targets){const auto query=PaddedPoint(grid,index);const auto &b=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];const Real h=(b.hi[0]-b.lo[0])/grid.n;V values;
    for(size_t p:row.donors){const auto &x=grid.geometry[p].x;const Real a=(x[0]-query[0])/h,c=(x[1]-query[1])/h,d=(x[2]-query[2])/h;values.push_back(1.23+.17*a+.11*c*d-.13*a*a*c+.07*std::pow(a,5)+.09*a*a*c*d*d+1.09*std::log(Radius(x)/Radius(query.data())));}
    Real value;if(!row.plan.Evaluate(values,point,grid.mesh.revision,value))throw std::runtime_error("actual target reproduction rejected");actual_reproduction=std::max(actual_reproduction,std::abs(value-1.23));}
  if(actual_reproduction>1e-8)throw std::runtime_error("actual target polynomial/log reproduction failed");
  V chi(grid.count,0),alpha(grid.count,0),stored(2*initial.size());
  for(size_t p=0;p<initial.size();++p){const auto &x=grid.geometry[p].x;chi[grid.active[p]]=stored[2*p]=std::log(initial[p].chi);alpha[grid.active[p]]=stored[2*p+1]=std::log(initial[p].alpha)-std::log(Chart(x,1).reference.value);}grid.Fill(chi,1);grid.Fill(alpha,1);const auto oldchi=chi,oldalpha=alpha;
  const auto fill_before=Clock::now();cache.FillPositiveLogs(chi,alpha,point);const Real fill_seconds=Seconds(fill_before);Real active_change=0;
  for(int p:grid.active)active_change=std::max({active_change,std::abs(chi[p]-oldchi[p]),std::abs(alpha[p]-oldalpha[p])});if(active_change!=0)throw std::runtime_error("cache changed active fields");
  V constant_chi(grid.count,1.23),constant_alpha(grid.count,-.42);cache.FillPositiveLogs(constant_chi,constant_alpha,point);for(int p=0;p<grid.count;++p)if(constant_chi[p]!=1.23||constant_alpha[p]!=-.42)throw std::runtime_error("constant changed");
  auto noisy_chi=oldchi,noisy_alpha=oldalpha;for(size_t p=0;p<grid.active.size();++p){const Real noise=1e-12*std::sin((p+1)*1.4142135623730951);noisy_chi[grid.active[p]]+=noise;noisy_alpha[grid.active[p]]+=noise;}cache.FillPositiveLogs(noisy_chi,noisy_alpha,point);Real maximum_noise=0;
  for(const auto &row:cache.rows)for(int p:row.targets)maximum_noise=std::max({maximum_noise,std::abs(noisy_chi[p]-chi[p]),std::abs(noisy_alpha[p]-alpha[p])});
  if(maximum_noise>cache.maximum_value_l1*1e-12+2e-14)throw std::runtime_error("deterministic noise amplification exceeds value L1 bound");
  std::cout<<"{\"phase\":\"cache\",\"radius\":"<<radius<<",\"rows\":"<<cache.rows.size()<<",\"assignments\":"<<cache.halo_assignments<<",\"build_seconds\":"<<build_seconds<<",\"fill_seconds\":"<<fill_seconds<<",\"maximum_condition\":"<<cache.maximum_condition<<",\"maximum_value_l1\":"<<cache.maximum_value_l1<<",\"maximum_reproduction\":"<<cache.maximum_reproduction<<",\"actual_target_reproduction\":"<<actual_reproduction<<",\"maximum_alias_distance\":"<<cache.maximum_alias_distance<<",\"maximum_output_noise\":"<<maximum_noise<<",\"active_change\":"<<active_change<<",\"support_counts\":{";
  bool first=true;for(const auto &entry:cache.support_counts){if(!first)std::cout<<',';first=false;std::cout<<'"'<<entry.first<<"\":"<<entry.second;}std::cout<<"},\"rejected_supports\":{";first=true;for(const auto &entry:cache.rejected_supports){if(!first)std::cout<<',';first=false;std::cout<<'"'<<entry.first<<"\":"<<entry.second;}std::cout<<"}}\n"<<std::flush;
  V oldpadded(2*grid.count),newpadded(2*grid.count);for(int i=0;i<grid.count;++i){oldpadded[i]=oldchi[i];oldpadded[grid.count+i]=oldalpha[i];newpadded[i]=chi[i];newpadded[grid.count+i]=alpha[i];}
  const auto plans_before=Clock::now();h::PunctureLogPlans existing(grid,point);const Real existing_seconds=Seconds(plans_before);std::map<size_t,const h::PunctureLogPlans::Row*> selected;for(const auto &row:existing.rows)selected[row.active]=&row;
  HNorm old_all,new_all,old_region[3],new_region[3],old_bulk,new_bulk,old_shell,new_shell;Real formula_error=0,old_at25482=0,new_at25482=0;size_t changed_active_derivative_cells=0;
  for(size_t p=0;p<grid.active.size();++p){const auto &x=grid.geometry[p].x;S oldq=Sample(grid,oldpadded,stored,p,0),newq=Sample(grid,newpadded,stored,p,0);if(selected.count(p))oldq=newq=existing.Sample(*selected[p],stored,2,0);
    const Real oldh=InitialHamiltonian(x,initial[p],oldq,formula_error),newh=InitialHamiltonian(x,initial[p],newq,formula_error),w=grid.volumes[p]/std::pow(initial[p].chi,1.5);old_all.Add(p,w,oldh);new_all.Add(p,w,newh);
    const int region=initial[p].r<.12?0:initial[p].r<.24?1:2;old_region[region].Add(p,w,oldh);new_region[region].Add(p,w,newh);
    if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0){old_bulk.Add(p,w,oldh);new_bulk.Add(p,w,newh);}else{old_shell.Add(p,w,oldh);new_shell.Add(p,w,newh);}
    if(oldh!=newh)++changed_active_derivative_cells;if(p==25482){old_at25482=oldh;new_at25482=newh;}}
  if(formula_error>5e-10)throw std::runtime_error("full tensor Hamiltonian disagrees with independent flat-metric identity");
  std::cout<<"{\"phase\":\"full_initial_H\",\"all_active_cells\":"<<grid.active.size()<<",\"existing_derivative_rows\":"<<existing.rows.size()<<",\"existing_derivative_build_seconds\":"<<existing_seconds<<",\"full_tensor_flat_identity_difference\":"<<formula_error<<",\"changed_active_derivative_cells\":"<<changed_active_derivative_cells<<",\"old_H_at25482\":"<<old_at25482<<",\"new_H_at25482\":"<<new_at25482<<",\"norms\":{";
  old_all.Print("old_all");std::cout<<',';new_all.Print("new_all");std::cout<<',';old_bulk.Print("old_bulk");std::cout<<',';new_bulk.Print("new_bulk");std::cout<<',';old_shell.Print("old_shell");std::cout<<',';new_shell.Print("new_shell");
  for(int r=0;r<3;++r){const std::string oldname="old_radius_partition_"+std::to_string(r),newname="new_radius_partition_"+std::to_string(r);std::cout<<',';old_region[r].Print(oldname.c_str());std::cout<<',';new_region[r].Print(newname.c_str());}
  const size_t peak=new_all.maximum_point;const auto &x=grid.geometry[peak].x;std::cout<<"},\"new_maximum_xyz\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"],\"new_maximum_radius\":"<<initial[peak].r<<",\"total_seconds\":"<<Seconds(start)<<",\"full_domain_H_evaluated\":true,\"coupled_evolution_run\":false,\"physics_rhs_changed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
