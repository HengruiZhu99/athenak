// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/puncture_log_value_plans.hpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include <chrono>
#include <type_traits>
static_assert(!std::is_constructible<h::PunctureLogValuePlans,Grid&&>::value,"temporary grid must be rejected");
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc<2)throw std::invalid_argument("usage: initial.dat [L1cap] [maximum_donors]");auto grid=MakeGrid();const auto initial=ReadInitial(argv[1],grid);h::PuncturePosition point;const Real cap=argc>2?std::stod(argv[2]):8;const int maximum_donors=argc>3?std::stoi(argv[3]):125;
  const auto targets=h::PunctureLogValuePlans::SelectTargets(grid,point,.12);std::cout<<std::setprecision(17)<<"{\"candidate_rows\":"<<targets.size()<<",\"row_budget\":10000,\"radius\":0.12,\"value_l1_cap\":"<<cap<<"}\n"<<std::flush;
  std::map<std::array<long long,4>,size_t> unique;Real min_spacing=1e99;for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0)min_spacing=std::min(min_spacing,Spacing(grid,p));
  for(int index:targets){const auto x=PaddedPoint(grid,index);const auto &b=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];std::array<long long,4> key;
    for(int a=0;a<3;++a)key[a]=std::llround((x[a]+grid.mesh.half_width)*2/min_spacing);key[3]=std::llround((b.hi[0]-b.lo[0])/(grid.n*min_spacing));++unique[key];}
  std::cout<<"{\"unique_physical_query_spacing_pairs\":"<<unique.size()<<",\"minimum_spacing\":"<<min_spacing<<"}\n"<<std::flush;
  const auto before=std::chrono::steady_clock::now();h::PunctureLogValuePlans cache(grid,point,.12,10000,cap,maximum_donors);const Real build_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
  Real actual_target_reproduction=0;
  for(const auto &row:cache.rows)for(int index:row.targets){const auto query=PaddedPoint(grid,index);const auto &b=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];const Real h=(b.hi[0]-b.lo[0])/grid.n;V polynomial;
    for(size_t p:row.donors){const auto &x=grid.geometry[p].x;const Real a=(x[0]-query[0])/h,c=(x[1]-query[1])/h,d=(x[2]-query[2])/h;
      polynomial.push_back(1.23+.17*a+.11*c*d-.13*a*a*c+.07*std::pow(a,5)+.09*a*a*c*d*d+1.09*std::log(Radius(x)/Radius(query.data())));}
    Real result;if(!row.plan.Evaluate(polynomial,point,grid.mesh.revision,result))throw std::runtime_error("actual target reproduction rejected");actual_target_reproduction=std::max(actual_target_reproduction,std::abs(result-1.23));}
  if(actual_target_reproduction>1e-8)throw std::runtime_error("actual query polynomial/log reproduction failed");
  V chi(grid.count,0),alpha(grid.count,0),stored(2*initial.size());for(size_t p=0;p<initial.size();++p){const auto &x=grid.geometry[p].x;chi[grid.active[p]]=stored[2*p]=std::log(initial[p].chi);alpha[grid.active[p]]=stored[2*p+1]=std::log(initial[p].alpha)-std::log(Chart(x,1).reference.value);}grid.Fill(chi,1);grid.Fill(alpha,1);
  const auto oldchi=chi,oldalpha=alpha;const auto fill_before=std::chrono::steady_clock::now();cache.FillPositiveLogs(chi,alpha,point);const Real fill_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-fill_before).count();
  Real active_change=0;for(int p:grid.active)active_change=std::max({active_change,std::abs(chi[p]-oldchi[p]),std::abs(alpha[p]-oldalpha[p])});if(active_change!=0)throw std::runtime_error("cache projected active fields");
  V constant_chi(grid.count,1.23),constant_alpha(grid.count,-.42);cache.FillPositiveLogs(constant_chi,constant_alpha,point);for(size_t p=0;p<constant_chi.size();++p)if(constant_chi[p]!=1.23||constant_alpha[p]!=-.42)throw std::runtime_error("cache changed constant");
  int rejections=0;auto reject=[&](auto action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}if(!caught)throw std::runtime_error("missing stale/cache rejection");++rejections;};
  auto rejected_chi=chi,rejected_alpha=alpha;auto stale=point;stale.position[0]+=.001;reject([&]{cache.FillPositiveLogs(rejected_chi,rejected_alpha,stale);});
  ++grid.mesh.revision;reject([&]{cache.FillPositiveLogs(rejected_chi,rejected_alpha,point);});--grid.mesh.revision;
  grid.mesh.blocks[0].lo[1]+=.001;reject([&]{cache.FillPositiveLogs(rejected_chi,rejected_alpha,point);});grid.mesh.blocks[0].lo[1]-=.001;
  // Restore exact saved descriptor values (floating add/sub need not be exact).
  auto rebuilt=MakeGrid();grid.mesh=rebuilt.mesh;
  if(rejected_chi!=chi||rejected_alpha!=alpha)throw std::runtime_error("rejection modified caller buffers");
  auto nonfinite_chi=chi,nonfinite_alpha=alpha;nonfinite_chi[grid.active[cache.rows.front().donors.front()]]=std::numeric_limits<Real>::quiet_NaN();const auto saved_alpha=nonfinite_alpha;
  reject([&]{cache.FillPositiveLogs(nonfinite_chi,nonfinite_alpha,point);});if(nonfinite_alpha!=saved_alpha)throw std::runtime_error("nonfinite input partially filled companion field");
  // Raw spatial-curvature partition only. Every r<.12 active cell is retained;
  // the full-domain coupled acceptance norms remain a separate unchanged gate.
  V oldpadded(2*grid.count),newpadded(2*grid.count);for(int i=0;i<grid.count;++i){oldpadded[i]=oldchi[i];oldpadded[grid.count+i]=oldalpha[i];newpadded[i]=chi[i];newpadded[grid.count+i]=alpha[i];}
  const auto plans_before=std::chrono::steady_clock::now();h::PunctureLogPlans existing(grid,point);const Real existing_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-plans_before).count();std::map<size_t,const h::PunctureLogPlans::Row*> selected;for(const auto &row:existing.rows)selected[row.active]=&row;
  size_t core=0,oldpoint=0,newpoint=0;Real oldmax=0,newmax=0,weight=0,oldsum=0,newsum=0;
  for(size_t p=0;p<grid.active.size();++p){const auto &x=grid.geometry[p].x;if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face>=0||Radius(x)>=.12)continue;++core;
    S oldq=Sample(grid,oldpadded,stored,p,0),newq=Sample(grid,newpadded,stored,p,0);if(selected.count(p))oldq=newq=existing.Sample(*selected[p],stored,2,0);
    const Real oldh=std::abs(HChi(x,oldq)),newh=std::abs(HChi(x,newq)),w=grid.volumes[p]/std::pow(initial[p].chi,1.5);weight+=w;oldsum+=w*oldh*oldh;newsum+=w*newh*newh;
    if(oldh>oldmax){oldmax=oldh;oldpoint=p;}if(newh>newmax){newmax=newh;newpoint=p;}}
  std::cout<<"{\"summary\":true,\"cache_rows\":"<<cache.rows.size()<<",\"halo_assignments\":"<<cache.halo_assignments<<",\"maximum_alias_distance\":"<<cache.maximum_alias_distance<<",\"maximum_condition\":"<<cache.maximum_condition<<",\"maximum_value_l1\":"<<cache.maximum_value_l1<<",\"maximum_reproduction_error\":"<<cache.maximum_reproduction<<",\"actual_target_polynomial_log_reproduction\":"<<actual_target_reproduction<<",\"enlarged_support_rows\":"<<cache.enlarged_support_rows<<",\"maximum_donors_used\":"<<cache.maximum_donors_used<<",\"support_counts\":{";
  bool first=true;for(const auto &entry:cache.support_counts){if(!first)std::cout<<',';first=false;std::cout<<'"'<<entry.first<<"\":"<<entry.second;}std::cout<<"},\"rejected_supports\":{";first=true;for(const auto &entry:cache.rejected_supports){if(!first)std::cout<<',';first=false;std::cout<<'"'<<entry.first<<"\":"<<entry.second;}
  std::cout<<"},\"build_seconds\":"<<build_seconds<<",\"fill_seconds\":"<<fill_seconds<<",\"active_change\":"<<active_change<<",\"stale_nonfinite_rejections\":"<<rejections<<",\"existing_derivative_rows\":"<<existing.rows.size()<<",\"existing_derivative_build_seconds\":"<<existing_seconds
    <<",\"core_partition_cells\":"<<core<<",\"old_core_H_max\":"<<oldmax<<",\"old_core_max_point\":"<<oldpoint<<",\"new_core_H_max\":"<<newmax<<",\"new_core_max_point\":"<<newpoint<<",\"old_core_H_rms\":"<<std::sqrt(oldsum/weight)<<",\"new_core_H_rms\":"<<std::sqrt(newsum/weight)<<",\"full_domain_acceptance_evaluated\":false,\"physics_rhs_changed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
