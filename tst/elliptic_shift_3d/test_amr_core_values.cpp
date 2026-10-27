// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Share the same actual-mesh/data/analytic controls without another executable.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/enriched_log_value.hpp"

std::vector<std::pair<Real,size_t>> NearestQuery(const Grid &grid,const Point &x){std::vector<std::pair<Real,size_t>> out;
  for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0){Real d=0;for(int a=0;a<3;++a)d+=std::pow(grid.geometry[p].x[a]-x[a],2);out.emplace_back(d,p);}
  std::partial_sort(out.begin(),out.begin()+125,out.end());out.resize(125);return out;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc!=3)throw std::invalid_argument("usage: initial.dat exact-jets.dat");const auto grid=MakeGrid();const auto initial=ReadInitial(argv[1],grid);const auto exact=ReadExact(argv[2]);const size_t target=34308;
  std::cout<<std::setprecision(17);V stored(2*initial.size()),ordinary(2*grid.count,0);
  for(size_t p=0;p<initial.size();++p)for(int f=0;f<2;++f){const auto &x=grid.geometry[p].x;stored[2*p+f]=std::log(f==0?initial[p].chi:initial[p].alpha)-std::log(Chart(x,f).reference.value);ordinary[f*grid.count+grid.active[p]]=stored[2*p+f];}grid.Fill(ordinary,2);
  auto oracle=ordinary,all=ordinary,coarsefine=ordinary,noisy=ordinary;std::set<int> active(grid.active.begin(),grid.active.end());std::map<int,const Grid::Row*> halos;
  for(const auto &row:grid.halos)halos[row.target]=&row;
  h::PuncturePosition point;size_t total=0,coarse_fine_rows=0,identity=0;Real maximum_l1=0,maximum_condition=0,maximum_value_error=0,maximum_noise=0,maximum_reproduction=0;
  int stale_rejections=0;const auto &x=grid.geometry[target].x;const Real spacing=Spacing(grid,target);const int per=grid.padded*grid.padded*grid.padded;
  for(int index:Stencil(grid,target))if(!active.count(index)){
    ++total;const auto query=PaddedPoint(grid,index);const auto &e=exact.at({2,size_t(index)});if(!(Radius(query.data())<.12))throw std::runtime_error("target outside prototype core");
    const auto near=NearestQuery(grid,query);std::vector<Point> nodes;for(const auto &entry:near){const auto &d=grid.geometry[entry.second].x;nodes.push_back({d[0],d[1],d[2]});}
    h::EnrichedLogValue plan;if(!plan.Build(query,nodes,spacing,point,grid.mesh.revision))throw std::runtime_error("explicit value plan rejection: "+plan.diagnostics().status);
    const auto &diag=plan.diagnostics();maximum_l1=std::max(maximum_l1,diag.value_weight_l1);maximum_condition=std::max(maximum_condition,diag.basis_condition);maximum_reproduction=std::max(maximum_reproduction,diag.value_reproduction);identity+=diag.coincident_active_identity;
    bool cf=false;const auto &receiving=grid.mesh.blocks[index/per];for(const auto &entry:halos.at(index)->entries)if(grid.mesh.blocks[entry.point/per].refinement!=receiving.refinement)cf=true;
    coarse_fine_rows+=cf;Real value_error[2]{},noise_error[2]{};
    for(int f=0;f<2;++f){V values,noised;for(const auto &entry:near){const Real value=stored[2*entry.second+f];values.push_back(value);noised.push_back(value+1e-12*std::sin((entry.second+1)*1.4142135623730951));}
      Real value=0,perturbed=0;if(!plan.Evaluate(values,point,grid.mesh.revision,value)||!plan.Evaluate(noised,point,grid.mesh.revision,perturbed))throw std::runtime_error("fresh value evaluation failed");
      all[f*grid.count+index]=value;noisy[f*grid.count+index]=perturbed;if(cf)coarsefine[f*grid.count+index]=value;
      oracle[f*grid.count+index]=StoredExact(query.data(),e,f).value;value_error[f]=std::abs(value-oracle[f*grid.count+index]);noise_error[f]=std::abs(perturbed-value);
      maximum_value_error=std::max(maximum_value_error,value_error[f]);maximum_noise=std::max(maximum_noise,noise_error[f]);
      if(noise_error[f]>diag.value_weight_l1*1e-12+2e-14)throw std::runtime_error("deterministic sample noise exceeded value-weight bound");
      V constant(125,1.23);Real unchanged=0;if(!plan.Evaluate(constant,point,grid.mesh.revision,unchanged)||unchanged!=1.23)throw std::runtime_error("constant value not preserved bitwise");
      auto stale=point;stale.time+=1;Real result=33;if(plan.Evaluate(values,stale,grid.mesh.revision,result)||result!=33)throw std::runtime_error("stale position stamp accepted");++stale_rejections;
      if(plan.Evaluate(values,point,grid.mesh.revision+1,result)||result!=33)throw std::runtime_error("stale topology revision accepted");++stale_rejections;
    }
    std::cout<<"{\"ghost\":"<<index<<",\"xyz\":["<<query[0]<<','<<query[1]<<','<<query[2]<<"],\"coarse_fine\":"<<cf<<",\"status\":\""<<diag.status<<"\",\"rank\":"<<diag.rank<<",\"condition\":"<<diag.basis_condition<<",\"value_l1\":"<<diag.value_weight_l1<<",\"reproduction\":"<<diag.value_reproduction<<",\"chi_log_value_error\":"<<value_error[0]<<",\"alpha_log_value_error\":"<<value_error[1]<<",\"chi_noise_amplification\":"<<noise_error[0]/1e-12<<",\"alpha_noise_amplification\":"<<noise_error[1]/1e-12<<"}\n";
  }
  Real active_change=0;for(size_t p=0;p<grid.active.size();++p)for(int f=0;f<2;++f)active_change=std::max({active_change,std::abs(all[f*grid.count+grid.active[p]]-ordinary[f*grid.count+grid.active[p]]),std::abs(coarsefine[f*grid.count+grid.active[p]]-ordinary[f*grid.count+grid.active[p]])});if(active_change!=0)throw std::runtime_error("value interpolation changed active fields");
  const auto &e=exact.at({0,target});for(int f=0;f<2;++f){const auto eq=StoredExact(x,e,f);Report("ordinary_mapped_fd",grid,target,f,Sample(grid,ordinary,stored,target,f),eq,e,total);Report("exact_internal_ghost_only",grid,target,f,Sample(grid,oracle,stored,target,f),eq,e,total);Report("value_all_used_ghosts",grid,target,f,Sample(grid,all,stored,target,f),eq,e,total);Report("value_coarsefine_only",grid,target,f,Sample(grid,coarsefine,stored,target,f),eq,e,total);Report("value_with_sample_noise_1e-12",grid,target,f,Sample(grid,noisy,stored,target,f),eq,e,total);}
  std::cout<<"{\"summary\":true,\"target\":"<<target<<",\"actual_cells\":"<<grid.active.size()<<",\"used_internal_ghosts\":"<<total<<",\"coarse_fine_rows\":"<<coarse_fine_rows<<",\"coincident_active_identity_rows\":"<<identity<<",\"maximum_value_l1\":"<<maximum_l1<<",\"maximum_basis_condition\":"<<maximum_condition<<",\"maximum_value_reproduction_error\":"<<maximum_reproduction<<",\"maximum_actual_log_value_error\":"<<maximum_value_error<<",\"maximum_output_noise\":"<<maximum_noise<<",\"noise_amplitude\":1e-12,\"active_field_change\":"<<active_change<<",\"stale_rejections\":"<<stale_rejections<<",\"physics_rhs_changed\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
