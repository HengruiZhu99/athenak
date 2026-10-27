// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <chrono>
#include <iomanip>
#include <iostream>
#include "multigrid/composite_topology_transfer.hpp"

using Mesh=cubed_sphere::CompositeMesh;
using Transfer=multigrid::CompositeLevelTransfer;
using Plan=multigrid::CompositeCoarseningPlan;
using Vector=std::vector<Real>;

void Require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
size_t Cells(const Mesh &m){return m.blocks.size()*m.cells*m.cells*m.cells;}
Real Polynomial(const std::array<Real,3> &x,int degree){
  Real p=.7+.4*x[0]-.3*x[1]+.2*x[2];
  if(degree==2)p+=.13*x[0]*x[0]-.11*x[1]*x[1]+.09*x[2]*x[2]
      +.07*x[0]*x[1]-.05*x[0]*x[2]+.03*x[1]*x[2];
  return p;
}
Vector Sample(const std::vector<std::array<Real,3>> &points,int degree){Vector v;for(const auto &x:points)v.push_back(Polynomial(x,degree));return v;}
Real MaxError(const Vector &a,const Vector &b){Require(a.size()==b.size(),"error shape");Real e=0;for(size_t i=0;i<a.size();++i)e=std::max(e,std::abs(a[i]-b[i]));return e;}
Real Mass(const Vector &u,const Vector &v){long double s=0;for(size_t i=0;i<u.size();++i)s+=static_cast<long double>(u[i])*v[i];return Real(s);}
Vector AveragePolynomial(const Mesh &mesh,const std::vector<std::array<Real,3>> &points){
  auto v=Sample(points,2);const size_t per=mesh.cells*mesh.cells*mesh.cells;
  for(size_t b=0;b<mesh.blocks.size();++b)if(mesh.blocks[b].face<0){const auto &block=mesh.blocks[b];
    const Real correction=(.13*std::pow((block.hi[0]-block.lo[0])/mesh.cells,2)
        -.11*std::pow((block.hi[1]-block.lo[1])/mesh.cells,2)
        +.09*std::pow((block.hi[2]-block.lo[2])/mesh.cells,2))/12;
    for(size_t p=b*per;p<(b+1)*per;++p)v[p]+=correction;
  }return v;
}
void SameShells(const Mesh &fine,const Mesh &coarse){std::vector<cubed_sphere::CompositeBlock> f,c;
  for(const auto &b:fine.blocks)if(b.face>=0)f.push_back(b);for(const auto &b:coarse.blocks)if(b.face>=0)c.push_back(b);
  Require(f.size()==c.size(),"shell count changed");
  for(size_t i=0;i<f.size();++i){Require(f[i].face==c[i].face&&f[i].layer==c[i].layer&&f[i].refinement==c[i].refinement,"shell identity changed");
    for(int a=0;a<3;++a)Require(f[i].lo[a]==c[i].lo[a]&&f[i].hi[a]==c[i].hi[a],"shell bounds changed");}
}
void CheckTransfer(const Mesh &fine,const Mesh &coarse,const char *name,int stage,bool topology){
  const auto start=std::chrono::steady_clock::now();Transfer t(fine,coarse);
  auto constant=t.Restrict(Vector(Cells(fine),3.25));auto back=t.Prolong(Vector(Cells(coarse),3.25));
  Require(MaxError(constant,Vector(constant.size(),3.25))==0&&MaxError(back,Vector(back.size(),3.25))==0,"constant transfer failed");
  const auto fl=Sample(t.FinePoints(),1),cl=Sample(t.CoarsePoints(),1),fq=Sample(t.FinePoints(),2),cq=Sample(t.CoarsePoints(),2);
  const Real pe1=MaxError(t.Prolong(cl),fl),pe2=MaxError(t.Prolong(cq),fq),re1=MaxError(t.Restrict(fl),cl);
  auto restricted=t.Restrict(fq);const Real mf=Mass(fq,t.fine_volume),mc=Mass(restricted,t.coarse_volume),mass=std::abs(mf-mc)/std::max(Real(1),std::abs(mf));
  Require(mass<3e-14&&t.maximum_coverage_error<2e-10,"represented mass/coverage failed");
  Real re2=-1;if(topology){SameShells(fine,coarse);re2=MaxError(t.Restrict(AveragePolynomial(fine,t.FinePoints())),AveragePolynomial(coarse,t.CoarsePoints()));
    Require(re1<2e-12&&re2<2e-12,"Cartesian topology cell-average polynomial restriction failed");}
  // Point-major multi-component contract, including an independent component.
  Vector pair(2*fl.size());for(size_t p=0;p<fl.size();++p){pair[2*p]=fl[p];pair[2*p+1]=-2*fl[p];}
  const auto rpair=t.Restrict(pair,2);for(size_t p=0;p<cl.size();++p)Require(std::abs(rpair[2*p+1]+2*rpair[2*p])<1e-13,"component transfer failed");
  std::cout<<"{\"case\":\""<<name<<"\",\"stage\":"<<stage<<",\"fine_blocks\":"<<fine.blocks.size()<<",\"coarse_blocks\":"<<coarse.blocks.size()
      <<",\"fine_n\":"<<fine.cells<<",\"coarse_n\":"<<coarse.cells<<",\"fine_cells\":"<<Cells(fine)<<",\"coarse_cells\":"<<Cells(coarse)
      <<",\"prolong_linear_max\":"<<pe1<<",\"prolong_quadratic_max\":"<<pe2<<",\"restrict_linear_nodal_max\":"<<re1
      <<",\"restrict_quadratic_cell_average_max\":"<<re2<<",\"represented_mass_relative\":"<<mass<<",\"coverage_max\":"<<t.maximum_coverage_error
      <<",\"quadratic_rows\":"<<t.quadratic_rows<<",\"linear_fallback_rows\":"<<t.linear_rows<<",\"identity_rows\":"<<t.identity_rows
      <<",\"cross_patch_rows\":"<<t.cross_patch_rows<<",\"maximum_prolongation_l1\":"<<t.maximum_prolongation_l1
      <<",\"maximum_moment_error\":"<<t.maximum_moment_error<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
  Require(pe1<2e-10&&pe2<2e-10&&t.linear_rows==0,"physical polynomial prolongation failed");
}
int main(){std::cout<<std::setprecision(17);try{
  Mesh mixed(.25,{.65,1},1,4,2,2);mixed.RefineCartesian(0);int child=-1;
  for(size_t b=0;b<mixed.blocks.size();++b)if(mixed.blocks[b].face<0&&mixed.blocks[b].refinement==1){child=b;break;}
  Require(child>=0,"missing refined child");mixed.RefineCartesian(child);Require(mixed.blocks.size()==34,"mixed leaf count");
  for(int r=0;r<2;++r){auto round=multigrid::CoarsenCompositeCartesian(mixed);Require(round.merged_families==1,"merge complete finest family only");CheckTransfer(mixed,round.coarse,"mixed_topology",r,true);mixed=std::move(round.coarse);}
  Require(mixed.blocks.size()==20,"mixed root recovery");Require(multigrid::CoarsenCompositeCartesian(mixed).merged_families==0,"base roots must remain");
  const std::vector<Real> radii={.45,.55,.60,.65,.70,.75,1};
  Mesh deep(.25,radii,1,2,2,2);
  for(int depth=0;depth<3;++depth){std::vector<int> ids;for(size_t b=0;b<deep.blocks.size();++b)if(deep.blocks[b].face<0)ids.push_back(b);
    for(auto i=ids.rbegin();i!=ids.rend();++i)deep.RefineCartesian(*i);}
  Require(Cells(deep)==33104,"deep cell budget");auto plan=Plan::Build(deep);Require(plan.dense_supported&&plan.TerminalCells()==400&&plan.transitions.size()==3,"deep hierarchy cap");
  for(size_t r=0;r<plan.transitions.size();++r)CheckTransfer(plan.levels[r],plan.levels[r+1],"deep_topology",r,true);
  Mesh odd(.25,radii,1,6,2,2);auto odd_plan=Plan::Build(odd);Require(odd_plan.dense_supported&&odd_plan.TerminalCells()==400&&odd_plan.levels[1].cells==3,"odd logical hierarchy");
  for(size_t r=0;r<odd_plan.transitions.size();++r)CheckTransfer(odd_plan.levels[r],odd_plan.levels[r+1],"odd_logical",r,false);
  Mesh wide(.25,radii,2,2,2,2);auto rejected=Plan::Build(wide);Require(!rejected.dense_supported&&rejected.TerminalCells()==1408&&!rejected.unsupported_reason.empty(),"unchanged shell cap guard");
  // An incomplete family must not be merged or silently declared supported.
  Mesh incomplete(.25,{1},1,2,2,2);incomplete.RefineCartesian(0);incomplete.blocks.pop_back();auto invalid=Plan::Build(incomplete);
  Require(!invalid.dense_supported&&!invalid.unsupported_reason.empty(),"incomplete sibling guard");
  std::cout<<"{\"status\":\"passed\",\"deep_initial_cells\":33104,\"deep_terminal_cells\":400,\"odd_n6_cells\":10800,\"odd_n3_cells\":1350,\"odd_n2_cells\":400,\"unsupported_shell_floor\":1408,\"dense_matrix_allocated\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
