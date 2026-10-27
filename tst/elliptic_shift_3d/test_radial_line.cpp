// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Standalone CPU prototype: no native/pilot/CompositePoisson edits.
#include <chrono>
#include <iomanip>
#include <iostream>
#include "multigrid/composite_poisson.hpp"
#include "multigrid/composite_radial_line.hpp"
#include "multigrid/flexible_gmres.hpp"

using V=std::vector<Real>;
using Grid=cubed_sphere::CompositeDifference;
void FullPoisson(const Grid &grid,const V &u,V &out){
  V field(grid.count,0);for(size_t p=0;p<u.size();++p)field[grid.active[p]]=u[p];grid.Fill(field,1);out.resize(u.size());
  for(size_t p=0;p<u.size();++p){Real d[3],dd[3][3];grid.Derivatives(field,0,p,d,dd);out[p]=-(dd[0][0]+dd[1][1]+dd[2][2]);}
}

// Test-only copy of the existing transfer/cycle structure. The ONLY algorithm
// change is the smoother; the coarse direct solve remains CompositePoisson.
// No hierarchy/operator implementation is installed into production code.
class LineHierarchy {
 public:
  explicit LineHierarchy(const Grid &grid,Real damping)
      :grid_(&grid),smoother_(grid),damping_(damping){BuildCoarse();}
  explicit LineHierarchy(cubed_sphere::CompositeMesh mesh,bool radial,Real damping)
      :owned_(std::make_unique<Grid>(std::move(mesh),2,radial)),grid_(owned_.get()),smoother_(*grid_),damping_(damping){BuildCoarse();}
  V Cycle(const V &rhs) const {
    if(direct_)return direct_->Cycle(rhs);
    const auto &grid=*grid_;V u(rhs.size(),0),au;
    const auto apply=[&](const V &x,V &y){FullPoisson(grid,x,y);};smoother_.Smooth(rhs,u,3,apply,damping_);apply(u,au);
    const auto &cg=*coarse_->grid_;const int nc=cg.n,nf=grid.n;V restricted(cg.active.size(),0);
    for(size_t p=0;p<restricted.size();++p){const int m=p/(nc*nc*nc),i=p%nc,j=p/nc%nc,k=p/(nc*nc)%nc;Real volume=0;
      for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di){const size_t f=((m*nf+2*k+dk)*nf+2*j+dj)*nf+2*i+di;
        restricted[p]+=grid.volumes[f]*(rhs[f]-au[f]);volume+=grid.volumes[f];}restricted[p]/=volume;}
    const auto correction=coarse_->Cycle(restricted);V field(cg.count,0);
    for(size_t p=0;p<correction.size();++p)field[cg.active[p]]=correction[p];cg.Fill(field,1);
    for(size_t p=0;p<u.size();++p){const int m=p/(nf*nf*nf),index[3]={int(p%nf),int(p/nf%nf),int(p/(nf*nf)%nf)};int first[3];Real fraction[3];
      for(int a=0;a<3;++a){const Real q=(index[a]+.5)/2-.5;first[a]=int(std::floor(q));fraction[a]=q-first[a];}
      for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i)u[p]+=(i?fraction[0]:1-fraction[0])*(j?fraction[1]:1-fraction[1])*(k?fraction[2]:1-fraction[2])*field[cg.Index(m,first[0]+i,first[1]+j,first[2]+k)];}
    smoother_.Smooth(rhs,u,3,apply,damping_);return u;
  }
  void PrintDiagnostics(std::ostream &out) const{const auto &d=smoother_.GetDiagnostics();
    out<<"{\"n\":"<<grid_->n<<",\"cells\":"<<grid_->active.size()<<",\"lines\":"<<d.lines<<",\"maximum_line_length\":"<<d.maximum_length
       <<",\"radial_ghost_rows\":"<<d.radial_ghost_rows<<",\"same_line_ghost_entries\":"<<d.same_line_ghost_entries<<",\"lagged_ghost_entries\":"<<d.lagged_ghost_entries
       <<",\"cross_layer_entries\":"<<d.cross_layer_entries<<",\"nonadjacent_layer_entries\":"<<d.nonadjacent_layer_entries<<",\"minimum_relative_pivot\":"<<d.minimum_relative_pivot
       <<",\"factor_residual\":"<<smoother_.MaximumFactorResidual()<<",\"assembly_residual\":"<<smoother_.MaximumAssemblyResidual()<<"}";
    if(coarse_){out<<',';coarse_->PrintDiagnostics(out);}}
  Real MaximumAuditError() const{return std::max({smoother_.MaximumFactorResidual(),smoother_.MaximumAssemblyResidual(),coarse_?coarse_->MaximumAuditError():Real(0)});}
 private:
  std::unique_ptr<Grid> owned_;const Grid *grid_;multigrid::CompositeRadialLine smoother_;Real damping_;
  std::unique_ptr<LineHierarchy> coarse_;std::unique_ptr<multigrid::CompositePoisson> direct_;
  void BuildCoarse(){auto mesh=grid_->mesh;if(mesh.cells>=4&&mesh.cells%2==0){mesh.cells/=2;coarse_=std::make_unique<LineHierarchy>(std::move(mesh),grid_->radial_interface_stencils,damping_);}
    else direct_=std::make_unique<multigrid::CompositePoisson>(std::move(mesh),grid_->radial_interface_stencils);}
};

cubed_sphere::Jet Exact(const Real x[3]){using cubed_sphere::Jet;Jet q[3];for(int a=0;a<3;++a)q[a]=Jet::Variable(x[a],a);
  return (Jet(1)-q[0]*q[0]-q[1]*q[1]-q[2]*q[2])*(Jet(1)+Jet(.2)*q[0]-Jet(.1)*q[1]+Jet(.15)*q[2]*q[2]+Jet(.05)*q[0]*q[1]);}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  std::string mode="line",source="discrete";int n=8,maxiter=120,restart=24,subdivisions=1;Real damping=.65;bool refine=false;
  for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--refine"){refine=true;continue;}if(i+1>=argc)throw std::invalid_argument("missing option value");const std::string value=argv[++i];
    if(key=="--mode")mode=value;else if(key=="--source")source=value;else if(key=="--n")n=std::stoi(value);else if(key=="--damping")damping=std::stod(value);else if(key=="--maxiter")maxiter=std::stoi(value);else if(key=="--restart")restart=std::stoi(value);else if(key=="--subdivisions")subdivisions=std::stoi(value);else throw std::invalid_argument("unknown radial-line test option");}
  if((mode!="line"&&mode!="point")||(source!="discrete"&&source!="continuum"))throw std::invalid_argument("mode/source value");
  cubed_sphere::CompositeMesh mesh(.25,{.45,.55,.60,.65,.70,.75,1.},subdivisions,n,2,2);
  if(refine)mesh.RefineCartesian(0);if(mesh.blocks.size()*size_t(n)*n*n>50000)throw std::runtime_error("radial-line test exceeds50000cells");
  const auto start=std::chrono::steady_clock::now();multigrid::CompositePoisson poisson(mesh,true);const auto &grid=poisson.Grid();
  LineHierarchy lines(grid,damping);const Real setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cerr<<"setup complete: "<<grid.active.size()<<" cells, original scalar operator retained\n";
  V exact(grid.active.size()),rhs(grid.active.size()),solution(grid.active.size(),0),random(grid.active.size()),a,b;
  Real volume=0;for(size_t p=0;p<exact.size();++p){const auto jet=Exact(grid.geometry[p].x);exact[p]=jet.f;rhs[p]=-jet.h[0][0]-jet.h[1][1]-jet.h[2][2];random[p]=std::sin(p*.31)+.1*std::cos(p*.17);volume+=grid.volumes[p];}
  if(source=="discrete")poisson.Operator(exact,rhs);
  poisson.Operator(random,a);FullPoisson(grid,random,b);Real operator_difference=0;for(size_t i=0;i<a.size();++i)operator_difference=std::max(operator_difference,std::abs(a[i]-b[i])/std::max(Real(1),std::abs(a[i])));
  const auto dot=[&](const V &x,const V &y){Real value=0;for(size_t p=0;p<x.size();++p)value+=grid.volumes[p]*x[p]*y[p];return value/volume;};
  const multigrid::Operator apply=[&](const V &x,V &y){poisson.Operator(x,y);};
  const multigrid::Operator precondition=[&](const V &x,V &y){y=mode=="line"?lines.Cycle(x):poisson.Cycle(x);};
  const auto before=std::chrono::steady_clock::now();const auto report=multigrid::FlexibleGMRES(apply,precondition,dot,rhs,solution,1e-9,0.,maxiter,restart);
  const Real solve_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();V residual,error(exact.size());apply(solution,residual);Real maximum=0;
  for(size_t p=0;p<residual.size();++p){residual[p]=rhs[p]-residual[p];maximum=std::max(maximum,std::abs(residual[p]));error[p]=solution[p]-exact[p];}
  const Real audit=lines.MaximumAuditError();if(!report.converged||operator_difference>1e-12||audit>1e-12)status=1;
  std::cout<<std::setprecision(17)<<"{\"mode\":\""<<mode<<"\",\"source\":\""<<source<<"\",\"n\":"<<n<<",\"cells\":"<<exact.size()<<",\"shells\":7,\"radii\":[0.45,0.55,0.60,0.65,0.70,0.75,1],\"angular_subdivisions\":"<<subdivisions
    <<",\"cartesian_refined\":"<<refine<<",\"damping\":"<<damping<<",\"maximum_iterations\":"<<maxiter<<",\"restart\":"<<restart<<",\"iterations\":"<<report.iterations<<",\"applications\":"<<report.applications
    <<",\"converged\":"<<report.converged<<",\"original_residual_rms\":"<<std::sqrt(dot(residual,residual))<<",\"original_residual_max\":"<<maximum<<",\"solution_error_rms\":"<<std::sqrt(dot(error,error))
    <<",\"operator_difference\":"<<operator_difference<<",\"maximum_block_audit_error\":"<<audit<<",\"setup_seconds\":"<<setup_seconds<<",\"solve_seconds\":"<<solve_seconds<<",\"residual_history\":[";
  for(size_t i=0;i<report.residual.size();++i){if(i)std::cout<<',';std::cout<<report.residual[i];}
  std::cout<<"],\"levels\":[";lines.PrintDiagnostics(std::cout);std::cout<<"],\"scope\":\"CPU scalar preconditioner prototype only; unchanged full operator; no native/GPU/vector/evolution validation\"}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';status=2;}Kokkos::finalize();return status;}
