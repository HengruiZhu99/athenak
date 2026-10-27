// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// CPU qualification of optional global AMR hierarchy; no native task dispatch.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include "multigrid/composite_poisson.hpp"
#include "multigrid/flexible_gmres.hpp"
#include "manufactured.hpp"

namespace h=z4c::hyperboloidal;
using V=multigrid::Vector;
using Grid=cubed_sphere::CompositeDifference;
using Clock=std::chrono::steady_clock;
void Require(bool b,const char *message){if(!b)throw std::runtime_error(message);}
double Seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
cubed_sphere::Jet Scalar(const Real x[3]){using cubed_sphere::Jet;Jet q[3];for(int a=0;a<3;++a)q[a]=Jet::Variable(x[a],a);
  return (Jet(1)-q[0]*q[0]-q[1]*q[1]-q[2]*q[2])*(Jet(1)+Jet(.2)*q[0]-Jet(.1)*q[1]+Jet(.15)*q[2]*q[2]+Jet(.05)*q[0]*q[1]);}
struct Region {Real volume=0,square=0,maximum=0;size_t cells=0;
  void Add(Real w,const V &v,size_t p,int components){volume+=w;++cells;for(int c=0;c<components;++c){const Real r=v[components*p+c];square+=w*r*r;maximum=std::max(maximum,std::abs(r));}}
  void Print(const char *name)const{std::cout<<"\""<<name<<"\":{\"cells\":"<<cells<<",\"residual_rms\":"<<(volume>0?std::sqrt(square/volume):0)<<",\"residual_max\":"<<maximum<<"}";}
};
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  int n=6,shells=1,subdivisions=1,maxiter=160,restart=40;bool mixed=false,lines=true,topology=true,curved=true,continuum=false;std::string equation="scalar";
  for(int i=1;i<argc;++i){const std::string key=argv[i];if(i+1>=argc)throw std::invalid_argument("missing test option value");const std::string value=argv[++i];
    if(key=="--n")n=std::stoi(value);else if(key=="--shells")shells=std::stoi(value);else if(key=="--subdivisions")subdivisions=std::stoi(value);
    else if(key=="--mixed")mixed=std::stoi(value);else if(key=="--lines")lines=std::stoi(value);else if(key=="--topology")topology=std::stoi(value);
    else if(key=="--curved")curved=std::stoi(value);else if(key=="--continuum")continuum=std::stoi(value);else if(key=="--equation")equation=value;
    else if(key=="--maxiter")maxiter=std::stoi(value);else if(key=="--restart")restart=std::stoi(value);else throw std::invalid_argument("unknown topology cycle test option");}
  Require((n==6||n==8)&&(shells==1||shells==2||shells==7)&&(subdivisions==1||subdivisions==2),"bounded topology cycle shape");
  Require(equation=="scalar"||equation=="vector","equation must be scalar or vector");
  std::vector<Real> radii=shells==1?std::vector<Real>{1}:shells==2?std::vector<Real>{.65,1}:std::vector<Real>{.45,.55,.60,.65,.70,.75,1};
  cubed_sphere::CompositeMesh mesh(.25,radii,subdivisions,n,2,2);
  if(mixed){mesh.RefineCartesian(0);int child=-1;for(size_t b=0;b<mesh.blocks.size();++b)if(mesh.blocks[b].face<0&&mesh.blocks[b].refinement==1){child=b;break;}mesh.RefineCartesian(child);}
  Require(mesh.blocks.size()*size_t(n)*n*n<=50000,"topology cycle cell budget");
  const auto start=Clock::now();multigrid::CompositePoisson poisson(mesh,true,lines,topology);
  Grid grid(mesh,equation=="scalar"?2:4,true);const size_t points=grid.active.size();const int components=equation=="scalar"?1:3;
  std::vector<h::MetricJet<Real>> metric;std::vector<h::SpatialGeometry<Real>> geometry;V weights=grid.volumes;
  V exact(components*points),continuum_rhs(exact.size()),solution(exact.size(),0),rhs,affine(exact.size(),0);
  for(size_t p=0;p<points;++p){
    if(components==1){const auto jet=Scalar(grid.geometry[p].x);exact[p]=jet.f;continuum_rhs[p]=-jet.h[0][0]-jet.h[1][1]-jet.h[2][2];}
    else{const auto state=manufactured::State(grid.geometry[p].x,curved);metric.push_back(state.metric);geometry.push_back(h::Geometry(state.metric));
      weights[p]*=std::sqrt(geometry.back().determinant);Real source[3];h::EllipticShiftSource(state,geometry.back(),0.,source);
      for(int c=0;c<3;++c){exact[3*p+c]=state.beta.value[c];continuum_rhs[3*p+c]=-source[c];}}
  }
  const Real volume=std::accumulate(weights.begin(),weights.end(),Real(0));
  const auto dot=[&](const V &x,const V &y){Real sum=0;for(size_t p=0;p<points;++p)for(int c=0;c<components;++c)sum+=weights[p]*x[components*p+c]*y[components*p+c];return sum/volume;};
  const auto vector_apply=[&](const V &u,V &out,bool boundary){V field(3*grid.count,0);for(size_t p=0;p<points;++p)for(int c=0;c<3;++c)field[c*grid.count+grid.active[p]]=u[3*p+c];
    grid.Fill(field,3,boundary?std::function<Real(int,const std::array<Real,3>&)>([](int c,const auto &x){return -x[c];}):nullptr);out.resize(u.size());
    for(size_t p=0;p<points;++p){h::VectorJet<Real> jet{};for(int c=0;c<3;++c){jet.value[c]=u[3*p+c];Real d[3],dd[3][3];grid.Derivatives(field,c,p,d,dd);
        for(int a=0;a<3;++a){jet.d[a][c]=d[a];for(int b=0;b<3;++b)jet.dd[a][b][c]=dd[a][b];}}
      Real result[3];h::ApplyEllipticShift(metric[p],geometry[p],jet,result);for(int c=0;c<3;++c)out[3*p+c]=-result[c];}
  };
  multigrid::Operator apply=components==1?multigrid::Operator([&](const V &u,V &out){poisson.Operator(u,out);}):multigrid::Operator([&](const V &u,V &out){vector_apply(u,out,false);});
  const auto precondition=[&](const V &u,V &out){out.resize(u.size());for(int c=0;c<components;++c){V scalar(points);for(size_t p=0;p<points;++p)scalar[p]=u[components*p+c];const auto correction=poisson.Cycle(scalar);for(size_t p=0;p<points;++p)out[components*p+c]=correction[p];}};
  if(components==3)vector_apply(V(exact.size(),0),affine,true);
  V exact_applied;if(components==1)apply(exact,exact_applied);else vector_apply(exact,exact_applied,true);
  rhs.resize(exact.size());V truncation(exact.size());for(size_t p=0;p<rhs.size();++p){rhs[p]=(continuum?continuum_rhs[p]:exact_applied[p])-affine[p];truncation[p]=exact_applied[p]-continuum_rhs[p];}
  // Independent scalar stencil evaluation checks that hierarchy selection leaves
  // the fine operator unchanged, including mixed leaves and all global halos.
  V random(points),a,b,field(poisson.Grid().count,0);for(size_t p=0;p<points;++p){random[p]=std::sin(.13*p)+.2*std::cos(.37*p);field[poisson.Grid().active[p]]=random[p];}
  poisson.Operator(random,a);poisson.Grid().Fill(field,1);b.resize(points);Real operator_difference=0;
  for(size_t p=0;p<points;++p){Real d[3],dd[3][3];poisson.Grid().Derivatives(field,0,p,d,dd);b[p]=-dd[0][0]-dd[1][1]-dd[2][2];operator_difference=std::max(operator_difference,std::abs(a[p]-b[p])/std::max(Real(1),std::abs(a[p])));}
  const Real setup=Seconds(start);std::cerr<<"setup "<<equation<<" N"<<n<<" shells="<<shells<<" blocks="<<mesh.blocks.size()<<" cells="<<points<<" levels="<<poisson.Levels()<<" seconds="<<setup<<'\n';
  const auto before=Clock::now();const auto report=multigrid::FlexibleGMRES(apply,precondition,dot,rhs,solution,1e-9,0,maxiter,restart);const Real solve=Seconds(before);
  V residual,error(exact.size());apply(solution,residual);for(size_t p=0;p<residual.size();++p){residual[p]=rhs[p]-residual[p];error[p]=solution[p]-exact[p];}
  Region all,bulk,shell,interface,scri,refined;
  for(size_t p=0;p<points;++p){const auto &block=mesh.blocks[p/(n*n*n)];const int i=p%n,j=p/n%n,k=p/(n*n)%n;all.Add(weights[p],residual,p,components);
    (block.face<0?bulk:shell).Add(weights[p],residual,p,components);if(block.refinement>0)refined.Add(weights[p],residual,p,components);
    if(i<2||j<2||k<2||i>=n-2||j>=n-2||k>=n-2)interface.Add(weights[p],residual,p,components);
    if(block.layer==shells-1&&i>=n-2)scri.Add(weights[p],residual,p,components);}
  const Real rms=std::sqrt(dot(residual,residual));
  std::cout<<std::setprecision(17)<<"{\"equation\":\""<<equation<<"\",\"source\":\""<<(continuum?"continuum_mms":"discrete_mms")<<"\",\"n\":"<<n<<",\"shells\":"<<shells<<",\"subdivisions\":"<<subdivisions
      <<",\"mixed_amr\":"<<mixed<<",\"curved_metric\":"<<(components==3&&curved)<<",\"lines\":"<<lines<<",\"topology\":"<<poisson.UsesTopologyCoarsening()<<",\"blocks\":"<<mesh.blocks.size()<<",\"cells\":"<<points<<",\"order\":"<<grid.order
      <<",\"setup_seconds\":"<<setup<<",\"solve_seconds\":"<<solve<<",\"iterations\":"<<report.iterations<<",\"applications\":"<<report.applications<<",\"converged\":"<<report.converged
      <<",\"operator_difference_relative\":"<<operator_difference<<",\"residual_rms\":"<<rms<<",\"error_rms\":"<<std::sqrt(dot(error,error))<<",\"continuum_truncation_rms\":"<<std::sqrt(dot(truncation,truncation))
      <<",\"coarse_cells\":"<<poisson.CoarseCells()<<",\"hierarchy_cells\":[";
  const auto hc=poisson.HierarchyCells(),hh=poisson.HierarchyHaloRows();for(size_t i=0;i<hc.size();++i){if(i)std::cout<<',';std::cout<<hc[i];}std::cout<<"],\"hierarchy_halo_rows\":[";
  for(size_t i=0;i<hh.size();++i){if(i)std::cout<<',';std::cout<<hh[i];}std::cout<<"],\"regions\":{";all.Print("all");std::cout<<',';bulk.Print("bulk");std::cout<<',';shell.Print("shell");std::cout<<',';interface.Print("interfaces_edges_corners");std::cout<<',';scri.Print("outer_boundary_collar");std::cout<<',';refined.Print("refined");
  std::cout<<"},\"residual_history\":[";for(size_t i=0;i<report.residual.size();++i){if(i)std::cout<<',';std::cout<<report.residual[i];}
  std::cout<<"],\"elliptic_atol\":1e-9,\"maximum_residual_target\":1e-7,\"scope\":\"CPU global elliptic hierarchy qualification; no native evolution/GPU/MPI validation\"}\n";
  if(!report.converged||rms>1e-9||all.maximum>1e-7||operator_difference>1e-12)status=1;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=2;}Kokkos::finalize();return status;}
