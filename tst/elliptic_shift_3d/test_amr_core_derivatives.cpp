// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Targeted READ-ONLY field differentiation diagnostic on the actual N6 AMR mesh.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/puncture_refinement.hpp"
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"
#include "z4c/hyperboloidal/positive_field_chart.hpp"

namespace h=z4c::hyperboloidal;
using Grid=cubed_sphere::CompositeDifference;using Point=std::array<Real,3>;using V=std::vector<Real>;
using S=h::ScalarJet<Real>;using Plan=h::EnrichedLogDerivative;
struct Datum {Real r,chi,alpha,arr,khat;};
struct Exact {Real r,chi,cp,cpp,alpha,ap,app,arr,arrp,khat,khatp,beta,bp,bpp;};
Grid MakeGrid(){auto mesh=cubed_sphere::CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,6,2,2);
  auto refined=h::RefinePunctureNeighborhood(std::move(mesh),h::PuncturePosition{},3,50000);return Grid(std::move(refined.mesh),4,true);}
std::vector<Datum> ReadInitial(const std::string &path,const Grid &grid){std::ifstream f(path);std::string magic;size_t count;f>>magic>>count;
  if(magic!="ISOTROPIC_M1_3"||count!=grid.active.size()||count!=47088)throw std::runtime_error("actual initial layout mismatch");std::vector<Datum> out(count);
  for(size_t p=0;p<count;++p){auto &d=out[p];f>>d.r>>d.chi>>d.alpha>>d.arr>>d.khat;Real r2=0;for(Real x:grid.geometry[p].x)r2+=x*x;
    if(!f||std::abs(d.r-std::sqrt(r2))>2e-13||!(d.chi>0&&d.alpha>0))throw std::runtime_error("actual initial radius/value mismatch");}return out;}
Real Radius(const Real x[3]){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
Real Spacing(const Grid &grid,size_t p){return 1/(grid.n*std::abs(grid.geometry[p].J[0][0]));}
h::PositiveFieldChart<Real> Chart(const Real x[3],int field){h::PositiveFieldChart<Real> c(h::PositiveFieldMode::log_reference);
  if(field==1){c.reference.value=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;for(int a=0;a<3;++a){c.reference.d[a]=x[a];c.reference.dd[a][a]=1;}}return c;}
std::vector<size_t> Targets(const Grid &grid){std::vector<Point> desired;const Real a=23./384,b=1./384,c=19./384,d=21./384;
  for(int s=0;s<8;++s)desired.push_back({s&1?a:-a,s&2?a:-a,s&4?b:-b});
  desired.push_back({-a,-a,-c});desired.push_back({a,a,c});desired.push_back({-a,-a,-d});
  desired.push_back({-a,-c,-b});desired.push_back({-a,-b,-b});desired.push_back({-.07,-.07,0});
  desired.push_back({-.044270833333333336,-.044270833333333336,-.044270833333333336});desired.push_back({b,b,b});
  std::vector<size_t> out;for(const auto &x:desired){Real best=1e99;size_t id=0;
    for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0){Real distance=0;for(int a=0;a<3;++a)distance+=std::pow(grid.geometry[p].x[a]-x[a],2);
      if(distance<best){best=distance;id=p;}}
    if(std::find(out.begin(),out.end(),id)==out.end())out.push_back(id);}
  if(out.front()!=34308)throw std::runtime_error("reported maximum point not reproduced");return out;}
std::vector<int> Stencil(const Grid &grid,size_t p){std::set<int> points;const int center=grid.active[p];points.insert(center);
  for(int a=0;a<3;++a)for(int i=-2;i<=2;++i)points.insert(center+grid.Offset(a,i));
  for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)for(int i=-2;i<=2;++i)for(int j=-2;j<=2;++j)points.insert(center+grid.Offset(a,i)+grid.Offset(b,j));
  return {points.begin(),points.end()};}
Point PaddedPoint(const Grid &grid,int index){const int m=index/(grid.padded*grid.padded*grid.padded),i=index%grid.padded-grid.ng,j=index/grid.padded%grid.padded-grid.ng,k=index/(grid.padded*grid.padded)%grid.padded-grid.ng;
  const Real q[3]={(i+.5)/grid.n,(j+.5)/grid.n,(k+.5)/grid.n};Point x;grid.mesh.Forward(m,q,x.data());return x;}
std::vector<std::pair<Real,size_t>> Nearest(const Grid &grid,size_t target,size_t count){std::vector<std::pair<Real,size_t>> near;const auto &x=grid.geometry[target].x;
  for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0){Real d2=0;for(int a=0;a<3;++a)d2+=std::pow(grid.geometry[p].x[a]-x[a],2);near.emplace_back(d2,p);}
  std::partial_sort(near.begin(),near.begin()+count,near.end());near.resize(count);return near;}
void Catalog(const Grid &grid,const std::vector<size_t> &targets,const std::string &path){std::ofstream f(path);f<<std::setprecision(17)<<"kind,id,x,y,z,r\n";
  std::set<int> active(grid.active.begin(),grid.active.end()),ghost;std::set<size_t> donors;
  auto line=[&](int kind,size_t id,const Real x[3]){f<<kind<<','<<id<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<Radius(x)<<'\n';};
  for(size_t p:targets){line(0,p,grid.geometry[p].x);for(int index:Stencil(grid,p))if(!active.count(index))ghost.insert(index);
    for(const auto &neighbor:Nearest(grid,p,256))donors.insert(neighbor.second);}
  for(int p:ghost){const auto x=PaddedPoint(grid,p);if(!(Radius(x.data())<.3))throw std::runtime_error("target oracle is not internal");line(2,p,x.data());}
  for(size_t p:donors)line(3,p,grid.geometry[p].x);
  // Existing independent generator treats the final unique radius as scri.
  // Include that real endpoint explicitly; otherwise the last lapse is wrong.
  const Real scri[3]={1,0,0};line(1,0,scri);
  std::cout<<"{\"targets\":"<<targets.size()<<",\"internal_ghosts\":"<<ghost.size()<<",\"active_donors\":"<<donors.size()<<",\"actual_cells\":"<<grid.active.size()<<"}\n";}
std::map<std::pair<int,size_t>,Exact> ReadExact(const std::string &path){std::ifstream f(path);std::string magic;size_t count;f>>magic>>count;if(magic!="EXACT_WORMHOLE_V1")throw std::runtime_error("exact jet magic");
  std::map<std::pair<int,size_t>,Exact> out;for(size_t i=0;i<count;++i){int kind;size_t id;Exact e;f>>kind>>id>>e.r>>e.chi>>e.cp>>e.cpp>>e.alpha>>e.ap>>e.app>>e.arr>>e.arrp>>e.khat>>e.khatp>>e.beta>>e.bp>>e.bpp;
    if(!f||!out.emplace(std::make_pair(kind,id),e).second)throw std::runtime_error("exact jet payload/duplicates");}return out;}
S Radial(const Real x[3],Real value,Real first,Real second){const Real r=Radius(x);S q{};q.value=value;
  for(int a=0;a<3;++a){q.d[a]=first*x[a]/r;for(int b=0;b<3;++b)q.dd[a][b]=(second-first/r)*x[a]*x[b]/(r*r)+(a==b?first/r:0);}return q;}
S Physical(const Real x[3],const Exact &e,int field){return field==0?Radial(x,e.chi,e.cp,e.cpp):Radial(x,e.alpha,e.ap,e.app);}
S StoredExact(const Real x[3],const Exact &e,int field){S result{};if(Chart(x,field).Encode(Physical(x,e,field),result)!=h::PositiveFieldStatus::valid)throw std::runtime_error("exact chart failed");return result;}
S Sample(const Grid &grid,const V &padded,const V &stored,size_t p,int field){S q{};q.value=stored[2*p+field];Real d[3],dd[3][3];grid.Derivatives(padded,field,p,d,dd);
  for(int a=0;a<3;++a){q.d[a]=d[a];for(int b=0;b<3;++b)q.dd[a][b]=dd[a][b];}return q;}
Real HChi(const Real x[3],const S &q){Real dot=0,g2=0,lap=0;for(int a=0;a<3;++a){dot+=x[a]*q.d[a];g2+=q.d[a]*q.d[a];lap+=q.dd[a][a];}
  const Real omega=(1-Radius(x)*Radius(x))/2;return std::exp(q.value)*(-6+2*omega*dot+omega*omega*(2*lap-.5*g2));}
void Report(const char *method,const Grid &grid,size_t p,int field,const S &q,const S &eq,const Exact &exact,size_t halo_count,const Plan::Diagnostics *diagnostic=nullptr,int donors=0){
  const auto &x=grid.geometry[p].x;const Real h=Spacing(grid,p);S physical{};if(Chart(x,field).Reconstruct(q,physical)!=h::PositiveFieldStatus::valid)throw std::runtime_error("physical reconstruction failed");const auto ep=Physical(x,exact,field);
  Real dg=0,dh=0,pg=0,ph=0;for(int a=0;a<3;++a){dg=std::max(dg,std::abs(q.d[a]-eq.d[a]));pg=std::max(pg,std::abs(physical.d[a]-ep.d[a]));for(int b=0;b<3;++b){dh=std::max(dh,std::abs(q.dd[a][b]-eq.dd[a][b]));ph=std::max(ph,std::abs(physical.dd[a][b]-ep.dd[a][b]));}}
  std::cout<<"{\"method\":\""<<method<<"\",\"point\":"<<p<<",\"field\":\""<<(field==0?"chi":"alpha")<<"\",\"xyz\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"],\"rho\":"<<exact.r<<",\"h\":"<<h<<",\"rho_over_h\":"<<exact.r/h
    <<",\"current_wrapper_selects\":"<<(exact.r<.1&&exact.r/h<=8)<<",\"inside_radius_point1\":"<<(exact.r<.1)<<",\"stencil_internal_ghosts\":"<<halo_count
    <<",\"stored_gradient_error_max\":"<<dg<<",\"stored_hessian_error_max\":"<<dh<<",\"stored_scaled_error\":"<<std::max(h*dg,h*h*dh)<<",\"physical_gradient_error_max\":"<<pg<<",\"physical_hessian_error_max\":"<<ph
    <<",\"physical_scaled_error\":"<<std::max(h*pg,h*h*ph)<<",\"active_value_change\":"<<std::abs(physical.value-ep.value)<<",\"chi_spatial_curvature\":"<<(field==0?HChi(x,q):0)<<",\"exact_chi_spatial_curvature\":"<<(field==0?HChi(x,eq):0);
  if(diagnostic)std::cout<<",\"donors\":"<<donors<<",\"rank\":"<<diagnostic->rank<<",\"condition\":"<<diagnostic->condition<<",\"weight_l1\":"<<diagnostic->maximum_weight_l1<<",\"reproduction\":"<<diagnostic->reproduction<<",\"orthogonality\":"<<diagnostic->orthogonality;
  std::cout<<"}\n";
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc<4)throw std::invalid_argument("usage: catalog initial.dat catalog.csv OR diagnose initial.dat exact.dat");const std::string mode=argv[1];auto grid=MakeGrid();auto initial=ReadInitial(argv[2],grid);const auto targets=Targets(grid);std::cout<<std::setprecision(17);
  if(mode=="catalog")Catalog(grid,targets,argv[3]);else if(mode=="diagnose"){
    const auto exact=ReadExact(argv[3]);V stored(2*initial.size()),padded(2*grid.count,0);for(size_t p=0;p<initial.size();++p)for(int field=0;field<2;++field){
      const auto &x=grid.geometry[p].x;stored[2*p+field]=std::log(field==0?initial[p].chi:initial[p].alpha)-std::log(Chart(x,field).reference.value);padded[field*grid.count+grid.active[p]]=stored[2*p+field];}
    grid.Fill(padded,2);auto oracle=padded;size_t oracle_rows=0;Real initial_difference=0;
    for(const auto &item:exact){const int kind=item.first.first;const size_t p=item.first.second;const auto &e=item.second;
      if(kind==2){const auto x=PaddedPoint(grid,p);for(int f=0;f<2;++f)oracle[f*grid.count+p]=StoredExact(x.data(),e,f).value;++oracle_rows;}
      if(kind==0||kind==3)initial_difference=std::max({initial_difference,std::abs(initial[p].chi-e.chi),std::abs(initial[p].alpha-e.alpha)});}
    std::set<int> active(grid.active.begin(),grid.active.end());
    for(size_t p:targets){const auto &x=grid.geometry[p].x;const auto &e=exact.at({0,p});size_t ghosts=0;for(int index:Stencil(grid,p))ghosts+=!active.count(index);
      for(int f=0;f<2;++f){const auto eq=StoredExact(x,e,f);Report("ordinary_mapped_fd",grid,p,f,Sample(grid,padded,stored,p,f),eq,e,ghosts);Report("exact_internal_ghost_only",grid,p,f,Sample(grid,oracle,stored,p,f),eq,e,ghosts);}
      for(int count:{125,175,256}){auto nearest=Nearest(grid,p,count);std::vector<Point> nodes;for(const auto &neighbor:nearest){const auto &y=grid.geometry[neighbor.second].x;nodes.push_back({y[0],y[1],y[2]});}
        Plan plan;h::PuncturePosition point;if(!plan.Build({x[0],x[1],x[2]},nodes,Spacing(grid,p),point)){const auto &d=plan.diagnostics();std::cout<<"{\"method\":\"enriched_rejected\",\"point\":"<<p<<",\"donors\":"<<count<<",\"status\":\""<<d.status<<"\",\"rank\":"<<d.rank<<",\"condition\":"<<d.condition<<",\"weight_l1\":"<<d.maximum_weight_l1<<"}\n";continue;}
        for(int f=0;f<2;++f){V values;for(const auto &neighbor:nearest)values.push_back(stored[2*neighbor.second+f]);Plan::Result result;if(!plan.Evaluate(values,point,result))throw std::runtime_error("fresh enriched evaluation failed");
          S q{};q.value=stored[2*p+f];for(int a=0;a<3;++a){q.d[a]=result.gradient[a];for(int b=0;b<3;++b)q.dd[a][b]=result.hessian[a][b];}Report("enriched_active_only",grid,p,f,q,StoredExact(x,e,f),e,ghosts,&plan.diagnostics(),count);}
      }
    }
    std::cout<<"{\"summary\":true,\"targets\":"<<targets.size()<<",\"oracle_internal_rows\":"<<oracle_rows<<",\"initial_oracle_value_difference_max\":"<<initial_difference<<",\"all_actual_active_cells\":"<<grid.active.size()<<",\"physics_rhs_changed\":false,\"production_headers_changed\":false}\n";
  }else throw std::invalid_argument("unknown mode");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
