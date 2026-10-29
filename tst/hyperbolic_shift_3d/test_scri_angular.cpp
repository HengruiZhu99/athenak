// Bounded manufactured jets and native parity; no evolution/stability claim.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"

namespace h=z4c::hyperboloidal;
namespace c=cubed_sphere;
using Device=h::HyperbolicScriDevice;

static void Require(bool good,const char*message){if(!good)throw std::runtime_error(message);}

static h::ScalarJet<Real> Manufactured(const Real*x){
  const Real a[3]={1.3,-.7,.9},b[3]={.4,1.1,-.6};Real u=0,v=0;
  for(int i=0;i<3;++i){u+=a[i]*x[i];v+=b[i]*x[i];}
  h::ScalarJet<Real> out{};out.value=std::sin(u)+.2*std::cos(v)+.17*x[0]*x[1]+.08*x[1]*x[2]*x[2];
  for(int i=0;i<3;++i){out.d[i]=std::cos(u)*a[i]-.2*std::sin(v)*b[i];
    for(int j=0;j<3;++j)out.dd[i][j]=-std::sin(u)*a[i]*a[j]-.2*std::cos(v)*b[i]*b[j];}
  out.d[0]+=.17*x[1];out.d[1]+=.17*x[0]+.08*x[2]*x[2];out.d[2]+=.16*x[1]*x[2];
  out.dd[0][1]+=.17;out.dd[1][0]+=.17;out.dd[1][2]+=.16*x[2];out.dd[2][1]+=.16*x[2];out.dd[2][2]+=.16*x[1];
  return out;
}

struct Errors{Real d=0,dd=0,mixed=0,normal_d=0,normal_dd=0;int rows=0;};
static std::array<Errors,3> CheckJets(int n){
  c::CompositeDifference grid(c::CompositeMesh(.125,{.45,.75,1},1,n,3),4,true);
  c::ScriSurface surface(grid,1,6,true);
  std::vector<bool> active(grid.count,false);for(int point:grid.active)active[point]=true;
  size_t terms=0,max_terms=0;int cross_volume=0,cross_surface=0;
  for(const auto&row:surface.rows){
    std::set<std::pair<bool,int>> seen;bool from_volume=false,from_surface=false;
    Require(row.terms.front().boundary&&row.terms.front().point==surface.base[row.block]+row.j+n*row.k,
            "new scri sampler does not anchor the direct trace");
    for(const auto&term:row.terms){
      Require(seen.insert({term.boundary,term.point}).second,"duplicate flattened scri donor");
      int donor_block;
      if(term.boundary){Require(term.point>=0&&size_t(term.point)<surface.rows.size(),"invalid surface donor");
        donor_block=surface.rows[term.point].block;}
      else{Require(term.point>=0&&term.point<grid.count&&active[term.point],"scri reconstruction uses a volume ghost");
        donor_block=term.point/(grid.padded*grid.padded*grid.padded);}
      if(grid.mesh.blocks[donor_block].face!=grid.mesh.blocks[row.block].face){
        if(term.boundary)from_surface=true;else from_volume=true;}
    }
    const bool near_edge=row.j<2||row.j>=n-2||row.k<2||row.k>=n-2;
    Require(from_surface==near_edge&&from_volume==near_edge,"missing/unexpected cross-face scri dependency");
    cross_volume+=from_volume;cross_surface+=from_surface;
    terms+=row.terms.size();max_terms=std::max(max_terms,row.terms.size());
  }
  std::vector<Real> volume(grid.count,std::numeric_limits<Real>::quiet_NaN());
  std::vector<Real> boundary(surface.rows.size(),1.2345);
  for(int point:grid.active)volume[point]=1.2345;
  for(size_t p=0;p<surface.rows.size();++p){const auto jet=surface.Sample(p,volume,boundary,1);
    Require(jet.value[0]==1.2345,"constant scri value changed");
    for(int a=0;a<3;++a){Require(jet.d[0][a]==0,"constant scri gradient nonzero");
      for(int b=0;b<3;++b)Require(jet.dd[0][a][b]==0,"constant scri Hessian nonzero");}}
  for(size_t p=0;p<grid.active.size();++p)volume[grid.active[p]]=Manufactured(grid.geometry[p].x).value;
  for(size_t p=0;p<surface.rows.size();++p)boundary[p]=Manufactured(surface.rows[p].geometry.x).value;
  std::array<Errors,3> errors{};
  for(size_t p=0;p<surface.rows.size();++p){const auto&row=surface.rows[p];
    const int category=int(row.j<2||row.j>=n-2)+int(row.k<2||row.k>=n-2);auto&e=errors[category];++e.rows;
    const auto sampled=surface.Sample(p,volume,boundary,1);const auto exact=Manufactured(row.geometry.x);
    Require(sampled.value[0]==boundary[p],"cross-face sampler interpolates its own trace value");
    Real normal_d=0,normal_dd=0;
    for(int a=0;a<3;++a){
      Require(std::isfinite(sampled.d[0][a]),"nonfinite manufactured scri gradient");
      const Real d=sampled.d[0][a]-exact.d[a];e.d=std::max(e.d,std::abs(d));normal_d+=row.geometry.x[a]*d;
      for(int b=0;b<3;++b){
        Require(std::isfinite(sampled.dd[0][a][b]),"nonfinite manufactured scri Hessian");
        const Real dd=sampled.dd[0][a][b]-exact.dd[a][b];e.dd=std::max(e.dd,std::abs(dd));
        if(a!=b)e.mixed=std::max(e.mixed,std::abs(dd));normal_dd+=row.geometry.x[a]*row.geometry.x[b]*dd;
      }
    }
    e.normal_d=std::max(e.normal_d,std::abs(normal_d));e.normal_dd=std::max(e.normal_dd,std::abs(normal_dd));
  }
  std::cout<<"n="<<n<<" rows="<<surface.rows.size()<<" terms="<<terms<<" max_terms="<<max_terms
           <<" term_bytes="<<terms*sizeof(c::ScriSurface::Term)<<" cross_volume_rows="<<cross_volume
           <<" cross_surface_rows="<<cross_surface<<" constant_error=0\n";
  for(int category=0;category<3;++category){const auto&e=errors[category];
    std::cout<<"n="<<n<<" region="<<(category==0?"interior":category==1?"edge":"corner")
             <<" rows="<<e.rows<<" gradient_error="<<e.d<<" hessian_error="<<e.dd
             <<" mixed_error="<<e.mixed<<" normal_gradient_error="<<e.normal_d
             <<" normal_hessian_error="<<e.normal_dd<<'\n';}
  if(n==6){
    c::ScriSurface implicit(grid,1),explicit_legacy(grid,1,6,false);
    for(size_t p=0;p<implicit.rows.size();++p){const auto a=implicit.Sample(p,volume,boundary,1);
      const auto b=explicit_legacy.Sample(p,volume,boundary,1);
      Require(a.value==b.value&&a.d==b.d&&a.dd==b.dd,"default scri sampler changed");}
  }
  return errors;
}

static Device::Buffer Upload(const std::vector<Real>&values){
  Device::Buffer out("angular scri test",values.size());auto host=Kokkos::create_mirror_view(out);
  for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(out,host);return out;
}
static Real Compare(const std::vector<Real>&expected,Device::ConstBuffer actual,int first=0){
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),actual);Real error=0;
  for(size_t i=0;i<actual.extent(0);++i){Require(std::isfinite(host(i)),"nonfinite native device scri output");
    const Real value=expected[31*first+i];error=std::max(error,std::abs(host(i)-value)/(1+std::abs(value)));}
  return error;
}
static void CheckNativeParity(){
  h::HyperbolicComposite model(c::CompositeMesh(.125,{.45,.75,1},1,6,3),false,true);
  const int total=model.scri.rows.size(),split=total/2+7;
  Device whole(model,0,total),left(model,0,split),right(model,split,total-split);
  for(bool perturb:{false,true}){
    auto seed=[&](const Real*x,int f){return (f==0?1.:0.)+(perturb?.0001*(1+f%4)*
      std::sin((1.1+.13*f)*x[0]+(.3+.11*f)*x[1]-(.7+.07*f)*x[2]):0.);};
    for(size_t p=0;p<model.grid.active.size();++p)for(int f=0;f<31;++f)
      model.state[31*p+f]=seed(model.grid.geometry[p].x,f);
    for(size_t p=0;p<model.scri.rows.size();++p)for(int f=0;f<31;++f)
      model.scri.values[31*p+f]=seed(model.scri.rows[p].geometry.x,f);
    const auto original=Upload(model.scri.values);model.BoundaryValues();const auto projected=Upload(model.scri.values);
    const auto projection=whole.Project(original);Require(projection.bad_rows==0,"cross-face projection failed");
    const Real project_error=Compare(model.scri.values,whole.LocalSurface());
    const auto active=Upload(model.state);const auto padded=model.ActivePadded(model.state);
    const auto expected=model.BoundaryRates(padded,.0001),trace=model.scri.values;
    const auto a=whole.Evaluate(active,projected,.0001),b=left.Evaluate(active,projected,.0001),
               d=right.Evaluate(active,projected,.0001);
    Require(a.bad_rows==0&&b.bad_rows==0&&d.bad_rows==0,"cross-face native scri rate failed");
    const Real rate_error=std::max({Compare(expected,whole.LocalRHS()),Compare(expected,left.LocalRHS()),Compare(expected,right.LocalRHS(),split)});
    const Real trace_error=std::max({Compare(trace,whole.LocalSurface()),Compare(trace,left.LocalSurface()),Compare(trace,right.LocalSurface(),split)});
    Require(project_error<2e-12&&rate_error<2e-10&&trace_error<2e-10,"cross-face CPU/device parity failed");
    Require(projection.null_after<3e-14&&a.compatibility<3e-13,"cross-face native null/D closure failed");
    std::cout<<"perturb="<<perturb<<" projection_error="<<project_error<<" rate_error="<<rate_error
             <<" dependent_trace_error="<<trace_error<<" null="<<projection.null_after
             <<" compatibility="<<a.compatibility<<'\n';
  }
}

int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{
    std::cout<<std::setprecision(17)<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    std::vector<std::array<Errors,3>> measured;
    for(int n:{6,8,12,16})measured.push_back(CheckJets(n));
    for(int region=0;region<3;++region){const auto&coarse=measured[0][region];const auto&fine=measured[3][region];
      Require(fine.d<.15*coarse.d&&fine.dd<.15*coarse.dd&&fine.mixed<.15*coarse.mixed,
              "cross-face manufactured jets fail refinement");
      const Real first_order=std::log(measured[2][region].d/fine.d)/std::log(16./12);
      const Real second_order=std::log(measured[2][region].dd/fine.dd)/std::log(16./12);
      const Real mixed_order=std::log(measured[2][region].mixed/fine.mixed)/std::log(16./12);
      std::cout<<"region="<<region<<" fine_gradient_order="<<first_order<<" fine_hessian_order="<<second_order
               <<" fine_mixed_order="<<mixed_order<<'\n';
      Require(first_order>2.8&&second_order>2.8&&mixed_order>2.8,"cross-face jets not approaching fourth order");
    }
    CheckNativeParity();
    std::cout<<"Cross-face scri constants, active donors, nonspherical jets and native device parity passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
