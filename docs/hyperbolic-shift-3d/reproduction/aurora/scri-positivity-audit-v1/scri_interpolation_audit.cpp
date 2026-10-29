// Read-only audit of the original ghost-boundary interpolation rejection.
// No evolution, projection, null repair, or checkpoint admission is performed.
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "athena.hpp"
#include "mesh/cubed_sphere/composite.hpp"
#include "z4c/hyperboloidal/conformal_constraints.hpp"

namespace h=z4c::hyperboloidal;
constexpr int nf=31,ng=3;
static_assert(sizeof(Real)==8,"failed-stage audit requires binary64 Real");
struct Target {int block,i,j,k;};
struct Plan {
  std::array<Real,3> x;int owner,first[2],point[36];Real q[3],weight[36];
  std::vector<Target> targets;
};
struct MetricCheck {
  bool finite=true,valid=false;Real chi=0,minor1=0,minor2=0,det=0,q=0,chiq=0;
  h::MetricJet<Real> metric{};
};
MetricCheck Check(const Real*v,const std::array<Real,3>&x){
  MetricCheck r;const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  for(int f=0;f<nf;++f)r.finite=r.finite&&std::isfinite(v[f]);
  r.chi=v[0];
  for(int f=0;f<6;++f)r.metric.g[ti[f]][tj[f]]=r.metric.g[tj[f]][ti[f]]=v[1+f]+(ti[f]==tj[f]);
  const auto g=h::Geometry(r.metric);r.valid=g.valid;r.det=g.determinant;
  r.minor1=r.metric.g[0][0];r.minor2=r.metric.g[0][0]*r.metric.g[1][1]-r.metric.g[0][1]*r.metric.g[1][0];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.q+=g.inverse[i][j]*x[i]*x[j];
  r.chiq=r.chi*r.q;return r;
}
bool Rejected(const MetricCheck&r){
  return !r.finite||!r.valid||!(r.chi>0&&r.q>0)||!std::isfinite(r.chiq);
}
void Number(std::ostream&out,Real x){if(std::isfinite(x))out<<x;else out<<"null";}
template<class T>void Array(std::ostream&out,const T*x,int n){
  out<<'[';for(int i=0;i<n;++i){if(i)out<<',';Number(out,x[i]);}out<<']';
}
void CheckJSON(std::ostream&out,const MetricCheck&r){
  out<<"{\"all31_finite\":"<<r.finite<<",\"metric_valid\":"<<r.valid<<",\"chi\":";Number(out,r.chi);
  out<<",\"minor1\":";Number(out,r.minor1);out<<",\"minor2\":";Number(out,r.minor2);
  out<<",\"determinant\":";Number(out,r.det);out<<",\"code_q\":";Number(out,r.q);
  out<<",\"code_chi_q\":";Number(out,r.chiq);out<<",\"metric\":[";
  for(int i=0;i<3;++i){if(i)out<<',';Array(out,r.metric.g[i],3);}out<<"]}";
}
int main(int argc,char**argv){
  try{
    if(argc!=4)throw std::invalid_argument("usage: scri_interpolation_audit n failed-stage.bin output.json");
    if(std::filesystem::exists(argv[3]))throw std::runtime_error("refusing to overwrite audit output");
    const int n=std::stoi(argv[1]);if(n<6||n%2)throw std::invalid_argument("even n>=6 required");
    const size_t nv=size_t(nf)*19*n*n*n,ns=size_t(nf)*6*n*n;
    std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
    if(!input||input.tellg()!=std::streamoff(8*(nv+ns)))throw std::runtime_error("wrong raw failed-stage size");
    input.seekg(8*nv);std::vector<Real> surface(ns);
    input.read(reinterpret_cast<char*>(surface.data()),8*ns);if(!input)throw std::runtime_error("surface read failed");
    cubed_sphere::CompositeMesh mesh(.125,{.45,.75,1},1,n,ng);
    std::vector<Plan> plans;std::map<std::array<Real,3>,int> ids;
    // Exactly the geometry/selection loop of CompositeDifference::BuildHalos.
    // Interior donor plans are unnecessary for this boundary-value rejection.
    for(int block=0;block<int(mesh.blocks.size());++block)
    for(int k=-ng;k<n+ng;++k)for(int j=-ng;j<n+ng;++j)for(int i=-ng;i<n+ng;++i){
      if(i>=0&&i<n&&j>=0&&j<n&&k>=0&&k<n)continue;
      const Real local[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(block,local,x);
      const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);if(radius<=mesh.radii.back())continue;
      std::array<Real,3> boundary{};for(int d=0;d<3;++d)boundary[d]=mesh.radii.back()*x[d]/radius;
      auto found=ids.find(boundary);
      if(found==ids.end()){
        Plan b{};b.x=boundary;b.owner=mesh.Owner(b.x.data());
        if(b.owner<13)throw std::runtime_error("unexpected non-outer surface owner");
        mesh.Inverse(b.owner,b.x.data(),b.q);Real w[2][6];
        // Exact six-by-six face-local plan in HyperbolicCompositeExchange.
        for(int a=0;a<2;++a){const Real t=n*b.q[a+1]-.5;
          b.first[a]=std::clamp(int(std::floor(t))-2,0,n-6);
          for(int ii=0;ii<6;++ii){w[a][ii]=1;
            for(int jj=0;jj<6;++jj)if(ii!=jj)w[a][ii]*=(t-b.first[a]-jj)/Real(ii-jj);}}
        for(int kk=0;kk<6;++kk)for(int jj=0;jj<6;++jj){const int p=6*kk+jj;
          b.point[p]=(b.owner-13)*n*n+b.first[0]+jj+n*(b.first[1]+kk);b.weight[p]=w[0][jj]*w[1][kk];}
        found=ids.emplace(boundary,plans.size()).first;plans.push_back(b);
      }
      plans[found->second].targets.push_back({block,i,j,k});
    }
    int nodal_bad=0,nodal_nonfinite=0;Real nodal_min_chi=1e300,nodal_min_det=1e300,nodal_min_minor1=1e300,nodal_min_minor2=1e300;
    for(int p=0;p<6*n*n;++p){
      const auto r=Check(&surface[nf*p],{1,0,0});nodal_bad+=!r.valid;nodal_nonfinite+=!r.finite;
      nodal_min_chi=std::min(nodal_min_chi,r.chi);nodal_min_det=std::min(nodal_min_det,r.det);
      nodal_min_minor1=std::min(nodal_min_minor1,r.minor1);nodal_min_minor2=std::min(nodal_min_minor2,r.minor2);
    }
    std::vector<std::array<Real,nf>> interpolated(plans.size());std::vector<MetricCheck> checks;
    std::vector<int> bad;int bad_finite=0,bad_metric=0,bad_chi=0,bad_q_with_valid_metric=0,bad_product=0;
    Real min_chi=1e300,min_det=1e300,min_q_valid=1e300;
    for(size_t p=0;p<plans.size();++p){const auto&b=plans[p];auto&v=interpolated[p];
      for(int f=0;f<nf;++f){const Real anchor=surface[b.point[0]*nf+f];v[f]=anchor;
        for(int k=0;k<36;++k)v[f]+=b.weight[k]*(surface[b.point[k]*nf+f]-anchor);}
      const auto r=Check(v.data(),b.x);checks.push_back(r);
      min_chi=std::min(min_chi,r.chi);min_det=std::min(min_det,r.det);if(r.valid)min_q_valid=std::min(min_q_valid,r.q);
      if(Rejected(r)){bad.push_back(p);bad_finite+=!r.finite;bad_metric+=!r.valid;bad_chi+=!(r.chi>0);
        bad_q_with_valid_metric+=r.valid&&!(r.q>0);bad_product+=!std::isfinite(r.chiq);}
    }
    std::ofstream out(argv[3]);if(!out)throw std::runtime_error("output open failed");out<<std::setprecision(17)<<std::boolalpha;
    out<<"{\"n\":"<<n<<",\"fields\":31,\"surface_nodes\":"<<6*n*n
       <<",\"unique_global_boundary_queries\":"<<plans.size()<<",\"rejected_queries\":"<<bad.size()
       <<",\"rejection_counts\":{\"nonfinite_field\":"<<bad_finite<<",\"metric_invalid\":"<<bad_metric
       <<",\"nonpositive_chi\":"<<bad_chi<<",\"nonpositive_q_with_valid_metric\":"<<bad_q_with_valid_metric
       <<",\"nonfinite_chi_q\":"<<bad_product<<"},\"nodal_surface\":{\"invalid_metric_count\":"<<nodal_bad
       <<",\"nonfinite_point_count\":"<<nodal_nonfinite<<",\"min_chi\":";Number(out,nodal_min_chi);
    out<<",\"min_determinant\":";Number(out,nodal_min_det);out<<",\"min_minor1\":";Number(out,nodal_min_minor1);
    out<<",\"min_minor2\":";Number(out,nodal_min_minor2);out<<"},\"interpolants\":{\"min_chi\":";Number(out,min_chi);
    out<<",\"min_determinant\":";Number(out,min_det);out<<",\"min_q_valid_metric\":";Number(out,min_q_valid);
    out<<"},\"rejected\":[";
    for(size_t index=0;index<bad.size();++index){if(index)out<<',';const int p=bad[index];const auto&b=plans[p];
      out<<"{\"query_id\":"<<p<<",\"boundary_xyz\":";Array(out,b.x.data(),3);
      out<<",\"donor_block\":"<<b.owner<<",\"donor_face\":"<<mesh.blocks[b.owner].face<<",\"donor_local\":";Array(out,b.q,3);
      out<<",\"first_donor_jk\":";Array(out,b.first,2);out<<",\"targets\":[";
      for(size_t t=0;t<b.targets.size();++t){if(t)out<<',';const auto&v=b.targets[t];out<<'['<<v.block<<','<<v.i<<','<<v.j<<','<<v.k<<']';}
      out<<"],\"check\":";CheckJSON(out,checks[p]);out<<",\"interpolated_fields\":";Array(out,interpolated[p].data(),nf);
      out<<",\"weights\":";Array(out,b.weight,36);out<<",\"donor_surface_ids\":";Array(out,b.point,36);
      Real lo[nf],hi[nf];for(int f=0;f<nf;++f){lo[f]=1e300;hi[f]=-1e300;for(int k=0;k<36;++k){const Real v=surface[b.point[k]*nf+f];lo[f]=std::min(lo[f],v);hi[f]=std::max(hi[f],v);}}
      out<<",\"donor_field_min\":";Array(out,lo,nf);out<<",\"donor_field_max\":";Array(out,hi,nf);
      Real donor_min_det=1e300,donor_min_minor1=1e300,donor_min_minor2=1e300;int invalid=0;
      for(int k=0;k<36;++k){const auto r=Check(&surface[b.point[k]*nf],b.x);invalid+=!r.valid;
        donor_min_det=std::min(donor_min_det,r.det);donor_min_minor1=std::min(donor_min_minor1,r.minor1);donor_min_minor2=std::min(donor_min_minor2,r.minor2);}
      out<<",\"donor_invalid_metrics\":"<<invalid<<",\"donor_min_determinant\":";Number(out,donor_min_det);
      out<<",\"donor_min_minor1\":";Number(out,donor_min_minor1);out<<",\"donor_min_minor2\":";Number(out,donor_min_minor2);out<<'}';
    }
    out<<"]}\n";out.close();if(!out)throw std::runtime_error("output write failed");
    std::cout<<"queries="<<plans.size()<<" rejected="<<bad.size()<<" metric="<<bad_metric<<" chi="<<bad_chi<<" finite="<<bad_finite<<'\n';
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
