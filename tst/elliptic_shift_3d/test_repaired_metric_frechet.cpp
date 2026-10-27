// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Independent finite-difference Frechet/continuum audit; no evolution or solve.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include "mesh/cubed_sphere/hermite_boundary_sampler.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"
#include "inject_repaired_interface.hpp"
namespace c=cubed_sphere;namespace h=z4c::hyperboloidal;using V=std::vector<Real>;using J=c::Jet;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
J Exp(const J&x){J o(std::exp(x.f));for(int a=0;a<3;++a){o.d[a]=o.f*x.d[a];for(int b=0;b<3;++b)o.h[a][b]=o.f*(x.h[a][b]+x.d[a]*x.d[b]);}return o;}
struct Data {h::MetricJet<Real> metric{};h::MetricRate<Real> rate{};};
Data Exact(const Real*p,int direction=0){J x[3];for(int a=0;a<3;++a)x[a]=J::Variable(p[a],a);const J f=Exp(J(.2)*x[0]-J(.3)*x[1]+J(.1)*x[2]);Data o;for(int t=0;t<6;++t){int i=ti[t],j=tj[t];J g,h;if(i==j)g=J(1.1+.03*i)+J(.025)*(x[i]*x[i]+J(.3)*x[(i+1)%3]*f);else g=J(.015)*(x[i]*x[j]+J(.4)*x[3-i-j]*f);h=J(.03*(t+1))*f*(J(1)+J(.2)*x[(t+1)%3]+J(.3)*x[0]*x[1]-J(.1)*x[1]*x[2]*x[2]);if(direction)h=J((t%2?-1:1)*.021*(t+1))*Exp(J(-.23)*x[2]+J(.17)*x[0]*x[1])*(x[(t+2)%3]+J(.4)*x[0]*x[0]*x[2]-J(.7)*x[1]*x[1]*x[1]*x[2]);o.metric.g[i][j]=o.metric.g[j][i]=g.f;o.rate.h[i][j]=o.rate.h[j][i]=h.f;for(int a=0;a<3;++a){o.metric.dg[a][i][j]=o.metric.dg[a][j][i]=g.d[a];o.rate.dh[a][i][j]=o.rate.dh[a][j][i]=h.d[a];for(int b=0;b<3;++b)o.metric.ddg[a][b][i][j]=o.metric.ddg[a][b][j][i]=g.h[a][b];}}return o;}
struct Probe {
 c::CompositeDifference grid;c::ScriSurface surface;c::HermiteBoundarySampler sampler;size_t N,S;int direction;V metric,trace,rate,ratetrace;std::vector<h::MetricJet<Real>> g;std::vector<h::MetricRate<Real>> H;V discrete,analytic;std::map<std::string,V> differences;Real min_det=1e100,max_det=0;
 static c::CompositeMesh Mesh(int n,bool refine){c::CompositeMesh m(.25,{.45,.55,.60,.65,.70,.75,1},1,n,2,2);if(refine)m.RefineCartesian(0);return m;}
 static c::CompositeDifference Repaired(int n,bool refine){c::CompositeDifference g(Mesh(n,refine),4,true);interface_probe::InjectRepairedInterface(g);return g;}
 Probe(int n,bool refine,int dir=0):grid(Repaired(n,refine)),surface(grid,6,7),sampler(surface),N(grid.active.size()),S(surface.rows.size()),direction(dir),metric(6*grid.count),trace(6*S),rate(metric.size()),ratetrace(trace.size()){
  if(N>50000)throw std::runtime_error("Frechet audit >50000cells");
  for(size_t p=0;p<N+S;++p){bool b=p>=N;size_t k=b?p-N:p;auto d=Exact(b?surface.rows[k].geometry.x:grid.geometry[k].x,direction);for(int t=0;t<6;++t){int i=ti[t],j=tj[t];if(b){trace[6*k+t]=d.metric.g[i][j];ratetrace[6*k+t]=d.rate.h[i][j];}else{metric[t*grid.count+grid.active[k]]=d.metric.g[i][j];rate[t*grid.count+grid.active[k]]=d.rate.h[i][j];}}}
  Fill(metric,trace);Fill(rate,ratetrace);g.resize(N+S);H.resize(N+S);discrete.resize(3*(N+S));analytic.resize(discrete.size());auto mp=sampler.Prepare(metric,trace,6,1),rp=sampler.Prepare(rate,ratetrace,6,1);
  for(size_t p=0;p<N+S;++p){bool b=p>=N;size_t k=b?p-N:p;c::ScriSurface::Jets ms,rs;if(b){ms=sampler.Sample(k,mp);rs=sampler.Sample(k,rp);}for(int t=0;t<6;++t){int i=ti[t],j=tj[t];Real dg[3],dh[3],dd[3][3];if(b){for(int a=0;a<3;++a){dg[a]=ms.d[t][a];dh[a]=rs.d[t][a];}}else{grid.Derivatives(metric,t,k,dg,dd);grid.Derivatives(rate,t,k,dh,dd);}g[p].g[i][j]=g[p].g[j][i]=b?trace[6*k+t]:metric[t*grid.count+grid.active[k]];H[p].h[i][j]=H[p].h[j][i]=b?ratetrace[6*k+t]:rate[t*grid.count+grid.active[k]];for(int a=0;a<3;++a){g[p].dg[a][i][j]=g[p].dg[a][j][i]=dg[a];H[p].dh[a][i][j]=H[p].dh[a][j][i]=dh[a];}}auto geo=h::Geometry(g[p]);if(!geo.valid)throw std::runtime_error("nonSPD synthetic metric");min_det=std::min(min_det,geo.determinant);max_det=std::max(max_det,geo.determinant);h::MetricConnectionVariation(g[p],geo,H[p],&discrete[3*p]);auto exact=Exact(b?surface.rows[k].geometry.x:grid.geometry[k].x,direction);h::MetricConnectionVariation(exact.metric,h::Geometry(exact.metric),exact.rate,&analytic[3*p]);}
 }
 void Fill(V&data,const V&boundary){grid.Fill(data,6,[&](int f,const std::array<Real,3>&x){return surface.Interpolate(boundary,6,x.data())[f];});}
 V Connection(const V&field,const V&boundary){auto mp=sampler.Prepare(field,boundary,6,1);V result(3*(N+S));for(size_t p=0;p<N+S;++p){bool b=p>=N;size_t k=b?p-N:p;h::MetricJet<Real> u{};c::ScriSurface::Jets ms;if(b)ms=sampler.Sample(k,mp);for(int t=0;t<6;++t){int i=ti[t],j=tj[t];u.g[i][j]=u.g[j][i]=b?boundary[6*k+t]:field[t*grid.count+grid.active[k]];Real d[3],dd[3][3];if(b)for(int a=0;a<3;++a)d[a]=ms.d[t][a];else grid.Derivatives(field,t,k,d,dd);for(int a=0;a<3;++a)u.dg[a][i][j]=u.dg[a][j][i]=d[a];}auto geo=h::Geometry(u);if(!geo.valid)throw std::runtime_error("finite perturbation lost SPD");for(int a=0;a<3;++a)result[3*p+a]=geo.contracted[a];}return result;}
 void Run(){V continuum(discrete.size());for(size_t i=0;i<continuum.size();++i)continuum[i]=discrete[i]-analytic[i];differences["continuum_error"]=continuum;for(Real eps:{1e-3,1e-4,1e-5,1e-6,1e-7}){V plus(metric),minus(metric),bp(trace),bm(trace);for(size_t p=0;p<N;++p)for(int t=0;t<6;++t){int a=t*grid.count+grid.active[p];plus[a]+=eps*rate[a];minus[a]-=eps*rate[a];}for(size_t i=0;i<trace.size();++i){bp[i]+=eps*ratetrace[i];bm[i]-=eps*ratetrace[i];}Fill(plus,bp);Fill(minus,bm);auto P=Connection(plus,bp),M=Connection(minus,bm);V err(P.size());for(size_t i=0;i<P.size();++i)err[i]=(P[i]-M[i])/(2*eps)-discrete[i];std::ostringstream name;name<<"finite_epsilon_"<<eps<<"_error";differences[name.str()]=std::move(err);}}
 void Write(const std::string&path){std::ofstream out(path);out<<std::setprecision(17)<<"surface,point,block,refinement,face,layer,x,y,z,map_weight,conformal_weight,determinant";for(auto name:{"discrete","analytic"})for(int a=0;a<3;++a)out<<','<<name<<'_'<<a;for(auto &[name,v]:differences)for(int a=0;a<3;++a)out<<','<<name<<'_'<<a;out<<'\n';for(size_t p=0;p<N+S;++p){bool b=p>=N;size_t k=b?p-N:p;int block=b?surface.rows[k].block:k/(grid.n*grid.n*grid.n);auto &bd=grid.mesh.blocks[block];const Real*x=b?surface.rows[k].geometry.x:grid.geometry[k].x;Real det=h::Geometry(g[p]).determinant,w=b?1:grid.volumes[k];out<<b<<','<<k<<','<<block<<','<<bd.refinement<<','<<bd.face<<','<<bd.layer<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<w<<','<<w*std::sqrt(det)<<','<<det;for(const auto*v:{&discrete,&analytic})for(int a=0;a<3;++a)out<<','<<(*v)[3*p+a];for(const auto&[name,v]:differences)for(int a=0;a<3;++a)out<<','<<v[3*p+a];out<<'\n';}}
 void Summary(){std::cout<<std::setprecision(17)<<"{\"direction\":"<<direction<<",\"cells\":"<<N<<",\"surface_points\":"<<S<<",\"minimum_det\":"<<min_det<<",\"maximum_det\":"<<max_det<<",\"errors\":{";bool first=true;for(const auto&[name,v]:differences){if(!first)std::cout<<',';first=false;Real bulk=0,volume=0,bmax=0,smax=0,srms=0;for(size_t p=0;p<N+S;++p){Real w=p<N?grid.volumes[p]*std::sqrt(h::Geometry(g[p]).determinant):1;if(p<N)volume+=w;for(int a=0;a<3;++a)if(p<N){bulk+=w*v[3*p+a]*v[3*p+a];bmax=std::max(bmax,std::abs(v[3*p+a]));}else{srms+=v[3*p+a]*v[3*p+a];smax=std::max(smax,std::abs(v[3*p+a]));}}std::cout<<'"'<<name<<"\":{\"body_conformal_rms\":"<<std::sqrt(bulk/volume)<<",\"body_max\":"<<bmax<<",\"surface_point_rms\":"<<std::sqrt(srms/S)<<",\"surface_max\":"<<smax<<'}';}std::cout<<"},\"arbitrary_nonspherical_nontracefree_direction\":true,\"projection_or_operator_adoption\":false}\n";}
};
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 if(argc!=4)throw std::invalid_argument("n refine outputprefix");
 for(int dir:{0,1}){
  Probe p(std::stoi(argv[1]),std::stoi(argv[2]),dir);p.Run();
  p.Write(std::string(argv[3])+std::to_string(dir)+".csv");p.Summary();
  // Independent centered finite-epsilon derivative, including every scri row.
  for(const char*name:{"finite_epsilon_0.0001_error","finite_epsilon_1e-05_error"})
    for(Real e:p.differences.at(name))if(!std::isfinite(e)||std::abs(e)>1e-7)code=1;
 }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();return code;}
