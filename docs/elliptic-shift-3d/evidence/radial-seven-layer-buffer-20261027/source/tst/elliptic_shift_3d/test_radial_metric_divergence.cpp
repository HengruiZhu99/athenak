// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Targeted actual-wormhole discretization diagnostic. No evolution/solve and
// no replacement of physical curvature by a discretely balanced fake source.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "z4c/hyperboloidal/composite_metric_divergence.hpp"
namespace h=z4c::hyperboloidal;
using Op=h::CompositeMetricDivergence;
using Vector=Op::Vector;
using Tensor=Op::Tensor;
struct Row {int kind=0,id=-1;Real r=0,chi=0,cp=0,cpp=0,alpha=0,ap=0,app=0,a=0,ar=0,k=0,kr=0,b=0,br=0,brr=0;};
void Check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
h::VectorJet<Real> Reference(const Real *x){h::VectorJet<Real> b{};for(int i=0;i<3;++i){b.value[i]=-x[i];b.d[i][i]=-1;}return b;}
h::Z4cJet<Real> Exact(const Real *x,const Row &q){h::Z4cJet<Real> u{};Real n[3];for(int i=0;i<3;++i)n[i]=x[i]/q.r;
 u.alpha.value=q.alpha;u.chi.value=q.chi;
 for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.alpha.d[i]=q.ap*n[i];u.beta.value[i]=q.b*n[i];
  for(int a=0;a<3;++a){u.beta.d[a][i]=q.br*n[a]*n[i]+q.b/q.r*((a==i?1.:0)-n[a]*n[i]);
   for(int b=0;b<3;++b){const Real n3=n[a]*n[b]*n[i];u.beta.dd[a][b][i]=q.brr*n3+(q.br/q.r-q.b/(q.r*q.r))*
    ((a==b?n[i]:0)+(a==i?n[b]:0)+(b==i?n[a]:0)-3*n3);}}
  for(int j=0;j<3;++j){const Real t=1.5*n[i]*n[j]-(i==j?.5:0);u.a.k[i][j]=q.a*t;
   for(int a=0;a<3;++a)u.a.dk[a][i][j]=q.ar*n[a]*t+1.5*q.a/q.r*
    (((a==i?1.:0)-n[a]*n[i])*n[j]+n[i]*((a==j?1.:0)-n[a]*n[j]));}}
 return u;}
struct Norm {
 Real physical2=0,conformal2=0,physical_volume=0,conformal_volume=0,maximum=0;size_t cells=0,worst=0;
 void Add(Real e,Real physical,Real conformal,size_t p){Check(std::isfinite(e)&&physical>0&&conformal>0,"invalid error/norm weight");
  physical2+=physical*e*e;conformal2+=conformal*e*e;physical_volume+=physical;conformal_volume+=conformal;++cells;if(e>maximum){maximum=e;worst=p;}}
 void Print(const cubed_sphere::CompositeDifference &g) const {
  std::cout<<"{\"cells\":"<<cells<<",\"physical_rms\":"<<(physical_volume?std::sqrt(physical2/physical_volume):0)
   <<",\"physical_l2\":"<<std::sqrt(physical2)<<",\"conformal_rms\":"<<(conformal_volume?std::sqrt(conformal2/conformal_volume):0)
   <<",\"physical_volume\":"<<physical_volume<<",\"conformal_volume\":"<<conformal_volume<<",\"maximum\":"<<maximum;
  if(cells){const auto &x=g.geometry[worst].x;std::cout<<",\"worst_point\":"<<worst<<",\"worst_xyz\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"],\"worst_r\":"<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
  std::cout<<'}';}
};
struct Diagnostics {
 Norm residual,operator_error,source_error,metric_rate;
 std::array<Norm,6> radial;
 Norm interface,scri;
 Real face_metric_rate_max=0;
};
Real VectorSize(const Vector &v,size_t p){Real sum=0;for(int a=0;a<3;++a)sum+=v[3*p+a]*v[3*p+a];return std::sqrt(sum);}
Diagnostics Diagnose(const cubed_sphere::CompositeDifference &g,const std::vector<Row> &q,
 const Vector &applied,const Vector &source,const Vector &truth,const Op::TensorField &metric_rate,std::ofstream &points,int mode){
 Diagnostics out;
 for(size_t p=0;p<q.size();++p){const Real wp=g.volumes[p]/std::pow(q[p].chi,1.5),wc=g.volumes[p];Real r2=0,l2=0,s2=0,h2=0;
  for(int a=0;a<3;++a){r2+=std::pow(applied[3*p+a]-source[3*p+a],2);l2+=std::pow(applied[3*p+a]-truth[3*p+a],2);s2+=std::pow(source[3*p+a]-truth[3*p+a],2);}
  for(int f=0;f<6;++f)h2+=(Op::ti[f]==Op::tj[f]?1:2)*std::pow(metric_rate.active[6*p+f],2);
  const Real residual=std::sqrt(r2),operator_error=std::sqrt(l2),source_error=std::sqrt(s2),rate=std::sqrt(h2);
  out.residual.Add(residual,wp,wc,p);out.operator_error.Add(operator_error,wp,wc,p);out.source_error.Add(source_error,wp,wc,p);out.metric_rate.Add(rate,wp,wc,p);
  const int region=q[p].r<.1?0:q[p].r<.3?1:q[p].r<.62?2:q[p].r<.70?3:q[p].r<.9?4:5;out.radial[region].Add(residual,wp,wc,p);
  const int n=g.n,m=p/(n*n*n),i=p%n,j=p/n%n,k=p/(n*n)%n;
  if(i<2||j<2||k<2||i>=n-2||j>=n-2||k>=n-2)out.interface.Add(residual,wp,wc,p);
  if(g.mesh.blocks[m].layer==int(g.mesh.radii.size())-1&&i>=n-2)out.scri.Add(residual,wp,wc,p);
  points<<mode<<','<<p<<','<<q[p].r<<','<<wp<<','<<wc<<','<<residual<<','<<operator_error<<','<<source_error<<','<<rate<<'\n';
 }
 for(size_t p=0;p<metric_rate.surface.size()/6;++p){Real sum=0;for(int f=0;f<6;++f)sum+=(Op::ti[f]==Op::tj[f]?1:2)*std::pow(metric_rate.surface[6*p+f],2);out.face_metric_rate_max=std::max(out.face_metric_rate_max,std::sqrt(sum));}
 return out;
}
void Print(const Diagnostics &d,const cubed_sphere::CompositeDifference &g){std::cout<<"{\"residual\":";d.residual.Print(g);std::cout<<",\"operator_vs_continuum\":";d.operator_error.Print(g);
 std::cout<<",\"source_vs_continuum\":";d.source_error.Print(g);std::cout<<",\"H_beta_minus_2alphaA\":";d.metric_rate.Print(g);
 std::cout<<",\"face_metric_rate_max\":"<<d.face_metric_rate_max<<",\"regions\":[";
 const char *names[]={"r_lt_01","r_01_03","r_03_062","r_062_070","r_070_090","r_ge_090"};
 for(int r=0;r<6;++r){if(r)std::cout<<',';std::cout<<"{\"name\":\""<<names[r]<<"\",\"norm\":";d.radial[r].Print(g);std::cout<<'}';}
 std::cout<<"],\"all_block_interface_two_cell_band\":";d.interface.Print(g);std::cout<<",\"scri_two_cell_band\":";d.scri.Print(g);std::cout<<'}';}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
 Check(argc>=5,"usage: n exact.dat|--coordinates radial_interface_flag point-errors.csv [radii] [datum_ghosts] [centered_only]");const int n=std::stoi(argv[1]);const bool radial=std::stoi(argv[3]);Check(n==6||n==8||n==10,"bounded N6/N8/N10 diagnostic only");
 std::vector<Real> radii;std::stringstream radii_input(argc>5?argv[5]:".62,.70,1");std::string token;while(std::getline(radii_input,token,','))radii.push_back(std::stod(token));
 Check(!radii.empty()&&radii.size()<=7&&radii.back()==1,"bounded 1..7 shells ending at actual scri");
 const int datum_ghosts=argc>6?std::stoi(argv[6]):2;Check(datum_ghosts==2||datum_ghosts==4,"exact datum ghost contract");
 const bool centered_only=argc>7?std::stoi(argv[7]):false;
 const int ghosts=centered_only?2:4;
 cubed_sphere::CompositeDifference g(cubed_sphere::CompositeMesh(.25,radii,1,n,ghosts,2),4,radial);
 Check(g.active.size()<=50000,"bounded maximum50000 active cells");
 auto coordinate=[&](int m,int i,int j,int k){const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};std::array<Real,3>x;g.mesh.Forward(m,q,x.data());return x;};
 if(std::string(argv[2])=="--coordinates"){
  std::ofstream out(argv[4]);Check(bool(out),"cannot write exact coordinates");out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";
  auto write=[&](int kind,int id,const Real *x){const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);Check(r>0&&r<=1+2e-13,"coordinate outside physical domain");out<<kind<<','<<id<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<'\n';};
  for(size_t p=0;p<g.active.size();++p)write(0,p,g.geometry[p].x);
  for(const auto &row:g.halos)if(row.boundary_weight==0){const int point=row.target,m=point/(g.padded*g.padded*g.padded),i=point%g.padded-g.ng,j=point/g.padded%g.padded-g.ng,k=point/(g.padded*g.padded)%g.padded-g.ng;const auto x=coordinate(m,i,j,k);write(1,point,x.data());}
  cubed_sphere::ScriSurface face(g,6);for(size_t p=0;p<face.rows.size();++p)write(2,p,face.rows[p].geometry.x);
  std::cout<<"{\"coordinates\":true,\"cells\":"<<g.active.size()<<",\"blocks\":"<<g.mesh.blocks.size()<<",\"datum_ghosts\":"<<g.ng<<"}\n";
 }else{
 std::ifstream file(argv[2]);std::string magic;int count;file>>magic>>count;Check(magic=="EXACT_WORMHOLE_V1","exact datum format");
 std::vector<Row> q(g.active.size()),surface(6*n*n),halos;std::vector<bool> seen(q.size(),false),face_seen(surface.size(),false);
 for(int i=0;i<count;++i){Row row;file>>row.kind>>row.id>>row.r>>row.chi>>row.cp>>row.cpp>>row.alpha>>row.ap>>row.app>>row.a>>row.ar>>row.k>>row.kr>>row.b>>row.br>>row.brr;
  Check(bool(file),"truncated exact datum");if(row.kind==0){q.at(row.id)=row;seen.at(row.id)=true;}else if(row.kind==2){surface.at(row.id)=row;face_seen.at(row.id)=true;}else if(row.kind==1)halos.push_back(row);}
 for(bool x:seen)Check(x,"missing exact active datum");for(bool x:face_seen)Check(x,"missing exact scri datum");
 std::vector<h::Z4cJet<Real>> exact(q.size());Vector beta(3*q.size()),truth(3*q.size()),compact(3*q.size()),compact_exact_ghosts(3*q.size()),product_rule_source(3*q.size());
 Vector padded_beta(3*g.count,0),sampled(7*g.count,0);
 Real analytic_identity=0,min_radius=1;
 for(size_t p=0;p<q.size();++p){const auto &x=g.geometry[p].x;const Real radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);Check(std::abs(radius-q[p].r)<2e-13,"exact data/grid coordinates mismatch");
  exact[p]=Exact(x,q[p]);min_radius=std::min(min_radius,radius);Real source[3],applied[3];h::EllipticShiftSource(exact[p],h::Geometry(exact[p].metric),0.,source);h::ApplyEllipticShift(exact[p].metric,h::Geometry(exact[p].metric),exact[p].beta,applied);
  for(int a=0;a<3;++a){truth[3*p+a]=source[a];analytic_identity=std::max(analytic_identity,std::abs(applied[a]-source[a]));beta[3*p+a]=(q[p].b/q[p].r+1)*x[a];padded_beta[a*g.count+g.active[p]]=beta[3*p+a];}
  sampled[g.active[p]]=q[p].alpha;for(int f=0;f<6;++f)sampled[(1+f)*g.count+g.active[p]]=exact[p].a.k[Op::ti[f]][Op::tj[f]];
 }
 g.Fill(padded_beta,3);g.Fill(sampled,7,[](int f,const auto&){return f==0?Real(1):Real(0);});
 Vector beta_exact_halos=padded_beta;std::vector<bool> exact_halo_covered(g.count,false);const int old_padded=n+2*datum_ghosts;
 for(const auto &row:halos){const int point=row.id,m=point/(old_padded*old_padded*old_padded),i=point%old_padded-datum_ghosts,j=point/old_padded%old_padded-datum_ghosts,k=point/(old_padded*old_padded)%old_padded-datum_ghosts;
  Check(m>=0&&m<int(g.mesh.blocks.size()),"exact beta halo block mismatch");
  if(i< -g.ng||j< -g.ng||k< -g.ng||i>=n+g.ng||j>=n+g.ng||k>=n+g.ng)continue; // unused ring in a wider archived datum
  const auto x=coordinate(m,i,j,k);const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);Check(std::abs(r-row.r)<2e-13,"exact beta halo radius mismatch");const int target=g.Index(m,i,j,k);
  for(int a=0;a<3;++a)beta_exact_halos[a*g.count+target]=(row.b/row.r+1)*x[a];exact_halo_covered[target]=true;}
 // Prove coverage of every internal ghost used by the centered first/second
 // pass, including mixed derivative corners. Physical outer ghosts stay under
 // the original direct vector boundary lift.
 for(size_t p=0;p<g.active.size();++p){const int m=p/(n*n*n),i=p%n,j=p/n%n,k=p/(n*n)%n;
  for(int dk=-2;dk<=2;++dk)for(int dj=-2;dj<=2;++dj)for(int di=-2;di<=2;++di){if(i+di>=0&&i+di<n&&j+dj>=0&&j+dj<n&&k+dk>=0&&k+dk<n)continue;
   const auto x=coordinate(m,i+di,j+dj,k+dk);const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);if(r<=1)Check(exact_halo_covered[g.Index(m,i+di,j+dj,k+dk)],"missing exact internal beta stencil donor");}}
 Op::TensorField first_pass_exact_halos;first_pass_exact_halos.active.resize(6*q.size());
 for(size_t p=0;p<q.size();++p){auto b=Reference(g.geometry[p].x);auto u=exact[p];
  for(int a=0;a<3;++a){b.value[a]+=beta[3*p+a];Real d[3],dd[3][3];g.Derivatives(padded_beta,a,p,d,dd);
   for(int i=0;i<3;++i){b.d[i][a]+=d[i];for(int j=0;j<3;++j)b.dd[i][j][a]+=dd[i][j];}}
  Real value[3];h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),b,value);for(int a=0;a<3;++a)compact[3*p+a]=value[a];
  auto bh=Reference(g.geometry[p].x);for(int a=0;a<3;++a){bh.value[a]+=beta[3*p+a];Real d[3],dd[3][3];g.Derivatives(beta_exact_halos,a,p,d,dd);
   for(int i=0;i<3;++i){bh.d[i][a]+=d[i];for(int j=0;j<3;++j)bh.dd[i][j][a]+=dd[i][j];}}
  h::ApplyEllipticShift(u.metric,h::Geometry(u.metric),bh,value);for(int a=0;a<3;++a)compact_exact_ghosts[3*p+a]=value[a];const auto hh=h::ShiftMetricRate(u.metric,bh);
  for(int f=0;f<6;++f)first_pass_exact_halos.active[6*p+f]=hh.h[Op::ti[f]][Op::tj[f]];
  Real d[3],dd[3][3];g.Derivatives(sampled,0,p,d,dd);for(int a=0;a<3;++a)u.alpha.d[a]=d[a];
  for(int f=0;f<6;++f){g.Derivatives(sampled,1+f,p,d,dd);for(int a=0;a<3;++a)u.a.dk[a][Op::ti[f]][Op::tj[f]]=u.a.dk[a][Op::tj[f]][Op::ti[f]]=d[a];}
  h::EllipticShiftSource(u,h::Geometry(u.metric),0.,value);for(int a=0;a<3;++a)product_rule_source[3*p+a]=value[a];
 }
 std::ofstream points(argv[4]);Check(bool(points),"cannot write point diagnostic");points<<std::setprecision(17)<<"mode,point,r,physical_weight,conformal_weight,residual,operator_error,source_error,metric_rate_error\n";
 std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"shell_radii\":[";for(size_t r=0;r<radii.size();++r){if(r)std::cout<<',';std::cout<<radii[r];}std::cout<<"],\"blocks\":"<<g.mesh.blocks.size()<<",\"cells\":"<<q.size()<<",\"ghosts\":"<<g.ng<<",\"radial_interface_stencils\":"<<radial
  <<",\"minimum_radius\":"<<min_radius<<",\"analytic_L_beta_vs_exact_source_max\":"<<analytic_identity<<",\"cases\":[";
 int mode=0;Real boundary_residual=0;
 std::vector<Op::Pair> pairs{Op::Pair::Centered};if(!centered_only)pairs.push_back(Op::Pair::ForwardAdjoint);
 for(auto pair:pairs){Op op(g,pair);op.SetGeometry(op.SampleTensor([](const auto&){return Tensor{1,0,0,1,0,1};}));
  auto curvature=op.CurvatureRate(exact,[](const auto&){return Tensor{};});
  for(const auto &row:surface)Check(std::abs(row.a)<1e-14&&std::abs(row.alpha-1)<1e-13,"outer physical curvature/lapse datum differs from assumed exact scri");
  const auto shift=op.ShiftRate(beta,{},Reference);auto rate=shift;for(size_t i=0;i<rate.active.size();++i)rate.active[i]+=curvature.active[i];for(size_t i=0;i<rate.surface.size();++i)rate.surface[i]+=curvature.surface[i];
  if(pair==Op::Pair::Centered){for(const auto &control:{std::pair<const char*,const Vector*>{"compact_operator_analytic_source",&truth},{"compact_operator_sampled_product_rule_source",&product_rule_source}}){
    const auto diagnostic=Diagnose(g,q,compact,*control.second,truth,rate,points,mode++);if(mode>1)std::cout<<',';std::cout<<"{\"name\":\""<<control.first<<"\",\"diagnostic\":";Print(diagnostic,g);std::cout<<'}';}}
  Vector applied,source;op.Variation(shift,applied);op.Source(curvature,0,source);const auto diagnostic=Diagnose(g,q,applied,source,truth,rate,points,mode++);
  std::cout<<",{\"name\":\""<<(pair==Op::Pair::Centered?"consistent_centered_checkerboard_control":"consistent_forward_adjoint_candidate")<<"\",\"diagnostic\":";Print(diagnostic,g);std::cout<<'}';
  if(pair==Op::Pair::Centered){auto ghost_rate=first_pass_exact_halos;ghost_rate.surface=shift.surface;auto ghost_defect=ghost_rate;
   for(size_t i=0;i<ghost_defect.active.size();++i)ghost_defect.active[i]+=curvature.active[i];for(size_t i=0;i<ghost_defect.surface.size();++i)ghost_defect.surface[i]+=curvature.surface[i];
   const auto compact_oracle=Diagnose(g,q,compact_exact_ghosts,truth,truth,ghost_defect,points,mode++);std::cout<<",{\"name\":\"compact_exact_internal_beta_ghosts_analytic_source\",\"diagnostic\":";Print(compact_oracle,g);std::cout<<'}';
   op.Variation(ghost_rate,applied);const auto first_pass_oracle=Diagnose(g,q,applied,source,truth,ghost_defect,points,mode++);
   std::cout<<",{\"name\":\"consistent_centered_exact_internal_beta_ghosts\",\"diagnostic\":";Print(first_pass_oracle,g);std::cout<<'}';}
  // Independent first-jet oracle: only beta derivatives come from the exact
  // radial quadrature/ODE datum. Physical alpha*A and the outer DV/halo code
  // are UNCHANGED. This is localization, not a usable unknown-shift operator.
  auto exact_rate=shift;
  for(size_t p=0;p<q.size();++p){const auto exact_h=h::ShiftMetricRate(exact[p].metric,exact[p].beta);for(int f=0;f<6;++f)exact_rate.active[6*p+f]=exact_h.h[Op::ti[f]][Op::tj[f]];}
  for(size_t p=0;p<surface.size();++p){const auto u=Exact(op.face.rows[p].geometry.x,surface[p]);const auto exact_h=h::ShiftMetricRate(u.metric,u.beta);
   for(int f=0;f<6;++f)exact_rate.surface[6*p+f]=exact_h.h[Op::ti[f]][Op::tj[f]];}
  op.Variation(exact_rate,applied);auto exact_defect=exact_rate;
  for(size_t i=0;i<exact_defect.active.size();++i)exact_defect.active[i]+=curvature.active[i];for(size_t i=0;i<exact_defect.surface.size();++i)exact_defect.surface[i]+=curvature.surface[i];
  const auto oracle=Diagnose(g,q,applied,source,truth,exact_defect,points,mode++);
  std::cout<<",{\"name\":\""<<(pair==Op::Pair::Centered?"exact_beta_first_jets_centered_DV_oracle":"exact_beta_first_jets_biased_DV_oracle")<<"\",\"diagnostic\":";Print(oracle,g);std::cout<<'}';
  for(size_t p=0;p<op.face.rows.size();++p)for(int a=0;a<3;++a)boundary_residual=std::max(boundary_residual,std::abs(op.face.ReconstructFromGhost(p,padded_beta,a)));
 }
 std::cout<<"],\"independent_direct_vector_boundary_max\":"<<boundary_residual<<",\"source_policy\":\"actual sampled -2alpha A only; no source replacement or RHS subtraction\",\"scope\":\"no solve, no evolution, prototype not admitted\"}\n";
 Check(analytic_identity<1e-8,"independent continuum shift/source identity failed");
 }
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=2;}Kokkos::finalize();return status;}
