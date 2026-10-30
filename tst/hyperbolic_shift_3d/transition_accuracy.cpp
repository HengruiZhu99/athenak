// Compare mapped finite differences with interpolated versus analytic ghosts.
#include <fstream>
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "fixed_composite_mesh.hpp"
namespace {
Real Radius(const Real*x){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
struct Exact {
 int kind;
 Real Value(const Real*x)const{
  return kind?std::exp(-20*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])):std::sin(x[0]+2*x[1]-3*x[2]);
 }
 void Jets(const Real*x,Real*d,Real h[3][3])const{
  const Real f=Value(x),wave[3]={1,2,-3};
  for(int a=0;a<3;++a){d[a]=kind?-40*x[a]*f:wave[a]*std::cos(x[0]+2*x[1]-3*x[2]);
   for(int b=0;b<3;++b)h[a][b]=kind?(1600*x[a]*x[b]-(a==b?40:0))*f:-wave[a]*wave[b]*f;}
 }
};
struct Errors {
 size_t points=0;Real weight=0,d2=0,h2=0,dmax=0,hmax=0;
 void Add(Real w,const Real*d,const Real h[3][3],const Real*de,const Real he[3][3]){
  ++points;weight+=w;
  for(int a=0;a<3;++a){const Real e=d[a]-de[a];d2+=w*e*e;dmax=std::max(dmax,std::abs(e));
   for(int b=0;b<3;++b){const Real e2=h[a][b]-he[a][b];h2+=w*e2*e2;hmax=std::max(hmax,std::abs(e2));}}
 }
 void Write(std::ostream&out)const{
  if(!points||!std::isfinite(d2+h2+weight)||!(weight>0))throw std::runtime_error("invalid regional errors");
  out<<"{\"points\":"<<points<<",\"volume\":"<<weight<<",\"gradient_rms\":"<<std::sqrt(d2/weight)
     <<",\"hessian_rms\":"<<std::sqrt(h2/weight)<<",\"gradient_max\":"<<dmax<<",\"hessian_max\":"<<hmax<<'}';
 }
};
struct Regional {Errors full,ideal,transfer;};
void Add(Regional&r,Real w,const Real*d,const Real h[3][3],const Real*di,const Real hi[3][3],const Real*de,const Real he[3][3]){
 r.full.Add(w,d,h,de,he);r.ideal.Add(w,di,hi,de,he);r.transfer.Add(w,d,h,di,hi);
}
void Output(std::ostream&out,std::map<std::string,Regional>&errors){
 bool comma=false;out<<'{';for(const auto&item:errors){if(comma)out<<',';comma=true;
  out<<'"'<<item.first<<"\":{\"coupled\":";item.second.full.Write(out);
  out<<",\"analytic_ghosts\":";item.second.ideal.Write(out);out<<",\"interpolation_contribution\":";item.second.transfer.Write(out);out<<'}';}out<<'}';
}
}
int main(int argc,char**argv){
 try{
  if(argc!=4)throw std::invalid_argument("scheme n fresh-output.json");
  const std::string scheme=argv[1];const int n=std::stoi(argv[2]);
  if(std::ifstream(argv[3]))throw std::runtime_error("refusing to overwrite evidence");
  std::map<std::string,Regional> errors;size_t points=0;Real minr=1,l1=0,constant=0;
  {
   auto mesh=hyperbolic_reproduction::ThinCoreMesh(n,scheme=="thin-l1"?1:3,scheme=="thin-l1");
   if(scheme=="legacy-v7")mesh=hyperbolic_reproduction::FixedMesh(n,1,0);
   else if(scheme=="matched-nonoverlap")mesh=cubed_sphere::CompositeMesh(.18,{.45,.75,1},1,n,3,2);
   else if(scheme!="thin-l1"&&scheme!="control-l3")throw std::invalid_argument("unknown scheme");
   cubed_sphere::CompositeDifference g(mesh,4,true);g.SetInnerTransitionLogical(true);points=g.active.size();
   for(const auto&geo:g.geometry)minr=std::min(minr,Radius(geo.x));
   for(const auto&row:g.halos){Real sum=row.boundary_weight,weight=std::abs(row.boundary_weight);
    for(const auto&t:row.entries){sum+=t.weight;weight+=std::abs(t.weight);}
    // Only interior-interface rows enter this donor conditioning audit.
    const int m=row.target/(g.padded*g.padded*g.padded);if(mesh.blocks[m].layer>0)continue;
    constant=std::max(constant,std::abs(sum-1));l1=std::max(l1,weight);
   }
   for(int kind=0;kind<2;++kind){Exact exact{kind};std::vector<Real> u(g.count),ideal(g.count);
    for(size_t p=0;p<g.active.size();++p)u[g.active[p]]=ideal[g.active[p]]=exact.Value(g.geometry[p].x);
    g.Fill(u,1,[&](int,const std::array<Real,3>&x){return exact.Value(x.data());});
    for(const auto&row:g.halos){const int m=row.target/(g.padded*g.padded*g.padded),local=row.target%(g.padded*g.padded*g.padded);
     const int i=local%g.padded-g.storage_halo,j=(local/g.padded)%g.padded-g.storage_halo,k=local/(g.padded*g.padded)-g.storage_halo;
     const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(m,q,x);ideal[row.target]=exact.Value(x);}
    for(size_t p=0;p<g.active.size();++p){const auto&cell=g.cells[p];const auto&mb=mesh.blocks[cell.block];
     const auto*x=g.geometry[p].x;const Real r=Radius(x);if(r>=.4)continue;
     Real d[3],h[3][3],di[3],hi[3][3],de[3],he[3][3];g.Derivatives(u,0,p,d,h);g.Derivatives(ideal,0,p,di,hi);exact.Jets(x,de,he);
     const Real w=g.volumes[p];const std::string prefix=kind?"radial/":"oblique/";
     // Include the extra evolved copies: their RHS can seed a later interface
     // disturbance even though their diagnostic partition weight is zero.
     if(mb.face<0){
      const Real dx=(mb.hi[0]-mb.lo[0])/n;
      if(std::max({std::abs(x[0]),std::abs(x[1]),std::abs(x[2])})>=mesh.half_width+(mesh.CoreOverlap()-2)*dx)
       Add(errors[prefix+"core-interface-stencil"],w,d,h,di,hi,de,he);
     }else if(mb.layer==0&&(cell.j<g.Lower(cell.block,1)+2||cell.k<g.Lower(cell.block,2)+2||
                           cell.j>=g.Upper(cell.block,1)-2||cell.k>=g.Upper(cell.block,2)-2)){
      Add(errors[prefix+"shell-interface-stencil"],w,d,h,di,hi,de,he);
     }
     if(p>=g.nominal_count)continue;
     if(mb.face<0){
      if(std::max({std::abs(x[0]),std::abs(x[1]),std::abs(x[2])})>=mesh.half_width*.14/.18)Add(errors[prefix+"core-outer"],w,d,h,di,hi,de,he);
     }else if(mb.layer==0){
      Add(errors[prefix+"shell-all"],w,d,h,di,hi,de,he);
      if(r<=.23)Add(errors[prefix+"shell-inner"],w,d,h,di,hi,de,he);
      Real sorted[3]={std::abs(x[0]),std::abs(x[1]),std::abs(x[2])};std::sort(sorted,sorted+3);
      // Fixed physical masks, not a shrinking number of angular rows.
      if(sorted[1]>=.8*sorted[2])Add(errors[prefix+"shell-edges"],w,d,h,di,hi,de,he);
      if(sorted[2]>=.18){
       Add(errors[prefix+"shell-common"],w,d,h,di,hi,de,he);
       if(sorted[1]>=.8*sorted[2])Add(errors[prefix+"shell-common-edges"],w,d,h,di,hi,de,he);
      }
     }
    }
   }
  }
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"scheme\":\""<<scheme<<"\",\"n\":"<<n<<",\"vertex_centered\":"<<"false"
    <<",\"evolved_points\":"<<points<<",\"minimum_radius\":"<<minr
    <<",\"max_halo_weight_l1\":"<<l1<<",\"constant_error\":"<<constant<<",\"errors\":";
  Output(out,errors);out<<",\"scope\":\"Interior spatial operator audit; no evolution, paired-penalty, scri, puncture-duration or AMR qualification. Analytic outer ghosts are diagnostic only.\"}\n";
  if(!out)throw std::runtime_error("evidence write failed");
  std::cout<<scheme<<" n="<<n<<" passed spatial audit, min_r="<<minr<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
