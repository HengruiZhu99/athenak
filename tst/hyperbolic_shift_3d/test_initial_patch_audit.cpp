// Initial snapshots only: isolate production exchange, algebraic projection,
// cutoff quadrature, and maximum locations without changing evolved equations.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"
#include "fixed_composite_mesh.hpp"
namespace h=z4c::hyperboloidal;
struct Norm {
  double volume=0,square[3]{},maximum[3]{},xyz[3][3]{};size_t points=0;int block[3]{};
  void Add(const double value[3],double weight,const double*x,int m){
    if(weight<=0)return;volume+=weight;++points;
    for(int f=0;f<3;++f){square[f]+=weight*value[f]*value[f];if(std::abs(value[f])>maximum[f]){
      maximum[f]=std::abs(value[f]);block[f]=m;for(int a=0;a<3;++a)xyz[f][a]=x[a];}}
  }
  void Write(std::ostream&out)const{
    out<<"{\"volume\":"<<volume<<",\"points\":"<<points;
    const char*names[3]={"H","M","Mr"};
    for(int f=0;f<3;++f){out<<",\""<<names[f]<<"\":{\"sum2\":"<<square[f]<<",\"rms\":"<<(volume?std::sqrt(square[f]/volume):0)
      <<",\"max\":"<<maximum[f]<<",\"max_block\":"<<block[f]<<",\"max_xyz\":["<<xyz[f][0]<<','<<xyz[f][1]<<','<<xyz[f][2]<<"]}";}
    out<<'}';
  }
};
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int status=0;
 try{
  if(argc!=3)throw std::invalid_argument("usage: initial_patch_audit N fresh_output.json");
  int n=std::stoi(argv[1]);if(std::filesystem::exists(argv[2]))throw std::runtime_error("refusing existing output");
  h::HyperbolicComposite model(hyperbolic_reproduction::ContinuousCoreMesh(n),false,true,2,
      h::HyperbolicComposite::NormalDerivative::sampled_numerator,h::HyperbolicComposite::RadialClosure::donor_seven);
  auto&g=model.grid;h::NativeWormhole data;const int nf=model.nf;
  for(size_t p=0;p<g.active.size();++p){const auto q=data.Fields(g.geometry[p].x);for(int f=0;f<nf;++f)model.state[nf*p+f]=q[f];}
  const auto limit=data.ScriFields();for(size_t p=0;p<model.scri.rows.size();++p)for(int f=0;f<nf;++f)model.scri.values[nf*p+f]=limit[f];
  const auto original=model.state,original_surface=model.scri.values;
  double projection_change=0,surface_projection_change=0,scri_limit_error=0;
  std::vector<std::array<double,3>> reference;
  const std::vector<std::string> names={"exterior","core","overlap_radial_016_018","shell_seams","initial_transition_045_075","scri_adjacent_last3","outer_0775_09"};
  const double cutoffs[]={.062,.063,.064,.13,.16,.18,.2,.3,.45,.56,.6,.675,.7,.75,.775,.8,.9,.95};
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"n\":"<<n<<",\"cutoff\":0.063,\"variants\":[";
  for(int mode=0;mode<3;++mode){
    model.state=original;model.scri.values=original_surface;
    if(mode==2){
      for(size_t p=0;p<g.active.size();++p){auto*q=&model.state[nf*p];auto result=h::ProjectAlgebraicNativeState(q);if(!result.valid)throw std::runtime_error("projection failed");
        for(int f=0;f<nf;++f)projection_change=std::max(projection_change,std::abs(q[f]-original[nf*p+f]));}
      for(size_t p=0;p<model.scri.rows.size();++p){auto*q=&model.scri.values[nf*p];auto result=h::ProjectAlgebraicNativeState(q);if(!result.valid)throw std::runtime_error("surface projection failed");
        for(int f=0;f<nf;++f)surface_projection_change=std::max(surface_projection_change,std::abs(q[f]-original_surface[nf*p+f]));}
    }
    // Exactly the production order for constraint-bearing surface and halo values.
    // Dependent D has no direct role in H/M and is nevertheless refreshed here.
    model.BoundaryValues();auto padded=model.Padded(model.state);model.BoundaryRates(padded,.0005*16/n);model.FillGhosts(padded);
    for(size_t p=0;p<model.scri.rows.size();++p)for(int f=0;f<18;++f)
      scri_limit_error=std::max(scri_limit_error,std::abs(model.scri.values[nf*p+f]-original_surface[nf*p+f]));
    size_t replaced=0;
    if(mode==1)for(const auto&row:g.halos){
      int index=row.target;const int i=index%g.padded-g.storage_halo;index/=g.padded;
      const int j=index%g.padded-g.storage_halo;index/=g.padded;
      const int k=index%g.padded-g.storage_halo,m=index/g.padded;
      const double local[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};double x[3];g.mesh.Forward(m,local,x);
      const double r=model.Radius(x);if(r<=0||r>=1)continue;
      const auto q=data.Fields(x);for(int f=0;f<18;++f)padded[f*g.count+row.target]=q[f];++replaced;
    }
    std::vector<Norm> regions(names.size()),cuts(std::size(cutoffs)),profile(100),difference(names.size());
    std::vector<std::array<double,3>> current(g.active.size());
    h::HyperbolicComposite::V v(nf);std::vector<std::array<double,3>>d(nf);
    std::vector<std::array<std::array<double,3>,3>>dd(nf);
    for(size_t p=0;p<g.active.size();++p){const double weight=g.DiagnosticVolume(p);if(weight<=0)continue;
      const auto*x=g.geometry[p].x;const double r=model.Radius(x);const auto&cell=g.cells[p];const auto&block=g.mesh.blocks[cell.block];
      for(int f=0;f<nf;++f){v[f]=model.state[nf*p+f];double df[3]{},ddf[3][3]{};
        if(f<18)model.BulkDerivatives(padded,f,p,df,ddf,f<=6);
        for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
      const auto u=model.Jet(x,v,d,dd);const auto c=h::EvolvedConstraints(u,model.Omega(x,u));
      if(!c.valid)throw std::runtime_error("invalid constraint sample");
      double mr=0;for(int a=0;a<3;++a)mr+=c.momentum[a]*x[a]/r;
      const double values[3]={c.hamiltonian,std::sqrt(c.momentum_conformal_norm2),mr};
      for(int f=0;f<3;++f)current[p][f]=values[f];
      const bool seam=block.face>=0&&(cell.j<2||cell.k<2||cell.j>=n-2||cell.k>=n-2);
      const bool mask[]={r>.063,r>.063&&block.face<0,r>=.16&&r<.18,r>.063&&seam,
        r>=.45&&r<.75,block.layer==int(g.mesh.radii.size())-1&&cell.i>=n-3,r>=.775&&r<.9};
      for(size_t k=0;k<names.size();++k)if(mask[k]){
        regions[k].Add(values,weight,x,cell.block);
        if(mode){double delta[3];for(int f=0;f<3;++f)delta[f]=values[f]-reference[p][f];difference[k].Add(delta,weight,x,cell.block);}}
      for(size_t k=0;k<std::size(cutoffs);++k)if(r>cutoffs[k])cuts[k].Add(values,weight,x,cell.block);
      if(r>.063)profile[std::min(99,int(r*100))].Add(values,weight,x,cell.block);
    }
    if(!mode)reference=std::move(current);
    if(mode)out<<',';out<<"{\"name\":\""<<(mode==0?"production":mode==1?"exact_interior_halos":"algebraic_projection")<<"\",\"replaced_halos\":"<<replaced<<",\"regions\":{";
    for(size_t k=0;k<names.size();++k){if(k)out<<',';out<<'"'<<names[k]<<"\":";regions[k].Write(out);}out<<"},\"difference_from_production\":{";
    for(size_t k=0;k<names.size();++k){if(k)out<<',';out<<'"'<<names[k]<<"\":";difference[k].Write(out);}out<<"},\"cutoffs\":[";
    for(size_t k=0;k<std::size(cutoffs);++k){if(k)out<<',';out<<"{\"r\":"<<cutoffs[k]<<",\"exact_volume\":"<<4*std::acos(-1.)*(1-std::pow(cutoffs[k],3))/3<<",\"measured\":";cuts[k].Write(out);out<<'}';}out<<"],\"radial_bins\":[";
    for(int k=0;k<100;++k){if(k)out<<',';out<<"{\"lo\":"<<k*.01<<",\"hi\":"<<(k+1)*.01<<",\"measured\":";profile[k].Write(out);out<<'}';}out<<"]}";
    std::cerr<<"N="<<n<<" mode="<<mode<<" completed\n";
  }
  out<<"],\"max_projection_field_change\":"<<projection_change<<",\"max_surface_projection_field_change\":"<<surface_projection_change<<",\"scri_constraint_field_change\":"<<scri_limit_error<<"}\n";
  if(!out)throw std::runtime_error("audit write failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
