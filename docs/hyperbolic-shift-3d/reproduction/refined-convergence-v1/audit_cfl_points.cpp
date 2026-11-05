// Read-only analysis of the native frozen-coefficient CFL estimate.
// No evolution, halo construction, or change of production CFL policy.
#include <iostream>
#include <iomanip>
#include <map>
#include "refined_configuration.hpp"
namespace h=z4c::hyperboloidal;
struct Maximum {
  double value=-1, x[3]{}, direction[3]{}, adv[3]{}, wave[3]{};
  int block=-1,layer=-1,refinement=-1,ijk[3]{};
  void Add(double v,const cubed_sphere::Point& p,const cubed_sphere::CompositeBlock& b,
           int m,int i,int j,int k,const double d[3],const double a[3],const double w[3]) {
    if(v<=value)return; value=v;block=m;layer=b.layer;refinement=b.refinement;
    ijk[0]=i;ijk[1]=j;ijk[2]=k;
    for(int q=0;q<3;++q){x[q]=p.x[q];direction[q]=d[q];adv[q]=a[q];wave[q]=w[q];}
  }
};
int main(int argc,char**argv){try{
  if(argc!=8)throw std::invalid_argument("N config leaves checkpoint offset count dt");
  const int n=std::stoi(argv[1]); const double dt=std::stod(argv[7]);
  hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[2],hyperbolic_reproduction::ContinuousCoreMesh(n));
  auto mesh=cfg.Mesh(n);const auto original=mesh.blocks;mesh.blocks.clear();
  std::ifstream leaves(argv[3]);int lx,ly,lz,level;
  const int root_level=int(std::log2(cfg.core_subdivisions));
  while(leaves>>lx>>ly>>lz>>level){
    cubed_sphere::CompositeBlock b;b.refinement=level-root_level;
    const int loc[3]={lx,ly,lz};
    const Real width=2*cfg.core_half_width/(cfg.core_subdivisions*(1<<b.refinement));
    for(int a=0;a<3;++a){b.lo[a]=-cfg.core_half_width+loc[a]*width;b.hi[a]=-cfg.core_half_width+(loc[a]+1)*width;}
    mesh.blocks.push_back(b);
  }
  for(const auto&b:original)if(b.face>=0)mesh.blocks.push_back(b);
  mesh.amr_cartesian_core=true;
  const size_t count=std::stoull(argv[6]);
  std::ifstream input(argv[4],std::ios::binary);input.seekg(std::stoull(argv[5]));
  std::map<std::string,Maximum> maxima;size_t consumed=0;
  auto point=[&](int m,int i,int j,int k){
    double v[31];input.read(reinterpret_cast<char*>(v),sizeof(v));
    if(!input||consumed+31>count)throw std::runtime_error("truncated array");consumed+=31;
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};
    const auto map=mesh.Geometry(m,q);const auto&b=mesh.blocks[m];
    double r2=0;for(auto x:map.x)r2+=x*x;
    const double r=std::sqrt(r2),alpha=v[18]+(1+r2)/2,chi=v[0];
    h::MetricJet<Real> metric{};int f=1;
    for(int a=0;a<3;++a)for(int c=a;c<3;++c)metric.g[a][c]=metric.g[c][a]=v[f++]+(a==c);
    const auto g=h::Geometry(metric);if(!(g.determinant>0))throw std::runtime_error("invalid metric");
    const auto c=h::ShiftCoefficients(r,alpha,chi,cfg.gauge);
    double direction[3]{},adv[3]{},wave[3]{},sum=0,squares=0,largest=0;
    for(int A=0;A<3;++A){
      double Q=0,E=0,transport=0;
      for(int a=0;a<3;++a){transport+=map.J[A][a]*(v[22+a]-map.x[a]);E+=map.J[A][a]*map.J[A][a];
        for(int d=0;d<3;++d)Q+=g.inverse[a][d]*map.J[A][a]*map.J[A][d];}
      double speed2=std::max(h::GaugeShiftSpeedSquared(c,Q),h::GaugeLapseSpeedSquared(alpha,chi,c,Q,E,cfg.gauge));
      if(cfg.gauge.enforce_cfl)speed2=std::max(speed2,alpha*alpha*chi*Q);
      if(!(speed2>=0&&std::isfinite(speed2)))throw std::runtime_error("invalid speed");
      adv[A]=std::abs(transport)*n*dt;wave[A]=std::sqrt(speed2)*n*dt;
      direction[A]=adv[A]+wave[A];sum+=direction[A];squares+=direction[A]*direction[A];largest=std::max(largest,direction[A]);
    }
    const std::string region=b.face<0?"core-level"+std::to_string(b.refinement):"shell-layer"+std::to_string(b.layer);
    for(const auto&entry:std::map<std::string,double>{{"l1",sum},{"l2",std::sqrt(squares)},{"linf",largest}})
      for(const auto&name:{entry.first,region+"/"+entry.first})
        maxima[name].Add(entry.second,map,b,m,i,j,k,direction,adv,wave);
  };
  for(int m=0;m<int(mesh.blocks.size());++m)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i)point(m,i,j,k);
  for(int m=0;m<int(mesh.blocks.size());++m){const auto&b=mesh.blocks[m];int lo[3],hi[3];
    const int overlap=b.layer<=0?mesh.evolved_overlap:0;
    for(int a=0;a<3;++a){
      lo[a]=(b.face<0&&b.lo[a]>-mesh.half_width+1e-14)||(mesh.spherical_inner_boundary&&b.layer==0&&a==0)?0:-overlap;
      hi[a]=b.face<0&&b.hi[a]<mesh.half_width-1e-14?n:n+overlap;}
    for(int k=lo[2];k<hi[2];++k)for(int j=lo[1];j<hi[1];++j)for(int i=lo[0];i<hi[0];++i)
      if(i<0||j<0||k<0||i>=n||j>=n||k>=n)point(m,i,j,k);
  }
  if(consumed!=count)throw std::runtime_error("native point ordering/count mismatch");
  std::cout<<std::setprecision(17)<<"{\"points\":"<<consumed/31<<",\"maxima\":{";bool first=true;
  for(const auto&entry:maxima){if(!first)std::cout<<',';first=false;const auto&v=entry.second;
    std::cout<<'"'<<entry.first<<"\":{\"cfl\":"<<v.value<<",\"block\":"<<v.block<<",\"layer\":"<<v.layer<<",\"refinement\":"<<v.refinement;
    auto array=[&](const char*name,const auto*values){std::cout<<",\""<<name<<"\":[";for(int a=0;a<3;++a)std::cout<<(a?",":"")<<values[a];std::cout<<']';};
    array("x",v.x);array("ijk",v.ijk);array("directional_cfl",v.direction);array("advection_cfl",v.adv);array("wave_cfl",v.wave);std::cout<<'}';
  }std::cout<<"}}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
