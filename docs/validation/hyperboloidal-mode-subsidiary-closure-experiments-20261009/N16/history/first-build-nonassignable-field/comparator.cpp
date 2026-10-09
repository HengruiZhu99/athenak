// Saved-mode diagnostic only; no evolution or replacement native closure.
// Exact pinned old Lift/Restrict implementation; its main is never invoked.
#define main PinnedOldJvMain
#include "inputs/old-jv-source.cpp"
#undef main
#include <complex>
#include <stdexcept>
using C=std::complex<double>;
// The frozen subsidiary uses only .v of stationary Omega-normal jets.
// Bind these to the identical analytic double-valued production helper.
struct NormalValue { double v; };
struct NormalJet { NormalValue normal,dnormal[3]; };
const hyp::Z4cJet<double>& Lift(const hyp::Z4cJet<double>&u) {return u;}
NormalJet Omega(const hyp::Z4cJet<double>&u,const hyp::LayerPoint<double>&p) {
  const auto o=hyp::CartesianOmega(u,p);
  NormalJet n{};n.normal.v=o.normal;
  for(int i=0;i<3;++i)n.dnormal[i].v=o.dnormal[i];
  return n;
}
#include "inputs/subsidiary.hpp"
struct Field {
  const std::vector<double>*v;int nx,ny,f;
  double operator()(int,int k,int j,int i)const{return (*v)[8*(i+nx*(j+ny*k))+f];}
  double operator()(int s)const{return (*v)[8*s+f];}
};
struct Velocity {
  const std::vector<hyp::LayerPoint<double>>*p;const std::vector<int>*map;int nx,ny;
  double operator()(int,int a,int k,int j,int i)const{return (*p)[(*map)[i+nx*(j+ny*k)]].beta[a];}
};
struct Comparator {
  hyp::CartesianConformalPatch patch;
  std::vector<Cell>cells;std::vector<hyp::LayerPoint<double>>points;
  std::vector<int>map,center,advection,nested;
  std::vector<std::array<int,3>>s2,s3;
  DvceArray5D<Real>delta;
  Comparator():patch(Grid(),.5,2,{true,.05,.95},Gauge(),true) {
    const auto&g=patch.grid;map.assign(g.n[0]*g.n[1]*g.n[2],-1);
    delta=patch.Allocate("pinned primitive linear operation");
    for(int k=0;k<g.n[2];++k)for(int j=0;j<g.n[1];++j)for(int i=0;i<g.n[0];++i)if(g.Interior(i,j,k)){
      Cell c{};c.s=g.Index(i,j,k);c.i=i;c.j=j;c.k=k;
      c.xyz[0]=g.first[0]+i*g.h[0];c.xyz[1]=g.first[1]+j*g.h[1];c.xyz[2]=g.first[2]+k*g.h[2];
      const auto p=patch.reference.At(c.xyz[0],c.xyz[1],c.xyz[2]);const auto geo=hyp::Geometry(p.state.metric);c.omega=p.omega;
      for(int a=0;a<3;++a)for(int b=0;b<3;++b){c.g[a][b]=p.state.metric.g[a][b];c.inv[a][b]=geo.inverse[a][b];c.a[a][b]=p.state.a.k[a][b];}
      map[c.s]=cells.size();cells.push_back(c);points.push_back(p);
    }
    s2.push_back({0,0,0});
    for(int a=0;a<3;++a)for(int n:{-2,-1,1,2}){std::array<int,3>d{};d[a]=n;s2.push_back(d);}
    for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)for(int x:{-2,-1,1,2})for(int y:{-2,-1,1,2}){
      std::array<int,3>d{};d[a]=x;d[b]=y;s2.push_back(d);
    }
    s3=s2;for(int a=0;a<3;++a)for(int n:{-3,3}){std::array<int,3>d{};d[a]=n;s3.push_back(d);}
    for(const auto&c:cells){
      center.push_back(Inside(c,s2));advection.push_back(Inside(c,s3));
      bool ok=true;for(const auto&a:s2)for(const auto&b:s3)if(!g.Interior(c.i+a[0]+b[0],c.j+a[1]+b[1],c.k+a[2]+b[2]))ok=false;
      nested.push_back(ok);
    }
  }
  static hyp::SphericalGhostGrid Grid(){hyp::SphericalGhostGrid g;g.radius=1;for(int a=0;a<3;++a){g.n[a]=22;g.h[a]=2.2/16;g.first[a]=-.5*21*g.h[a];}return g;}
  static hyp::LayerGaugeParameters Gauge(){hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=2;return g;}
  bool Inside(const Cell&c,const std::vector<std::array<int,3>>&st)const{for(const auto&d:st)if(!patch.grid.Interior(c.i+d[0],c.j+d[1],c.k+d[2]))return false;return true;}
  void PrintMetadata(){const auto&g=patch.grid;auto gh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.ghosts);double sumerr=0;size_t refs=0;
    for(size_t a=0;a<gh.extent(0);++a){double sum=0;for(int b=0;b<gh(a).count;++b){int s=gh(a).donors[b];if(map[s]<0)throw std::runtime_error("recursive or exterior ghost donor");sum+=gh(a).weights[b];++refs;}sumerr=std::max(sumerr,std::abs(sum-1));}
    std::cout<<std::setprecision(17)<<"{\"points\":"<<cells.size()<<",\"spacing\":"<<g.h[0]<<",\"ghost_plans\":"<<gh.extent(0)<<",\"strict_nonrecursive_donor_references\":"<<refs<<",\"ghost_weight_sum_error\":"<<sumerr<<",\"xyz_omega\":[";
    for(size_t a=0;a<cells.size();++a){const auto&c=cells[a];std::cout<<(a?",":"")<<'['<<c.xyz[0]<<','<<c.xyz[1]<<','<<c.xyz[2]<<','<<c.omega<<']';}
    auto mask=[&](const char*name,const std::vector<int>&v){std::cout<<"],\""<<name<<"\":[";for(size_t i=0;i<v.size();++i)std::cout<<(i?",":"")<<v[i];};
    mask("centered_active_stencil",center);mask("centered_plus_Lx_active_stencil",advection);mask("fully_nested_primitive_active_stencil",nested);std::cout<<"]}\n"<<std::flush;
  }
  void ConstraintOperator(const double*input,double*out,bool extension){
    const auto&g=patch.grid;std::vector<double>field(map.size()*8,0);
    for(size_t p=0;p<cells.size();++p)for(int f=0;f<8;++f)field[8*cells[p].s+f]=input[8*p+f];
    if(extension)for(int f=0;f<8;++f)hyp::FillSphericalGhosts(Field{&field,g.n[0],g.n[1],f},patch.ghosts);
    const double idx[3]={1/g.h[0],1/g.h[1],1/g.h[2]},spacing[3]={g.h[0],g.h[1],g.h[2]};int stride[3]={1,g.n[0],g.n[0]*g.n[1]};
    Velocity beta{&points,&map,g.n[0],g.n[1]};std::fill(out,out+cells.size()*24,0);
    for(size_t p=0;p<cells.size();++p){const auto&c=cells[p];ConstraintJet q{};
      if(extension||center[p])for(int f=0;f<8;++f){Field v{&field,g.n[0],g.n[1],f};q.q[f]=v(0,c.k,c.j,c.i);for(int a=0;a<3;++a){q.d[a][f]=Dx<3>(a,idx,v,0,c.k,c.j,c.i);for(int b=a;b<3;++b)q.dd[a][b][f]=q.dd[b][a][f]=a==b?Dxx<3>(a,idx,v,0,c.k,c.j,c.i):Dxy<3>(a,b,idx,v,0,c.k,c.j,c.i);}}
      if(extension||center[p]){auto k=Subsidiary(points[p],10,q);for(int f=0;f<8;++f){if(std::abs(k[f].imag())>1e-20)throw std::runtime_error("unexpected imaginary real-input output");out[p*24+f]=k[f].real();}}
      for(int f=0;f<8;++f){Field v{&field,g.n[0],g.n[1],f};
        if(extension||advection[p])out[p*24+8+f]=hyp::ScalarUpwindCorrection<3>(v,beta,idx,0,c.k,c.j,c.i);
        out[p*24+16+f]=.1*hyp::InteriorKOSixth(v,patch.mask,c.s,stride,spacing);
      }
    }
  }
  void PrimitiveCorrections(const double*input,double*out){
    Kokkos::deep_copy(delta,0.);const auto&g=patch.grid;
    for(size_t p=0;p<cells.size();++p){double raw[22];Lift(input+20*p,cells[p],raw);for(int f=0;f<22;++f)delta(0,f,cells[p].k,cells[p].j,cells[p].i)=raw[f];}
    for(int f=0;f<22;++f)hyp::FillSphericalGhosts(hyp::CartesianComponent{delta,f,g.n[0],g.n[1]},patch.ghosts);
    auto dev=hyp::BindCartesianFields(delta);const double idx[3]={1/g.h[0],1/g.h[1],1/g.h[2]},spacing[3]={g.h[0],g.h[1],g.h[2]};int stride[3]={1,g.n[0],g.n[0]*g.n[1]};
    for(size_t p=0;p<cells.size();++p){const auto&c=cells[p];hyp::Z4cRHS<double>r{};hyp::GaugeRHS<double>a{};
      struct B{const hyp::LayerPoint<double>*p;double operator()(int,int d,int,int,int)const{return p->beta[d];}}beta{&points[p]};
      hyp::AddMeshUpwindAdvectionWithVelocity<3>(dev,beta,idx,0,c.k,c.j,c.i,r,a);
      double raw[22]{};raw[0]=r.chi;raw[7]=r.trace;raw[17]=r.theta;raw[18]=a.alpha;for(int f=0;f<6;++f){raw[1+f]=r.metric[ti[f]][tj[f]];raw[8+f]=r.a[ti[f]][tj[f]];}for(int d=0;d<3;++d){raw[14+d]=r.lambda[d];raw[19+d]=a.beta[d];}Restrict(raw,c,out+40*p);
      for(int f=0;f<22;++f)raw[f]=.1*hyp::InteriorKOSixth(hyp::CartesianComponent{delta,f,g.n[0],g.n[1]},patch.mask,c.s,stride,spacing);
      Restrict(raw,c,out+40*p+20);
    }
  }
};
int main(int argc,char**argv){Kokkos::ScopeGuard guard(argc,argv);try{Comparator a;a.PrintMetadata();char mode;
  while(std::cin.read(&mode,1)){size_t n=a.cells.size(),size=mode=='p'?20*n:8*n;std::vector<double>v(size),out((mode=='p'?40:24)*n);
    if(!std::cin.read(reinterpret_cast<char*>(v.data()),size*8))throw std::runtime_error("truncated comparator protocol");
    if(mode=='p')a.PrimitiveCorrections(v.data(),out.data());else if(mode=='k'||mode=='g')a.ConstraintOperator(v.data(),out.data(),mode=='g');else throw std::runtime_error("unknown comparator operation");
    std::cout.write(reinterpret_cast<char*>(out.data()),out.size()*8);std::cout.flush();
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
