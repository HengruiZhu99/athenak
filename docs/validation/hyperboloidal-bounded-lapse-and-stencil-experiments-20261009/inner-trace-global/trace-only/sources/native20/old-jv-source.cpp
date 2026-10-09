#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp = z4c::hyperboloidal;
using z4c::Z4c;
const int key[20] = {0,1,2,3,4,5,7,8,9,10,11,12,14,15,16,17,18,19,20,21};
const int ti[6] = {0,0,0,1,1,2}, tj[6] = {0,1,2,1,2,2};
struct Cell {
  int s, i,j,k;
  double xyz[3], omega;
  double g[3][3], inv[3][3], a[3][3];
};

// Twenty independent tangent components: determinant-one g and trace-free A.
// Build dependent zz components with the actual nonflat reference metric/A.
void Lift(const double* v,const Cell& c,double out[22]) {
  std::fill(out,out+22,0);
  for(int f=0;f<20;++f)out[key[f]]=v[f];
  double dg[3][3]{}, da[3][3]{};
  for(int f=0;f<5;++f) {
    dg[ti[f]][tj[f]]=dg[tj[f]][ti[f]]=out[1+f];
    da[ti[f]][tj[f]]=da[tj[f]][ti[f]]=out[8+f];
  }
  double tg=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)tg+=c.inv[i][j]*dg[i][j];
  dg[2][2]=out[6]=-tg/c.inv[2][2];
  double ta=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    ta+=c.inv[i][j]*da[i][j];
    for(int k=0;k<3;++k)for(int l=0;l<3;++l)
      ta-=c.inv[i][k]*dg[k][l]*c.inv[l][j]*c.a[i][j];
  }
  out[13]=-ta/c.inv[2][2];
}

// Derivative of the actual final-stage algebraic projector at the reference.
void Restrict(const double in[22],const Cell& c,double* out) {
  double dg[3][3]{}, da[3][3]{};
  for(int f=0;f<6;++f) {
    dg[ti[f]][tj[f]]=dg[tj[f]][ti[f]]=in[1+f];
    da[ti[f]][tj[f]]=da[tj[f]][ti[f]]=in[8+f];
  }
  double tg=0,ta=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    tg+=c.inv[i][j]*dg[i][j];
    ta+=c.inv[i][j]*da[i][j];
    for(int k=0;k<3;++k)for(int l=0;l<3;++l)
      ta-=c.inv[i][k]*dg[k][l]*c.inv[l][j]*c.a[i][j];
  }
  for(int f=0;f<20;++f)out[f]=in[key[f]];
  for(int f=0;f<5;++f) {
    out[1+f]-=c.g[ti[f]][tj[f]]*tg/3;
    out[7+f]-=c.g[ti[f]][tj[f]]*ta/3;
  }
}

int main(int argc,char** argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  try {
    const int n=std::atoi(argv[1]), degree=std::atoi(argv[2]);
    const double a=std::atof(argv[3]);
    const bool layer=std::atoi(argv[4]),sym=std::atoi(argv[5]);
    const double span=std::atof(argv[6]),r0=std::atof(argv[7]),r1=std::atof(argv[8]);
    const double eps_scale=argc>9?std::atof(argv[9]):1e-5;
    hyp::SphericalGhostGrid grid;
    grid.radius=1;
    for(int k=0;k<3;++k) {grid.n[k]=n+6;grid.h[k]=span/n;grid.first[k]=-.5*(n+5)*grid.h[k];}
    hyp::LayerParameters lp;lp.enabled=layer;lp.r0=r0;lp.r1=r1;
    hyp::LayerGaugeParameters gauge;gauge.physical_trace_lapse=true;gauge.preferred_source=false;
    hyp::CartesianConformalPatch patch(grid,a,degree,lp,gauge,sym);
    auto q=patch.Allocate("Jv state"),base=patch.Allocate("reference"),rhs=patch.Allocate("Jv rhs");
    patch.InitializeReference(base);
    std::vector<Cell> cells;
    for(int k=0;k<grid.n[2];++k)for(int j=0;j<grid.n[1];++j)for(int i=0;i<grid.n[0];++i) {
      if(!grid.Interior(i,j,k))continue;
      Cell c{};c.s=grid.Index(i,j,k);c.i=i;c.j=j;c.k=k;
      c.xyz[0]=grid.first[0]+i*grid.h[0];c.xyz[1]=grid.first[1]+j*grid.h[1];c.xyz[2]=grid.first[2]+k*grid.h[2];
      const auto p=patch.reference.At(c.xyz[0],c.xyz[1],c.xyz[2]);
      const auto geo=hyp::Geometry(p.state.metric);c.omega=p.omega;
      for(int i=0;i<3;++i)for(int j=0;j<3;++j) {c.g[i][j]=p.state.metric.g[i][j];c.inv[i][j]=geo.inverse[i][j];c.a[i][j]=p.state.a.k[i][j];}
      cells.push_back(c);
    }
    Kokkos::deep_copy(q,base);patch.RHS(q,rhs);
    double reference_residual=0;
    for(const auto& c:cells)for(int f=0;f<22;++f)reference_residual=std::max(reference_residual,std::abs(rhs(0,f,c.k,c.j,c.i)));
    std::cout<<std::setprecision(17)<<"{\"dimension\":"<<20*cells.size()<<",\"points\":"<<cells.size()
             <<",\"min_omega\":"<<patch.min_omega<<",\"reference_rhs_max\":"<<reference_residual<<",\"field_ids\":[";
    for(int i=0;i<20;++i)std::cout<<(i?",":"")<<key[i];
    std::cout<<"],\"xyz_omega\":[";
    for(size_t p=0;p<cells.size();++p) {const auto&c=cells[p];std::cout<<(p?",":"")<<'['<<c.xyz[0]<<','<<c.xyz[1]<<','<<c.xyz[2]<<','<<c.omega<<']';}
    std::cout<<"]}\n"<<std::flush;
    std::vector<double> v(20*cells.size()),rplus(v.size()),rminus(v.size()),output(v.size());
    while(std::cin.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(double))) {
      double maximum=0;for(double x:v)maximum=std::max(maximum,std::abs(x));
      if(maximum==0){std::fill(output.begin(),output.end(),0);}
      else {
        const double eps=eps_scale/maximum;
        for(int sign:{1,-1}) {
          Kokkos::deep_copy(q,base);
          for(size_t p=0;p<cells.size();++p) {
            const auto&c=cells[p];double lifted[22];Lift(v.data()+20*p,c,lifted);
            for(int f=0;f<22;++f)q(0,f,c.k,c.j,c.i)+=sign*eps*lifted[f];
          }
          patch.ProjectAlgebraic(q);patch.RHS(q,rhs);
          auto& result=sign==1?rplus:rminus;
          for(size_t p=0;p<cells.size();++p) {
            const auto&c=cells[p];double raw[22];
            for(int f=0;f<22;++f)raw[f]=rhs(0,f,c.k,c.j,c.i);
            Restrict(raw,c,result.data()+20*p);
          }
        }
        for(size_t i=0;i<v.size();++i)output[i]=(rplus[i]-rminus[i])/(2*eps);
      }
      std::cout.write(reinterpret_cast<char*>(output.data()),output.size()*sizeof(double));std::cout.flush();
    }
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
