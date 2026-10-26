// Standalone massless test scalar. Does not link or allocate a Z4c system.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include "geometry.hpp"
#include "sbp63.hpp"
namespace ts=test_scalar;
using State=Kokkos::View<double*[5],Kokkos::LayoutRight>;
constexpr int NG=4;
struct Block {
  int offset,n[3],stride[3],size;double dx[3];int patch;
  int index(int i,int j,int k)const{return offset+i*stride[0]+j*stride[1]+k*stride[2];}
};
struct Cell {
  ts::Coefficients physical,reference;
  double idx[3]{},weight=0,ko[3]{};int stride[3]{},block=0;
  double boundary_rate=0,radial_speeds[2]{},outer_norm=1,boundary_amplitude_gradient=0;int outer_distance=-1,radial_sbp63=0,radial_end=-1,normal_upwind=0,normal_sbp=0;
  int one[3]{},trim_ko[3]{}; // one-sided direction only when an excision control removes points
};
struct KernelPhysical {
  double g[4][4]{},b[4]{},beta_gradient[3][3]{},alpha_gradient[3]{},potential=0,x[3]{};int active=0;
  struct {double J[3][3]{};} map;
};
struct KernelCell {
  KernelPhysical physical;
  double idx[3]{},ko[3]{},boundary_rate=0,radial_speeds[2]{},outer_norm=1,boundary_amplitude_gradient=0;int outer_distance=-1,radial_sbp63=0,radial_end=-1,normal_upwind=0,normal_sbp=0;
  int stride[3]{},one[3]{},trim_ko[3]{};
};
struct Row {int target=0,start=0,stride[3]{},count[3]{6,6,6};double w[3][6]{};
  int own_start=0,own_stride[3]{},own_count[3]{},characteristic=0;
  double own_w[3][6]{},normal[3]{},ginvn[3]{},beta[3]{},alpha=1,root=1,speeds[3]{};
};
template<class T,class C> KOKKOS_INLINE_FUNCTION double deriv(const T &u,int l,int v,int A,const C &c) {
  int s=c.stride[A],one=c.one[A];
  if(A==0 && c.normal_upwind && c.radial_end>=0 && c.radial_end<2)one=-1;
  if(A==0 && c.radial_sbp63) {
    if(c.outer_distance>=0) {int end=l+c.outer_distance*s;double value=0;
      for(int j=0;j<9;++j)value-=radial_sbp::left[c.outer_distance][j]*u(end-j*s,v);return c.idx[A]*value;}
    return c.idx[A]*(-u(l-3*s,v)/60+3*u(l-2*s,v)/20-3*u(l-s,v)/4+3*u(l+s,v)/4-3*u(l+2*s,v)/20+u(l+3*s,v)/60);
  }
  if(A==0 && c.outer_distance>=0) {
    const double rows[4][6]={{24./17,-59./34,4./17,3./34,0,0},{.5,0,-.5,0,0,0},{-4./43,59./86,0,-59./86,4./43,0},{-3./98,0,59./98,0,-32./49,4./49}};
    int end=l+c.outer_distance*s;double value=0;for(int j=0;j<6;++j)value+=rows[c.outer_distance][j]*u(end-j*s,v);return c.idx[A]*value;
  }
  if(one) {
    s*=one;return one*c.idx[A]*(-25*u(l,v)/12+4*u(l+s,v)-3*u(l+2*s,v)+4*u(l+3*s,v)/3-u(l+4*s,v)/4);
  }
  return c.idx[A]*(u(l-2*s,v)-8*u(l-s,v)+8*u(l+s,v)-u(l+2*s,v))/12;
}
template<class T,class C> KOKKOS_INLINE_FUNCTION double gradient(const T &u,int l,int v,int a,const C &c) {
  double r=0;for(int A=0;A<3;++A)r+=c.physical.map.J[A][a]*deriv(u,l,v,A,c);return r;
}
template<class T,class C> KOKKOS_INLINE_FUNCTION double radial_sbp_dissipation(const T &u,int l,int v,const C &c) {
  int stride=c.stride[0];double q=0;
  if(c.radial_end>=4) {
    const double stencil[9]={1,-8,28,-56,70,-56,28,-8,1};
    for(int k=0;k<9;++k)q+=stencil[k]*u(l+(k-4)*stride,v);
  } else {
    const double difference[5]={1,-4,6,-4,1};
    for(int m=0;m<5;++m)if(4-m<=c.radial_end) {
      double delta=0;for(int j=0;j<5;++j)delta+=difference[j]*u(l+(j-m)*stride,v);
      q+=difference[m]*delta;
    }
  }
  return c.idx[0]*q/(128*c.outer_norm);
}
template<class T,class C> KOKKOS_INLINE_FUNCTION double upwind(const T &u,int l,int v,int A,double speed,const C &c) {
  if(A==0 && c.normal_sbp)return deriv(u,l,v,A,c)+(speed>=0?1:-1)*radial_sbp_dissipation(u,l,v,c);
  int s=c.stride[A],one=c.one[A];
  if(A==0 && c.radial_end>=0 && c.radial_end<(speed>=0?1:3))one=-1;
  if(one) {s*=one;return one*c.idx[A]*(-25*u(l,v)/12+4*u(l+s,v)-3*u(l+2*s,v)+4*u(l+3*s,v)/3-u(l+4*s,v)/4);}
  if(speed>=0)return c.idx[A]*(-u(l-3*s,v)/12+u(l-2*s,v)/2-3*u(l-s,v)/2+5*u(l,v)/6+u(l+s,v)/4);
  return c.idx[A]*(-u(l-s,v)/4-5*u(l,v)/6+3*u(l+s,v)/2-u(l+2*s,v)/2+u(l+3*s,v)/12);
}
static double det(const double a[3][3]) {
  return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}
class Solver {
 public:
  const ts::Parameters p;
  const int NG; // Extra halo only for the normal SBP fourth-difference pair.
  const cubed_sphere::Map map;
  std::vector<Block> blocks;std::vector<Cell> host;
  Kokkos::View<const Cell*> geometry;
  Kokkos::View<const KernelCell*> kernel;
  Kokkos::View<int*> active;
  Kokkos::View<Row*> ghosts;
  State u,rhs,base,stage,reduction_work;
  double dt=1e99; std::string output;
  uint64_t initial_physical=0,initial_reference=0,initial_physical_derivatives=0,initial_reference_derivatives=0;double stage_time=0;
  Solver(ts::Parameters parameters,std::string directory):p(parameters),NG(parameters.normal_upwind && parameters.outer_sbp==2?4:3),map{p.L,p.S*p.cap,p.radial_stretch},output(directory) {
    if((p.normal_upwind && (p.compatible_reduction || p.characteristic_transfer || p.outer_sbp==1 || p.ko!=0 || (p.cap<1 && p.boundary=="exact"))) || p.n<12 || p.S<=20 || p.cap<=.5 || p.cap>1 || p.cfl<=0 || p.cfl>.5 || p.mass<0 || (p.compatible_reduction && p.characteristic_transfer) || (p.compatible_reduction && p.outer_sbp && p.cap<1 && p.boundary=="exact") || !std::isfinite(p.start_time) || !std::isfinite(p.tlim) || p.start_time<0 || p.tlim<p.start_time || !std::isfinite(p.wall_seconds) || p.wall_seconds<0 || p.outer_sbp<0 || p.outer_sbp>2 || !std::isfinite(p.ko) || p.ko<0 || p.ko>1 || !std::isfinite(p.radial_stretch) || p.radial_stretch < -4 || p.radial_stretch>0 || p.reduction_damping<0 || std::abs(p.spin)>=1)
      throw std::invalid_argument("invalid scalar configuration");
    if(p.boundary!="exact" && p.boundary!="radiation")throw std::invalid_argument("unknown scalar boundary");
    if(p.mass==0 && p.data!="angular")throw std::invalid_argument("Minkowski requires angular analytic data");
    if(p.mass!=0 && p.boundary=="exact")throw std::invalid_argument("black holes have no analytic outer data");
    std::filesystem::create_directories(output);
    int total=0;
    for(int b=0;b<7;++b) {
      Block z{};z.patch=b-1;z.offset=total;
      z.n[0]=2*p.n+(b?1:0);z.n[1]=z.n[2]=b?p.n:2*p.n;
      z.dx[0]=b?1./(z.n[0]-1):2*p.L/z.n[0];
      z.dx[1]=z.dx[2]=b?2./p.n:z.dx[0];
      z.stride[0]=1;z.stride[1]=z.n[0]+2*NG;z.stride[2]=z.stride[1]*(z.n[1]+2*NG);
      z.size=z.stride[2]*(z.n[2]+2*NG);total+=z.size;blocks.push_back(z);
    }
    host.resize(total);std::vector<int> ids;
    std::ofstream chars(output+"/characteristics.csv");chars<<std::setprecision(17)<<"block,x,y,z,face,cminus,cplus\n";
    ts::Parameters ref=p;ref.mass=ref.spin=0;ref.smooth_layer=false;
    for(int b=0;b<7;++b) {
      const auto &z=blocks[b];
      for(int k=0;k<z.n[2]+2*NG;++k)for(int j=0;j<z.n[1]+2*NG;++j)for(int i=0;i<z.n[0]+2*NG;++i) {
        int l=z.index(i,j,k);Cell &c=host[l];c.block=b;
        double q[3],y[3];
        q[0]=b?(i-NG)*z.dx[0]:-p.L+(i-NG+.5)*z.dx[0];
        q[1]=b?-1+(j-NG+.5)*z.dx[1]:-p.L+(j-NG+.5)*z.dx[1];
        q[2]=b?-1+(k-NG+.5)*z.dx[2]:-p.L+(k-NG+.5)*z.dx[2];
        if(b)map.Forward(z.patch,q,y);else for(int a=0;a<3;++a)y[a]=q[a];
        if(b && y[z.patch/2]*(z.patch%2==0?1:-1)<=0)throw std::invalid_argument("folded radial ghost map");
        for(int a=0;a<3;++a) {c.physical.x[a]=y[a];c.idx[a]=1/z.dx[a];c.stride[a]=z.stride[a];}
        c.physical.active=i>=NG&&i<NG+z.n[0]&&j>=NG&&j<NG+z.n[1]&&k>=NG&&k<NG+z.n[2];
        if(!b && p.mass!=0 && std::max({fabs(y[0]),fabs(y[1]),fabs(y[2])})<p.excision)c.physical.active=0;
        if(!c.physical.active)continue;
        c.physical=ts::coefficients(y,p);c.reference=ts::coefficients(y,ref);
        if(b){c.physical.map=map.Geometry(z.patch,q);c.radial_end=NG+z.n[0]-1-i;}
        else {for(int a=0;a<3;++a)c.physical.map.J[a][a]=1;}c.normal_upwind=p.normal_upwind;c.normal_sbp=p.normal_upwind && p.outer_sbp==2 && b>0;
        c.reference.map=c.physical.map;
        if(p.outer_sbp==2 && b)c.radial_sbp63=1;
        if(p.outer_sbp && b && NG+z.n[0]-1-i<(p.outer_sbp==2?6:4)) {
          const double weights[4]={17./48,59./48,43./48,49./48};
          c.outer_distance=NG+z.n[0]-1-i;c.outer_norm=p.outer_sbp==2?radial_sbp::H[c.outer_distance]:weights[c.outer_distance];
          if(c.outer_distance<3)c.trim_ko[0]=1;
        }
        c.weight=z.dx[0]*z.dx[1]*z.dx[2]/fabs(det(c.physical.map.J));
        if(b && (i==NG || i==NG+z.n[0]-1))c.weight*=.5;
        double invdt=0;
        for(int A=0;A<3;++A) {
          auto speed=ts::speeds(c.physical,c.physical.map.J[A]);
          double rate=std::max(fabs(speed[0]),fabs(speed[1]))*c.idx[A];
          invdt+=rate;c.ko[A]=p.ko*rate/64;
          if(b && A==0 && i==NG+z.n[0]-1)chars<<b<<","<<y[0]<<","<<y[1]<<","<<y[2]<<",outer,"<<speed[0]<<","<<speed[1]<<"\n";
        }
        if(b && i==NG+z.n[0]-1) {
          double rho=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]),normal[3];for(int a=0;a<3;++a)normal[a]=y[a]/rho;
          auto unit_speed=ts::speeds(c.physical,normal);chars<<b<<","<<y[0]<<","<<y[1]<<","<<y[2]<<",outer_unit,"<<unit_speed[0]<<","<<unit_speed[1]<<"\n";
        }
        if(b && i==NG+z.n[0]-1 && p.cap<1 && p.boundary=="radiation") {
          double rho=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]),normal[3];
          for(int a=0;a<3;++a)normal[a]=y[a]/rho;
          auto speed=ts::speeds(c.physical,normal);
          c.radial_speeds[0]=speed[0];c.radial_speeds[1]=speed[1];
          double scale=0;for(int a=0;a<3;++a)scale+=c.physical.map.J[0][a]*normal[a];
          c.boundary_rate=(p.outer_sbp?1/c.outer_norm:2)*std::max(0.,-speed[0])*std::abs(scale)*c.idx[0];
          ts::AD radius=ts::AD::var(rho,0),om=ts::omega(radius,p);
          c.boundary_amplitude_gradient=(1+p.mass*om.d[0])/(rho+p.mass*om.f);
        }
        dt=std::min(dt,p.cfl/invdt);ids.push_back(l);
      }
    }
    // Excision is optional. Check the actual Cartesian faces, not merely r<r+.
    bool inner_ok=true;
    auto removed=[&](int l) {auto &x=host[l].physical.x;
      return std::max({fabs(x[0]),fabs(x[1]),fabs(x[2])})<p.excision;};
    for(int l:ids)if(p.mass!=0 && p.excision>0 && host[l].block==0)for(int A=0;A<3;++A) {
      auto &c=host[l];int s=c.stride[A];
      for(int sign:{-1,1})if(removed(l+sign*s)) {
        double normal[3]{},face[3];normal[A]=sign;
        for(int a=0;a<3;++a)face[a]=c.physical.x[a];face[A]+=sign/(2*c.idx[A]);
        auto sp=ts::speeds(ts::coefficients(face,p),normal);
        chars<<0<<","<<face[0]<<","<<face[1]<<","<<face[2]<<",inner,"<<sp[0]<<","<<sp[1]<<"\n";
        inner_ok &= sp[0]>=-1e-12;
        int tangent[2],next_tangent=0;for(int a=0;a<3;++a)if(a!=A)tangent[next_tangent++]=a;
        for(int u0=-1;u0<=1;++u0)for(int v0=-1;v0<=1;++v0)if(u0 || v0) {
          double probe[3];for(int a=0;a<3;++a)probe[a]=face[a];probe[tangent[0]]+=u0/(2*c.idx[tangent[0]]);probe[tangent[1]]+=v0/(2*c.idx[tangent[1]]);
          auto corner_speed=ts::speeds(ts::coefficients(probe,p),normal);chars<<0<<","<<probe[0]<<","<<probe[1]<<","<<probe[2]<<",inner_face_probe,"<<corner_speed[0]<<","<<corner_speed[1]<<"\n";inner_ok &= corner_speed[0]>=-1e-12;
        }
      }
      if(removed(l-2*s) || removed(l-s))c.one[A]=1;
      if(removed(l+2*s) || removed(l+s))c.one[A]=-1;
      for(int off=-3;off<=3;++off)if(removed(l+off*s))c.trim_ko[A]=1;
    }
    chars.close();if(!inner_ok)throw std::runtime_error("excision has an incoming physical scalar characteristic");
    if(p.mass>0) {
      double bulk_ko[3]{};
      for(int l:ids)if(host[l].block==0)for(int A=0;A<3;++A)bulk_ko[A]=std::max(bulk_ko[A],host[l].ko[A]);
      for(int l:ids)if(host[l].block==0)for(int A=0;A<3;++A)host[l].ko[A]=bulk_ko[A];
    }
    if(p.outer_sbp) {
      double patch_ko[7]{};for(int l:ids)if(host[l].block>0)patch_ko[host[l].block]=std::max(patch_ko[host[l].block],host[l].ko[0]);
      for(int l:ids)if(host[l].block>0)host[l].ko[0]=patch_ko[host[l].block];
    }
    if(p.reduction_damping>0)dt=std::min(dt,.1/p.reduction_damping);
    Kokkos::View<Cell*> g("frozen scalar and reference geometry",total);
    auto gh=Kokkos::create_mirror_view(g);for(int l=0;l<total;++l)gh(l)=host[l];Kokkos::deep_copy(g,gh);geometry=g;
    Kokkos::View<KernelCell*> compact("frozen evolution coefficients",total);auto kh=Kokkos::create_mirror_view(compact);
    for(int l=0;l<total;++l) {
      auto &k=kh(l);const auto &f=host[l];k.physical.active=f.physical.active;k.physical.potential=f.physical.potential;
      for(int a=0;a<4;++a){k.physical.b[a]=f.physical.b[a];for(int b=0;b<4;++b)k.physical.g[a][b]=f.physical.g[a][b];}
      for(int a=0;a<3;++a) {k.physical.alpha_gradient[a]=f.physical.alpha_gradient[a];k.physical.x[a]=f.physical.x[a];k.idx[a]=f.idx[a];k.ko[a]=f.ko[a];k.stride[a]=f.stride[a];k.one[a]=f.one[a];k.trim_ko[a]=f.trim_ko[a];
        for(int b=0;b<3;++b){k.physical.map.J[a][b]=f.physical.map.J[a][b];k.physical.beta_gradient[a][b]=f.physical.beta_gradient[a][b];}}
      k.normal_sbp=f.normal_sbp;k.normal_upwind=f.normal_upwind;k.radial_end=f.radial_end;k.radial_sbp63=f.radial_sbp63;k.outer_distance=f.outer_distance;k.outer_norm=f.outer_norm;k.boundary_amplitude_gradient=f.boundary_amplitude_gradient;k.boundary_rate=f.boundary_rate;for(int a=0;a<2;++a)k.radial_speeds[a]=f.radial_speeds[a];
    }
    Kokkos::deep_copy(compact,kh);kernel=compact;
    active=Kokkos::View<int*>("active scalar cells",ids.size());auto ah=Kokkos::create_mirror_view(active);
    for(size_t i=0;i<ids.size();++i)ah(i)=ids[i];Kokkos::deep_copy(active,ah);
    u=State("scalar psi v d_i",total);rhs=State("scalar rhs",total);base=State("scalar RK base",total);stage=State("scalar RK stage",total);if(p.compatible_reduction)reduction_work=State("derived scalar reduction work",total);
    std::vector<Row> plan;
    for(int b=0;b<7;++b) {auto z=blocks[b];
      for(int k=0;k<z.n[2]+2*NG;++k)for(int j=0;j<z.n[1]+2*NG;++j)for(int i=0;i<z.n[0]+2*NG;++i) {
        if(i>=NG&&i<NG+z.n[0]&&j>=NG&&j<NG+z.n[1]&&k>=NG&&k<NG+z.n[2])continue;
        int l=z.index(i,j,k);Row row=donor(host[l].physical.x);row.target=l;
        if(p.characteristic_transfer && !(b && i>=NG+z.n[0])) {
          int indices[3]={i,j,k},A=0,sign=0;double localq[3];
          for(int a=0;a<3;++a) {
            localq[a]=b?(a==0?(indices[a]-NG)*z.dx[a]:-1+(indices[a]-NG+.5)*z.dx[a]):-p.L+(indices[a]-NG+.5)*z.dx[a];
            if(!sign && indices[a]<NG){A=a;sign=-1;}
            if(!sign && indices[a]>=NG+z.n[a]){A=a;sign=1;}
          }
          int ownstart[3];
          for(int a=0;a<3;++a) {
            double loc=(b&&a==0)?(indices[a]-NG):(indices[a]-NG);
            int near=std::round(loc),start=std::clamp(int(std::floor(loc))-2,0,z.n[a]-6);
            row.own_count[a]=6;
            if(near>=0 && near<z.n[a]) {start=near;row.own_count[a]=1;row.own_w[a][0]=1;}
            else for(int u0=0;u0<6;++u0) {row.own_w[a][u0]=1;for(int v0=0;v0<6;++v0)if(u0!=v0)row.own_w[a][u0]*=(loc-start-v0)/double(u0-v0);}
            ownstart[a]=start+NG;row.own_stride[a]=z.stride[a];
          }
          row.own_start=z.index(ownstart[0],ownstart[1],ownstart[2]);
          // Evaluate the actual boundary face, with other directions clipped to it.
          for(int a=0;a<3;++a)localq[a]=std::clamp(localq[a],b?(a==0?0.:-1.):-p.L,b?1.:p.L);
          localq[A]=b?(A==0?(sign<0?0:1):double(sign)):sign*p.L;
          double face[3],normal[3]{};
          if(b) {auto geom=map.Geometry(b-1,localq);double length=0;
            for(int a=0;a<3;++a)length+=geom.J[A][a]*geom.J[A][a];
            for(int a=0;a<3;++a)normal[a]=sign*geom.J[A][a]/std::sqrt(length);
            map.Forward(b-1,localq,face);
          }else {for(int a=0;a<3;++a)face[a]=localq[a];normal[A]=sign;}
          auto f=ts::coefficients(face,p);row.alpha=1/std::sqrt(-f.g[0][0]);double bn=0,nn=0;
          for(int a=0;a<3;++a) {row.beta[a]=-f.g[0][a+1]/f.g[0][0];bn+=row.beta[a]*normal[a];row.normal[a]=normal[a];
            for(int c0=0;c0<3;++c0)row.ginvn[a]+=(f.g[a+1][c0+1]-f.g[0][a+1]*f.g[0][c0+1]/f.g[0][0])*normal[c0];
            nn+=normal[a]*row.ginvn[a];}
          row.root=std::sqrt(nn);row.speeds[0]=-bn+row.alpha*row.root;row.speeds[1]=-bn-row.alpha*row.root;row.speeds[2]=-bn;row.characteristic=1;
        }
        plan.push_back(row);
      }
    }
    ghosts=Kokkos::View<Row*>("scalar active-donor ghosts",plan.size());auto ph=Kokkos::create_mirror_view(ghosts);
    for(size_t i=0;i<plan.size();++i)ph(i)=plan[i];Kokkos::deep_copy(ghosts,ph);
    auto uh=Kokkos::create_mirror_view(u);
    for(int l:ids) {
      std::array<double,5> v{};auto &c=host[l];
      if(p.mass==0)v=ts::exact(c.physical.x,0,p);
      else {
        ts::AD x[3];for(int i=0;i<3;++i)x[i]=ts::AD::var(c.physical.x[i],i);
        auto rho=ts::power(x[0]*x[0]+x[1]*x[1]+x[2]*x[2],.5L),o=ts::omega(rho,p);
        if(o.f>1e-10) {
          auto r=rho/o,q=(r-ts::AD(6))/ts::AD(1.2);
          auto f=x[0]*x[1]/(rho*rho)*ts::exp(-q*q)*ts::layer_smooth((r-ts::AD(1.2))/ts::AD(.8),p)/rho;
          v={double(f.f),0,double(f.d[0]),double(f.d[1]),double(f.d[2])};
        }
      }
      if(p.normal_upwind){double normal_pi=-v[1];for(int a=0;a<3;++a)normal_pi+=-c.physical.g[0][a+1]/c.physical.g[0][0]*v[2+a];v[1]=normal_pi*std::sqrt(-c.physical.g[0][0]);}
      for(int v0=0;v0<5;++v0)uh(l,v0)=v[v0];
    }
    Kokkos::deep_copy(u,uh);
    std::ofstream meta(output+"/grid.txt");meta<<std::setprecision(17)<<p.n<<" "<<p.mass<<" "<<p.spin<<" "<<p.cap<<" "<<p.cfl<<" "<<dt<<" "<<ids.size()<<"\n";
    std::ofstream runtime(output+"/backend.txt");Kokkos::print_configuration(runtime,true);
    std::ofstream config(output+"/parameters.json");
    config<<std::setprecision(17)<<"{\"n\":"<<p.n<<",\"M\":"<<p.mass<<",\"a_over_M\":"<<p.spin
      <<",\"L\":"<<p.L<<",\"S\":"<<p.S<<",\"C\":"<<p.C<<",\"cap\":"<<p.cap
      <<",\"halo_width\":"<<NG<<",\"normal_upwind\":"<<(p.normal_upwind?"true":"false")<<",\"stored_auxiliary\":\""<<(p.normal_upwind?"normal_Pi":"coordinate_v")<<"\""<<",\"compatible_reduction\":"<<(p.compatible_reduction?"true":"false")<<",\"outer_sbp\":"<<p.outer_sbp<<",\"smooth_physical_layer\":"<<(p.smooth_layer?"true":"false")<<",\"reference_layer\":\"original_quintic\",\"radial_stretch\":"<<p.radial_stretch<<",\"characteristic_transfer\":"<<(p.characteristic_transfer?"true":"false")<<",\"reduction_damping\":"<<p.reduction_damping<<",\"excision\":"<<p.excision<<",\"cfl\":"<<p.cfl<<",\"ko\":"<<p.ko<<",\"tlim\":"<<p.tlim
      <<",\"boundary\":\""<<p.boundary<<"\",\"data\":\""<<p.data<<"\"}\n";
    initial_physical=fingerprint(false);initial_reference=fingerprint(true);initial_physical_derivatives=fingerprint(false,true);initial_reference_derivatives=fingerprint(true,true);
    if(!p.restart_file.empty()) {
      if(p.restart_physical_hash.empty() || p.restart_reference_hash.empty())throw std::invalid_argument("restart requires both frozen geometry fingerprints");
      if(std::stoull(p.restart_physical_hash,nullptr,16)!=initial_physical || std::stoull(p.restart_reference_hash,nullptr,16)!=initial_reference)throw std::runtime_error("restart frozen geometry fingerprint mismatch");
      std::ifstream saved(p.restart_file,std::ios::binary|std::ios::ate);
      if(!saved || saved.tellg()!=static_cast<std::streamoff>(u.size()*sizeof(double)))throw std::runtime_error("restart payload size mismatch");
      auto restored=Kokkos::create_mirror_view(u);saved.seekg(0);saved.read(reinterpret_cast<char*>(restored.data()),restored.size()*sizeof(double));
      if(!saved)throw std::runtime_error("restart payload read failed");Kokkos::deep_copy(u,restored);
    }
    std::cout<<"cells="<<ids.size()<<" dt="<<dt<<" frozen trumpet M="<<p.mass<<" a/M="<<p.spin<<"\n"<<std::flush;
  }
  uint64_t fingerprint(bool reference,bool derivatives=false)const {
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),geometry);
    auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),active);
    uint64_t value=14695981039346656037ULL;
    auto add=[&](double x) {uint64_t bits;std::memcpy(&bits,&x,sizeof(bits));
      for(int j=0;j<8;++j){value^=(bits>>(8*j))&255;value*=1099511628211ULL;}};
    for(size_t i=0;i<ids.extent(0);++i) {
      const auto &c=reference?h(ids(i)).reference:h(ids(i)).physical;
      if(derivatives){for(auto &row:c.beta_gradient)for(double x:row)add(x);for(double x:c.alpha_gradient)add(x);continue;}
      for(auto &row:c.g)for(double x:row)add(x);for(double x:c.b)add(x);
      add(c.potential);add(c.density);add(c.omega);for(double x:c.x)add(x);
      for(auto &row:c.map.J)for(double x:row)add(x);
      for(auto &mat:c.map.H)for(auto &row:mat)for(double x:row)add(x);
    }
    return value;
  }
  Row donor(const double x[3])const {
    double q[3],zeta[3];int b=0;
    if(std::max({fabs(x[0]),fabs(x[1]),fabs(x[2])})>=p.L) {b=1+map.Owner(x);map.Inverse(b-1,x,q);}
    else for(int a=0;a<3;++a)q[a]=x[a];
    const auto z=blocks[b];Row r{};int start[3];
    for(int a=0;a<3;++a) {
      double lo=b?(a==0?0:-1):-p.L;
      zeta[a]=(q[a]-lo)/z.dx[a]-(b&&a==0?0:.5);
      start[a]=std::clamp(int(floor(zeta[a]))-2,0,z.n[a]-6);
      int nearest=std::round(zeta[a]);
      if(std::abs(zeta[a]-nearest)<1e-10 && nearest>=0 && nearest<z.n[a]) {
        start[a]=nearest;r.count[a]=1;r.w[a][0]=1;
      }
      for(int i=0;i<r.count[a] && r.count[a]>1;++i) {r.w[a][i]=1;for(int j=0;j<6;++j)if(i!=j)r.w[a][i]*=(zeta[a]-start[a]-j)/double(i-j);}
      r.stride[a]=z.stride[a];start[a]+=NG;
    }
    r.start=z.index(start[0],start[1],start[2]);return r;
  }
  void fill(double t) {
    stage_time=t;auto state=u;auto plan=ghosts;
    Kokkos::parallel_for("scalar interpatch transfer",plan.extent(0),KOKKOS_LAMBDA(int l) {
      auto r=plan(l);
      double donor[5]{},own[5]{},anchor[5];
      for(int v=0;v<5;++v)donor[v]=anchor[v]=state(r.start,v);
      for(int k=0;k<r.count[2];++k)for(int j=0;j<r.count[1];++j)for(int i=0;i<r.count[0];++i) {
        double weight=r.w[0][i]*r.w[1][j]*r.w[2][k];int src=r.start+i*r.stride[0]+j*r.stride[1]+k*r.stride[2];
        for(int v=0;v<5;++v)donor[v]+=weight*(state(src,v)-anchor[v]);
      }
      for(int v=0;v<5;++v)own[v]=donor[v];
      if(r.characteristic) {
        for(int v=0;v<5;++v)own[v]=anchor[v]=state(r.own_start,v);
        for(int k=0;k<r.own_count[2];++k)for(int j=0;j<r.own_count[1];++j)for(int i=0;i<r.own_count[0];++i) {
          double weight=r.own_w[0][i]*r.own_w[1][j]*r.own_w[2][k];int src=r.own_start+i*r.own_stride[0]+j*r.own_stride[1]+k*r.own_stride[2];
          for(int v=0;v<5;++v)own[v]+=weight*(state(src,v)-anchor[v]);
        }
      }
      if(!r.characteristic) {for(int v=0;v<5;++v)state(r.target,v)=donor[v];}
      else {
        double pin=0,pout=0,nd=0,no=0;
        for(int a=0;a<3;++a) {pin+=r.beta[a]*donor[2+a];pout+=r.beta[a]*own[2+a];nd+=r.ginvn[a]*donor[2+a]/r.root;no+=r.ginvn[a]*own[2+a]/r.root;}
        pin=(pin-donor[1])/r.alpha;pout=(pout-own[1])/r.alpha;
        double up=r.speeds[0]<0?pin+nd:pout+no,um=r.speeds[1]<0?pin-nd:pout-no;
        double pi=(up+um)/2,qn=(up-um)/2,v=-r.alpha*pi;
        for(int a=0;a<3;++a) {
          double tangent=r.speeds[2]<0?donor[2+a]-r.normal[a]*nd/r.root:own[2+a]-r.normal[a]*no/r.root;
          state(r.target,2+a)=tangent+r.normal[a]*qn/r.root;v+=r.beta[a]*state(r.target,2+a);
        }
        state(r.target,0)=r.speeds[2]<0?donor[0]:own[0];state(r.target,1)=v;
      }
    });
    // Exact incoming data is a Minkowski verification control only. At exact
    // scri no incoming data is set: the polynomial closure extends active data.
    if(p.boundary=="exact" && p.cap<1) {
      Kokkos::fence();auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
      for(int b=1;b<7;++b) {auto z=blocks[b];
        for(int k=0;k<z.n[2]+2*NG;++k)for(int j=0;j<z.n[1]+2*NG;++j)for(int i=NG+z.n[0];i<z.n[0]+2*NG;++i) {
          int l=z.index(i,j,k);double r=0;for(double x:host[l].physical.x)r+=x*x;
          if(r>=p.S*p.S)continue;
          auto v=ts::exact(host[l].physical.x,t,p);for(int a=0;a<5;++a)h(l,a)=v[a];
        }
      }
      Kokkos::deep_copy(u,h);
    }
  }
  void evaluate_normal() {
    auto state=u,rate=rhs;auto g=kernel;auto ids=active;double kappa=p.reduction_damping;
    Kokkos::parallel_for("normal scalar characteristic upwind",ids.extent(0),KOKKOS_LAMBDA(int z) {
      int l=ids(z);const auto &c=g(l);const auto &f=c.physical;double alpha=1/std::sqrt(-f.g[0][0]),beta[3],gamma[3][3];
      for(int i=0;i<3;++i){beta[i]=-f.g[0][i+1]/f.g[0][0];for(int j=0;j<3;++j)gamma[i][j]=f.g[i+1][j+1]-f.g[0][i+1]*f.g[0][j+1]/f.g[0][0];}
      for(int v=0;v<5;++v)rate(l,v)=0;rate(l,0)=-alpha*state(l,1);
      for(int A=0;A<3;++A) {
        double n[3],gn[3]{},bn=0,nn=0;for(int i=0;i<3;++i){n[i]=f.map.J[A][i];bn+=beta[i]*n[i];for(int j=0;j<3;++j)gn[i]+=gamma[i][j]*f.map.J[A][j];nn+=n[i]*gn[i];}
        double root=std::sqrt(nn),cp=-bn+alpha*root,cm=-bn-alpha*root;
        double dp[5],dm[5],dtan[5];for(int v=0;v<5;++v){
          if(A==0 && c.normal_sbp){double dc=deriv(state,l,v,A,c),q=radial_sbp_dissipation(state,l,v,c);dp[v]=dc+(cp>=0?1:-1)*q;dm[v]=dc+(cm>=0?1:-1)*q;dtan[v]=dc+(-bn>=0?1:-1)*q;}
          else{dp[v]=upwind(state,l,v,A,cp,c);dm[v]=upwind(state,l,v,A,cm,c);dtan[v]=upwind(state,l,v,A,-bn,c);}
        }
        double plus=dp[1],minus=dm[1],normal_tan=0;
        for(int i=0;i<3;++i){plus+=gn[i]/root*dp[2+i];minus-=gn[i]/root*dm[2+i];normal_tan+=gn[i]/root*dtan[2+i];}
        rate(l,0)+=bn*dtan[0];rate(l,1)-=.5*(cp*plus+cm*minus);
        for(int i=0;i<3;++i)rate(l,2+i)+=-n[i]/(2*root)*(cp*plus-cm*minus)+bn*(dtan[2+i]-n[i]/root*normal_tan);
      }
      double pi_coefficient=alpha*alpha*f.b[0];for(int i=0;i<3;++i)pi_coefficient+=beta[i]*f.alpha_gradient[i]/alpha;
      rate(l,1)+=pi_coefficient*state(l,1)+alpha*f.potential*state(l,0);
      for(int i=0;i<3;++i) {
        double qcoefficient=-alpha*(f.b[0]*beta[i]+f.b[1+i]);for(int j=0;j<3;++j)qcoefficient-=beta[j]*f.beta_gradient[j][i]/alpha;
        rate(l,1)+=qcoefficient*state(l,2+i);
        rate(l,2+i)-=f.alpha_gradient[i]*state(l,1)+kappa*(state(l,2+i)-gradient(state,l,0,i,c));
        for(int j=0;j<3;++j)rate(l,2+i)+=f.beta_gradient[i][j]*state(l,2+j);
      }
      if(c.boundary_rate>0) {
        double rho=0;for(int i=0;i<3;++i)rho+=f.x[i]*f.x[i];rho=std::sqrt(rho);double normal[3],gn[3]{},nn=0,delta=state(l,1)-c.radial_speeds[1]*c.boundary_amplitude_gradient*state(l,0)/alpha;
        for(int i=0;i<3;++i){normal[i]=f.x[i]/rho;for(int j=0;j<3;++j)gn[i]+=gamma[i][j]*f.x[j]/rho;nn+=normal[i]*gn[i];}
        double root=std::sqrt(nn);for(int i=0;i<3;++i)delta-=gn[i]/root*state(l,2+i);
        rate(l,1)-=.5*c.boundary_rate*delta;for(int i=0;i<3;++i)rate(l,2+i)+=.5*c.boundary_rate*normal[i]/root*delta;
      }
    });
  }
  void evaluate() {
    if(p.normal_upwind){evaluate_normal();return;}

    auto state=u,rate=rhs;auto g=kernel;auto ids=active;double kappa=p.reduction_damping;bool compatible=p.compatible_reduction;
    Kokkos::parallel_for("frozen physical scalar wave",ids.extent(0),KOKKOS_LAMBDA(int z) {
      int l=ids(z);const auto &c=g(l);const auto &f=c.physical;
      double gradv[3]{},gradq[3][3]{},gradpsi[3]{},logical[5][3];
      for(int v=0;v<4;++v)for(int A=0;A<3;++A)logical[v][A]=deriv(state,l,v+1,A,c);
      if(kappa>0)for(int A=0;A<3;++A)logical[4][A]=deriv(state,l,0,A,c);
      for(int a=0;a<3;++a)for(int A=0;A<3;++A) {
        gradv[a]+=f.map.J[A][a]*logical[0][A];
        if(kappa>0)gradpsi[a]+=f.map.J[A][a]*logical[4][A];
        for(int b=0;b<3;++b)gradq[a][b]+=f.map.J[A][a]*logical[1+b][A];
      }
      rate(l,0)=state(l,1);double vdot=f.b[0]*state(l,1)-f.potential*state(l,0);
      for(int a=0;a<3;++a) {
        rate(l,2+a)=gradv[a];
        for(int b=0;b<3;++b)rate(l,2+a)+=-f.g[0][1+b]/f.g[0][0]*(gradq[b][a]-gradq[a][b]);
        vdot+=2*f.g[0][1+a]*gradv[a]+f.b[1+a]*state(l,2+a);
        for(int b=0;b<3;++b)vdot+=f.g[1+a][1+b]*gradq[a][b];
      }
      rate(l,1)=-vdot/f.g[0][0];
      if(kappa>0)for(int a=0;a<3;++a) {
        double constraint=state(l,2+a)-gradpsi[a],beta=-f.g[0][a+1]/f.g[0][0];
        rate(l,0)-=beta*constraint;rate(l,2+a)-=kappa*constraint;rate(l,1)-=kappa*beta*constraint;
      }
      if(c.boundary_rate>0) {
        double rho=std::sqrt(f.x[0]*f.x[0]+f.x[1]*f.x[1]+f.x[2]*f.x[2]),qn=0;
        for(int a=0;a<3;++a)qn+=f.x[a]/rho*state(l,2+a);
        double win=state(l,1)+c.radial_speeds[1]*c.boundary_amplitude_gradient*state(l,0),alpha=1/std::sqrt(-f.g[0][0]),nn=0,gn[3]{};
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)gn[a]+=(f.g[a+1][b+1]-f.g[0][a+1]*f.g[0][b+1]/f.g[0][0])*f.x[b]/rho;
        for(int a=0;a<3;++a)nn+=gn[a]*f.x[a]/rho;
        for(int a=0;a<3;++a)win+=(f.g[0][a+1]/f.g[0][0]+alpha*gn[a]/std::sqrt(nn))*state(l,2+a);
        double correction=c.boundary_rate*win/(c.radial_speeds[1]-c.radial_speeds[0]);
        if(compatible)rate(l,1)-=c.boundary_rate*win;else rate(l,1)+=c.radial_speeds[0]*correction;
        for(int a=0;a<3;++a)rate(l,2+a)-=f.x[a]/rho*correction;
      }
      for(int v=0;v<5;++v)for(int A=0;A<3;++A) {
        int s=c.stride[A];
        if(!c.trim_ko[A])rate(l,v)+=c.ko[A]/(A==0?c.outer_norm:1)*(state(l-3*s,v)-6*state(l-2*s,v)+15*state(l-s,v)-20*state(l,v)+15*state(l+s,v)-6*state(l+2*s,v)+state(l+3*s,v));
        else {
          const double d3[4]={-1,3,-3,1};
          for(int m=0;m<4;++m) {
            int start=l-m*s;bool valid=true;double edge=0,delta=0;
            for(int j=0;j<4;++j){valid &= g(start+j*s).physical.active;edge+=g(start+j*s).ko[A]/4;delta+=d3[j]*state(start+j*s,v);}
            if(valid)rate(l,v)-=edge*d3[m]*delta/(A==0?c.outer_norm:1);
          }
        }
      }
    });
    if(compatible) {
      auto work=reduction_work;auto plan=ghosts;
      Kokkos::parallel_for("derive scalar auxiliary differential reduction",ids.extent(0),KOKKOS_LAMBDA(int z) {
        int l=ids(z);const auto &c=g(l);work(l,0)=rate(l,0);work(l,4)=0;
        for(int a=0;a<3;++a)work(l,1+a)=state(l,2+a)-gradient(state,l,0,a,c);
      });
      Kokkos::parallel_for("transfer derived scalar RHS and reduction residual",plan.extent(0),KOKKOS_LAMBDA(int l) {
        auto r=plan(l);double anchor[4],value[4];for(int v=0;v<4;++v)value[v]=anchor[v]=work(r.start,v);
        for(int k=0;k<r.count[2];++k)for(int j=0;j<r.count[1];++j)for(int i=0;i<r.count[0];++i) {
          double weight=r.w[0][i]*r.w[1][j]*r.w[2][k];int src=r.start+i*r.stride[0]+j*r.stride[1]+k*r.stride[2];
          for(int v=0;v<4;++v)value[v]+=weight*(work(src,v)-anchor[v]);
        }
        for(int v=0;v<4;++v)work(r.target,v)=value[v];
      });
      // Legacy analytic finite-radius verification has time-dependent scalar halos.
      if(p.boundary=="exact" && p.cap<1) {
        auto wh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),work);
        for(int b=1;b<7;++b){auto z=blocks[b];for(int k=0;k<z.n[2]+2*NG;++k)for(int j=0;j<z.n[1]+2*NG;++j)for(int i=NG+z.n[0];i<z.n[0]+2*NG;++i){int l=z.index(i,j,k);double r=0;for(double x:host[l].physical.x)r+=x*x;if(r<p.S*p.S)wh(l,0)=ts::exact(host[l].physical.x,stage_time,p)[1];}}
        Kokkos::deep_copy(work,wh);
      }
      Kokkos::parallel_for("compatible scalar auxiliary update",ids.extent(0),KOKKOS_LAMBDA(int z) {
        int l=ids(z);const auto &c=g(l);const auto &f=c.physical;
        for(int a=0;a<3;++a) {
          double qdot=gradient(work,l,0,a,c)-kappa*work(l,1+a);
          for(int j=0;j<3;++j)qdot+=-f.g[0][1+j]/f.g[0][0]*gradient(work,l,1+a,j,c)+f.beta_gradient[a][j]*work(l,1+j);
          for(int A=0;A<3;++A) {
            int s=c.stride[A];
            if(!c.trim_ko[A])qdot+=c.ko[A]/(A==0?c.outer_norm:1)*(work(l-3*s,1+a)-6*work(l-2*s,1+a)+15*work(l-s,1+a)-20*work(l,1+a)+15*work(l+s,1+a)-6*work(l+2*s,1+a)+work(l+3*s,1+a));
            else {const double d3[4]={-1,3,-3,1};for(int m=0;m<4;++m){int start=l-m*s;bool valid=true;double edge=0,delta=0;for(int j=0;j<4;++j){valid &= g(start+j*s).physical.active;edge+=g(start+j*s).ko[A]/4;delta+=d3[j]*work(start+j*s,1+a);}if(valid)qdot-=edge*d3[m]*delta/(A==0?c.outer_norm:1);}}
          }
          rate(l,2+a)=qdot;
        }
      });
    }
  }
  void step(double t,double h) {
    Kokkos::deep_copy(base,u);auto state=u,reg=base,tmp=stage,rate=rhs;auto ids=active;
    for(int s=0;s<4;++s) {
      fill(t+(s==0?0:(s==3?h:h/2)));evaluate();double weight=s==0||s==3?1./6:1./3,advance=s==2?1:.5;
      Kokkos::parallel_for("scalar RK4",ids.extent(0),KOKKOS_LAMBDA(int z) {
        int l=ids(z);for(int v=0;v<5;++v) {
          if(s==0)tmp(l,v)=reg(l,v);
          tmp(l,v)+=h*weight*rate(l,v);
          state(l,v)=s==3?tmp(l,v):reg(l,v)+advance*h*rate(l,v);
        }
      });
    }
  }
  void measure(double t,std::ostream &history,bool snapshot) {
    fill(t);Kokkos::fence();auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    std::ofstream out;if(snapshot) {std::ostringstream name;name<<output<<"/fields-t"<<std::fixed<<std::setprecision(3)<<t<<".f64";out.open(name.str(),std::ios::binary);}
    double sums[4][6]{},maxima[4][6]{},volume[4]{};
    auto ah=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),active);
    for(size_t z=0;z<ah.extent(0);++z) {
      int l=ah(z);auto c=host[l];int region=c.block?2:1;
      double grad[3];for(int i=0;i<3;++i)grad[i]=gradient(h,l,0,i,c);
      std::array<double,5> ex{};if(p.mass==0)ex=ts::exact(c.physical.x,t,p);
      double coordinate_v=h(l,1);if(p.normal_upwind){coordinate_v=-h(l,1)/std::sqrt(-c.physical.g[0][0]);for(int i=0;i<3;++i)coordinate_v+=-c.physical.g[0][i+1]/c.physical.g[0][0]*grad[i];}
      double err[6]={h(l,0)-ex[0],coordinate_v-ex[1],0,0,h(l,0),coordinate_v};
      for(int i=0;i<3;++i) {err[2]+=pow(grad[i]-ex[2+i],2);err[3]+=pow(h(l,2+i)-grad[i],2);}
      err[2]=sqrt(err[2]);err[3]=sqrt(err[3]);
      for(int r:{0,region}) {volume[r]+=c.weight;for(int i=0;i<6;++i) {sums[r][i]+=c.weight*err[i]*err[i];maxima[r][i]=std::max(maxima[r][i],fabs(err[i]));}}
      double compact_radius=0;for(double x:c.physical.x)compact_radius+=x*x;
      if(p.mass>0 && compact_radius>=1.2*1.2){volume[3]+=c.weight;for(int i=0;i<6;++i){sums[3][i]+=c.weight*err[i]*err[i];maxima[3][i]=std::max(maxima[3][i],fabs(err[i]));}}
      if(snapshot) {double row[12]={c.physical.x[0],c.physical.x[1],c.physical.x[2],c.weight,h(l,0),coordinate_v,h(l,2),h(l,3),h(l,4),grad[0],grad[1],grad[2]};out.write(reinterpret_cast<char*>(row),sizeof(row));}
    }
    for(int r=0;r<(p.mass>0?4:3);++r) {const char *names[4]={"all","bulk","shell","exterior"};history<<t<<","<<names[r];for(int i=0;i<6;++i)history<<","<<sqrt(sums[r][i]/volume[r])<<","<<maxima[r][i];history<<"\n";}
    history.flush();if(!std::isfinite(sums[0][0]) || maxima[0][4]>10) {
      std::ofstream failure(output+"/failure-checkpoint.f64",std::ios::binary);failure.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(double));
      throw std::runtime_error("scalar nonfinite or amplitude guard");
    }
    if(snapshot)std::cout<<"t="<<t<<(p.mass==0?" psi_error=":" psi_norm=")<<sqrt(sums[0][0]/volume[0])<<(p.mass==0?" Dpsi_error=":" Dpsi_norm=")<<sqrt(sums[0][2]/volume[0])<<" reduction="<<sqrt(sums[0][3]/volume[0])<<std::endl;
    // Fixed rho=10, hence fixed physical r=10/Omega(10) within each map family.
    bool wave_header=!std::filesystem::exists(output+"/waveform.csv");std::ofstream wave(output+"/waveform.csv",std::ios::app);
    if(wave_header)wave<<"time,real22,imag22\n";
    double real=0,imag=0;const double pi=acos(-1.),rho=10;
    // Midpoint angular quadrature; independent radial benchmark uses the same projection.
    for(int j=0;j<24;++j)for(int k=0;k<48;++k) {
      double ct=-1+(j+.5)*2/24,phi=(k+.5)*2*pi/48,st=sqrt(1-ct*ct);
      double x[3]={rho*st*cos(phi),rho*st*sin(phi),rho*ct};auto row=donor(x);double value=0;
      for(int a=0;a<row.count[0];++a)for(int b=0;b<row.count[1];++b)for(int d=0;d<row.count[2];++d)value+=row.w[0][a]*row.w[1][b]*row.w[2][d]*h(row.start+a*row.stride[0]+b*row.stride[1]+d*row.stride[2],0);
      double norm=sqrt(15./(32*pi))*(1-ct*ct)*2/24*2*pi/48;
      real+=norm*cos(2*phi)*value;imag-=norm*sin(2*phi)*value;
    }
    wave<<std::setprecision(17)<<t<<","<<real<<","<<imag<<"\n";
  }
  void run() {
    std::ofstream history(output+"/history.csv");history<<std::setprecision(17)<<"time,region,psi_L2,psi_Linf,v_L2,v_Linf,Dpsi_L2,Dpsi_Linf,reduction_L2,reduction_Linf,psi_norm,psi_max,v_norm,v_max\n";
    measure(p.start_time,history,true);double t=p.start_time,next=p.start_time+.5;int steps=0;bool checkpointed=false;auto began=std::chrono::steady_clock::now();
    while(t<p.tlim-1e-12) {
      double target=std::min(next,p.tlim),h=std::min(dt,target-t);step(t,h);t+=h;++steps;
      if(t>=target-1e-12) {bool save=fabs(t-2)<1e-9||fabs(t-6)<1e-9||fabs(t-12)<1e-9||fabs(t-p.tlim)<1e-9;measure(t,history,save);next+=.5;
        if(p.wall_seconds>0 && t<p.tlim-1e-12 && std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()>=p.wall_seconds){checkpointed=true;break;}
      }
    }
    Kokkos::fence();std::ofstream ck(output+"/checkpoint.f64",std::ios::binary);auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    ck.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(double));
    uint64_t final_physical=fingerprint(false),final_reference=fingerprint(true),final_physical_derivatives=fingerprint(false,true),final_reference_derivatives=fingerprint(true,true);
    std::ofstream integrity(output+"/frozen-integrity.json");
    bool unchanged=initial_physical==final_physical && initial_reference==final_reference && initial_physical_derivatives==final_physical_derivatives && initial_reference_derivatives==final_reference_derivatives;
    integrity<<"{\"algorithm\":\"FNV1a64 over numeric frozen cache, external SHA256 manifest covers files\",\"physical_before\":\""<<std::hex<<initial_physical
      <<"\",\"physical_after\":\""<<final_physical<<"\",\"reference_before\":\""<<initial_reference
      <<"\",\"reference_after\":\""<<final_reference<<"\",\"physical_derivatives_before\":\""<<initial_physical_derivatives<<"\",\"physical_derivatives_after\":\""<<final_physical_derivatives<<"\",\"reference_derivatives_before\":\""<<initial_reference_derivatives<<"\",\"reference_derivatives_after\":\""<<final_reference_derivatives<<"\",\"unchanged\":"<<(unchanged?"true":"false")<<"}\n";
    if(!unchanged)throw std::runtime_error("frozen geometry changed");
    std::ofstream receipt(output+(checkpointed?"/checkpointed.txt":"/complete.txt"));receipt<<std::setprecision(17)<<"time="<<t<<" steps="<<steps<<" dt="<<dt<<"\n";
  }
};
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);int status=0;
  try {
    ts::Parameters p;std::string out="scalar-output";bool audit=false;
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];if(key=="--audit"){audit=true;continue;}
      if(i+1>=argc)throw std::invalid_argument("missing argument value");std::string v=argv[++i];
      if(key=="--n")p.n=std::stoi(v);else if(key=="--mass")p.mass=std::stod(v);else if(key=="--spin")p.spin=std::stod(v);
      else if(key=="--cap")p.cap=std::stod(v);else if(key=="--cfl")p.cfl=std::stod(v);else if(key=="--tlim")p.tlim=std::stod(v);
      else if(key=="--excision")p.excision=std::stod(v);else if(key=="--boundary")p.boundary=v;else if(key=="--output")out=v;
      else if(key=="--normal-upwind")p.normal_upwind=std::stoi(v)!=0;else if(key=="--compatible-reduction")p.compatible_reduction=std::stoi(v)!=0;else if(key=="--restart")p.restart_file=v;else if(key=="--start-time")p.start_time=std::stod(v);else if(key=="--wall-seconds")p.wall_seconds=std::stod(v);else if(key=="--restart-physical-hash")p.restart_physical_hash=v;else if(key=="--restart-reference-hash")p.restart_reference_hash=v;else if(key=="--outer-sbp")p.outer_sbp=std::stoi(v);else if(key=="--smooth-layer")p.smooth_layer=std::stoi(v)!=0;else if(key=="--radial-stretch")p.radial_stretch=std::stod(v);else if(key=="--characteristic-transfer")p.characteristic_transfer=std::stoi(v)!=0;else if(key=="--reduction-damping")p.reduction_damping=std::stod(v);else if(key=="--data")p.data=v;else if(key=="--ko")p.ko=std::stod(v);else throw std::invalid_argument("unknown scalar option "+key);
    }
    if(audit) {
      std::filesystem::create_directories(out);std::ofstream f(out+"/geometry.csv");f<<std::setprecision(17)<<"mass,spin,x,y,z,omega,density,potential";
      for(int mu=0;mu<4;++mu)for(int nu=0;nu<4;++nu)f<<",g"<<mu<<nu;for(int mu=0;mu<4;++mu)f<<",b"<<mu;for(int i=0;i<3;++i){f<<",alpha"<<i;for(int j=0;j<3;++j)f<<",beta"<<i<<j;}for(int i=0;i<4;++i)for(int j=0;j<4;++j)f<<",reference_g"<<i<<j<<",delta_g"<<i<<j;f<<",reference_potential,delta_potential,reference_density,delta_density\n";
      for(double mass:{0.,1.})for(double spin:{0.,.5,-.5,.9})for(double r:{.01,.1,.5,1.,2.,7.9,8.1,12.,15.9,16.1,20.,23.,23.9,23.999,24.})for(int dir=0;dir<3;++dir) {
        auto q=p;q.mass=mass;q.spin=spin;double n[3]={.6,.8,0};if(dir==1){n[0]=.3;n[1]=.4;n[2]=sqrt(.75);}if(dir==2){n[0]=0;n[1]=0;n[2]=1;}
        double x[3];for(int a=0;a<3;++a)x[a]=r*n[a];auto c=ts::coefficients(x,q);
        f<<mass<<","<<spin;for(double z:x)f<<","<<z;f<<","<<c.omega<<","<<c.density<<","<<c.potential;
        for(auto &row:c.g)for(double z:row)f<<","<<z;for(double z:c.b)f<<","<<z;for(int i=0;i<3;++i){f<<","<<c.alpha_gradient[i];for(int j=0;j<3;++j)f<<","<<c.beta_gradient[i][j];}auto reference_parameters=q;reference_parameters.mass=reference_parameters.spin=0;reference_parameters.smooth_layer=false;auto reference=ts::coefficients(x,reference_parameters);auto delta=ts::difference(c,reference);for(int i=0;i<4;++i)for(int j=0;j<4;++j)f<<","<<reference.g[i][j]<<","<<delta.g[i][j];f<<","<<reference.potential<<","<<delta.potential<<","<<reference.density<<","<<delta.density<<"\n";
      }
    } else {Solver solver(p,out);solver.run();}
  } catch(const std::exception &e) {std::cerr<<"scalar failure: "<<e.what()<<"\n";status=1;}
  Kokkos::finalize();return status;
}
