#include <Kokkos_Core.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
using Real = double;
#include "pc_gh/lapse_gradient.hpp"

// Test the production helper and production curl on device. Scalar ghosts are
// copied periodically; target ghosts are computed from the same scalar data.
template<int S>
bool check(int dim, int n) {
  constexpr int r=S-1, g=2*r;
  int const extent=n+2*g;
  using Scalar=Kokkos::View<Real****>;
  using Vector=Kokkos::View<Real*****>;
  Scalar rho("rho",1,extent,extent,extent), w("w",1,extent,extent,extent);
  Vector target("target",1,3,extent,extent,extent), old("old",1,3,extent,extent,extent);
  auto rh=Kokkos::create_mirror_view(rho), wh=Kokkos::create_mirror_view(w);
  Real max_product=0;
  for(int k=0;k<extent;++k) for(int j=0;j<extent;++j) for(int i=0;i<extent;++i) {
    double x=2*M_PI*((i-g+n)%n)/n, y=2*M_PI*((j-g+n)%n)/n;
    double z=dim==3 ? 2*M_PI*((k-g+n)%n)/n : 0;
    rh(0,k,j,i)=1+.3*std::sin(x+2*y-z)+.1*std::cos(3*x-y+z);
    wh(0,k,j,i)=1+.2*std::cos(2*x-y+2*z)+.15*std::sin(x+y+z);
    max_product=std::fmax(max_product,std::fabs(rh(0,k,j,i)*wh(0,k,j,i)));
  }
  Kokkos::deep_copy(rho,rh); Kokkos::deep_copy(w,wh);
  Real const idx[3]={n/4.,n/5.,n/6.};
  using Policy=Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
  Kokkos::parallel_for("target including ghosts",Policy({r,r,r},{extent-r,extent-r,extent-r}),
    KOKKOS_LAMBDA(int k,int j,int i) {
      for(int d=0;d<dim;++d) {
        target(0,d,k,j,i)=pc_gh::DirectLapseGradient<S>(d,idx,rho,w,0,k,j,i);
        old(0,d,k,j,i)=2*(w(0,k,j,i)*Dx<S>(d,idx,rho,0,k,j,i)
                           +rho(0,k,j,i)*Dx<S>(d,idx,w,0,k,j,i));
      }
    });
  Vector curls("curls",1,6,n,n,n);
  Kokkos::parallel_for("curl",Policy({g,g,g},{g+n,g+n,g+n}),
    KOKKOS_LAMBDA(int k,int j,int i) {
      int p=0;
      for(int a=0;a<dim;++a) for(int b=a+1;b<dim;++b,++p) {
        curls(0,p,k-g,j-g,i-g)=Dx<S>(a,idx,target,0,b,k,j,i)-Dx<S>(b,idx,target,0,a,k,j,i);
        curls(0,p+3,k-g,j-g,i-g)=Dx<S>(a,idx,old,0,b,k,j,i)-Dx<S>(b,idx,old,0,a,k,j,i);
      }
    });
  auto th=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),target);
  auto ch=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),curls);
  // Independent coefficient summation, different summation order from Dx.
  Real c[4]={0,.5,0,0};
  if constexpr(S==3) {c[1]=2./3; c[2]=-1./12;}
  if constexpr(S==4) {c[1]=3./4; c[2]=-3./20; c[3]=1./60;}
  double l1=0; for(int q=1;q<=r;++q) l1+=2*std::abs(c[q]);
  double eps=std::numeric_limits<Real>::epsilon(), scaled=0, target_scaled=0, old_scaled=0;
  double absolute=0, old_absolute=0;
  bool finite=true;
  for(int k=g;k<g+n;++k) for(int j=g;j<g+n;++j) for(int i=g;i<g+n;++i) {
    for(int d=0;d<dim;++d) {
      double ref=0;
      for(int q=-r;q<=r;++q) if(q) {
        int kk=k+(d==2)*q,jj=j+(d==1)*q,ii=i+(d==0)*q;
        ref+=(q>0 ? c[q] : -c[-q])*(rh(0,kk,jj,ii)*wh(0,kk,jj,ii));
      }
      finite=finite && std::isfinite(th(0,d,k,j,i));
      target_scaled=std::fmax(target_scaled,std::abs(th(0,d,k,j,i)-2*idx[d]*ref)/(2*eps*max_product*l1*idx[d]));
    }
    int p=0;
    for(int a=0;a<dim;++a) for(int b=a+1;b<dim;++b,++p) {
      double scale=2*eps*max_product*l1*l1*idx[a]*idx[b];
      double v=std::abs(ch(0,p,k-g,j-g,i-g)), o=std::abs(ch(0,p+3,k-g,j-g,i-g));
      finite=finite && std::isfinite(v) && std::isfinite(o);
      scaled=std::fmax(scaled,v/scale); old_scaled=std::fmax(old_scaled,o/scale);
      absolute=std::fmax(absolute,v); old_absolute=std::fmax(old_absolute,o);
    }
  }
  bool ok=finite&&std::isfinite(scaled)&&scaled<=64&&target_scaled<=64&&old_scaled>1000;
  std::cout<<2*(S-1)<<','<<dim<<','<<n<<','<<absolute<<','<<scaled<<','<<target_scaled<<','<<old_absolute<<','<<old_scaled<<','<<(ok?"PASS":"FAIL")<<'\n';
  return ok;
}
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);
  bool ok=true;
  {
    std::cout<<"# execution_space="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    std::cout<<"order,dim,n,curl_max,curl_scaled,target_scaled,old_curl_max,old_curl_scaled,decision\n"<<std::setprecision(17);
    for(int dim:{2,3}) for(int n:{16,32}) {
      ok=check<2>(dim,n)&&ok; ok=check<3>(dim,n)&&ok; ok=check<4>(dim,n)&&ok;
    }
  }
  Kokkos::finalize(); return ok?0:1;
}
