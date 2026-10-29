#include <iostream>
#include <iomanip>
#include "z4c/hyperboloidal/paired_trace_device.hpp"
#include "fixed_composite_mesh.hpp"
using Device=z4c::hyperboloidal::PairedTraceDevice;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int code=0;
 try{
  std::cout<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
  const int n=12; cubed_sphere::CompositeDifference g(hyperbolic_reproduction::ThinCoreMesh(n,1,true),4,true);
  cubed_sphere::PairedTracePenalty plan(g);const int total=g.active.size();
  Device::Buffer state("test state",31*total),rhs("test rate",31*total);
  auto host=Kokkos::create_mirror_view(state);std::vector<Real> expected(31*total,0),q(total);
  for(int f=0;f<31;++f){
   for(int p=0;p<total;++p){const auto*x=g.geometry[p].x;const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2],A=(1+r2)/2;
    const Real chart=.001*std::sin((f+1)*x[0]+3*x[1]-2*x[2])+.0001*(g.mesh.blocks[g.cells[p].block].face<0?1:-1);
    host(31*p+f)=f==0?std::exp(chart):f==18?A*(std::exp(chart)-1):chart;
    q[p]=f==0?std::log(host(31*p+f)):f==18?std::log((host(31*p+f)+A)/A):host(31*p+f);
   }
   const auto rate=plan.Apply(q,1,f<=6||f==18||(f>=22&&f<=24)?.25:0);
   for(int p=0;p<total;++p){const auto*x=g.geometry[p].x;const Real A=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
    expected[31*p+f]=rate[p]*(f==0?host(31*p):f==18?host(31*p+18)+A:1);}
  }
  Kokkos::deep_copy(state,host);Kokkos::deep_copy(rhs,Real(0));
  Real error=0,scale=0;std::array<Real,6> partition_norm{};
  for(int part=0;part<2;++part){const int first=total*part/2,last=total*(part+1)/2;
   Device device(g,plan,first,last-first);
   Device::Buffer local=Kokkos::subview(rhs,std::make_pair(size_t(31*first),size_t(31*last)));
   device.Add(state,local,1,.25);const auto norm=device.Measure(state);
   for(int i=0;i<6;++i)partition_norm[i]+=norm[i];
  }
  auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  for(int i=0;i<31*total;++i){error=std::max(error,std::abs(result(i)-expected[i]));scale=std::max(scale,std::abs(expected[i]));}
  if(error>2e-12*std::max(Real(1),scale))throw std::runtime_error("CPU/device paired lift differs");
  Device all(g,plan,0,total);const auto norm=all.Measure(state);
  Real norm_error=0;for(int i=0;i<6;++i)norm_error=std::max(norm_error,std::abs(norm[i]-partition_norm[i]));
  if(norm_error>2e-12)throw std::runtime_error("diagnostic partition double counts trace");
  Kokkos::deep_copy(rhs,Real(0));all.Add(state,rhs,0,.25);
  result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  for(int i=0;i<31*total;++i)if(result(i)!=0)throw std::runtime_error("zero penalty not exact");
  for(int p=0;p<total;++p)for(int f=0;f<31;++f)host(31*p+f)=f==0?1:0;
  Kokkos::deep_copy(state,host);all.Add(state,rhs,1,.25);result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
  for(int i=0;i<31*total;++i)if(result(i)!=0)throw std::runtime_error("Minkowski penalty not zero");
  std::cout<<std::setprecision(17)<<"PASS: CPU/device physical RHS, partition norms, zero coefficient and exact Minkowski; max_error="<<error<<" scaled_error="<<error/std::max(Real(1),scale)<<" norm_error="<<norm_error<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();return code;
}
