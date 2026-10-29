#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
namespace h=z4c::hyperboloidal;
using Exchange=h::HyperbolicCompositeExchange;

static void Require(bool condition,const char*message) {
  if(!condition)throw std::runtime_error(message);
}
static Real Difference(Real expected,Real value) {
  Require(std::isfinite(value),"nonfinite valid exchange sample");
  return std::abs(value-expected)/(1+std::abs(expected));
}
static Exchange::Buffer Upload(const std::vector<Real>&values) {
  Exchange::Buffer out("test active stage",values.size());auto host=Kokkos::create_mirror_view(out);
  for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(out,host);return out;
}
static void Seed(h::HyperbolicComposite&cpu,int stage) {
  auto sample=[&](const Real*x,int f) {
    const Real wave=std::sin((2+.13*f)*x[0]+(1.3+.11*f)*x[1]-(1.9+.07*f)*x[2]+.2*stage);
    return (f==0 ? 1. : 0.)+.006*(1+f%4)*wave;
  };
  for(size_t p=0;p<cpu.grid.active.size();++p)for(int f=0;f<cpu.nf;++f)
    cpu.state[cpu.nf*p+f]=sample(cpu.grid.geometry[p].x,f);
  for(size_t p=0;p<cpu.scri.rows.size();++p)for(int f=0;f<cpu.nf;++f)
    cpu.scri.values[cpu.nf*p+f]=sample(cpu.scri.rows[p].geometry.x,f);
  cpu.BoundaryValues();
}
static Real CheckSparseScri(h::HyperbolicComposite&cpu,Exchange&exchange,
                           Exchange::ConstBuffer state,const std::vector<Real>&padded) {
  constexpr Real sentinel=-9123.25;
  std::vector<Real> sparse(cpu.nf*cpu.grid.count,sentinel);
  exchange.DownloadScriVolume(state,sparse);
  std::set<int> donors;
  for(const auto&row:cpu.scri.rows)for(const auto&term:row.terms)
    if(!term.boundary)donors.insert(term.point);
  for(int p=0;p<cpu.grid.count;++p)for(int f=0;f<cpu.nf;++f) {
    const auto i=f*cpu.grid.count+p;
    if(donors.count(p))Require(sparse[i]==padded[i],"scri active donor copy differs");
    else Require(sparse[i]==sentinel,"scri download overwrote an unrelated padded value");
  }
  Real error=0;
  for(size_t p=0;p<cpu.scri.rows.size();++p) {
    const auto expected=cpu.scri.Sample(p,padded,cpu.scri.values,cpu.nf);
    const auto actual=cpu.scri.Sample(p,sparse,cpu.scri.values,cpu.nf);
    for(int f=0;f<cpu.nf;++f) {
      error=std::max(error,Difference(expected.value[f],actual.value[f]));
      for(int a=0;a<3;++a) {
        error=std::max(error,Difference(expected.d[f][a],actual.d[f][a]));
        for(int b=0;b<3;++b)error=std::max(error,Difference(expected.dd[f][a][b],actual.dd[f][a][b]));
      }
    }
  }
  return error;
}
static void CheckRange(h::HyperbolicComposite&cpu,int first,int end,int stages) {
  Exchange exchange(cpu,first,end);
  const int size=cpu.grid.count,blocksize=cpu.grid.padded*cpu.grid.padded*cpu.grid.padded;
  std::vector<char> active(size,0);for(int p:cpu.grid.active)active[p]=1;
  for(int stage=0;stage<stages;++stage) {
    Seed(cpu,stage);
    const auto padded=cpu.Padded(cpu.state);const auto state=Upload(cpu.state);
    exchange.SetSurface(cpu.scri.values);
    constexpr Real sentinel=-1729.125;
    Kokkos::deep_copy(exchange.padded,sentinel);
    const auto result=exchange.Fill(state);
    const auto actual=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),exchange.padded);
    Require(result.invalid==0,"valid scri rejected by exchange");
    Real error=0,before=0,after=0;int selected_halos=0;
    for(int p=0;p<size;++p)for(int f=0;f<cpu.nf;++f) {
      const bool owned=p/blocksize>=first&&p/blocksize<end;
      if(active[p]||owned)error=std::max(error,Difference(padded[f*size+p],actual(f*size+p)));
      else Require(actual(f*size+p)==sentinel,"unowned halo was evaluated");
    }
    for(const auto&row:cpu.grid.halos) {
      const int block=row.target/blocksize;
      if(block<first||block>=end)continue;
      ++selected_halos;
      if(row.boundary_weight==0)continue;
      auto value=cpu.scri.Interpolate(cpu.scri.values,cpu.nf,row.boundary.data());
      before=std::max(before,cpu.NullResidual(row.boundary.data(),value.data()));
      cpu.EnforceBoundaryValues(row.boundary.data(),value.data());
      after=std::max(after,cpu.NullResidual(row.boundary.data(),value.data()));
    }
    Require(error<3e-12,"GPU halo differs from CPU mapped interpolation");
    Require(Difference(before,result.before)<3e-12&&Difference(after,result.after)<3e-12,
            "off-node null diagnostics differ from CPU");
    Require(result.after<3e-14,"off-node null closure is not at roundoff");
    if(first==0&&end==int(cpu.grid.mesh.blocks.size()))
      Require(result.before>1e-8,"nonspherical surface does not exercise interpolation closure");
    const Real scri_error=CheckSparseScri(cpu,exchange,state,padded);
    Require(scri_error<3e-12,"sparse scri volume does not reproduce CPU surface jets");
    std::cout<<"blocks=["<<first<<','<<end<<") stage="<<stage<<" halos="<<selected_halos
             <<" scaled_error="<<error<<" scri_jet_error="<<scri_error
             <<" null_before="<<result.before<<" null_after="<<result.after<<'\n';
  }
  if(first==0&&end==int(cpu.grid.mesh.blocks.size())) {
    // A failed surface must overwrite the old interpolants, not reuse the last
    // stage's valid values. The caller must reject result.invalid before RHS.
    for(size_t p=0;p<cpu.scri.rows.size();++p)cpu.scri.values[cpu.nf*p]=-1;
    exchange.SetSurface(cpu.scri.values);const auto result=exchange.Fill(Upload(cpu.state));
    Require(result.invalid>0,"negative scri chi was accepted");
    const auto actual=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),exchange.padded);
    for(const auto&row:cpu.grid.halos)if(row.boundary_weight!=0)
      Require(!std::isfinite(actual(row.target)),"invalid scri reused a stale interpolant");
  }
}
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);int result=0;
  try {
    std::cout<<std::setprecision(17);
    std::cout<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<std::endl;
    h::HyperbolicComposite cpu(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3));
    const int blocks=cpu.grid.mesh.blocks.size();
    CheckRange(cpu,0,blocks,2);
    CheckRange(cpu,0,1,1);                 // Core only: no scri plans.
    CheckRange(cpu,1,blocks-6,1);          // Transition and inner shells.
    CheckRange(cpu,blocks-7,blocks-3,1);   // Mixed interior/outer blocks.
    CheckRange(cpu,blocks-3,blocks,1);
    CheckRange(cpu,blocks,blocks,1);      // Empty ownership still supplies scri donors.
    bool invalid_range=false;
    try{Exchange invalid(cpu,-1,blocks);}catch(const std::invalid_argument&){invalid_range=true;}
    Require(invalid_range,"invalid halo block range accepted");
    Exchange empty(cpu,blocks,blocks);Exchange::Buffer short_state("short state",1);
    bool invalid_shape=false;
    try{empty.Fill(short_state);}catch(const std::invalid_argument&){invalid_shape=true;}
    Require(invalid_shape,"invalid active-state shape accepted");
    std::cout<<"CPU/device halo, off-node null closure and sparse scri donor parity passed\n";
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
