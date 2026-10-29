#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_device.hpp"
namespace h=z4c::hyperboloidal;

static void Require(bool condition,const char *message) {
  if (!condition) throw std::runtime_error(message);
}
static Real Compare(const std::vector<Real> &reference,const std::vector<Real>&value,int first) {
  Real maximum=0;
  for (size_t i=0;i<value.size();++i) {
    const Real expected=reference[h::HyperbolicComposite::nf*first+i];
    const Real scaled=std::abs(value[i]-expected)/(1+std::abs(expected));
    maximum=std::max(maximum,scaled);
  }
  return maximum;
}
using Device=h::HyperbolicCompositeDevice;
using Sums=Device::DiagnosticSums;
static Sums Add(const Sums&a,const Sums&b) {
  return {a.volume+b.volume,a.h2+b.h2,a.mr2+b.mr2,a.m2+b.m2,
          a.m2_physical+b.m2_physical,a.theta2+b.theta2,a.zr2+b.zr2,
          a.z2+b.z2,a.z2_physical+b.z2_physical};
}
static Real CompareSums(const Sums&a,const Sums&b) {
  const Real av[]={a.volume,a.h2,a.mr2,a.m2,a.m2_physical,a.theta2,a.zr2,a.z2,a.z2_physical};
  const Real bv[]={b.volume,b.h2,b.mr2,b.m2,b.m2_physical,b.theta2,b.zr2,b.z2,b.z2_physical};
  Real maximum=0;
  for(int f=0;f<9;++f)maximum=std::max(maximum,std::abs(av[f]-bv[f])/(1+std::abs(av[f])));
  return maximum;
}
static Device::DiagnosticSummary CPUConstraints(const h::HyperbolicComposite &cpu,
                                               const std::vector<Real>&padded,Real cutoff) {
  Device::DiagnosticSummary result;
  std::vector<Real>values(cpu.nf);
  std::vector<std::array<Real,3>>d(cpu.nf);
  std::vector<std::array<std::array<Real,3>,3>>dd(cpu.nf);
  for(size_t p=0;p<cpu.grid.active.size();++p) {
    const auto*x=cpu.grid.geometry[p].x;const Real r=cpu.Radius(x),w=cpu.grid.volumes[p];
    for(int f=0;f<cpu.nf;++f) {
      values[f]=cpu.state[cpu.nf*p+f];Real df[3],ddf[3][3];
      cpu.grid.Derivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
      for(int i=0;i<3;++i){d[f][i]=df[i];for(int j=0;j<3;++j)dd[f][i][j]=ddf[i][j];}
      result.max_state_deviation=std::max(result.max_state_deviation,
                                         std::abs(values[f]-(f==0 ? 1 : 0)));
    }
    const auto u=cpu.Jet(x,values,d,dd);const auto c=h::EvolvedConstraints(u,cpu.Omega(x,u));
    Require(c.valid,"CPU constraint sample invalid");
    Real mr=0,zr=0;
    if(r>0)for(int i=0;i<3;++i){mr+=x[i]*c.momentum[i]/r;zr+=x[i]*c.z4.z_covector[i]/r;}
    Sums sums{w,w*c.hamiltonian*c.hamiltonian,w*mr*mr,w*c.momentum_conformal_norm2,
              w*c.momentum_physical_norm2,w*u.theta.value*u.theta.value,w*zr*zr,
              w*c.z4.z_conformal_norm2,w*c.z4.z_physical_norm2};
    result.total=Add(result.total,sums);
    if(r>cutoff)result.exterior=Add(result.exterior,sums);
    result.min_chi=std::min(result.min_chi,u.chi.value);
    result.min_lapse=std::min(result.min_lapse,u.alpha.value);
    result.max_determinant_residual=std::max(result.max_determinant_residual,
                                            std::abs(c.z4.determinant_residual));
    result.max_tracefree_residual=std::max(result.max_tracefree_residual,
                                          std::abs(c.z4.tracefree_residual));
  }
  return result;
}
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);int result=0;
  try {
    std::cout<<std::setprecision(17);
    std::cout<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<std::endl;
    h::HyperbolicComposite cpu(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3),false);
    const int cells=cpu.grid.active.size(),split=cells/2+7;
    h::HyperbolicCompositeDevice full(cpu.grid,0,cells),left(cpu.grid,0,split),right(cpu.grid,split,cells-split);
    h::HyperbolicCompositeDevice empty(cpu.grid,cells,0);
    // Nonzero nonspherical data populate every scalar, symmetric tensor and
    // vector component, including the otherwise dormant adaptive-source slots.
    for(int p=0;p<cells;++p) {
      const auto*x=cpu.grid.geometry[p].x;const Real r=cpu.Radius(x);
      const Real taper=std::pow(1-r*r,4);
      for(int f=0;f<cpu.nf;++f) {
        const Real wave=std::sin((1+.19*f)*x[0]+(.2+.17*f)*x[1]-(.4+.13*f)*x[2]);
        cpu.state[p*cpu.nf+f]=(f==0?1.:0.)+.0001*(1+f%5)*wave*taper;
      }
    }
    Real worst=0;
    for(bool adaptive:{false,true}) for(Real ko:{0.,.5}) {
      cpu.adaptive=adaptive;cpu.ko=ko;std::vector<Real>edge;
      const auto expected=cpu.RHS(cpu.state,.0001,edge);
      const auto padded=cpu.Padded(cpu.state);const auto&device_padded=full.UploadPadded(padded);
      const auto stats=full.Evaluate(device_padded,.0001,adaptive,ko);
      Require(stats.bad_cells==0 && stats.first_bad==-1,"device valid state rejected");
      const Real error=Compare(expected,full.DownloadRHS(),0);worst=std::max(worst,error);
      const auto a=left.Evaluate(device_padded,.0001,adaptive,ko);
      const auto b=right.Evaluate(device_padded,.0001,adaptive,ko);
      const auto zero=empty.Evaluate(device_padded,.0001,adaptive,ko);
      Require(a.bad_cells==0 && b.bad_cells==0 && zero.bad_cells==0 && zero.first_bad==-1,
              "partition diagnostics differ");
      const Real partition_error=std::max(Compare(expected,left.DownloadRHS(),0),
                                         Compare(expected,right.DownloadRHS(),split));
      worst=std::max(worst,partition_error);
      Require(std::abs(std::min(a.min_gap,b.min_gap)-stats.min_gap)<1e-14,
              "partition characteristic reduction");
      Require(std::abs(std::max(a.max_cfl,b.max_cfl)-stats.max_cfl)<1e-14,
              "partition CFL reduction");
      std::cout<<"adaptive="<<adaptive<<" ko="<<ko<<" scaled_error="<<error
               <<" partition_error="<<partition_error<<" min_gap="<<stats.min_gap
               <<" max_cfl="<<stats.max_cfl<<'\n';
    }
    // Tight enough to expose missing terms while permitting different fused
    // math choices on device. This is a parity audit, not a trajectory claim.
    Require(worst<2e-11,"device bulk differs from CPU equations/stencils");
    const Real cutoff=.3;
    const auto padded=cpu.Padded(cpu.state);
    const auto&device_padded=full.UploadPadded(padded);
    const auto expected=CPUConstraints(cpu,padded,cutoff);
    const auto diagnostics=full.Diagnostics(device_padded,cutoff);
    const auto da=left.Diagnostics(device_padded,cutoff),db=right.Diagnostics(device_padded,cutoff);
    const auto dzero=empty.Diagnostics(device_padded,cutoff);
    const auto no_exterior=full.Diagnostics(device_padded,1);
    const Real diagnostic_error=std::max(CompareSums(expected.total,diagnostics.total),
                                         CompareSums(expected.exterior,diagnostics.exterior));
    const Real split_error=std::max(CompareSums(diagnostics.total,Add(da.total,db.total)),
                                    CompareSums(diagnostics.exterior,Add(da.exterior,db.exterior)));
    Require(diagnostic_error<2e-11 && split_error<2e-11,"device constraint sums differ from CPU");
    Require(diagnostics.invalid_cells==0 && diagnostics.first_bad==-1 &&
            da.invalid_cells==0 && db.invalid_cells==0 && dzero.invalid_cells==0 &&
            dzero.first_bad==-1,"valid diagnostic sample rejected");
    Require(dzero.total.volume==0 && no_exterior.exterior.volume==0 &&
            CompareSums(no_exterior.exterior,Sums{})==0,"empty diagnostic region not empty");
    Require(std::abs(expected.min_chi-diagnostics.min_chi)<1e-14 &&
            std::abs(expected.min_lapse-diagnostics.min_lapse)<1e-14 &&
            std::abs(expected.max_state_deviation-diagnostics.max_state_deviation)<1e-14 &&
            std::abs(expected.max_determinant_residual-diagnostics.max_determinant_residual)<2e-11 &&
            std::abs(expected.max_tracefree_residual-diagnostics.max_tracefree_residual)<2e-11,
            "device constraint extrema differ from CPU");
    std::cout<<"diagnostic_scaled_error="<<diagnostic_error<<" partition_sum_error="<<split_error
             <<" coordinate_volume="<<diagnostics.total.volume
             <<" exterior_volume="<<diagnostics.exterior.volume<<'\n';
    auto invalid=cpu.Padded(cpu.state);const int bad=split+3;
    invalid[cpu.grid.active[bad]]=-1;
    auto stats=right.Evaluate(full.UploadPadded(invalid),.0001,true,.5);
    Require(stats.first_bad==bad && stats.reason==h::HyperbolicCompositeDevice::invalid_input
            && stats.bad_cells==1,"invalid cell global index/status not preserved");
    const auto invalid_diagnostic=right.Diagnostics(full.UploadPadded(invalid),cutoff);
    Require(invalid_diagnostic.first_bad==bad && invalid_diagnostic.invalid_cells==1 &&
            invalid_diagnostic.reason==Device::invalid_input,"invalid diagnostic cell not reported");
    stats=right.Evaluate(full.UploadPadded(cpu.Padded(cpu.state)),.01,true,.5);
    Require(stats.bad_cells==cells-split && stats.reason==h::HyperbolicCompositeDevice::cfl_exceeded,
            "device CFL rejection missing");
    std::cout<<"CPU/device and partition parity passed on "<<cells<<" cells; worst_scaled_error="<<worst<<'\n';
  } catch(const std::exception&ex) {std::cerr<<ex.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
