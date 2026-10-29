#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"
namespace h=z4c::hyperboloidal;
using Device=h::HyperbolicScriDevice;

static void Require(bool condition,const char*message) {
  if(!condition)throw std::runtime_error(message);
}
static Device::Buffer Upload(const std::vector<Real>&values) {
  Device::Buffer out("test scri input",values.size());auto host=Kokkos::create_mirror_view(out);
  for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(out,host);return out;
}
static std::vector<Real> Download(Device::ConstBuffer values) {
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);
  std::vector<Real>out(values.extent(0));for(size_t i=0;i<out.size();++i)out[i]=host(i);return out;
}
static Real Compare(const std::vector<Real>&expected,const std::vector<Real>&actual,int first=0) {
  Real error=0;
  for(size_t i=0;i<actual.size();++i) {
    Require(std::isfinite(actual[i]),"nonfinite device surface sample");
    const Real e=expected[Device::nf*first+i];error=std::max(error,std::abs(e-actual[i])/(1+std::abs(e)));
  }
  return error;
}
static void Seed(h::HyperbolicComposite&cpu,bool perturb) {
  auto sample=[&](const Real*x,int f) {
    const Real wave=std::sin((1.1+.13*f)*x[0]+(.3+.11*f)*x[1]-(.7+.07*f)*x[2]);
    return (f==0?1.:0.)+(perturb?.0001*(1+f%4)*wave:0.);
  };
  for(size_t p=0;p<cpu.grid.active.size();++p)for(int f=0;f<cpu.nf;++f)
    cpu.state[cpu.nf*p+f]=sample(cpu.grid.geometry[p].x,f);
  for(size_t p=0;p<cpu.scri.rows.size();++p)for(int f=0;f<cpu.nf;++f)
    cpu.scri.values[cpu.nf*p+f]=sample(cpu.scri.rows[p].geometry.x,f);
}
static void ResetDiagnostics(h::HyperbolicComposite&cpu) {
  cpu.null_max=cpu.pole_max=cpu.clock_max=cpu.compatibility_max=0;
}
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);int result=0;
  try {
    std::cout<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<std::endl;
    std::cout<<std::setprecision(17);
    h::HyperbolicComposite cpu(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3));
    const int total=cpu.scri.rows.size(),split=total/2+7;const Real dt=.0001;
    Device full(cpu,0,total),left(cpu,0,split),right(cpu,split,total-split),empty(cpu,total,0);
    for(bool perturb:{false,true}) {
      Seed(cpu,perturb);ResetDiagnostics(cpu);
      const auto original=cpu.scri.values;cpu.BoundaryValues();const auto projected=cpu.scri.values;
      const auto input=Upload(original),active=Upload(cpu.state),surface=Upload(projected);
      const auto projection=full.Project(input);const auto pa=left.Project(input),pb=right.Project(input);
      const auto pempty=empty.Project(input);
      Require(projection.bad_rows==0&&pa.bad_rows==0&&pb.bad_rows==0&&pempty.bad_rows==0&&
              projection.first_bad==-1&&pempty.first_bad==-1,"valid boundary projection rejected");
      const Real projection_error=std::max({Compare(projected,Download(full.LocalSurface())),
          Compare(projected,Download(left.LocalSurface())),Compare(projected,Download(right.LocalSurface()),split)});
      Require(projection_error<2e-12,"device null projection differs from CPU");
      Require(projection.null_after<3e-14&&std::abs(projection.null_after-cpu.null_max)<2e-14,
              "device surface null residual differs from CPU");
      const auto padded=cpu.ActivePadded(cpu.state);const auto expected=cpu.BoundaryRates(padded,dt);
      const auto expected_surface=cpu.scri.values;
      const auto summary=full.Evaluate(active,surface,dt),sa=left.Evaluate(active,surface,dt),
                 sb=right.Evaluate(active,surface,dt),sz=empty.Evaluate(active,surface,dt);
      Require(summary.bad_rows==0&&sa.bad_rows==0&&sb.bad_rows==0&&sz.bad_rows==0&&
              summary.first_bad==-1&&sz.first_bad==-1,"valid native scri rates rejected");
      const Real rate_error=std::max({Compare(expected,Download(full.LocalRHS())),
          Compare(expected,Download(left.LocalRHS())),Compare(expected,Download(right.LocalRHS()),split)});
      const Real trace_error=std::max({Compare(expected_surface,Download(full.LocalSurface())),
          Compare(expected_surface,Download(left.LocalSurface())),Compare(expected_surface,Download(right.LocalSurface()),split)});
      Require(rate_error<2e-10&&trace_error<2e-10,"device scri native rates/dependent trace differ from CPU");
      Require(std::abs(summary.pole-cpu.pole_max)<2e-11&&std::abs(summary.clock-cpu.clock_max)<2e-11&&
              std::abs(summary.compatibility-cpu.compatibility_max)<2e-11,"native scri diagnostics differ");
      Require(std::abs(std::max(sa.pole,sb.pole)-summary.pole)<2e-11&&
              std::abs(std::max(sa.clock,sb.clock)-summary.clock)<2e-11&&
              std::abs(std::max(sa.max_cfl,sb.max_cfl)-summary.max_cfl)<2e-14,
              "partitioned scri reductions differ");
      // D's contribution to HyperbolicGauge.shift_regular is discarded by
      // BoundaryRates. Alter both volume/surface D and reverse CPU row order to
      // prove that sequential dependent-D writes do not affect another row.
      auto changed_state=cpu.state,changed_surface=projected;
      for(size_t p=0;p<cpu.grid.active.size();++p)for(int a=0;a<3;++a)
        changed_state[cpu.nf*p+25+a]=13+3*a+std::sin(Real(p));
      for(int p=0;p<total;++p)for(int a=0;a<3;++a)
        changed_surface[cpu.nf*p+25+a]=-17-2*a+std::cos(Real(p));
      cpu.scri.values=changed_surface;const auto changed_padded=cpu.ActivePadded(changed_state);
      const auto changed_rate=cpu.BoundaryRates(changed_padded,dt);
      Require(Compare(expected,changed_rate)==0&&Compare(expected_surface,cpu.scri.values)==0,
              "CPU scri outputs depend on old D");
      const auto changed_summary=full.Evaluate(Upload(changed_state),Upload(changed_surface),dt);
      Require(changed_summary.bad_rows==0&&Compare(expected,Download(full.LocalRHS()))<2e-10&&
              Compare(expected_surface,Download(full.LocalSurface()))<2e-10,
              "device scri outputs depend on old D");
      cpu.scri.values=changed_surface;std::reverse(cpu.scri.rows.begin(),cpu.scri.rows.end());
      const auto reverse_rate=cpu.BoundaryRates(changed_padded,dt);
      std::reverse(cpu.scri.rows.begin(),cpu.scri.rows.end());
      for(int p=0;p<total;++p) {
        for(int f=0;f<cpu.nf;++f)Require(reverse_rate[(total-1-p)*cpu.nf+f]==expected[p*cpu.nf+f],
                                       "CPU scri rate depends on evaluation order");
        for(int a=0;a<3;++a)Require(cpu.scri.values[(total-1-p)*cpu.nf+25+a]==expected_surface[p*cpu.nf+25+a],
                                    "dependent D depends on evaluation order");
      }
      std::cout<<"perturb="<<perturb<<" projection_error="<<projection_error<<" rate_error="<<rate_error
               <<" dependent_trace_error="<<trace_error<<" null="<<projection.null_after
               <<" pole="<<summary.pole<<" clock="<<summary.clock
               <<" compatibility="<<summary.compatibility<<" max_cfl="<<summary.max_cfl
               <<" D_input_and_order_independent=1\n";
      const auto cfl=full.Evaluate(active,surface,.01);
      Require(cfl.bad_rows==total&&cfl.first_bad==0&&cfl.reason==Device::cfl_exceeded,
              "scri penalty CFL rejection missing");
      auto invalid=original;const int bad=split+3;invalid[cpu.nf*bad]=-1;
      const auto invalid_projection=right.Project(Upload(invalid));
      Require(invalid_projection.bad_rows==1&&invalid_projection.first_bad==bad&&
              invalid_projection.reason==Device::invalid_input,"scri invalid-row index lost");
    }
    std::cout<<"CPU/device scri projection, native rates, D independence and partition parity passed on "
             <<total<<" rows\n";
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
