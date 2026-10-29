// Component operator tests only. No evolution or coupled stability claim.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_angular_dissipation_device.hpp"
#include "angular_dissipation_reference.hpp"

namespace c=cubed_sphere;
using Plan=c::AngularDissipationPlan;
using Device=z4c::hyperboloidal::HyperbolicAngularDissipationDevice;
static void Require(bool good,const char*message){if(!good)throw std::runtime_error(message);}
template<class F>static void Reject(F callback,const char*message){
  bool rejected=false;try{callback();}catch(const std::exception&){rejected=true;}Require(rejected,message);
}
static Device::Buffer Upload(const std::vector<Real>&v){
  Device::Buffer out("angular test input",v.size());auto host=Kokkos::create_mirror_view(out);
  for(size_t i=0;i<v.size();++i)host(i)=v[i];Kokkos::deep_copy(out,host);return out;
}
static std::vector<Real> Download(Device::ConstBuffer v){
  const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),v);std::vector<Real>out(v.extent(0));
  for(size_t i=0;i<out.size();++i)out[i]=host(i);return out;
}
static Real Compare(const std::vector<Real>&a,const std::vector<Real>&b,int offset=0){
  Require(size_t(offset)+b.size()<=a.size(),"test comparison shape");Real error=0;
  for(size_t i=0;i<b.size();++i){Require(std::isfinite(b[i]),"nonfinite test output");
    error=std::max(error,std::abs(a[offset+i]-b[i])/(1+std::abs(a[offset+i])));}
  return error;
}
static Real Inner(const Plan&p,const std::vector<Real>&a,const std::vector<Real>&b){
  Real value=0;for(int row=0;row<p.Points();++row)for(int f=0;f<Plan::free_fields;++f){
    const int field=Plan::FreeField(f);
    const bool off_diagonal=field==2||field==3||field==5||field==8||field==9||field==11;
    value+=(off_diagonal?2:1)*p.Areas()[row]*a[31*row+field]*b[31*row+field];}
  return value;
}
static void Check(int n,int degree){
  c::CompositeDifference grid(c::CompositeMesh(.125,{.45,.75,1},1,n,3),4,true);
  c::ScriSurface surface(grid,31,6,true);Plan plan(surface,n,degree);
  const int total=plan.Points();const auto&stats=plan.Stats();
  Require(std::abs(std::accumulate(plan.Areas().begin(),plan.Areas().end(),Real(0))-4*std::acos(-1.))<2e-13,
          "angular area sum mismatch");
  Require(stats.cross_face_rows>0&&stats.max_moment_error<1e-11&&stats.max_condition_bound<1e8,
          "angular neighbor/moment statistics invalid");
  Require(plan.TransposeTerms().size()==2*plan.Terms().size(),"transpose lost paired incidences");
  std::vector<Real> u(31*total),v(u.size()),constant(u.size());
  for(int row=0;row<total;++row)for(int f=0;f<31;++f){
    u[31*row+f]=std::sin(.37*row+.17*f)+std::cos(.013*row*row+.23*f);
    v[31*row+f]=std::cos(.23*row-.11*f)+.2*std::sin(.019*row*row+.3*f);
    constant[31*row+f]=1.2345+.7*f;
  }
  const Real epsilon=1.25;const auto residual=plan.Residual(u),source=plan.ApplyResidual(residual,epsilon);
  const auto other=plan.Apply(v,epsilon),zero=plan.Apply(constant,epsilon);
  Real constant_error=0;for(Real value:zero)constant_error=std::max(constant_error,std::abs(value));
  Require(constant_error==0,"constant angular data changed");
  Require(Compare(source,plan.Apply(u,epsilon))==0,"host factorized apply mismatch");
  for(int row=0;row<total;++row)for(int f:{18,22,23,24,25,26,27,28,29,30})
    Require(source[31*row+f]==0,"dependent/prescribed field received angular source");
  Real dissipation=0;for(int row=0;row<total;++row)for(int f=0;f<21;++f){
    const int field=Plan::FreeField(f);const bool off=field==2||field==3||field==5||field==8||field==9||field==11;
    dissipation-=(off?2:1)*epsilon*n*plan.Areas()[row]*residual[21*row+f]*residual[21*row+f];
  }
  const Real energy=Inner(plan,u,source),energy_error=std::abs(energy-dissipation)/(1+std::abs(dissipation));
  const Real left=Inner(plan,u,other),right=Inner(plan,source,v),adjoint_error=std::abs(left-right)/(1+std::abs(left)+std::abs(right));
  Require(energy<0&&energy_error<2e-12&&adjoint_error<2e-12,"angular weighted energy/adjoint failed");
  Real reference_coeff=0,reference_source=0;
  if(n==6&&degree==7){
    for(const auto&reference:angular_reference::terms){
      bool found=false;for(int k=plan.Offsets()[reference.row];k<plan.Offsets()[reference.row+1];++k){
        const auto&t=plan.Terms()[k];if(t.point==reference.donor){found=true;
          reference_coeff=std::max(reference_coeff,std::abs(t.coefficient-reference.coefficient));}}
      Require(found,"native neighbor list differs from NumPy SVD reference");
    }
    const auto unscaled=plan.Apply(u,1);
    for(int row=0;row<total;++row)reference_source=std::max(reference_source,
        std::abs(unscaled[31*row]-angular_reference::source[row])/(1+std::abs(angular_reference::source[row])));
    Require(reference_coeff<2e-11&&reference_source<2e-11,"native QR differs from NumPy SVD reference");
  }
  const auto input=Upload(u),global_residual=Upload(residual);Device full(plan,0,total);
  Require(full.Residual(input).bad_values==0&&full.Apply(global_residual,epsilon).bad_values==0,
          "valid device angular input rejected");
  const Real device_residual=Compare(residual,Download(full.LocalResidual()));
  const Real device_source=Compare(source,Download(full.LocalRHS()));
  Real partition_error=0;
  for(int rank=0;rank<12;++rank){
    const int begin=total*rank/12,end=total*(rank+1)/12;Device part(plan,begin,end-begin);
    Require(part.Residual(input).bad_values==0&&part.Apply(global_residual,epsilon).bad_values==0,
            "owned angular operation rejected");
    partition_error=std::max({partition_error,Compare(residual,Download(part.LocalResidual()),21*begin),
                             Compare(source,Download(part.LocalRHS()),31*begin)});
  }
  Device empty(plan,total,0);
  Require(empty.Residual(input).bad_values==0&&empty.Apply(global_residual,epsilon).bad_values==0
          &&empty.LocalResidual().extent(0)==0&&empty.LocalRHS().extent(0)==0,"empty angular partition failed");
  Require(device_residual<2e-11&&device_source<2e-11&&partition_error<2e-11,"angular CPU/device or partition parity failed");
  const auto constants=Upload(constant);Require(full.Residual(constants).bad_values==0,"device constant input rejected");
  const auto constant_residual=Download(full.LocalResidual());
  Require(std::all_of(constant_residual.begin(),constant_residual.end(),[](Real x){return x==0;}),"device constant residual nonzero");
  Require(full.Apply(Upload(constant_residual),epsilon).bad_values==0,"constant transpose rejected");
  const auto constant_source=Download(full.LocalRHS());
  Require(std::all_of(constant_source.begin(),constant_source.end(),[](Real x){return x==0;}),"device constant source nonzero");
  Reject([&]{Plan wrong(surface,n+1,degree);},"mismatched angular n accepted");
  Reject([&]{Plan wrong(surface,n,6);},"unsupported polynomial degree accepted");
  Reject([&]{Device wrong(plan,-1,1);},"negative owned range accepted");
  Reject([&]{Device wrong(plan,total,1);},"out-of-range owned row accepted");
  Reject([&]{plan.Apply(u,-1);},"negative host epsilon accepted");
  Reject([&]{full.Apply(global_residual,std::numeric_limits<Real>::quiet_NaN());},"nonfinite device epsilon accepted");
  Reject([&]{full.Apply(global_residual,-1);},"negative device epsilon accepted");
  Reject([&]{plan.Residual(std::vector<Real>(u.size()-1));},"bad host surface shape accepted");
  Reject([&]{plan.ApplyResidual(std::vector<Real>(residual.size()-1),1);},"bad host residual shape accepted");
  Reject([&]{full.Residual(Upload(std::vector<Real>(u.size()-1)));},"bad device surface shape accepted");
  Reject([&]{full.Apply(Upload(std::vector<Real>(residual.size()-1)),1);},"bad device residual shape accepted");
  auto invalid=u;invalid[0]=std::numeric_limits<Real>::quiet_NaN();
  Reject([&]{plan.Apply(invalid,1);},"nonfinite free host state accepted");
  const auto bad=full.Residual(Upload(invalid));Require(bad.bad_values>0&&bad.first_bad==0,"device invalid state not reported");
  auto invalid_residual=residual;invalid_residual[0]=std::numeric_limits<Real>::infinity();
  Reject([&]{plan.ApplyResidual(invalid_residual,1);},"nonfinite host residual accepted");
  Require(full.Apply(Upload(invalid_residual),1).bad_values>0,"device invalid residual not reported");
  const auto disabled=plan.Apply(u,0);Require(std::all_of(disabled.begin(),disabled.end(),[](Real x){return x==0;}),"host epsilon zero not zero");
  Require(full.Apply(global_residual,0).bad_values==0,"device epsilon zero rejected");
  const auto disabled_device=Download(full.LocalRHS());Require(std::all_of(disabled_device.begin(),disabled_device.end(),[](Real x){return x==0;}),"device epsilon zero not zero");
  std::cout<<"n="<<n<<" degree="<<degree<<" points="<<total<<" epsilon="<<epsilon
    <<" coefficient_reference_error="<<reference_coeff<<" source_reference_error="<<reference_source
    <<" device_residual_error="<<device_residual<<" device_source_error="<<device_source
    <<" partition_error="<<partition_error<<" energy_error="<<energy_error<<" adjoint_error="<<adjoint_error
    <<" constant_error="<<constant_error<<" max_moment_error="<<stats.max_moment_error
    <<" max_condition_bound="<<stats.max_condition_bound<<" max_coefficient_l1="<<stats.max_coefficient_l1
    <<" max_neighbors="<<stats.max_neighbors<<" spectral_upper_bound="<<plan.SpectralUpperBound()<<'\n';
}
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{
    std::cout<<std::setprecision(17)<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    Check(6,7);Check(8,7);Check(6,5);
    std::cout<<"Angular dissipation CPU/device, energy, constants, partitions and reference parity passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
