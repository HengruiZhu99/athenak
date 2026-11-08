// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Compare actual extracted legacy bulk operator and legacy endpoint/reference
// functions with the native point API, using deterministic nonspherical jets.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/endpoint.hpp"
#include "z4c_hyperboloidal/dissipation.hpp"
#include "z4c/hyperboloidal/spherical_reference_3d.hpp"
#include "z4c/hyperboloidal/spherical_policy.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"

namespace cubed_sphere {inline constexpr bool scri_first6=false;}
namespace z4c::hyperboloidal {
// Storage/jet adapter only. The operator() text below is read directly from
// HyperbolicCompositeDevice::PointKernel in the production source during build.
template<class T,int N>struct OracleView {
  mutable T data[N]{};
  T &operator()(int i)const{return data[i];}
};
struct LegacyPointOracle {
  static constexpr int nf=31;
  enum {success=0,invalid_input=1,invalid_geometry=2,invalid_assembly=3,
        invalid_gauge=4,invalid_auxiliary=5,cfl_exceeded=6,nonfinite_rhs=7};
  OracleView<int,1> status,active;
  OracleView<Real,1> gap,cfl;
  OracleView<Real,31> rhs,padded;
  OracleView<cubed_sphere::Point,1> geometry;
  OracleView<StationaryGaugeSource<Real>,1> stationary_gauge;
  HyperbolicShiftParameters<Real> parameters;
  CartesianChartJets jets;
  Real transport[31]{},dissipation[31]{};
  int padded_count=1,n=12;
  Real dt=1e-6;
  bool adaptive=false,amr_cfl=false;
  HyperbolicShiftParameters<Real> Parameters()const{return parameters;}
  Real Dissipation(int,int f)const{return dissipation[f];}
  int BuildJet(int,const cubed_sphere::Point &map,Real,Z4cJet<Real> &u,
      VectorJet<Real> &B,VectorJet<Real> &D,Real *correction)const {
    u=SphericalPolicy::Physical(map.x,jets);
    B=SphericalPolicy::Vector(jets,19);D=SphericalPolicy::Vector(jets,25);
    for(int f=0;f<31;++f)correction[f]=transport[f];
    return u.chi.value>0&&u.alpha.value>0?success:invalid_input;
  }
#include "physics_legacy_operator.hpp"
};
struct LegacySampler {
  OracleView<Real,31*9*9*9> padded;
  OracleView<int,1> active;
  int n=12,width=9,padded_count=9*9*9,diagnostic_outer_first_order=0;
  Real outer_radius2=0;
  struct NoRadialRow {
    bool HasRow(int)const{return false;}
    void Derivatives(int,int,Real*,Real(*)[3],bool)const{}
  } radial;
#include "physics_legacy_sampler.hpp"
};
}

namespace old=z4c::hyperboloidal;
namespace nh=z4c_hyperboloidal;
int comparisons=0,bulk_cases=0,scri_cases=0,invalid_cases=0;
double worst_relative=0;
void Require(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
void Compare(double a,double b,const char *message,double tolerance=2e-13) {
  ++comparisons;
  const double error=std::abs(a-b)/(1+std::max(std::abs(a),std::abs(b)));
  worst_relative=std::max(worst_relative,error);
  if(!std::isfinite(a)||!std::isfinite(b)||error>tolerance) {
    std::cerr<<message<<": "<<std::setprecision(17)<<a<<" vs "<<b<<" error="<<error<<"\n";
    throw std::runtime_error(message);
  }
}
old::CartesianChartJets OldJets(const nh::CartesianChartJets &a) {
  old::CartesianChartJets b;
  for(int f=0;f<31;++f){b[f].f=a[f].f;for(int i=0;i<3;++i){b[f].d[i]=a[f].d[i];
    for(int j=0;j<3;++j)b[f].h[i][j]=a[f].h[i][j];}}
  return b;
}
cubed_sphere::Point OldMap(const nh::mapped::Point &a) {
  cubed_sphere::Point b;
  for(int i=0;i<3;++i){b.x[i]=a.x[i];for(int A=0;A<3;++A){b.J[A][i]=a.J[A][i];
    for(int j=0;j<3;++j)b.H[A][i][j]=a.H[A][i][j];}}
  return b;
}
void CompareJets(const nh::CartesianChartJets &a,const old::CartesianChartJets &b) {
  for(int f=0;f<31;++f){Compare(a[f].f,b[f].f,"reference/closure value");
    for(int i=0;i<3;++i){Compare(a[f].d[i],b[f].d[i],"reference/closure first");
      for(int j=0;j<3;++j)Compare(a[f].h[i][j],b[f].h[i][j],"reference/closure second");}}
}

void Bulk(const nh::mapped::Point &map,const nh::CartesianChartJets &jets,
    const nh::HyperbolicShiftParameters<Real> &np,const old::HyperbolicShiftParameters<Real> &op,
    const nh::StationaryGaugeSource<Real> &ns,const old::StationaryGaugeSource<Real> &os,
    const nh::PointForcing &forcing) {
  old::LegacyPointOracle oracle;oracle.parameters=op;oracle.jets=OldJets(jets);
  oracle.geometry(0)=OldMap(map);oracle.stationary_gauge(0)=os;
  for(int f=0;f<31;++f){oracle.transport[f]=forcing.transport[f];oracle.dissipation[f]=forcing.dissipation[f];}
  oracle(0);
  const auto out=nh::EvaluateBulkPoint(map,jets,np,ns,forcing);
  Require(static_cast<int>(out.status)==oracle.status(0),"bulk invalid status differs");
  if(out.status!=nh::PointStatus::success){++invalid_cases;return;}
  for(int f=0;f<31;++f)Compare(out.rate[f],oracle.rhs(f),"bulk RHS");
  Compare(out.gap,oracle.gap(0),"gauge gap");
  Compare(oracle.dt*oracle.n*(out.speed[0]+out.speed[1]+out.speed[2]),oracle.cfl(0),"mapped characteristic speed");
  ++bulk_cases;
}

void TestEndpointPolynomial() {
  nh::mapped::Point map;for(int a=0;a<3;++a)map.J[a][a]=1;
  const Real nodes[3][7]={{.7,.75,.8,.85,.9,.95,1},{-.2,-.1,0,.1,.2,0,0},{-.2,-.1,0,.1,.2,0,0}};
  const int width[3]={7,5,5};const Real q[3]={.97,.03,-.04};
  const auto f=[](Real r,Real u,Real v){return r*r*r*r*r*r+u*u*u*u+v*v*v*v+r*u*v;};
  nh::mapped::Jet total;
  const Real anchor=f(q[0],q[1],q[2]);
  for(int k=0;k<5;++k)for(int j=0;j<5;++j)for(int i=0;i<7;++i) {
    const auto w=nh::EndpointWeight(map,q,nodes,width,i,j,k);
    total=total+w*nh::mapped::Jet(f(nodes[0][i],nodes[1][j],nodes[2][k])-anchor);
  }
  const auto r=nh::mapped::Jet::Variable(q[0],0),u=nh::mapped::Jet::Variable(q[1],1),v=nh::mapped::Jet::Variable(q[2],2);
  const auto exact=r*r*r*r*r*r+u*u*u*u+v*v*v*v+r*u*v;
  for(int a=0;a<3;++a){Compare(total.d[a],exact.d[a],"endpoint first polynomial",1e-11);
    for(int b=0;b<3;++b)Compare(total.h[a][b],exact.h[a][b],"endpoint second polynomial",1e-11);}
  const Real dr=.01,gap=1;
  for(int n=0;n<7;++n){map.x[0]=n==0?1:1-(n-1+gap)*dr;
    const auto lift=nh::NormalLift(map,nh::mapped::Jet(2),dr,gap);
    Compare(lift.f,0,"normal lift zeros",1e-11);
    if(n==0)Compare(lift.d[0],2,"normal lift derivative");}
}

void TestKO() {
  const double q[9]={.2,-.4,.8,.1,-.7,.3,.9,-.2,.6};
  double rates[9]{},energy=0,expected=0;
  for(int begin=0;begin<5;++begin){double values[5],difference=0;
    for(int a=0;a<5;++a){values[a]=q[begin+a];difference+=nh::PhysicalKO8::Coefficient(a)*(values[a]-q[begin]);}
    for(int target=0;target<5;++target)
      rates[begin+target]+=nh::PhysicalKO8::Line(values,target,1,1,12,.1);
    expected-=.1*12*difference*difference/256;
  }
  for(int i=0;i<9;++i)energy+=q[i]*rates[i];
  Compare(energy,expected,"KO weighted quadratic identity");Require(energy<0,"KO must dissipate");
  double cubic[5];for(int i=0;i<5;++i)cubic[i]=i*i*i;
  Compare(nh::PhysicalKO8::Line(cubic,2,1,1,12,.1),0,"KO annihilates cubic");
}

void TestNativeReader() {
  DvceArray5D<Real> state("oracle native fields",2,31,9,9,9),ref("oracle reference",2,31,9,9,9);
  DvceArray4D<nh::mapped::Point> geometry("oracle geometry",2,9,9,9);
  auto hs=Kokkos::create_mirror_view(state),hr=Kokkos::create_mirror_view(ref);
  auto hg=Kokkos::create_mirror_view(geometry);
  for(int m=0;m<2;++m)for(int k=0;k<9;++k)for(int j=0;j<9;++j)for(int i=0;i<9;++i){
    hg(m,k,j,i).x[0]=.2+.001*i;
    for(int f=0;f<31;++f){hr(m,f,k,j,i)=100*m+f;
      hs(m,f,k,j,i)=hr(m,f,k,j,i)+(f==0?hg(m,k,j,i).x[0]*hg(m,k,j,i).x[0]:1)*(.01*i+.02*j+.03*k);}}
  Kokkos::deep_copy(state,hs);Kokkos::deep_copy(ref,hr);Kokkos::deep_copy(geometry,hg);
  DvceArray1D<Real> result("reader result",3);
  Kokkos::parallel_for("native block reader oracle",1,KOKKOS_LAMBDA(int){
    const nh::BlockResidualReader reader{state,ref,geometry,1,4,4,4};
    result(0)=reader(0,1,-1,0);result(1)=reader(22,1,-1,0);result(2)=reader(6);});
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
  Compare(h(0),.23,"native chi residual layout",2e-12);Compare(h(1),.23,"native field residual layout");Compare(h(2),0,"dependent field masked");
}

void TestStencil(const nh::HyperbolicShiftParameters<Real> &p,
                 nh::SphericalReference3D &reference) {
  // Exercise native packed stencils against the actual legacy Sample method,
  // including mixed Hessians and the biased transport correction.
  old::LegacySampler old_sample;old_sample.active(0)=4+9*(4+9*4);
  const auto value=[](int f,int i,int j,int k){
    const double x=(i-4)/12.,y=(j-4)/12.,z=(k-4)/12.;
    return (f+1)*1e-5*(.2+x*x*x*x*x+x*y*y+z*z*z*z+y*z);
  };
  DvceArray5D<Real> state("stencil oracle native fields",1,31,9,9,9);
  auto host=Kokkos::create_mirror_view(state);
  for(int f=0;f<31;++f)for(int k=0;k<9;++k)for(int j=0;j<9;++j)for(int i=0;i<9;++i){
    host(0,f,k,j,i)=value(f,i,j,k);
    old_sample.padded(f*9*9*9+i+9*(j+9*k))=value(f,i,j,k);
  }
  Kokkos::deep_copy(state,host);
  const nh::BlockPointReader reader{state,0,4,4,4};
  chart_atlas::BlockMap block;block.face=2;block.chart_id=3;block.inner_radius=.7;block.outer_radius=.9;
  for(int a=0;a<3;++a){block.lo[a]=0;block.hi[a]=1;}
  const Real local[3]={.5,.25,.15};const auto map=block.Geometry(local);
  const Real inv_h[3]={12,12,12},beta[3]={-.1,.2,-.3};
  for(int f=0;f<28;++f)if(nh::Fields::Independent(f)) {
    const bool second=f<=6||f==18||(f>=22&&f<25);
    Real transport=0,old_transport=0;
    const auto jet=nh::SampleResidualJet(reader,f,map,inv_h,beta,second,transport);
    const auto legacy=old_sample.Sample(0,f,OldMap(map),beta,second,old_transport);
    Compare(jet.f,legacy.value,"stencil value");Compare(transport,old_transport,"stencil transport");
    for(int a=0;a<3;++a){Compare(jet.d[a],legacy.d[a],"stencil first");
      for(int b=0;b<3;++b)Compare(jet.h[a][b],legacy.dd[a][b],"stencil second");}
  }
  const auto ref=reference.At(map.x);
  const auto built=nh::BuildPointState(reader,map,inv_h,ref,p);
  Require(built.status==nh::PointStatus::success,"native BuildPointState");
  // Legacy BuildJet policy path, using the extracted legacy stencil.
  Real physical_beta[3];for(int a=0;a<3;++a)physical_beta[a]=reader(22+a)+ref[22+a].f-map.x[a];
  old::CartesianChartJets j{};Real correction[31]{};
  for(int f=0;f<31;++f)if(old::CartesianIndependent(f)){
    const auto q=old_sample.Sample(0,f,OldMap(map),physical_beta,
       f<=6||f==18||(f>=22&&f<25),correction[f]);
    j[f].f=q.value;for(int a=0;a<3;++a){j[f].d[a]=q.d[a];
      for(int b=0;b<3;++b)j[f].h[a][b]=q.dd[a][b];}}
  const Real r=nh::SphericalPolicy::Radius(map.x),w=nh::SphericalPolicy::CompatibleWeight(r,p);
  if(w>0){Real composed=0;
    for(int k=-2;k<=2;++k)if(k){const Real c=(k==1||k==-1)?2.*k/3:-k/24.;
      composed+=12*c*old_sample.First(18*729+old_sample.active(0)+k,0);}
    Real inverse=0;for(int a=0;a<3;++a)inverse+=map.J[0][a]*map.x[a]/r;
    const Real delta=w*inverse*inverse*(composed-old_sample.Second(18*729+old_sample.active(0),0,0));
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)j[18].h[a][b]+=delta*map.x[a]*map.x[b]/(r*r);}
  Require(old::SphericalPolicy::Restore(map.x,j,OldJets(ref)),"legacy BuildJet restore");
  correction[0]*=r*r;CompareJets(built.jets,j);
  for(int f=0;f<31;++f)Compare(built.transport[f],correction[f],"BuildPointState transport");
}

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);int rc=0;
  try {
    int independent=0;for(int f=0;f<31;++f)independent+=nh::Fields::Independent(f);
    Require(independent==26,"independent field count");
    nh::CentralZ4cConfiguration nc;nc.compact_slicing_end=.97;
    old::CentralZ4cConfiguration oc;oc.compact_slicing_end=.97;
    auto np=nc.Resolve().gauge;auto op=oc.Resolve().gauge;
    np.outer_lapse_damping=op.outer_lapse_damping=.25;
    nh::SphericalReference3D nr(np);old::SphericalReference3D ore(op);
    std::mt19937_64 rng(0x3112262026ULL);std::uniform_real_distribution<double> random(-1,1);
    const Real radii[]={.015,.08,.15,.32,.55,.75,.90,.97,.995};
    for(Real radius:radii)for(int sample=0;sample<48;++sample) {
      const int face=sample%6;
      chart_atlas::BlockMap nm;nm.chart_id=face+1;nm.face=face;
      nm.half_width=.005;nm.outer_radius=radius*1.03;nm.cube_transition=true;
      for(int a=0;a<3;++a)nm.hi[a]=1;
      cubed_sphere::Map om{.005,radius*1.03,0};
      const Real q[3]={.9,.7*random(rng),.7*random(rng)};
      auto map=nm.Geometry(q);const auto omap=om.Geometry(face,q);
      for(int a=0;a<3;++a){Compare(map.x[a],omap.x[a],"map value");
        for(int b=0;b<3;++b){Compare(map.J[a][b],omap.J[a][b],"map inverse Jacobian");
          for(int c=0;c<3;++c)Compare(map.H[a][b][c],omap.H[a][b][c],"map inverse Hessian");}}
      auto jets=nr.At(map.x);auto oj=ore.At(map.x);CompareJets(jets,oj);
      nh::CartesianChartJets residual{};auto old_residual=OldJets(residual);
      for(int f=0;f<31;++f)if(nh::Fields::Independent(f)){
        residual[f].f=1e-5*random(rng);
        for(int a=0;a<3;++a){residual[f].d[a]=1e-4*random(rng);
          for(int b=0;b<3;++b)residual[f].h[a][b]=1e-4*random(rng);}}
      old_residual=OldJets(residual);
      Require(nh::SphericalPolicy::Restore(map.x,residual,jets),"native restore");
      Require(old::SphericalPolicy::Restore(map.x,old_residual,oj),"old restore");
      CompareJets(residual,old_residual);jets=residual;
      nh::PointForcing forcing;
      for(int f=0;f<28;++f)if(nh::Fields::Independent(f)){
        forcing.transport[f]=1e-5*random(rng);forcing.dissipation[f]=1e-5*random(rng);}
      const double r=nh::SphericalPolicy::Radius(map.x);
      Bulk(map,jets,np,op,nr.source.At(r),ore.source.At(r),forcing);
      if(sample==0){auto bad=jets;bad[0].f=-1;Bulk(map,bad,np,op,nr.source.At(r),ore.source.At(r),{});
        bad=jets;bad[1].f=-2;Bulk(map,bad,np,op,nr.source.At(r),ore.source.At(r),{});
        bad=jets;bad[18].f=-2;Bulk(map,bad,np,op,nr.source.At(r),ore.source.At(r),{});
        bad=jets;bad[7].d[0]=std::numeric_limits<double>::quiet_NaN();Bulk(map,bad,np,op,nr.source.At(r),ore.source.At(r),{});}
    }
    for(int sample=0;sample<256;++sample) {
      nh::mapped::Point map;Real norm=0;for(auto &x:map.x){x=random(rng);norm+=x*x;}
      for(auto &x:map.x)x/=std::sqrt(norm);
      auto jets=nr.At(map.x,true);CompareJets(jets,ore.At(map.x,true));
      for(int f=0;f<28;++f)if(nh::Fields::Independent(f)){
        jets[f].f+=.001*random(rng);for(int a=0;a<3;++a){jets[f].d[a]+=.01*random(rng);
          for(int b=0;b<3;++b)jets[f].h[a][b]+=.01*random(rng);}}
      nh::ExpandCartesianChart(jets);
      const auto a=nh::EvaluateScriPoint(map,1./240,jets,np,nr.source.At(1),.5);
      const auto b=old::SphericalPolicy::Boundary(OldMap(map),1./240,OldJets(jets),op,ore.source.At(1),.5);
      Require(a.valid==b.valid&&a.valid,"scri validity");
      for(int f=0;f<31;++f)Compare(a.rate[f],b.rate[f],"scri RHS");
      for(int i=0;i<3;++i)Compare(a.defect[i],b.defect[i],"normal compatibility");
      Compare(a.clock,b.clock,"scri clock");++scri_cases;
      if(sample==0){jets[0].f=-1;
        const auto na=nh::EvaluateScriPoint(map,1./240,jets,np,nr.source.At(1),.5);
        const auto ob=old::SphericalPolicy::Boundary(OldMap(map),1./240,OldJets(jets),op,ore.source.At(1),.5);
        Require(na.valid==ob.valid&&!na.valid,"invalid scri chi");++invalid_cases;}
    }
    TestEndpointPolynomial();TestKO();TestNativeReader();TestStencil(np,nr);
    std::cout<<"PASS physics oracle: bulk="<<bulk_cases<<" scri="<<scri_cases
      <<" invalid="<<invalid_cases<<" comparisons="<<comparisons
      <<" max_scaled_error="<<std::setprecision(17)<<worst_relative<<"\n";
  }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";rc=1;}
  Kokkos::finalize();return rc;
}
