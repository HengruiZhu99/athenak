#include <iostream>
#include <iomanip>
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/scri_radial_device.hpp"
namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;using Device=h::ScriRadialDevice;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;
try{
  constexpr int n=8;
  Model model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false,true,2,
      Model::NormalDerivative::sampled_numerator,Model::RadialClosure::donor_seven);
  const auto&g=model.grid;const auto&plan=*model.radial_plan;
  Model::V padded(31*g.count,0),surface(model.scri.values.size(),0);
  auto value=[](int f,const Real*x){return (f+1)*.01*std::exp(.3*x[0]-.2*x[1]+.4*x[2]);};
  for(size_t m=0;m<g.mesh.blocks.size();++m)
  for(int k=-g.ng;k<n+g.ng;++k)for(int j=-g.ng;j<n+g.ng;++j)for(int i=-g.ng;i<n+g.ng;++i){
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];g.mesh.Forward(m,q,x);
    for(int f=0;f<31;++f)padded[f*g.count+g.Index(m,i,j,k)]=model.scri.base[m]>=0&&i>=n?
        std::numeric_limits<Real>::quiet_NaN():value(f,x);
  }
  for(size_t p=0;p<model.scri.rows.size();++p)for(int f=0;f<31;++f)surface[31*p+f]=value(f,model.scri.rows[p].geometry.x);
  auto upload=[](const Model::V&source){Device::Buffer out("test input",source.size());auto host=Kokkos::create_mirror_view(out);
    for(size_t i=0;i<source.size();++i)host(i)=source[i];Kokkos::deep_copy(out,host);return out;};
  const auto volume=upload(padded),edge=upload(surface);Real worst=0;size_t tested=0;
  auto check=[&](const std::vector<size_t>&points){
    Device device(plan,points);const auto kernel=device.Bind(volume,edge);
    Device::Buffer result("derivative control",points.size()*31*12);
    Kokkos::parallel_for("radial plan control",Kokkos::RangePolicy<>(0,points.size()*31),KOKKOS_LAMBDA(int index){
      const int local=index/31,f=index%31;Real d[3]{},dd[3][3]{};
      if(kernel.HasRow(local))kernel.Derivatives(f,local,d,dd,f<7||f==18);
      for(int a=0;a<3;++a){result(index*12+a)=d[a];for(int b=0;b<3;++b)result(index*12+3+3*a+b)=dd[a][b];}
    });
    const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
    for(size_t i=0;i<points.size();++i)for(int f=0;f<31;++f){Real d[3]{},dd[3][3]{};
      if(plan.Contains(points[i])){plan.Derivatives(padded,surface,f,points[i],d,dd,f<7||f==18);++tested;}
      for(int a=0;a<3;++a){const Real got=host((31*i+f)*12+a);if(!std::isfinite(got))throw std::runtime_error("device read poisoned radial ghost");
        worst=std::max(worst,std::abs(got-d[a])/(1+std::abs(d[a])));
        for(int b=0;b<3;++b){const Real v=host((31*i+f)*12+3+3*a+b);if(!std::isfinite(v))throw std::runtime_error("nonfinite second derivative");
          worst=std::max(worst,std::abs(v-dd[a][b])/(1+std::abs(dd[a][b])));}}
    }
  };
  for(int rank=0;rank<12;++rank){std::vector<size_t>points;
    for(size_t p=g.active.size()*rank/12;p<g.active.size()*(rank+1)/12;++p)points.push_back(p);check(points);}
  // Numerator ownership includes sparse donors and may be empty.
  std::vector<size_t>sparse;for(size_t p=0;p<g.active.size();p+=37)sparse.push_back(p);check(sparse);check({});
  bool shape_rejected=false,index_rejected=false;
  try{Device p(plan,{});p.Bind(Device::Buffer("bad",1),edge);}catch(const std::invalid_argument&){shape_rejected=true;}
  try{Device p(plan,{g.active.size()});}catch(const std::invalid_argument&){index_rejected=true;}
  if(worst>2e-12||!shape_rejected||!index_rejected||tested==0)throw std::runtime_error("radial device parity/guard failed");
  std::cout<<std::setprecision(17)<<"{\"passed\":true,\"backend\":\""<<Device::ExecutionSpace::name()
    <<"\",\"max_scaled_error\":"<<worst<<",\"tested_field_rows\":"<<tested
    <<",\"simulated_partitions\":12,\"sparse_and_empty_ownership\":true,\"radial_ghost_poison\":true,\"invalid_inputs_rejected\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
Kokkos::finalize();return code;}
