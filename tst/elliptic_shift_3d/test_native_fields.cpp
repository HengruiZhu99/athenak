// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include "z4c/hyperboloidal/native_composite_fields.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2));
 const int cells=grid.n+2*grid.ng;const size_t points=grid.active.size();
 DvceArray5D<Real> bulk("native bulk",10,25,cells,cells,cells),shell("native shells",12,25,cells,cells,cells);
 std::vector<h::NativeCompositeFields::Block> bm,sm;
 for(int m=0;m<8;++m)bm.push_back({7-m,m});for(int m=0;m<12;++m)sm.push_back({11-m,8+m});
 Real roundtrip=0,rate_error=0,halo_error=0,physical_error=0,positive_rate_error=0,tiny_relative_error=0,tiny_coefficient_error=0;int cases=0,invalid_storage=0;
 for(int mode=-1;mode<6;++mode){const bool puncture=mode>=0;const int positive=mode>=3?mode-2:0;
  h::NativeCompositeFields adapter(grid,puncture,positive?0:std::max(0,mode),-4,positive);
  std::vector<Real> y(22*points),b(3*points),dy(y.size());
  for(size_t p=0;p<points;++p){const auto &x=grid.geometry[p].x;
   for(int f=0;f<22;++f){y[22*p+f]=.001*(f+1)*(1+x[f%3]);dy[22*p+f]=.0007*(f+1);}
   if(puncture)y[22*p+13]=2.8;
   if(positive){const Real chi=.4+.03*x[0]+.01*x[1]*x[1],alpha=.7+.04*x[2];
    const Real A=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
    y[22*p]=positive==1?chi:(positive==2?std::log(chi):chi*chi);
    y[22*p+18]=positive==1?alpha:(positive==2?std::log(alpha)-std::log(A):alpha*alpha);
    y[22*p+13]=3-.2+.03*x[0];y[22*p+14]=-.004*(1+x[1]);
    for(int t=0;t<6;++t)y[22*p+7+t]=.002*(t%2?-1:1)*(1+x[t%3]);
    for(int v=0;v<3;++v){y[22*p+15+v]=.003*(v%2?-1:1)*(1+x[v]);y[22*p+19+v]=-.0003*(v+1)*(1+x[v]);}}

   for(int v=0;v<3;++v)b[3*p+v]=.003*(v+1)*(1+x[v]);}
  Kokkos::deep_copy(bulk,-91.);Kokkos::deep_copy(shell,-91.);
  adapter.Write(bulk,bm,y,b);adapter.Write(shell,sm,y,b);
  std::vector<Real> again,bb;adapter.Read(shell,sm,again,bb);adapter.Read(bulk,bm,again,bb);
  for(size_t p=0;p<y.size();++p)roundtrip=std::max(roundtrip,std::abs(y[p]-again[p]));
  for(size_t p=0;p<b.size();++p)roundtrip=std::max(roundtrip,std::abs(b[p]-bb[p]));
  const auto saved=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bulk);
  if(positive)for(const auto &block:bm)for(int k=0;k<grid.n;++k)for(int j=0;j<grid.n;++j)for(int i=0;i<grid.n;++i){
   const size_t p=((block.composite*grid.n+k)*grid.n+j)*grid.n+i;const auto &x=grid.geometry[p].x;
   for(int f=0;f<22;++f){Real expected=y[22*p+f];
    if(f==0)expected=.4+.03*x[0]+.01*x[1]*x[1];if(f==18)expected=.7+.04*x[2];if(f==13)expected-=3;
    for(int t=0;t<6;++t)if(f==1+t&&h::kCompositeTensorI[t]==h::kCompositeTensorJ[t])expected+=1;
    physical_error=std::max(physical_error,std::abs(saved(block.native,h::NativeCompositeFields::scalar_native[f],k+grid.ng,j+grid.ng,i+grid.ng)-expected));}
  }
  for(int m=0;m<10;++m)for(int f=0;f<25;++f)for(int k=0;k<cells;++k)for(int j=0;j<cells;++j)for(int i=0;i<cells;++i)
   if(m>=8||i<grid.ng||j<grid.ng||k<grid.ng||i>=grid.ng+grid.n||j>=grid.ng+grid.n||k>=grid.ng+grid.n)
    halo_error=std::max(halo_error,std::abs(saved(m,f,k,j,i)+91));
  DvceArray5D<Real> plus("plus",10,25,cells,cells,cells),minus("minus",10,25,cells,cells,cells),rate("rate",10,25,cells,cells,cells);
  Kokkos::deep_copy(plus,-37.);Kokkos::deep_copy(minus,-37.);Kokkos::deep_copy(rate,-37.);
  const Real eps=1e-5;auto yp=y,ym=y;for(size_t p=0;p<y.size();++p){yp[p]+=eps*dy[p];ym[p]-=eps*dy[p];}
  adapter.Write(plus,bm,yp,b);adapter.Write(minus,bm,ym,b);adapter.Write(rate,bm,y,b,&dy);
  const auto hp=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),plus),hm=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),minus),hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rate);
  for(const auto &block:bm)for(int f=0;f<25;++f)for(int k=grid.ng;k<grid.ng+grid.n;++k)for(int j=grid.ng;j<grid.ng+grid.n;++j)for(int i=grid.ng;i<grid.ng+grid.n;++i)
   rate_error=std::max(rate_error,std::abs((hp(block.native,f,k,j,i)-hm(block.native,f,k,j,i))/(2*eps)-hr(block.native,f,k,j,i)));
  for(int m=0;m<10;++m)for(int f=0;f<25;++f)for(int k=0;k<cells;++k)for(int j=0;j<cells;++j)for(int i=0;i<cells;++i)
   if(m>=8||i<grid.ng||j<grid.ng||k<grid.ng||i>=grid.ng+grid.n||j>=grid.ng+grid.n||k>=grid.ng+grid.n)
    halo_error=std::max(halo_error,std::abs(hr(m,f,k,j,i)+37));
  if(positive)for(const auto &block:bm)for(int k=0;k<grid.n;++k)for(int j=0;j<grid.n;++j)for(int i=0;i<grid.n;++i){
   const size_t p=((block.composite*grid.n+k)*grid.n+j)*grid.n+i;const auto &x=grid.geometry[p].x;
   for(int f=0;f<22;++f){Real expected=dy[22*p+f];
    if(f==0||f==18){const Real value=f==0?.4+.03*x[0]+.01*x[1]*x[1]:.7+.04*x[2];
      if(positive==2)expected*=value;if(positive==3)expected/=2*value;}
    positive_rate_error=std::max(positive_rate_error,std::abs(hr(block.native,h::NativeCompositeFields::scalar_native[f],k+grid.ng,j+grid.ng,i+grid.ng)-expected));}}
  ++cases;
 }
 // Tiny representable positive fields have no prescribed radial exponent and
 // must not be floored. Keep their square and rates representable explicitly.
 for(int positive=1;positive<=3;++positive){h::NativeCompositeFields adapter(grid,true,0,-4,positive);
  const Real chi=1.e-80,alpha=2.e-70;std::vector<Real> y(22*points,0),dy(y.size(),0),b(3*points,0);
  for(size_t p=0;p<points;++p){const auto &x=grid.geometry[p].x;const Real A=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
   y[22*p+13]=3;y[22*p]=positive==1?chi:(positive==2?std::log(chi):chi*chi);
   y[22*p+18]=positive==1?alpha:(positive==2?std::log(alpha)-std::log(A):alpha*alpha);
   dy[22*p]=positive==1?.125*chi:(positive==2?.125:.25*chi*chi);
   dy[22*p+18]=positive==1?-.25*alpha:(positive==2?-.25:-.5*alpha*alpha);}
  adapter.Write(bulk,bm,y,b);const auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bulk);
  std::vector<Real> gathered,delta;adapter.Read(bulk,bm,gathered,delta);
  for(const auto &block:bm)for(int k=0;k<grid.n;++k)for(int j=0;j<grid.n;++j)for(int i=0;i<grid.n;++i){
   const size_t p=((block.composite*grid.n+k)*grid.n+j)*grid.n+i;
   for(int f:{0,18})tiny_coefficient_error=std::max(tiny_coefficient_error,std::abs(gathered[22*p+f]-y[22*p+f])/std::max(Real(1),std::abs(y[22*p+f])));}
  for(const auto &block:bm)for(int k=grid.ng;k<grid.ng+grid.n;++k)for(int j=grid.ng;j<grid.ng+grid.n;++j)for(int i=grid.ng;i<grid.ng+grid.n;++i){
   tiny_relative_error=std::max(tiny_relative_error,std::abs(host(block.native,z4c::Z4c::I_Z4C_CHI,k,j,i)/chi-1));
   tiny_relative_error=std::max(tiny_relative_error,std::abs(host(block.native,z4c::Z4c::I_Z4C_ALPHA,k,j,i)/alpha-1));}
  adapter.Write(bulk,bm,y,b,&dy);const auto rate=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bulk);
  for(const auto &block:bm)for(int k=grid.ng;k<grid.ng+grid.n;++k)for(int j=grid.ng;j<grid.ng+grid.n;++j)for(int i=grid.ng;i<grid.ng+grid.n;++i){
   tiny_relative_error=std::max(tiny_relative_error,std::abs(rate(block.native,z4c::Z4c::I_Z4C_CHI,k,j,i)/(.125*chi)-1));
   tiny_relative_error=std::max(tiny_relative_error,std::abs(rate(block.native,z4c::Z4c::I_Z4C_ALPHA,k,j,i)/(-.25*alpha)-1));}
  const std::vector<Real> saved_rate(rate.data(),rate.data()+rate.size());
  auto bad=y;bad[0]=positive==1?0:(positive==2?-std::numeric_limits<Real>::infinity():-1);
  bool rejected=false;try{adapter.Write(bulk,bm,bad,b);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("invalid positive state accepted");++invalid_storage;
  auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bulk);
  if(!std::equal(saved_rate.begin(),saved_rate.end(),after.data()))throw std::runtime_error("failed positive state write changed device state");
  if(positive==1){rejected=false;try{adapter.Write(bulk,bm,bad,b,&dy);}catch(const std::runtime_error&){rejected=true;}
   if(!rejected)throw std::runtime_error("zero active field accepted for direct positive rate");++invalid_storage;}
 }
 bool duplicate=false,wrong_chi=false;int bad_positive=0;
 try{h::NativeCompositeFields adapter(grid);std::vector<Real> y,b;adapter.Read(bulk,{{0,0},{0,1}},y,b);}catch(const std::invalid_argument&){duplicate=true;}
 try{h::NativeCompositeFields adapter(grid,false,0,4);}catch(const std::invalid_argument&){wrong_chi=true;}
 for(int bad:{-1,4})try{h::NativeCompositeFields adapter(grid,true,0,-4,bad);}catch(const std::invalid_argument&){++bad_positive;}
 try{h::NativeCompositeFields adapter(grid,false,0,-4,2);}catch(const std::invalid_argument&){++bad_positive;}
 if(roundtrip>2e-14||rate_error>1e-9||physical_error>2e-14||positive_rate_error>2e-14||tiny_relative_error>5e-14||tiny_coefficient_error>5e-14||invalid_storage!=4||halo_error!=0||!duplicate||!wrong_chi||bad_positive!=3)throw std::runtime_error("native field adapter audit failed");
 std::cout<<std::setprecision(17)<<"{\"chart_cases\":"<<cases<<",\"blocks\":"<<grid.mesh.blocks.size()<<",\"native_field_count\":25,\"roundtrip_max\":"<<roundtrip<<",\"physical_rate_FD_error\":"<<rate_error<<",\"independent_positive_state_error\":"<<physical_error<<",\"independent_positive_rate_error\":"<<positive_rate_error
  <<",\"tiny_positive_relative_error\":"<<tiny_relative_error<<",\"tiny_read_coefficient_error\":"<<tiny_coefficient_error
  <<",\"invalid_positive_storage_rejected\":"<<invalid_storage<<",\"positive_modes\":3,\"bad_positive_modes_rejected\":"<<bad_positive
  <<",\"untouched_ghost_and_block_error\":"<<halo_error<<",\"bad_layouts_rejected\":true}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
Kokkos::finalize();return status;}
