// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/native_composite_fields.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2));
 const int cells=grid.n+2*grid.ng;const size_t points=grid.active.size();
 DvceArray5D<Real> bulk("native bulk",10,25,cells,cells,cells),shell("native shells",12,25,cells,cells,cells);
 std::vector<h::NativeCompositeFields::Block> bm,sm;
 for(int m=0;m<8;++m)bm.push_back({7-m,m});for(int m=0;m<12;++m)sm.push_back({11-m,8+m});
 Real roundtrip=0,rate_error=0,halo_error=0;int cases=0;
 for(int mode=-1;mode<3;++mode){const bool puncture=mode>=0;h::NativeCompositeFields adapter(grid,puncture,std::max(0,mode));
  std::vector<Real> y(22*points),b(3*points),dy(y.size());
  for(size_t p=0;p<points;++p){const auto &x=grid.geometry[p].x;
   for(int f=0;f<22;++f){y[22*p+f]=.001*(f+1)*(1+x[f%3]);dy[22*p+f]=.0007*(f+1);}
   if(puncture)y[22*p+13]=2.8;
   for(int v=0;v<3;++v)b[3*p+v]=.003*(v+1)*(1+x[v]);}
  Kokkos::deep_copy(bulk,-91.);Kokkos::deep_copy(shell,-91.);
  adapter.Write(bulk,bm,y,b);adapter.Write(shell,sm,y,b);
  std::vector<Real> again,bb;adapter.Read(shell,sm,again,bb);adapter.Read(bulk,bm,again,bb);
  for(size_t p=0;p<y.size();++p)roundtrip=std::max(roundtrip,std::abs(y[p]-again[p]));
  for(size_t p=0;p<b.size();++p)roundtrip=std::max(roundtrip,std::abs(b[p]-bb[p]));
  const auto saved=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bulk);
  for(int m=0;m<10;++m)for(int f=0;f<25;++f)for(int k=0;k<cells;++k)for(int j=0;j<cells;++j)for(int i=0;i<cells;++i)
   if(m>=8||i<grid.ng||j<grid.ng||k<grid.ng||i>=grid.ng+grid.n||j>=grid.ng+grid.n||k>=grid.ng+grid.n)
    halo_error=std::max(halo_error,std::abs(saved(m,f,k,j,i)+91));
  DvceArray5D<Real> plus("plus",10,25,cells,cells,cells),minus("minus",10,25,cells,cells,cells),rate("rate",10,25,cells,cells,cells);
  const Real eps=1e-5;auto yp=y,ym=y;for(size_t p=0;p<y.size();++p){yp[p]+=eps*dy[p];ym[p]-=eps*dy[p];}
  adapter.Write(plus,bm,yp,b);adapter.Write(minus,bm,ym,b);adapter.Write(rate,bm,y,b,&dy);
  const auto hp=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),plus),hm=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),minus),hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rate);
  for(const auto &block:bm)for(int f=0;f<25;++f)for(int k=grid.ng;k<grid.ng+grid.n;++k)for(int j=grid.ng;j<grid.ng+grid.n;++j)for(int i=grid.ng;i<grid.ng+grid.n;++i)
   rate_error=std::max(rate_error,std::abs((hp(block.native,f,k,j,i)-hm(block.native,f,k,j,i))/(2*eps)-hr(block.native,f,k,j,i)));
  ++cases;
 }
 bool duplicate=false,wrong_chi=false;
 try{h::NativeCompositeFields adapter(grid);std::vector<Real> y,b;adapter.Read(bulk,{{0,0},{0,1}},y,b);}catch(const std::invalid_argument&){duplicate=true;}
 try{h::NativeCompositeFields adapter(grid,false,0,4);}catch(const std::invalid_argument&){wrong_chi=true;}
 if(roundtrip>2e-14||rate_error>1e-9||halo_error!=0||!duplicate||!wrong_chi)throw std::runtime_error("native field adapter audit failed");
 std::cout<<std::setprecision(17)<<"{\"chart_cases\":"<<cases<<",\"blocks\":"<<grid.mesh.blocks.size()<<",\"native_field_count\":25,\"roundtrip_max\":"<<roundtrip<<",\"physical_rate_FD_error\":"<<rate_error<<",\"untouched_ghost_and_block_error\":"<<halo_error<<",\"bad_layouts_rejected\":true}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";status=1;}
Kokkos::finalize();return status;}
