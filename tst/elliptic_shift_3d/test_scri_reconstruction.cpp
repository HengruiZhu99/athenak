// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/scri_surface.hpp"
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{.65,1},1,8));
 std::vector<Real> volume(grid.count,0);const Real coefficient[8]={1,.2,.1,.03,.04,.02,.01,.005};
 auto polynomial=[&](Real r){Real f=0;for(int p=7;p>=0;--p)f=f*r+coefficient[p];return f;};
 for(size_t p=0;p<grid.active.size();++p){const auto &x=grid.geometry[p].x;volume[grid.active[p]]=polynomial(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]));}
 Real errors[2]{};Real f=0,df=0,ddf=0;
 for(int p=0;p<8;++p){f+=coefficient[p];df+=p*coefficient[p];ddf+=p*(p-1)*coefficient[p];}
 for(int method=0;method<2;++method){cubed_sphere::ScriSurface surface(grid,1,method?8:6);std::fill(surface.values.begin(),surface.values.end(),f);
  for(size_t p=0;p<surface.rows.size();++p){const auto &x=surface.rows[p].geometry.x;const auto jet=surface.Sample(p,volume,surface.values,1);
   errors[method]=std::max(errors[method],std::abs(jet.value[0]-f));
   for(int a=0;a<3;++a){errors[method]=std::max(errors[method],std::abs(jet.d[0][a]-df*x[a]));for(int b=0;b<3;++b)
    errors[method]=std::max(errors[method],std::abs(jet.dd[0][a][b]-(ddf*x[a]*x[b]+df*((a==b?1.:0)-x[a]*x[b]))));}
  }
 }
 if(errors[1]>3e-10||errors[0]<100*errors[1])throw std::runtime_error("scri reconstruction degree/chain-rule audit");
 std::cout<<std::setprecision(17)<<"{\"degree5_error\":"<<errors[0]<<",\"degree7_error\":"<<errors[1]<<",\"full_second_chain_rule\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
