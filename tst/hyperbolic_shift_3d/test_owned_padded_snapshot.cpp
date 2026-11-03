#include <iostream>
#include "z4c/hyperboloidal/owned_padded_snapshot.hpp"
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int code=0;
  try{
    using S=z4c::hyperboloidal::OwnedPaddedSnapshot;
    constexpr int bs=64,count=bs*9,fields=31;
    S::Buffer source("fixture",fields*count);auto host=Kokkos::create_mirror_view(source);
    for(auto blocks:{std::vector<int>{0,8,3,3},std::vector<int>{},std::vector<int>{1,2,3,4,5,6,7,8}}){
      S snapshot(count,bs,fields,blocks);std::vector<Real> output;
      for(int stage=0;stage<3;++stage){
        for(int i=0;i<fields*count;++i)host(i)=stage*100000+i+.125;
        Kokkos::deep_copy(source,host);snapshot.Download(source,output,true);
        for(int f=0;f<fields;++f)for(int b=0;b<9;++b)for(int i=0;i<bs;++i){
          const int at=f*count+b*bs+i;const bool owned=std::find(blocks.begin(),blocks.end(),b)!=blocks.end();
          if(owned?output[at]!=host(at):!std::isnan(output[at]))throw std::runtime_error("stale, missing or extraneous diagnostic value");
        }
      }
    }
    bool rejected=false;try{S bad(count,bs,fields,{9});}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid block accepted");
    std::cout<<"exact diagnostic blocks: all31, gaps, duplicate/empty ownership, three changed stages, poisoning passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
  Kokkos::finalize();return code;
}
