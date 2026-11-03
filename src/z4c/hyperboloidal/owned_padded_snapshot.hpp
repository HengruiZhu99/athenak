// Persistent exact-block diagnostic download. Logical padded indexing is retained.
#ifndef Z4C_HYPERBOLOIDAL_OWNED_PADDED_SNAPSHOT_HPP_
#define Z4C_HYPERBOLOIDAL_OWNED_PADDED_SNAPSHOT_HPP_
#include <algorithm>
#include <limits>
#include <vector>
#include "athena.hpp"
namespace z4c::hyperboloidal {
class OwnedPaddedSnapshot {
 public:
  using Space=Kokkos::DefaultExecutionSpace::memory_space;
  using Buffer=Kokkos::View<Real*,Space>;
  using ConstBuffer=Kokkos::View<const Real*,Space>;
  using Host=Kokkos::View<Real*,Kokkos::HostSpace>;
  using Indices=Kokkos::View<int*,Space>;
 private:
  int count_,blocksize_,fields_;
  std::vector<int> blocks_;
  Indices ids_;
  Buffer packed_;
  Host host_;
 public:
  OwnedPaddedSnapshot(int count,int blocksize,int fields,std::vector<int> blocks)
      :count_(count),blocksize_(blocksize),fields_(fields),blocks_(std::move(blocks)){
    if(count<0||blocksize<=0||count%blocksize||fields<=0)
      throw std::invalid_argument("invalid diagnostic padded shape");
    std::sort(blocks_.begin(),blocks_.end());
    blocks_.erase(std::unique(blocks_.begin(),blocks_.end()),blocks_.end());
    for(int b:blocks_)if(b<0||b>=count/blocksize)throw std::invalid_argument("invalid diagnostic block");
    // On CPU the source is already host accessible; avoid an unnecessary pack.
    if constexpr(!Kokkos::SpaceAccessibility<Kokkos::HostSpace,Space>::accessible){
      ids_=Indices("diagnostic selected blocks",blocks_.size());
      auto ids=Kokkos::create_mirror_view(ids_);for(size_t i=0;i<blocks_.size();++i)ids(i)=blocks_[i];
      Kokkos::deep_copy(ids_,ids);
      packed_=Buffer("diagnostic packed fields",size_t(fields)*blocksize*blocks_.size());
      host_=Host("diagnostic persistent host fields",packed_.extent(0));
    }
  }
  size_t Values()const{return size_t(fields_)*blocksize_*blocks_.size();}
  void Download(ConstBuffer source,std::vector<Real>&destination,bool poison=false){
    if(source.extent(0)!=size_t(fields_)*count_)throw std::invalid_argument("diagnostic source shape");
    if(destination.size()!=source.extent(0))destination.assign(source.extent(0),0);
    if(poison)std::fill(destination.begin(),destination.end(),std::numeric_limits<Real>::quiet_NaN());
    if constexpr(Kokkos::SpaceAccessibility<Kokkos::HostSpace,Space>::accessible){
      // Match deep_copy's completion contract, including asynchronous host backends.
      Kokkos::fence("diagnostic host source ready");
      for(int f=0;f<fields_;++f)for(int b:blocks_){const size_t offset=size_t(f)*count_+size_t(b)*blocksize_;
        std::copy_n(source.data()+offset,blocksize_,destination.data()+offset);}
    }else{
      const int bs=blocksize_,count=count_,selected=blocks_.size();const auto ids=ids_;const auto packed=packed_;
      Kokkos::parallel_for("pack exact diagnostic blocks",Kokkos::RangePolicy<>(0,Values()),KOKKOS_LAMBDA(size_t i){
        const size_t cell=i%bs,slot=(i/bs)%selected,field=i/(size_t(bs)*selected);
        packed(i)=source(field*count+ids(slot)*bs+cell);
      });
      Kokkos::deep_copy(host_,packed_);
      for(int f=0;f<fields_;++f)for(size_t b=0;b<blocks_.size();++b)
        std::copy_n(host_.data()+(size_t(f)*blocks_.size()+b)*blocksize_,blocksize_,
                    destination.data()+size_t(f)*count_+size_t(blocks_[b])*blocksize_);
    }
  }
};
}
#endif
