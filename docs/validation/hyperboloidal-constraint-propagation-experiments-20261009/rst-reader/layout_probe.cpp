#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include "mesh/mesh.hpp"
#include "outputs/io_wrapper.hpp"
#include "z4c/z4c.hpp"
int main() {
 const std::uint16_t one=1;
 std::cout<<"{\"little_endian\":"<<(*reinterpret_cast<const unsigned char*>(&one)==1)
  <<",\"Real\":"<<sizeof(Real)<<",\"int\":"<<sizeof(int)<<",\"float\":"<<sizeof(float)
  <<",\"IOWrapperSizeT\":"<<sizeof(IOWrapperSizeT)<<",\"RegionSize\":"<<sizeof(RegionSize)
  <<",\"RegionIndcs\":"<<sizeof(RegionIndcs)<<",\"LogicalLocation\":"<<sizeof(LogicalLocation)
  <<",\"nz4c\":"<<z4c::Z4c::nz4c
  <<",\"LayoutRight\":"<<std::is_same<LayoutWrapper,Kokkos::LayoutRight>::value
  <<",\"RegionSize_offsets\":{\"x1min\":"<<offsetof(RegionSize,x1min)
  <<",\"x2min\":"<<offsetof(RegionSize,x2min)<<",\"x3min\":"<<offsetof(RegionSize,x3min)
  <<",\"x1max\":"<<offsetof(RegionSize,x1max)<<",\"x2max\":"<<offsetof(RegionSize,x2max)
  <<",\"x3max\":"<<offsetof(RegionSize,x3max)<<",\"dx1\":"<<offsetof(RegionSize,dx1)
  <<",\"dx2\":"<<offsetof(RegionSize,dx2)<<",\"dx3\":"<<offsetof(RegionSize,dx3)
  <<"},\"RegionIndcs_offsets\":{";
#define OFFSET(name) <<"\"" #name "\":"<<offsetof(RegionIndcs,name)
 std::cout OFFSET(ng)<<',' OFFSET(nx1)<<',' OFFSET(nx2)<<',' OFFSET(nx3)
  <<',' OFFSET(is)<<',' OFFSET(ie)<<',' OFFSET(js)<<',' OFFSET(je)<<',' OFFSET(ks)<<',' OFFSET(ke)
  <<',' OFFSET(cnx1)<<',' OFFSET(cnx2)<<',' OFFSET(cnx3)<<',' OFFSET(cis)<<',' OFFSET(cie)
  <<',' OFFSET(cjs)<<',' OFFSET(cje)<<',' OFFSET(cks)<<',' OFFSET(cke)
  <<"},\"LogicalLocation_offsets\":{\"lx1\":"<<offsetof(LogicalLocation,lx1)
  <<",\"lx2\":"<<offsetof(LogicalLocation,lx2)<<",\"lx3\":"<<offsetof(LogicalLocation,lx3)
  <<",\"level\":"<<offsetof(LogicalLocation,level)<<"}}\n";
}
