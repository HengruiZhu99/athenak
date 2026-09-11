#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/vertex_parent_states.hpp"
void Check(bool ok) { if (!ok) throw std::runtime_error("parent state regression"); }
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    std::vector<subcycling::BlockKey> leaves{{1,2,0,0},{1,3,0,0},{1,2,1,0},{1,3,1,0},{1,1,0,0},{1,0,1,0},{1,1,1,0},
        {2,0,0,0},{2,1,0,0},{2,0,1,0},{2,1,1,0}};
    subcycling::Hierarchy hierarchy(leaves,0,2);
    z4c::Z4cGridLayout l;
    l.centering=z4c::Z4cGridCentering::vertex;
    l.nx1=l.nx2=8; l.nx3=1; l.is=l.js=2; l.ie=l.je=10;
    l.ks=l.ke=0; l.n1=l.n2=13; l.n3=1;
    DvceArray5D<Real> u("leaf test state",leaves.size(),2,1,13,13);
    auto host=Kokkos::create_mirror_view(u);
    for (int m=0; m<static_cast<int>(leaves.size()); ++m) {
      const double width=std::ldexp(1.0,-leaves[m][0]);
      for (int v=0; v<2; ++v) for (int j=0; j<13; ++j) for (int i=0; i<13; ++i) {
        const double x=width*(leaves[m][1]+(i-2)/8.0);
        const double z=width*(leaves[m][2]+(j-2)/8.0);
        host(m,v,0,j,i)=v+x*x+z*z*z;
      }
    }
    Kokkos::deep_copy(u,host);
    subcycling::VertexParentStates parents;
    parents.Initialize(hierarchy,u,l);
    Check(parents.Values().extent(0)==3);
    auto coarse=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),parents.Values());
    for (int p=0; p<3; ++p) {
      auto key=hierarchy.Nodes()[parents.ParentNodes()[p]].key;
      const double width=std::ldexp(1.0,-key[0]);
      for (int v=0; v<2; ++v) for (int j=0; j<13; ++j) for (int i=0; i<13; ++i) {
        if (i<2 || i>10 || j<2 || j>10) { Check(std::isnan(coarse(p,v,0,j,i))); continue; }
        const double x=width*(key[1]+(i-2)/8.0),z=width*(key[2]+(j-2)/8.0);
        Check(coarse(p,v,0,j,i)==v+x*x+z*z*z);
      }
    }
    auto coverage=parents.FillSameLevelGhosts(hierarchy,u,l);
    Check(coverage.copied>0 && coverage.unavailable>0);
    auto filled=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),parents.Values());
    int copied=0, unavailable=0;
    for (int p=0; p<3; ++p) {
      auto key=hierarchy.Nodes()[parents.ParentNodes()[p]].key;
      const double width=std::ldexp(1.0,-key[0]);
      for (int v=0; v<2; ++v) for (int j=0; j<13; ++j) for (int i=0; i<13; ++i) {
        const bool ghost=i<2 || i>10 || j<2 || j>10;
        if (std::isnan(filled(p,v,0,j,i))) { Check(ghost); ++unavailable; continue; }
        if (ghost) ++copied;
        const double x=width*(key[1]+(i-2)/8.0),z=width*(key[2]+(j-2)/8.0);
        Check(filled(p,v,0,j,i)==v+x*x+z*z*z);
      }
    }
    Check(copied==2*coverage.copied && unavailable==2*coverage.unavailable);
    auto invalid=l; ++invalid.is;
    bool rejected=false;
    try { parents.FillSameLevelGhosts(hierarchy,u,invalid); }
    catch (const std::invalid_argument &) { rejected=true; }
    Check(rejected);
    auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    for (std::size_t i=0; i<u.size(); ++i) Check(after.data()[i]==host.data()[i]);
    std::cout << "PASS: parent injection, same-level ghosts, missing-donor accounting, unchanged leaves\n";
  }
  Kokkos::finalize();
}
