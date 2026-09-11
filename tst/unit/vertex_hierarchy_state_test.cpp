#include <cmath>
#include <iostream>
#include <stdexcept>
#include "driver/vertex_parent_states.hpp"
#include "driver/hierarchy_geometry.hpp"
#include "driver/hierarchy_vertex_exchange.hpp"
void Check(bool ok) {if(!ok) throw std::runtime_error("hierarchy state regression");}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    std::vector<subcycling::BlockKey> leaves{{1,2,0,0},{1,3,0,0},{1,2,1,0},{1,3,1,0},
      {1,1,0,0},{1,0,1,0},{1,1,1,0},{2,0,0,0},{2,1,0,0},{2,0,1,0},{2,1,1,0}};
    subcycling::Hierarchy tree(leaves,0,2);
    z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
    l.nx1=l.nx2=8;l.nx3=l.n3=1;l.ks=l.ke=0;l.is=l.js=2;l.ie=l.je=10;l.n1=l.n2=13;
    subcycling::HierarchyGeometry geometry;
    RegionSize domain{};domain.x1min=0;domain.x1max=2;
    domain.x2min=-.5;domain.x2max=.5;domain.x3min=-.5;domain.x3max=.5;
    const std::array<BoundaryFlag,6> faces{BoundaryFlag::axis,BoundaryFlag::outflow,
      BoundaryFlag::outflow,BoundaryFlag::outflow,BoundaryFlag::periodic,BoundaryFlag::periodic};
    geometry.Initialize(tree,domain,0,2,1,l,faces);
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;const double width=std::ldexp(1.,-key[0]);
      const auto &g=geometry.sizes.h_view(n);
      Check(g.x1min==width*key[1] && g.x1max==width*(key[1]+1));
      Check(g.x2min==-.5+width*key[2] && g.x2max==-.5+width*(key[2]+1));
      Check(g.dx1==width/8 && g.dx2==width/8 && g.dx3==1);
      Check(geometry.boundaries.h_view(n,0)==(key[1]==0 ? BoundaryFlag::axis : BoundaryFlag::block));
      Check(geometry.boundaries.h_view(n,1)==(key[1]+1==(2<<key[0]) ? BoundaryFlag::outflow : BoundaryFlag::block));
      Check(geometry.boundaries.h_view(n,2)==(key[2]==0 ? BoundaryFlag::outflow : BoundaryFlag::block));
      Check(geometry.boundaries.h_view(n,3)==(key[2]+1==(1<<key[0]) ? BoundaryFlag::outflow : BoundaryFlag::block));
    }
    bool bad_domain=false;
    try {geometry.Initialize(tree,domain,0,1,1,l,faces);}
    catch(const std::invalid_argument &) {bad_domain=true;}
    Check(bad_domain && geometry.sizes.extent_int(0)==static_cast<int>(tree.Nodes().size()));
    // Give coincident active vertices deliberately different values. Resolve
    // expected authority independently by scanning containing same-level nodes.
    subcycling::HierarchyVertexExchange exchange;
    const auto exchange_coverage=exchange.Build(tree,l);
    Check(exchange_coverage.shared>0 && exchange_coverage.ghosts>0 &&
          exchange_coverage.unavailable>0);
    DvceArray5D<Real> stage("exchange stage",tree.Nodes().size(),2,1,l.n2,l.n1);
    auto seed=Kokkos::create_mirror_view(stage);
    for(int n=0;n<stage.extent_int(0);++n) for(int v=0;v<2;++v)
      for(int j=0;j<l.n2;++j) for(int i=0;i<l.n1;++i)
        seed(n,v,0,j,i)=(i>=l.is && i<=l.ie && j>=l.js && j<=l.je)
          ? 100000*n+10000*v+100*j+i : -99;
    Kokkos::deep_copy(stage,seed);
    exchange.Apply(stage,1,1);
    auto received=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),stage);
    for(int n=0;n<stage.extent_int(0);++n) for(int v=0;v<2;++v)
      for(int j=0;j<l.n2;++j) for(int i=0;i<l.n1;++i) {
        const auto key=tree.Nodes()[n].key;
        int donor=-1,di=0,dj=0;
        const int gx=key[1]*l.nx1+i-l.is,gy=key[2]*l.nx2+j-l.js;
        if(key[0]==1) for(int d=0;d<stage.extent_int(0);++d) {
          const auto k=tree.Nodes()[d].key;
          if(k[0]!=key[0]) continue;
          const int x=gx-k[1]*l.nx1+l.is,y=gy-k[2]*l.nx2+l.js;
          if(x<l.is || x>l.ie || y<l.js || y>l.je) continue;
          if(donor<0 || k<tree.Nodes()[donor].key) {donor=d;di=x;dj=y;}
        }
        const Real expected=donor<0 ? seed(n,v,0,j,i) : seed(donor,v,0,dj,di);
        Check(received(n,v,0,j,i)==expected);
      }
    // Repeated application must be idempotent and use the cached topology.
    exchange.Apply(stage,1,1);
    auto again=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),stage);
    for(std::size_t k=0;k<stage.size();++k) Check(again.data()[k]==received.data()[k]);
    bool bad_exchange=false;
    try {exchange.Apply(stage,2,1);} catch(const std::invalid_argument &) {bad_exchange=true;}
    Check(bad_exchange);
    auto invalid_layout=l;invalid_layout.nx3=2;
    try {exchange.Build(tree,invalid_layout);Check(false);}
    catch(const std::invalid_argument &) {}
    bad_exchange=false;
    try {exchange.Apply(stage,1,1);} catch(const std::invalid_argument &) {bad_exchange=true;}
    Check(bad_exchange);
    DvceArray5D<Real> u("leaves",leaves.size()+20,2,1,13,13);
    auto h=Kokkos::create_mirror(u);Kokkos::deep_copy(h,-3.0);
    for(int m=0;m<static_cast<int>(leaves.size());++m) {
      double width=std::ldexp(1.,-leaves[m][0]);
      for(int v=0;v<2;++v) for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
        double x=width*(leaves[m][1]+(i-2)/8.),z=width*(leaves[m][2]+(j-2)/8.);
        h(m,v,0,j,i)=v+x*x+z*z*z;
      }
    }
    Kokkos::deep_copy(u,h);
    subcycling::VertexParentStates state;state.InitializeAll(tree,u,l);
    Check(state.StoredNodes().size()==tree.Nodes().size());
    auto all=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state.Values());
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto node=tree.Nodes()[n];Check(state.StoredNodes()[n]==n);
      Check(state.AllLevels().h_view(n)==node.key[0]);
      const double width=std::ldexp(1.,-node.key[0]);
      for(int v=0;v<2;++v) for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
        if(i<2 || i>10 || j<2 || j>10) {Check(std::isnan(all(n,v,0,j,i)));continue;}
        const double x=width*(node.key[1]+(i-2)/8.),z=width*(node.key[2]+(j-2)/8.);
        Check(all(n,v,0,j,i)==v+x*x+z*z*z);
        if(!node.Covered()) all(n,v,0,j,i)+=7;
      }
    }
    Kokkos::deep_copy(state.Values(),all);
    // Simulate changed leaf states. Restriction must read the owned hierarchy
    // fields, not the unchanged external production leaf arrays.
    const auto *storage=state.Values().data();
    state.RestrictLevel(l,1);
    auto middle=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state.Values());
    Check(middle(0,0,0,2,2)==0); // coarse root unchanged until its synchronization
    state.RestrictLevel(l,0);Check(storage==state.Values().data());
    auto final=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state.Values());
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;const double width=std::ldexp(1.,-key[0]);
      for(int v=0;v<2;++v) for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
        if(i<2 || i>10 || j<2 || j>10) {Check(std::isnan(final(n,v,0,j,i)));continue;}
        double x=width*(key[1]+(i-2)/8.),z=width*(key[2]+(j-2)/8.);
        Check(final(n,v,0,j,i)==v+x*x+z*z*z+7);
      }
    }
    const auto coverage=state.FillSameLevelGhosts(tree,u,l,1,1);Check(coverage.copied>0);
    auto ghosts=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state.Values());
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) {
      const auto key=tree.Nodes()[n].key;const double width=std::ldexp(1.,-key[0]);
      for(int v=0;v<2;++v) for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
        if(i>=2 && i<=10 && j>=2 && j<=10) continue;
        if(key[0]!=1) Check(std::isnan(ghosts(n,v,0,j,i)));
        else if(std::isfinite(ghosts(n,v,0,j,i))) {
          double x=width*(key[1]+(i-2)/8.),z=width*(key[2]+(j-2)/8.);
          Check(ghosts(n,v,0,j,i)==v+x*x+z*z*z+7);
        }
      }
    }
    state.FillPhysicalGhosts<4>(l,0,2,1,{true,true,true,true},1,1);
    state.FillAxisAtLogicalZero(l,{1,1},2,2);
    auto selective=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state.Values());
    for(int n=0;n<static_cast<int>(tree.Nodes().size());++n) if(tree.Nodes()[n].key[0]==0)
      for(int v=0;v<2;++v) for(int j=0;j<13;++j) for(int i=0;i<13;++i)
        if(i<2 || i>10 || j<2 || j>10) Check(std::isnan(selective(n,v,0,j,i)));
    // Explicit synchronized scatter writes active leaves only, preserving ghosts.
    auto before=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    for(std::size_t i=0;i<u.size();++i) Check(before.data()[i]==h.data()[i]);
    state.CopyLeavesTo(l,u);
    auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    for(int m=0;m<static_cast<int>(leaves.size());++m) for(int v=0;v<2;++v)
      for(int j=0;j<13;++j) for(int i=0;i<13;++i)
        Check(after(m,v,0,j,i)==h(m,v,0,j,i)+(i>=2 && i<=10 && j>=2 && j<=10 ? 7 : 0));
    for(int m=leaves.size();m<u.extent_int(0);++m) for(int v=0;v<2;++v)
      for(int j=0;j<13;++j) for(int i=0;i<13;++i) Check(after(m,v,0,j,i)==-3);
    state.Initialize(tree,u,l);
    bool rejected=false;try{state.RestrictLevel(l,0);}catch(const std::invalid_argument &){rejected=true;}
    Check(rejected);
    std::cout << "PASS: populated hierarchy, selective restriction/ghost fills, explicit leaf scatter\n";
  }
  Kokkos::finalize();
}
