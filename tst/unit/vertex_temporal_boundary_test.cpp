#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include "driver/vertex_temporal_boundary.hpp"
#include "driver/hierarchy_temporal_ghosts.hpp"
void Check(bool ok) { if (!ok) throw std::runtime_error("temporal boundary regression"); }
template<class F> void Reject(F f) {
  bool rejected=false;try {f();} catch(const std::exception &) {rejected=true;} Check(rejected);
}
double Shape(double x,double y,int degree) {return 1+.02*std::pow(x,degree)+.003*std::pow(y,degree);}
void TestHierarchyScatter() {
  std::vector<subcycling::BlockKey> leaves;
  for(int x=0;x<4;++x) for(int y=0;y<4;++y) {
    if(x==1 && y==1) {
      for(int a=0;a<2;++a) for(int b=0;b<2;++b) leaves.push_back({1,2+a,2+b,0});
    } else leaves.push_back({0,x,y,0});
  }
  subcycling::Hierarchy tree(leaves,0,2);
  z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
  l.nx1=l.nx2=8;l.nx3=l.n3=1;l.ks=l.ke=0;
  l.is=l.js=2;l.ie=l.je=10;l.n1=l.n2=13;
  subcycling::HierarchyTemporalGhosts ghosts;ghosts.Build<6>(tree,l,1,0,4,4);
  Check(ghosts.TargetCount()>0 && ghosts.SourceBlocks().size()==16);
  DvceArray5D<Real> u("scatter state",tree.Nodes().size(),2,1,13,13);
  DvceArray5D<Real> rhs("scatter rhs",tree.Nodes().size(),2,1,13,13);
  auto initial=Kokkos::create_mirror_view(u);
  for(int n=0;n<u.extent_int(0);++n) for(int v=0;v<2;++v)
    for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
      const auto key=tree.Nodes()[n].key;
      initial(n,v,0,j,i)=key[0]==0 ? (v+1)*Shape(key[1]+(i-2)/8.,key[2]+(j-2)/8.,5) : -99;
    }
  Kokkos::deep_copy(u,initial);
  subcycling::RK4PredictorStates predictor;const double dt=.1;
  predictor.Begin(u,l,ghosts.SourceBlocks(),0,dt);
  const double factors[4]={1,1+.5*dt,1+.5*dt+.25*dt*dt,1+dt+.5*dt*dt+.25*dt*dt*dt};
  auto hr=Kokkos::create_mirror_view(rhs);
  for(int stage=1;stage<=4;++stage) {
    for(std::size_t k=0;k<rhs.size();++k) hr.data()[k]=initial.data()[k]*factors[stage-1];
    Kokkos::deep_copy(rhs,hr);predictor.Capture(rhs,stage);
  }
  for(double q:{0.,.5}) for(int stage=1;stage<=4;++stage) {
    Kokkos::deep_copy(u,initial);
    ghosts.Apply(predictor,q,dt/2,stage,u);
    auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u);
    int changed=0;
    for(int n=0;n<u.extent_int(0);++n) for(int v=0;v<2;++v)
      for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
        const auto key=tree.Nodes()[n].key;
        const int x=key[1]*8+i-2,y=key[2]*8+j-2;
        // This refined patch occupies [16,32] squared in fine vertex units.
        const bool target=key[0]==1 && (x<16 || x>32 || y<16 || y>32);
        if(!target) {Check(result(n,v,0,j,i)==initial(n,v,0,j,i));continue;}
        ++changed;
        // Analytic fine RK stages starting from exp(q*dt). The coarse dense
        // predictor has O(dt^4) error; spatial degree-five data interpolate exactly.
        const double f=dt/2;
        const double fine[4]={1,1+.5*f,1+.5*f+.25*f*f,1+f+.5*f*f+.25*f*f*f};
        const double exact=(v+1)*Shape(x/16.,y/16.,5)*std::exp(q*dt)*fine[stage-1];
        Check(std::abs(result(n,v,0,j,i)-exact)<2e-5);
        if(q==0 && stage==1) Check(std::abs(result(n,v,0,j,i)-exact)<2e-14);
      }
    Check(changed==2*ghosts.TargetCount());
  }
  DvceArray5D<Real> wrong("wrong component count",tree.Nodes().size(),1,1,13,13);
  Kokkos::deep_copy(wrong,-17.);
  Reject([&]{ghosts.Apply(predictor,0,dt/2,1,wrong);});
  auto unchanged=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),wrong);
  for(std::size_t k=0;k<wrong.size();++k) Check(unchanged.data()[k]==-17);
  Reject([&]{ghosts.Build<6>(tree,l,0,0,4,4);});
  Reject([&]{ghosts.Apply(predictor,0,dt/2,1,u);});
}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    TestHierarchyScatter();
    std::vector<subcycling::BlockKey> keys;
    for(int j=0;j<3;++j) for(int i=0;i<3;++i) keys.push_back({2,i,j,0});
    // Deliberately reorder keys and source mapping. Stencils cross faces and corners.
    std::reverse(keys.begin(),keys.end());
    std::vector<int> ids{2,5,1,7,8,3,0,4,6};
    std::vector<subcycling::FineVertex2D> targets{{15,15},{16,16},{17,17},{31,33}};
    subcycling::VertexTemporalBoundary plan;
    plan.Build<6>(keys,ids,8,8,targets);
    z4c::Z4cGridLayout l;l.is=l.js=2;l.ie=l.je=10;l.ks=l.ke=0;
    DvceArray5D<Real> state("state",9,1,1,13,13),rhs("rhs",9,1,1,13,13);
    auto h=Kokkos::create_mirror(state);
    DvceArray2D<Real> out;
    for (int order : {4,6,8}) {
    for(int b=0;b<9;++b) for(int j=0;j<13;++j) for(int i=0;i<13;++i)
      h(ids[b],0,0,j,i)=Shape(keys[b][1]+(i-2)/8.,keys[b][2]+(j-2)/8.,order-1);
    if(order==4) plan.Build<4>(keys,ids,8,8,targets);
    if(order==6) plan.Build<6>(keys,ids,8,8,targets);
    if(order==8) plan.Build<8>(keys,ids,8,8,targets);
    std::vector<double> errors;
    for(double dt : {.2,.1,.05,.025}) {
      Kokkos::deep_copy(state,h);
      subcycling::RK4PredictorStates predictor;predictor.Begin(state,l,ids,0,dt);
      double factor[4]={1,1+.5*dt,1+.5*dt+.25*dt*dt,1+dt+.5*dt*dt+.25*dt*dt*dt};
      for(int stage=1;stage<=4;++stage) {
        auto hr=Kokkos::create_mirror_view(rhs);
        for(int b=0;b<9;++b) for(int j=0;j<13;++j) for(int i=0;i<13;++i)
          hr(b,0,0,j,i)=h(b,0,0,j,i)*factor[stage-1];
        Kokkos::deep_copy(rhs,hr);predictor.Capture(rhs,stage);
      }
      double error=0;
      for(double q : {0.,.5}) for(int stage=1;stage<=4;++stage) {
        plan.Evaluate(predictor,q,dt/2,stage,out);
        auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
        const double f=dt/2;
        double fine_factor[4]={1,1+.5*f,1+.5*f+.25*f*f,1+f+.5*f*f+.25*f*f*f};
        for(int p=0;p<4;++p) {
          const double exact=Shape(targets[p][0]/16.,targets[p][1]/16.,order-1)*
                             std::exp(q*dt)*fine_factor[stage-1];
          error=std::max(error,std::abs(ho(p,0)-exact));
          if(q==0 && stage==1) Check(std::abs(ho(p,0)-exact)<2e-14);
        }
      }
      errors.push_back(error);
      // Reject an otherwise identical predictor with a different donor ordering.
      auto wrong=ids;std::swap(wrong[0],wrong[1]);predictor.Begin(state,l,wrong,0,dt);
      Reject([&]{plan.Evaluate(predictor,0,dt/2,1,out);});
    }
    for(int n=0;n<3;++n) {const double ratio=errors[n]/errors[n+1];
      std::cout << "stage interpolation ratio " << ratio << '\n';Check(ratio>12 && ratio<20);}
    }
    Reject([&]{plan.Build<6>(keys,ids,8,8,{{1,1}});});
    subcycling::RK4PredictorStates invalid;
    Reject([&]{plan.Evaluate(invalid,0,.1,1,out);});
    Reject([&]{plan.Build<6>(keys,ids,8,8,{{-1,16}});});
    std::cout << "PASS: native spatial stencil plus temporal predictor across faces/corners; missing donors rejected\n";
  }
  Kokkos::finalize();
}
