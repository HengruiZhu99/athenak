#include <cmath>
#include <iostream>
#include <cstring>
#include "driver/hierarchy_rk4.hpp"
void Check(bool ok) {if(!ok) throw std::runtime_error("hierarchy RK regression");}
struct Transport {
  subcycling::VertexParentStates &storage;
  z4c::Z4cGridLayout l;
  int calls[2]={0,0};
  void Prepare(const subcycling::StepContext &s,int stage,
               const subcycling::BlockBatches &,const DvceArray5D<Real> &) {
    storage.FillPhysicalGhosts<4>(l,0,4,4,{{true,true,true,true}},s.minimum_level,s.maximum_level);
  }
  void RHS(const subcycling::StepContext &s,int stage,const subcycling::BlockBatches &b,
           const DvceArray5D<Real> &u,const DvceArray5D<Real> &rhs) {
    ++calls[s.maximum_level];
    const Real dx=std::ldexp(1./8,-s.maximum_level);
    b.For5("transport and reaction",0,0,0,0,l.js,l.je,l.is,l.ie,
      KOKKOS_LAMBDA(int m,int v,int k,int j,int i) {
        rhs(m,v,k,j,i)=u(m,v,k,j,i)+.1*(u(m,v,k,j,i+1)-u(m,v,k,j,i-1)+
                                      u(m,v,k,j+1,i)-u(m,v,k,j-1,i))/(2*dx);
      });
  }
  void Project(const subcycling::StepContext &,int,const subcycling::BlockBatches &,
               const DvceArray5D<Real> &) {}
};
int main(int argc,char **argv) {
 Kokkos::initialize(argc,argv);
 {
  std::vector<subcycling::BlockKey> leaves;
  for(int x=0;x<4;++x) for(int y=0;y<4;++y) {
    if(x==1 && y==1) for(int a=0;a<2;++a) for(int b=0;b<2;++b)
      leaves.push_back({1,2+a,2+b,0});
    else leaves.push_back({0,x,y,0});
  }
  subcycling::Hierarchy tree(leaves,0,2);
  z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
  l.nx1=l.nx2=8;l.nx3=l.n3=1;l.ks=l.ke=0;
  l.is=l.js=2;l.ie=l.je=10;l.n1=l.n2=13;
  std::vector<double> errors;
  for(int steps:{2,4,8,16}) {
    DvceArray5D<Real> leaf("initial leaves",leaves.size(),1,1,13,13);
    auto seed=Kokkos::create_mirror_view(leaf);
    for(int m=0;m<leaf.extent_int(0);++m) for(int j=0;j<13;++j) for(int i=0;i<13;++i) {
      const auto key=leaves[m];const double w=std::ldexp(1.,-key[0]);
      seed(m,0,0,j,i)=1+.1*w*(key[1]+(i-2)/8.)+.2*w*(key[2]+(j-2)/8.);
    }
    Kokkos::deep_copy(leaf,seed);
    subcycling::VertexParentStates storage;storage.InitializeAll(tree,leaf,l);
    subcycling::HierarchyRK4 engine;engine.Initialize<6>(tree,l,0,4,4,{},4,{{true,true,true,true}});
    Transport physics{storage,l};const double dt=.2/steps;
    if(steps==2) {
      const auto before=Kokkos::create_mirror(storage.Values());
      Kokkos::deep_copy(before,storage.Values());
      // Force a nonconvergent feedback contract at the first interval length.
      // This exercises real evolution+rollback, then compares its half-step
      // retry to a direct evolution from exactly the original hierarchy.
      int empty_histories=0;
      const auto begin=[&](const subcycling::HierarchyRK4::Histories &h) {
        if(h.empty()) ++empty_histories;
      };
      const auto feedback=[&](const subcycling::HierarchyRK4::Histories &a,
                             const subcycling::HierarchyRK4::Histories &) {
        return a.begin()->second.Dt()>dt*.75 ? Real(2) : Real(0);
      };
      const auto accepted=engine.RunWithRetry(0,dt,storage,physics,2,{}, {},begin,feedback);
      Check(accepted.dt==dt*.5 && accepted.attempts==2 && empty_histories==2 &&
            accepted.corrector.converged && accepted.total_passes>accepted.corrector.passes);
      const auto retried=Kokkos::create_mirror(storage.Values());
      Kokkos::deep_copy(retried,storage.Values());
      Kokkos::deep_copy(storage.Values(),before);
      engine.RunCorrected(0,dt*.5,storage,physics,2);
      const auto direct=Kokkos::create_mirror(storage.Values());
      Kokkos::deep_copy(direct,storage.Values());
      Check(std::memcmp(retried.data(),direct.data(),direct.size()*sizeof(Real))==0);
      Kokkos::deep_copy(storage.Values(),before);
      subcycling::IntervalRetryControl no_retry;no_retry.maximum_halvings=0;
      bool failed=false;
      try {engine.RunWithRetry(0,dt,storage,physics,2,{},no_retry,begin,feedback);}
      catch(const subcycling::CorrectorFailure &) {failed=true;}
      const auto after=Kokkos::create_mirror(storage.Values());
      Kokkos::deep_copy(after,storage.Values());
      Check(failed && !engine.LastCorrectorReport().converged &&
            std::memcmp(before.data(),after.data(),after.size()*sizeof(Real))==0);
      int failure_calls=0;
      bool propagated=false;
      try {
        engine.RunWithRetry(0,dt,storage,physics,2,{}, {},
          [&](const auto &){++failure_calls;throw std::runtime_error("code failure");});
      } catch(const std::runtime_error &e) {propagated=std::string(e.what())=="code failure";}
      Check(propagated && failure_calls==1);
      physics.calls[0]=physics.calls[1]=0;
      std::cout << "PASS: bounded retry, half-step equivalence, rollback, fatal propagation\n";
    }
    int passes=0;
    for(int n=0;n<steps;++n) {
      if(argc>1 && std::string(argv[1])=="--corrected") {
        const auto report=engine.RunCorrected(n*dt,dt,storage,physics,2);
        Check(report.converged);passes+=report.passes;
      } else {engine.Run(n*dt,dt,storage,physics);++passes;}
    }
    Check(physics.calls[0]==4*passes && physics.calls[1]==8*passes);
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
    double error=0;
    for(int n=0;n<h.extent_int(0);++n) for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i)
    {
      const auto key=tree.Nodes()[n].key;const double w=std::ldexp(1.,-key[0]);
      const double exact=std::exp(.2)*(1+.1*w*(key[1]+(i-2)/8.)+
                                      .2*w*(key[2]+(j-2)/8.)+.03*.2);
      error=std::max(error,std::abs(h(n,0,0,j,i)-exact));
    }
    errors.push_back(error);std::cout << "steps=" << steps << " error=" << error << '\n';
  }
  for(int n=1;n<4;++n) {double ratio=errors[n-1]/errors[n];
    std::cout << "ratio=" << ratio << '\n';Check(ratio>10 && ratio<22);}
 }
 Kokkos::finalize();
}
