#include <cmath>
#include <iostream>
#include <stdexcept>
#include "z4c/bulk_rhs.hpp"
#include "z4c/dissipation.hpp"
#include "z4c/state_views.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
void Check(bool ok) {if(!ok) throw std::runtime_error("parent bulk RHS regression");}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  {
    using Z=z4c::Z4c;
    std::vector<subcycling::BlockKey> keys{{1,0,0,0},{1,1,0,0},{1,0,1,0},{1,1,1,0}};
    subcycling::Hierarchy hierarchy(keys,0,2);
    z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
    l.nx1=l.nx2=8;l.nx3=l.n3=1;l.ks=l.ke=0;l.is=l.js=4;l.ie=l.je=12;l.n1=l.n2=17;
    DvceArray5D<Real> leaves("leaves",4,Z::nz4c,1,17,17);
    auto h=Kokkos::create_mirror_view(leaves);
    for(int m=0;m<4;++m) for(int n=0;n<Z::nz4c;++n) for(int j=0;j<17;++j) for(int i=0;i<17;++i) {
      double rho=keys[m][1]+(i-4)/8., z=-1+keys[m][2]+(j-4)/8.;
      double value=0;
      if(n==Z::I_Z4C_CHI || n==Z::I_Z4C_GXX || n==Z::I_Z4C_GYY || n==Z::I_Z4C_GZZ) value=1;
      if(n==Z::I_Z4C_ALPHA) value=1+.01*rho*rho+.02*z*z;
      h(m,n,0,j,i)=value;
    }
    Kokkos::deep_copy(leaves,h);
    subcycling::VertexParentStates parents;parents.Initialize(hierarchy,leaves,l);
    parents.FillSameLevelGhosts(hierarchy,leaves,l);
    std::vector<int> parity(Z::nz4c);
    for(int n=0;n<Z::nz4c;++n) parity[n]=z4c::Z4cStateAxisParitySignFromPackedIndex(n);
    parents.FillAxisAtLogicalZero(l,parity);
    parents.FillPhysicalGhosts<4>(l,0,1,1,{false,true,true,true});
    DvceArray5D<Real> rhs("parent rhs",1,Z::nz4c,1,17,17),mu("mu",1,1,1,17,17),unused;
    Kokkos::deep_copy(rhs,-99.0);
    DualArray1D<RegionSize> sizes("parent geometry",1);
    auto &size=sizes.h_view(0);
    size.x1min=0;size.x1max=2;size.x2min=-1;size.x2max=1;size.x3min=-.5;size.x3max=.5;
    size.dx1=size.dx2=.25;size.dx3=1;
    sizes.modify_host();sizes.sync_device();
    DualArray1D<int> levels("parent levels",1);levels.h_view(0)=0;levels.modify_host();levels.sync_device();
    subcycling::BlockBatches batches;batches.Update(true,levels,1);
    Z::Options opt{};opt.chi_psi_power=-4;opt.chi_div_floor=1e-12;opt.chi_min_floor=1e-12;
    opt.lapse_oplog=2;opt.lapse_harmonicf=1;opt.sss_damping_time=1;opt.ssl_damping_time=1;
    opt.shift_mode=z4c::Z4cShiftMode::prescribed_zero;opt.use_z4c=true;
    auto state_views=z4c::BindStateViews(parents.Values()),rhs_views=z4c::BindStateViews(rhs);
    Tmunu::Tmunu_vars vacuum;
    z4c::EvaluateZ4cBulkRHS<z4c::VertexCenteredZ4c,z4c::CartoonSO2,3>(l,sizes,batches,
        state_views,rhs_views,mu,opt,true,vacuum,0,0,0,1,false,unused,false,unused);
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
    double worst=0;
    for(int n=0;n<Z::nz4c;++n) for(int j=0;j<17;++j) for(int i=0;i<17;++i) {
      if(i<4 || i>12 || j<4 || j>12) {Check(out(0,n,0,j,i)==-99);continue;}
      double expected=0;
      if(n==Z::I_Z4C_KHAT) expected=-.08;
      // Packed directions are (rho, axial z, suppressed y).
      if(n==Z::I_Z4C_AXX || n==Z::I_Z4C_AZZ) expected=-.02+.08/3;
      if(n==Z::I_Z4C_AYY) expected=-.04+.08/3;
      const double diff=std::abs(out(0,n,0,j,i)-expected);
      Check(std::isfinite(diff));worst=std::max(worst,diff);
    }
    std::cout << "maximum analytic parent bulk RHS error " << worst << '\n';Check(worst<1e-11);
    DualArray2D<BoundaryFlag> bcs("parent boundary flags",1,6);
    for(int f=0;f<6;++f) bcs.h_view(0,f)=BoundaryFlag::outflow;
    bcs.h_view(0,0)=BoundaryFlag::axis;bcs.modify_host();bcs.sync_device();
    z4c::AddZ4cDissipation<z4c::VertexCenteredZ4c,z4c::CartoonSO2,3>(
        l,sizes,bcs,batches,parents.Values(),rhs,.02/64,false,unused);
    auto smooth=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
    for(int j=4;j<=12;++j) for(int i=4;i<=12;++i)
      Check(std::abs(smooth(0,Z::I_Z4C_KHAT,0,j,i)+.08)<1e-11);
    // A grid-frequency even scalar mode must be damped by the sixth difference
    // in both retained spatial directions. Suppressed-direction KO is zero.
    auto ph=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),parents.Values());
    for(int j=0;j<17;++j) for(int i=0;i<17;++i)
      ph(0,Z::I_Z4C_CHI,0,j,i)=1+((i+j)%2 ? -.001 : .001);
    Kokkos::deep_copy(parents.Values(),ph);Kokkos::deep_copy(rhs,0.0);
    z4c::AddZ4cDissipation<z4c::VertexCenteredZ4c,z4c::CartoonSO2,3>(
        l,sizes,bcs,batches,parents.Values(),rhs,.02/64,false,unused);
    auto damped=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rhs);
    double ko_error=0;
    for(int j=4;j<=12;++j) for(int i=4;i<=12;++i) {
      const double expected=(i+j)%2 ? .00016 : -.00016;
      ko_error=std::max(ko_error,std::abs(damped(0,Z::I_Z4C_CHI,0,j,i)-expected));
    }
    Check(ko_error<1e-12);
    std::cout << "parent grid-mode KO error " << ko_error << '\n';
    std::cout << "PASS: actual VC Cartoon bulk RHS and dissipation on injected parent state including axis and outer ghosts\n";
  }
  Kokkos::finalize();
}
