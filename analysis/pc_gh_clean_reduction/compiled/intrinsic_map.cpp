#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include "pc_gh/intrinsic_state_map.hpp"
#include "pc_gh/pc_gh.hpp"
using namespace pc_gh::intrinsic;
static_assert(pc_gh::PcGh::I_W==0 && pc_gh::PcGh::I_GTXX==1
    && pc_gh::PcGh::I_K==7 && pc_gh::PcGh::I_ATXX==8
    && pc_gh::PcGh::I_ZX==14 && pc_gh::PcGh::I_CPERP==17
    && pc_gh::PcGh::I_RHO==18 && pc_gh::PcGh::I_BETAX==19
    && pc_gh::PcGh::I_P1==22 && pc_gh::PcGh::I_Q1XX==25
    && pc_gh::PcGh::I_L1==43 && pc_gh::PcGh::I_B11==46
    && pc_gh::PcGh::npcgh==55);
int main(int argc,char **argv) {
  if (argc!=4) return 2;
  std::ifstream file(argv[1]); int count; file>>count;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double**> input("input",count,105), output("output",count,457);
    auto host=Kokkos::create_mirror_view(input);
    for (int r=0;r<count;++r) for (int n=0;n<105;++n) file>>host(r,n);
    if (!file) throw std::runtime_error("incomplete map input");
    Kokkos::deep_copy(input,host);
    Kokkos::parallel_for("intrinsic map oracle",count,KOKKOS_LAMBDA(int r) {
      double u[50],old[55],back[50];
      for (int n=0;n<50;++n) u[n]=input(r,n);
      Geometry<double> g; BuildGeometry(u,g); int pos=0;
      for (int block=0;block<6;++block) for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        double val=block==0?g.tri[i][j] : block==1?g.inverse_tri[i][j]
          : block==2?g.metric[i][j] : block==3?g.inverse_metric[i][j]
          : block==4?g.ahat[i][j] : g.curvature[i][j];
        output(r,pos++)=val;
      }
      for (int k=0;k<3;++k) for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        output(r,pos++)=g.gradient[k][i][j];
      for (int a=0;a<5;++a) for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        output(r,pos++)=g.jacobian[a][i][j];
      for (int a=0;a<5;++a) for (int b=0;b<5;++b)
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) output(r,pos++)=g.hessian[a][b][i][j];
      ToLegacy(u,old);
      for (int n=0;n<55;++n) output(r,pos++)=old[n];
      for (int n=0;n<55;++n) old[n]=input(r,50+n);
      for (int n=0;n<50;++n) back[n]=-9876.0;
      bool valid=FromLegacy(old,back,2e-12);
      output(r,pos++)=valid?1:0;
      for (int n=0;n<50;++n) output(r,pos++)=back[n];
    });
    Kokkos::fence();
    auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
    std::ofstream out(argv[2]); out<<std::setprecision(17);
    for (int r=0;r<count;++r) {
      for (int n=0;n<457;++n) out<<(n?" ":"")<<result(r,n);
      out<<'\n';
    }
    std::ofstream config(argv[3]); Kokkos::print_configuration(config,true);
  }
  Kokkos::finalize();
}
