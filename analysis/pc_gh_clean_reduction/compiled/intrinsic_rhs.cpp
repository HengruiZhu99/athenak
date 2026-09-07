#include <fstream>
#include <iomanip>
#include <stdexcept>
#include "pc_gh/intrinsic_rhs.hpp"
using namespace pc_gh::intrinsic;
int main(int argc,char **argv) {
  if (argc!=4) return 2;
  std::ifstream file(argv[1]); int count; file>>count;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double**> input("input",count,203), output("output",count,128);
    auto host=Kokkos::create_mirror_view(input);
    for (int r=0;r<count;++r) for (int n=0;n<203;++n) file>>host(r,n);
    if (!file) throw std::runtime_error("incomplete RHS input");
    Kokkos::deep_copy(input,host);
    Kokkos::parallel_for("intrinsic RHS oracle",count,KOKKOS_LAMBDA(int r) {
      double u[50],du[3][50],rhs[50],f[10],ell[3][3];
      for (int n=0;n<50;++n) {u[n]=input(r,n);for (int k=0;k<3;++k) du[k][n]=input(r,50+50*k+n);}
      PointRHS(u,du,input(r,200),input(r,201),input(r,202),rhs);
      ConfigurationSources(u,input(r,201),f,ell);int pos=0;
      for (double v:rhs) output(r,pos++)=v;
      for (double v:f) output(r,pos++)=v;
      for (int i=0;i<3;++i) for (int j=0;j<3;++j) output(r,pos++)=ell[i][j];
      for (int k=0;k<3;++k) {
        double df[10],de[3][3]; DifferentiateConfiguration(u,du[k],input(r,201),df,de);
        for (double v:df) output(r,pos++)=v;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) output(r,pos++)=de[i][j];
      }
      Jet plateau=GaugePlateau(Jet(u[0]*u[1]*u[0]*u[0],1));
      output(r,pos++)=plateau.value; output(r,pos++)=plateau.derivative;
    });
    Kokkos::fence(); auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
    std::ofstream out(argv[2]);out<<std::setprecision(17);
    for (int r=0;r<count;++r) {for (int n=0;n<128;++n) out<<(n?" ":"")<<result(r,n);out<<'\n';}
    std::ofstream config(argv[3]);Kokkos::print_configuration(config,true);
  }
  Kokkos::finalize();
}
