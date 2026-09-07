#include <fstream>
#include <iomanip>
#include "pc_gh/intrinsic_rhs.hpp"
using namespace pc_gh::intrinsic;
int main(int argc,char **argv) {
  std::ifstream file(argv[1]);int count;file>>count;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double**> input("input",count,203), output("probe",count,128);
    auto host=Kokkos::create_mirror_view(input);
    for (int r=0;r<count;++r) for (int n=0;n<203;++n) file>>host(r,n);
    Kokkos::deep_copy(input,host);
    Kokkos::parallel_for("probe",count,KOKKOS_LAMBDA(int r) {
      double u[50],du[50],f[10],ell[3][3],df[10],de[3][3];
      for (int n=0;n<50;++n) {u[n]=input(r,n);du[n]=input(r,50+n);}
      output(r,0)=input(r,0);output(r,1)=input(r,1);output(r,2)=input(r,50);output(r,3)=input(r,201);
      ConfigurationSources(u,input(r,201),f,ell);
      for (int n=0;n<10;++n) output(r,4+n)=f[n];
      DifferentiateConfiguration(u,du,input(r,201),df,de);
      output(r,14)=df[0];output(r,15)=u[0];
    });
    Kokkos::parallel_for("RHS probe",count,KOKKOS_LAMBDA(int r) {
      double u[50],du[3][50],rhs[50];
      for (int n=0;n<50;++n) {u[n]=input(r,n);for (int k=0;k<3;++k) du[k][n]=input(r,50+50*k+n);}
      output(r,16)=u[0];output(r,17)=input(r,0);
      PointRHS(u,du,input(r,200),input(r,201),input(r,202),rhs);
      output(r,18)=u[0];output(r,19)=input(r,0);output(r,20)=rhs[0];output(r,21)=rhs[10];output(r,22)=du[0][0];output(r,23)=u[1];
    });
    Kokkos::fence();auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
    std::ofstream stream(argv[2]);stream<<std::setprecision(17);
    for (int r=0;r<count;++r) {for (int n=0;n<24;++n) stream<<(n?" ":"")<<out(r,n);stream<<'\n';}
  }
  Kokkos::finalize();
}
