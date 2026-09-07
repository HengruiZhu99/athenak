#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include "pc_gh/intrinsic_physical_constraints.hpp"

struct CurvedPrimary {
  double phase,h[3];
  int dimensions,radius,fixture;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int, int field, int k, int j, int i) const {
    if (abs(i)>radius || abs(j)>radius || abs(k)>radius
        || (dimensions<3 && k!=0)) return std::numeric_limits<double>::quiet_NaN();
    double n[3]={0.6,0.8,0},v[3]={0.8,-0.6,0};
    if (dimensions==3) {
      n[0]=2.0/7;n[1]=3.0/7;n[2]=6.0/7;
      v[0]=3/std::sqrt(13.0);v[1]=-2/std::sqrt(13.0);v[2]=0;
    }
    double theta=phase+n[0]*i*h[0]+n[1]*j*h[1]+n[2]*k*h[2];
    double b=.15*cos(theta),w=1+.1*cos(theta);
    if (field==0) return w;
    if (field==1) return .8+.03*sin(theta);
    if (field==2) return .03*sin(theta);
    int row=((field-3)%9)/3,col=(field-3)%3;
    if (fixture==1) {
      if (field>=12) return 0;
      double t[3]={n[1]*v[2]-n[2]*v[1],n[2]*v[0]-n[0]*v[2],n[0]*v[1]-n[1]*v[0]};
      double f=.12*cos(theta);
      return exp(2*f)*v[row]*v[col]+exp(-2*f)*t[row]*t[col]+n[row]*n[col];
    }
    double cross=v[row]*n[col]+n[row]*v[col];
    if (field<12) return (row==col?1.:0.)+b*cross+b*b*n[row]*n[col];
    return .02*(v[row]*v[col]+b*cross+(b*b-1)*n[row]*n[col]);
  }
};

int main(int argc,char **argv) {
  if (argc!=4) return 2;
  std::ifstream file(argv[1]);int count;file>>count;
  Kokkos::initialize(argc,argv);
  {
    Kokkos::View<double**> in("input",count,6),out("constraints",count,7);
    auto host=Kokkos::create_mirror_view(in);
    for (int r=0;r<count;++r) for (int n=0;n<6;++n) file>>host(r,n);
    if (!file) return 3;
    Kokkos::deep_copy(in,host);
    Kokkos::parallel_for("primary constraint stencil",count,KOKKOS_LAMBDA(int r) {
      double h=2*3.14159265358979323846/in(r,2);
      CurvedPrimary field{in(r,3),{h,1.3*h,.7*h},static_cast<int>(in(r,1)),static_cast<int>(in(r,4)),static_cast<int>(in(r,5))};
      Real idx[3]={1/h,1/(1.3*h),1/(.7*h)};double result[7];
      if (in(r,0)==2) pc_gh::intrinsic::PhysicalConstraints<2>(field,0,0,0,0,idx,field.dimensions,result);
      else if (in(r,0)==4) pc_gh::intrinsic::PhysicalConstraints<3>(field,0,0,0,0,idx,field.dimensions,result);
      else pc_gh::intrinsic::PhysicalConstraints<4>(field,0,0,0,0,idx,field.dimensions,result);
      for (int n=0;n<7;++n) out(r,n)=result[n];
    });
    Kokkos::fence();auto values=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
    std::ofstream output(argv[2]);output<<std::setprecision(17);
    for (int r=0;r<count;++r) {for (int n=0;n<7;++n) output<<(n?" ":"")<<values(r,n);output<<'\n';}
    std::ofstream config(argv[3]);Kokkos::print_configuration(config,true);
  }
  Kokkos::finalize();
}
