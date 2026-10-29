// Diagnostic geometry adapter only. No evolution or boundary equations.
#include "mesh/cubed_sphere/composite.hpp"
extern "C" int angular_native_rows(int n, double *out) {
  if (n < 6 || n > 32) return 1;
  cubed_sphere::CompositeMesh mesh(.125,{.45,.75,1},1,n,3);
  int row=0;
  for (int block=0;block<int(mesh.blocks.size());++block) {
    if (mesh.blocks[block].layer!=2) continue;
    for(int k=0;k<n;++k)for(int j=0;j<n;++j,++row) {
      const Real local[3]={1,(j+.5)/n,(k+.5)/n};
      const auto p=mesh.Geometry(block,local);const auto &a=p.J;
      const Real det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
        -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
        +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
      Real normal=0;for(int d=0;d<3;++d)normal+=a[0][d]*a[0][d];
      for(int d=0;d<3;++d)out[7*row+d]=p.x[d];
      out[7*row+3]=mesh.blocks[block].face;out[7*row+4]=j;out[7*row+5]=k;
      out[7*row+6]=std::sqrt(normal)/(std::abs(det)*n*n);
    }
  }
  return row==6*n*n?0:2;
}
