// Exact coordinates for the static core-resolution initial-data controls.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "fixed_composite_mesh.hpp"
int main(int argc,char**argv) {
  try {
    if(argc!=4)throw std::invalid_argument("usage: core_coordinates n core_subdivisions output.csv");
    const int n=std::stoi(argv[1]),subdivisions=std::stoi(argv[2]);
    if(n<6||n%2)throw std::invalid_argument("even n >=6 required");
    const auto mesh=hyperbolic_reproduction::FixedMesh(n,subdivisions);
    if(std::filesystem::exists(argv[3]))throw std::runtime_error("refusing to overwrite coordinate file");
    std::ofstream out(argv[3]);out<<std::setprecision(17)<<"x,y,z,r,volume\n";
    for(size_t m=0;m<mesh.blocks.size();++m)
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
      const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};const auto g=mesh.Geometry(m,q);
      const auto*x=g.x;const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<','<<cubed_sphere::PhysicalCellVolume(g,n)<<'\n';
    }
    out.close();if(!out)throw std::runtime_error("coordinate output failed");
    std::cout<<"n="<<n<<" cartesian_core_subdivisions="<<subdivisions<<" mesh_blocks="<<mesh.blocks.size()<<'\n';
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
