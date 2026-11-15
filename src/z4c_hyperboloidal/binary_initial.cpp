// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "globals.hpp"
#include "z4c_hyperboloidal/binary_coordinates.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"

namespace z4c_hyperboloidal {
// Per-global-block records make offline export independent of MPI ownership.
// This format contains physical stored fields, never representation residuals.
void Z4cHyperboloidal::InitializeBinary(ParameterInput *pin) {
  const bool prepare=pin->GetOrAddBoolean("problem","binary_prepare",false);
  if(prepare && pin->GetString("time","evolution")!="static")
    throw std::invalid_argument("Binary sampling preparation requires static mode");
  const std::filesystem::path dir=pin->GetString("problem","binary_directory");
  if(prepare)std::filesystem::create_directories(dir);
  InitializeReference();
  auto g=Kokkos::create_mirror_view_and_copy(HostMemSpace(),geometry);
  auto sg=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface_geometry);
  auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u0);
  auto s=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s0);
  const std::string identity=prepare?"prepare":pin->GetString("problem","binary_identity");
  if(!prepare && (identity.size()!=64 || identity.find_first_not_of("0123456789abcdef")!=std::string::npos))
    throw std::invalid_argument("Binary import needs a SHA256 export identity");
  double max_coordinate_delta=0;
  for(int m=0;m<nblocks;++m) {
    std::vector<int> bulk,edge;
    for(int p=0;p<npoints;++p)if(points.h_view(p).m==m)bulk.push_back(p);
    for(int p=0;p<nsurface_points;++p)
      if(face_blocks.h_view(surface_points.h_view(p).face)==m)edge.push_back(p);
    const int gid=pmy_pack->pmb->mb_gid.h_view(m);
    const auto path=dir/("block-"+std::to_string(gid)+(prepare?".points":".hbi"));
    std::ofstream out;std::ifstream in;
    if(prepare) {
      if(std::filesystem::exists(path))throw std::runtime_error("Refusing to overwrite binary sample points");
      out.open(path);out<<std::setprecision(17)<<"HYPERBOL_POINTS_V1 "<<gid<<' '<<bulk.size()<<' '<<edge.size()<<'\n';
    } else {
      in.open(path);std::string magic,bound;int id=-1;std::size_t nb=0,ns=0;
      in>>magic>>bound>>id>>nb>>ns;
      if(!in || magic!="HYPERBOL_BINARY_V1" || bound!=identity || id!=gid ||
          nb!=bulk.size() || ns!=edge.size())
        throw std::runtime_error("Binary block header/identity/mesh mismatch: "+path.string());
    }
    auto row=[&](const Real *x,auto get,auto set) {
      if(prepare) {
        for(int a=0;a<3;++a)out<<x[a]<<' ';
        for(int f=0;f<nvar;++f)out<<get(f)<<(f==nvar-1?'\n':' ');
      } else {
        Real pos[3]{};CartesianChartJets raw{};
        for(int a=0;a<3;++a) {
          in>>pos[a];
          if(!in || !BinaryCoordinateMatches(pos[a],x[a])) {
            std::ostringstream message;message<<std::setprecision(17)
                <<"Binary coordinate/order mismatch: "<<path.string()<<" axis="<<a
                <<" supplied="<<pos[a]<<" expected="<<x[a];
            throw std::runtime_error(message.str());
          }
          max_coordinate_delta=std::max(max_coordinate_delta,std::abs(pos[a]-x[a]));
        }
        for(auto &v:raw){in>>v.f;if(!in || !std::isfinite(v.f))throw std::runtime_error("Invalid binary field");}
        const auto original=raw;
        if(!(raw[0].f>0 && raw[18].f+(1+SphericalPolicy::Radius2(x).f)/2>0) || !TryExpandCartesianChart(raw))
          throw std::runtime_error("Nonpositive binary geometry/lapse");
        for(int f:{6,12})if(std::abs(raw[f].f-original[f].f)>1e-10*(1+std::abs(original[f].f)))
          throw std::runtime_error("Binary determinant/tracefree identity mismatch");
        if(raw[14].f!=0)throw std::runtime_error("Binary Theta must initially vanish");
        for(int a=0;a<3;++a)if(raw[19+a].f!=0 || raw[28+a].f!=0 || raw[25+a].f!=raw[15+a].f)
          throw std::runtime_error("Binary auxiliary initialization convention mismatch");
        for(int f=0;f<nvar;++f)set(f,original[f].f);
      }
    };
    for(int p:bulk){auto c=points.h_view(p);row(g(p).x,
      [&](int f){return u(c.m,f,c.k,c.j,c.i);},[&](int f,Real v){u(c.m,f,c.k,c.j,c.i)=v;});}
    for(int p:edge){auto c=surface_points.h_view(p);row(sg(p).x,
      [&](int f){return s(c.face,f,c.k,c.j);},[&](int f,Real v){s(c.face,f,c.k,c.j)=v;});}
    if(prepare){out.close();if(!out)throw std::runtime_error("Writing binary points failed");}
    else {std::string extra;if(in>>extra)throw std::runtime_error("Trailing binary block records");}
  }
  if(!prepare) {
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&max_coordinate_delta,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
    if(global_variable::my_rank==0)
      std::cout<<std::setprecision(17)<<"binary import max_coordinate_delta="
               <<max_coordinate_delta<<std::endl;
    Kokkos::deep_copy(u0,u);Kokkos::deep_copy(s0,s);
    InvalidateDiagnostics();FormResidual();
  }
}
}
