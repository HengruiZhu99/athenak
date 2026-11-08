// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Native donor rows and actual peer fills, with the legacy selector as oracle.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/fields.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"

namespace native=z4c_hyperboloidal;
void Require(bool pass,const char *what) {if (!pass) throw std::runtime_error(what);}
double Payload(const bvals::DonorPoint &p,int field,int stage,bool physical) {
  // Deliberately different physical/residual values catch conflating their paths.
  return std::sin(.19*p.gid+.021*p.i+.031*p.j+.043*p.k+.17*field+.11*stage)
      +(physical?2.3:-1.1)+field*.003;
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  const std::string input=argv[1];
  bool shell=false,full=false;
  for (int a=2;a<argc;++a) {
    shell=shell||std::string(argv[a])=="shell";
    full=full||std::string(argv[a])=="full";
  }
  Kokkos::initialize(argc,argv);
  try {{
    ParameterInput pin;IOWrapper file;file.Open(input.c_str(),IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);
    constexpr int origin=7,n=12,size=26;
    native::NativeDonorSelector selector(mesh,origin);
    native::MappedHalo halo;
    halo.Rebuild(mesh.pmb_pack,origin);
    auto descriptor=cubed_sphere::CompositeMesh(.1,
        {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,12,4,3,.08);
    descriptor.evolved_overlap=3;descriptor.amr_cartesian_core=true;
    descriptor.outer_radial_endpoint_grid=true;descriptor.RefineCartesian(13);
    // The runner suppresses only automatic halo materialization in this legacy
    // test header. Active-cell construction and Interpolate are unmodified.
    std::unique_ptr<cubed_sphere::CompositeDifference> oracle;
    if (!shell) oracle=std::make_unique<cubed_sphere::CompositeDifference>(descriptor,4,true,true);
    std::vector<int> native_id(descriptor.blocks.size(),-1);
    for (std::size_t old=0;!shell&&old<descriptor.blocks.size();++old) {
      const auto &b=descriptor.blocks[old];
      for (int gid=0;gid<mesh.nmb_total;++gid) {
        const auto &a=selector.Map(gid);bool match=a.face==b.face&&a.layer==b.layer;
        for (int d=0;d<3;++d) match=match&&std::abs(a.lo[d]-b.lo[d])<1e-14
            &&std::abs(a.hi[d]-b.hi[d])<1e-14;
        if (match) {Require(native_id[old]<0,"ambiguous legacy/native leaf");native_id[old]=gid;}
      }
      Require(native_id[old]>=0,"missing legacy/native leaf match");
    }
    std::uint64_t tested_rows=0,weights=0;double weight_error=0,payload_error=0;
    for (std::size_t row=0;row<halo.Rows().size();++row) {
      // Cover the full local row range, including each block's first and last
      // faces, without allocating the old global padded halo plan.
      if (!full&&row%97!=0 && row+1!=halo.Rows().size()) continue;
      const auto c=halo.Rows()[row].target;const auto &map=selector.Map(mesh.pmb_pack->gids+c.m);
      const Real q[3]={(c.i-origin+map.node_offset[0])/n,
                        (c.j-origin+map.node_offset[1])/n,
                        (c.k-origin+map.node_offset[2])/n};
      Real x[3];map.Forward(q,x);
      if (shell) {
        const double radius=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        if (radius<.24) continue;
        const auto &owner=selector.Map(selector.Owner(x));
        const int a=owner.face/2;
        auto polynomial=[&](const Real y[3],int degree) {
          const double r=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
          const double scale=owner.face%2?-y[a]:y[a];
          const double u=4*std::atan(y[(a+1)%3]/scale)/std::acos(-1.);
          const double v=4*std::atan(y[(a+2)%3]/scale)/std::acos(-1.);
          return std::pow(r,degree)*(1+.23*u*u+.17*v*v*v);
        };
        for (int degree=0;degree<=5;++degree) {
          double actual=0;
          for (const auto &entry:halo.Rows()[row].donors) {
            const auto &dm=selector.Map(entry.point.gid);
            const Real dq[3]={(entry.point.i-origin+dm.node_offset[0])/n,
                               (entry.point.j-origin+dm.node_offset[1])/n,
                               (entry.point.k-origin+dm.node_offset[2])/n};
            Real donor[3];dm.Forward(dq,donor);
            actual+=entry.weight*polynomial(donor,degree);++weights;
          }
          const double expected=polynomial(x,degree);
          weight_error=std::max(weight_error,std::abs(actual-expected));
          Require(std::abs(actual-expected)<=1e-12+5e-10*std::abs(expected),
                  "shell-SMR mapped polynomial reproduction mismatch");
        }
        ++tested_rows;continue;
      }
      std::map<bvals::DonorPoint,double> expected,actual;
      std::map<int,double> legacy_weights;
      const auto donor_key=[&](int point) {
        const int old=point/(size*size*size),i=point%size,
                  j=point/size%size,k=point/(size*size)%size;
        return bvals::DonorPoint{native_id.at(old),k,j,i};
      };
      for (const auto &entry:oracle->Interpolate(x,6)) {
        expected[donor_key(entry.point)]+=entry.weight;
        legacy_weights[entry.point]+=entry.weight;
      }
      for (const auto &entry:halo.Rows()[row].donors) {
        Require(selector.IsEvolved(entry.point),"donor is not evolved");
        actual[entry.point]+=entry.weight;
      }
      for (const auto &[key,value]:expected) {
        const double error=std::abs(actual[key]-value);
        weight_error=std::max(weight_error,error);
        Require(error<=1e-12+5e-10*std::abs(value),"native/legacy donor weight mismatch");
        actual.erase(key);++weights;
      }
      for (const auto &[key,value]:actual) Require(std::abs(value)<1e-12,
                                                  "native selector added a nonzero donor");
      // The old coalesced rows sum in old padded-block order. The new rows sum
      // in native GID order. Check the actual anchored interpolation, including
      // that ordering difference, for independent non-polynomial payloads.
      for (int f:{0,13,27}) {
        const double old_anchor=Payload(donor_key(legacy_weights.begin()->first),f,0,true);
        const double new_anchor=Payload(halo.Rows()[row].donors[0].point,f,0,true);
        double old_value=old_anchor,new_value=new_anchor;
        for (const auto &[point,weight]:legacy_weights)
          old_value+=weight*(Payload(donor_key(point),f,0,true)-old_anchor);
        for (const auto &entry:halo.Rows()[row].donors)
          new_value+=entry.weight*(Payload(entry.point,f,0,true)-new_anchor);
        const double error=std::abs(new_value-old_value);
        payload_error=std::max(payload_error,error);
        Require(error<=1e-12+5e-10*std::abs(old_value),
                "native/legacy anchored interpolation mismatch");
      }
      ++tested_rows;
    }
    const int blocks=mesh.pmb_pack->nmb_thispack;
    DvceArray5D<Real> residual("test residual",blocks,31,size,size,size),
                     physical("test physical",blocks,31,size,size,size);
    double fill_error=0;std::uint64_t filled=0;
    for (int stage=0;stage<2;++stage) {
      Kokkos::deep_copy(residual,std::numeric_limits<Real>::quiet_NaN());
      Kokkos::deep_copy(physical,std::numeric_limits<Real>::quiet_NaN());
      auto hr=Kokkos::create_mirror_view(residual),hp=Kokkos::create_mirror_view(physical);
      for (int m=0;m<blocks;++m) {
        const int gid=mesh.pmb_pack->gids+m;
        for (int k=origin+selector.Lower(gid,2);k<origin+selector.Upper(gid,2);++k)
          for (int j=origin+selector.Lower(gid,1);j<origin+selector.Upper(gid,1);++j)
            for (int i=origin+selector.Lower(gid,0);i<origin+selector.Upper(gid,0);++i)
              for (int f=0;f<31;++f) {
                hr(m,f,k,j,i)=native::Fields::Independent(f)?Payload({gid,k,j,i},f,stage,false):0;
                hp(m,f,k,j,i)=Payload({gid,k,j,i},f,stage,true);
              }
      }
      Kokkos::deep_copy(residual,hr);Kokkos::deep_copy(physical,hp);
      halo.StartReceives();halo.PackAndSend(residual,physical);halo.FillLocal(residual,physical);
      while (halo.Receive()!=TaskStatus::complete) {}
      halo.FillRemote(residual,physical);
      while (halo.ClearSend()!=TaskStatus::complete) {}
      while (halo.ClearRecv()!=TaskStatus::complete) {}
      hr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),residual);
      hp=Kokkos::create_mirror_view_and_copy(HostMemSpace(),physical);
      for (const auto &row:halo.Rows()) for (int f=0;f<31;++f) {
        const auto c=row.target;
        for (bool raw:{false,true}) {
          const double anchor=Payload(row.donors[0].point,f,stage,raw);
          double expected=anchor;
          for (const auto &entry:row.donors)
            expected+=entry.weight*(Payload(entry.point,f,stage,raw)-anchor);
          if (!raw&&!native::Fields::Independent(f)) expected=0;
          const double actual=raw?hp(c.m,f,c.k,c.j,c.i):hr(c.m,f,c.k,c.j,c.i);
          Require(std::isfinite(actual),"halo fill left poison or read unevolved donor");
          fill_error=std::max(fill_error,std::abs(actual-expected));
          Require(std::abs(actual-expected)<1e-12,"native donor exchange/fill mismatch");++filled;
        }
      }
      for (const auto &row:halo.ExteriorTargets()) {
        const auto c=row.target;
        Require(std::isnan(hr(c.m,0,c.k,c.j,c.i))&&std::isnan(hp(c.m,0,c.k,c.j,c.i)),
                "outer radial dependency silently filled without scri state");
      }
    }
    std::uint64_t counts[9]={halo.Rows().size(),halo.ExteriorTargets().size(),tested_rows,
        weights,filled,halo.InteriorPoints().size(),halo.LocalHaloPoints().size(),
        halo.RemoteHaloPoints().size(),halo.SurfaceDependentPoints().size()};
    MPI_Allreduce(MPI_IN_PLACE,counts,9,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&weight_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&fill_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&payload_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    std::uint64_t expected_points=0;
    for (int gid=0;gid<mesh.nmb_total;++gid) {
      std::uint64_t points=1;
      for (int a=0;a<3;++a) points*=selector.Upper(gid,a)-selector.Lower(gid,a);
      expected_points+=points;
    }
    Require(counts[5]+counts[6]+counts[7]+counts[8]==expected_points,
            "readiness classes omit evolved cells");
    if (!shell) Require(expected_points==229392,"N12 evolved workload changed");
    bool rejected=false;
    try {halo.Rebuild(mesh.pmb_pack,global_variable::my_rank==0?0:origin);}
    catch (const std::exception &) {rejected=true;}
    Require(rejected,"one-rank invalid geometry did not fail collectively");
    rejected=false;
    try {halo.StartReceives();} catch (const std::exception &) {rejected=true;}
    Require(rejected,"failed halo rebuild reused stale stage buffers");
    if (global_variable::my_rank==0) std::cout<<std::setprecision(17)
      <<"PASS "<<(shell?"shell-SMR ":"")<<"mapped halos: rows="<<counts[0]<<" exterior="<<counts[1]
      <<" oracle_rows="<<counts[2]<<" weights="<<counts[3]
      <<" fill_values="<<counts[4]<<" max_weight_error="<<weight_error
      <<" max_payload_error="<<payload_error<<" max_fill_error="<<fill_error
      <<" ready="<<counts[5]<<','<<counts[6]<<','
      <<counts[7]<<','<<counts[8]<<" collective_failure=passed\n";
  }} catch (const std::exception &e) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
