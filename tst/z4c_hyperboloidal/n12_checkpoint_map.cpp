// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Offline geometry proof. No evolution, native field allocation or production I/O.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "outputs/io_wrapper.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"

namespace old=z4c::hyperboloidal;
constexpr int n=12,overlap=3,padded=18,bulk_slots=padded*padded*padded;
void Require(bool yes,const char *why) {if (!yes)throw std::runtime_error(why);}
template<class T>void Put(std::ofstream &out,const T *values,std::size_t count) {
  out.write(reinterpret_cast<const char *>(values),count*sizeof(T));
  Require(out.good(),"cannot write mapping artifact");
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==3 && global_variable::nranks==1,"usage: n12_checkpoint_map fixture output-directory (one rank)");
    ParameterInput pin;IOWrapper input;input.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(input);input.Close();
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);
    // This geometry-only helper deliberately omits AddCoordinatesAndPhysics().
    // BuildTree creates pmb, but pcoord otherwise remains unset while the pack
    // destructor owns both pointers. Mark the omitted coordinates owner empty.
    mesh.pmb_pack->pcoord=nullptr;
    Require(mesh.nmb_total==106 && mesh.mb_indcs.nx1==n && mesh.atlas,
            "mapping requires fixed N12 native atlas");
    z4c_hyperboloidal::NativeDonorSelector selector(mesh,mesh.mb_indcs.ng+overlap);
    cubed_sphere::CompositeMesh oldmesh(.1,
        {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,n,4,3,.08);
    oldmesh.evolved_overlap=3;oldmesh.RefineCartesian(13);oldmesh.amr_cartesian_core=true;
    oldmesh.outer_radial_endpoint_grid=true;
    old::CentralZ4cConfiguration configuration;configuration.inner_z4c_radius_M=20;
    configuration.compact_slicing_end=.97;auto parameters=configuration.Resolve().gauge;
    parameters.outer_lapse_damping=.25;
    old::SphericalTransfer3D model(std::move(oldmesh),parameters);
    Require(model.grid.cells.size()==229392 && model.scri.rows.size()==864,
            "legacy N12 point counts changed");
    std::vector<int> native_id(model.grid.mesh.blocks.size(),-1),offset(mesh.nmb_total+1),
        edge_offset(mesh.nmb_total,-1);
    int edges=0;
    for(int gid=0;gid<mesh.nmb_total;++gid) {
      int cells=1;for(int a=0;a<3;++a)cells*=selector.Upper(gid,a)-selector.Lower(gid,a);
      offset[gid+1]=offset[gid]+cells;
      if(selector.Map(gid).outer_surface) {edge_offset[gid]=edges;edges+=n*n;}
    }
    Require(offset.back()==229392 && edges==864,"native N12 point counts changed");
    for(std::size_t old=0;old<native_id.size();++old) {
      const auto &b=model.grid.mesh.blocks[old];
      for(int gid=0;gid<mesh.nmb_total;++gid) {
        const auto &a=selector.Map(gid);bool match=a.face==b.face && a.layer==b.layer;
        for(int d=0;d<3;++d)match=match && a.lo[d]==b.lo[d] && a.hi[d]==b.hi[d];
        if(match) {Require(native_id[old]<0,"ambiguous native leaf");native_id[old]=gid;}
      }
      Require(native_id[old]>=0,"legacy leaf has no exact native match");
    }
    auto ids=native_id;std::sort(ids.begin(),ids.end());
    Require(ids.size()==std::size_t(mesh.nmb_total),"leaf cardinality differs");
    for(int gid=0;gid<mesh.nmb_total;++gid)Require(ids[gid]==gid,"leaf map is not bijective");
    const std::string directory=argv[2];std::ofstream out(directory+"/mapping.bin",std::ios::binary);
    const char magic[16]="N12_MAP_V1";Put(out,magic,16);
    const std::uint64_t header[4]={229392,864,106,64};Put(out,header,4);
    std::set<std::pair<int,int>> slots;std::set<int> bulk_order,surface_order;
    auto row=[&](int gid,int slot,int i,int j,int k,int ordinal,const Real *expected,
                 double weight,double diagnostic,bool surface) {
      const auto &map=selector.Map(gid);
      const Real q[3]={surface?1:(i+map.node_offset[0])/n,
                      (j+map.node_offset[1])/n,(k+map.node_offset[2])/n};
      Real x[3];map.Forward(q,x);
      Require(std::memcmp(x,expected,3*sizeof(Real))==0,"native/legacy coordinates differ");
      Require(slots.emplace(gid,slot).second,"duplicate native restart slot");
      Require((surface?surface_order:bulk_order).insert(ordinal).second,"duplicate canonical ordinal");
      const std::int32_t indices[6]={gid,slot,i,j,k,ordinal};
      const double values[5]={x[0],x[1],x[2],weight,diagnostic};
      Put(out,indices,6);Put(out,values,5);
    };
    for(std::size_t p=0;p<model.grid.cells.size();++p) {
      const auto c=model.grid.cells[p];const int gid=native_id[c.block];
      const int wi=selector.Upper(gid,0)-selector.Lower(gid,0),
                wj=selector.Upper(gid,1)-selector.Lower(gid,1);
      const int ordinal=offset[gid]+((c.k-selector.Lower(gid,2))*wj+
          c.j-selector.Lower(gid,1))*wi+c.i-selector.Lower(gid,0);
      const int slot=((c.k+overlap)*padded+c.j+overlap)*padded+c.i+overlap;
      Require(slot>=0 && slot<bulk_slots,"bulk restart slot out of range");
      row(gid,slot,c.i,c.j,c.k,ordinal,model.grid.geometry[p].x,
          model.grid.volumes[p],model.grid.DiagnosticVolume(p),false);
    }
    for(const auto &r:model.scri.rows) {
      const int gid=native_id[r.block];Require(edge_offset[gid]>=0,"scri leaf is not outer");
      row(gid,bulk_slots+r.k*n+r.j,n,r.j,r.k,edge_offset[gid]+r.k*n+r.j,
          r.geometry.x,cubed_sphere::PhysicalCellVolume(r.geometry,n)/r.radial_spacing,0,true);
    }
    Require(*bulk_order.begin()==0 && *bulk_order.rbegin()==229391 &&
            *surface_order.begin()==0 && *surface_order.rbegin()==863,"canonical coverage incomplete");
    std::ofstream topology(directory+"/topology.bin",std::ios::binary);
    for(int gid=0;gid<mesh.nmb_total;++gid) {
      const auto &l=mesh.lloc_eachmb[gid];
      const std::int32_t a[5]={l.lx1,l.lx2,l.lx3,l.level,l.chart_id};Put(topology,a,5);
    }
    std::ofstream atlas(directory+"/atlas.txt");atlas<<mesh.atlas->Identity();
    std::ofstream fields(directory+"/initial-fields.bin",std::ios::binary),
                  boundary(directory+"/initial-boundary.bin",std::ios::binary);
    Put(fields,model.state.data(),model.state.size());
    Put(boundary,model.scri.values.data(),model.scri.values.size());
  }}catch(const std::exception &error) {
    std::cerr<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  std::cout<<"PASS N12 mapping leaves=106 bulk=229392 scri=864 coordinate_mismatches=0"
           <<std::endl;
  Kokkos::finalize();MPI_Finalize();return 0;
}
