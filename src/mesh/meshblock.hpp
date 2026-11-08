#ifndef MESH_MESHBLOCK_HPP_
#define MESH_MESHBLOCK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock.hpp
//! \brief defines MeshBlock class, a very lightweight class to store data about
//! MeshBlocks inside a given MeshBlockPack.
//! The Mesh is the overall grid structure while MeshBlocks are local patches of data
//! (potentially on different levels) that tile the entire domain and are stored in
//! containers called MashBlockPack.

#include <memory>

#include "bvals/bvals.hpp"
#include "meshblock_pack.hpp"
#include "chart_atlas.hpp"

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//! \brief data/functions associated with each MeshBlock

class MeshBlock {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class Mesh;
  friend class MeshBlockPack;
  friend class MeshBlockTree;

 public:
  MeshBlock(MeshBlockPack *ppack, int igids, int nmb);
  ~MeshBlock() {}  // only default destructor needed

  // data
  int nnghbr;           // maximum number of neighbors for each MeshBlock

  // DualArrays are used to store data used on both device and host
  // First dimension of each array will be [# of MeshBlocks in this MeshBlockPack]
  DualArray1D<int> mb_gid;           // grid ID, unique identifier for each MeshBlock
  DualArray1D<int> mb_lev;           // logical level of each MeshBlock
  DualArray1D<int> mb_chart;         // chart root, zero for Cartesian blocks
  DualArray1D<chart_atlas::BlockMap> mb_map;  // local [0,1]^3 map for atlas blocks
  // Physical bounding box. Uniform physical dx exists only on Cartesian blocks;
  // shell derivatives must use mb_map and logical spacing 1/nx, never these dxs.
  DualArray1D<RegionSize> mb_size;
  DualArray2D<BoundaryFlag> mb_bcs;  // boundary conditions at 6 faces of each MeshBlock
  DualArray2D<NeighborBlock> nghbr;  // data on all (up to 56) neighbors for each MB

  // function to set data describing neighbors
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);

 private:
  // data
  MeshBlockPack* pmy_pack;
};
#endif // MESH_MESHBLOCK_HPP_
