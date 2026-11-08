#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_tree.hpp
//  \brief defines the MeshBlockTree class
// The MeshBlockTree stores the logical grid structure in a binary tree, and is used for
// neighbor searches, storing global IDs, etc. Levels refer to "logical level", where the
// logical root (single block) level is 0 (see Figs. 1 and 3 of method paper). Note the
// logical level of the physical root grid (user-specified root grid) will be greater than
// zero if it contains more than one MeshBlock.
//
// Original version of this foundational class written c2015-2016 by K. Tomida.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "mesh/logical_location.hpp"

class Mesh;
enum class BoundaryFlag;
namespace chart_atlas {struct Seam;}

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in a binary tree structure.  Thus, the class name does not
//  refer to the overall tree itself, but rather each MeshBlockTree is a node. A pointer
//  to the root node is stored in the Mesh class.

class MeshBlockTree {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class Mesh;
  friend class MeshBlock;
  friend class MeshBlockPack;

 public:
  // Each chart has its own Cartesian logical lattice. All charts in a forest use
  // one root_level, so refinement levels have the same meaning across charts.
  struct RootGrid {
    int chart_id = 0;
    int root_level = 0;
    int ndim = 3;
    std::array<int, 3> nmb_root{1, 1, 1};
    std::array<BoundaryFlag, 6> bcs;
    RootGrid();
  };

  explicit MeshBlockTree(Mesh *pmesh);
  explicit MeshBlockTree(const RootGrid &root_grid);
  MeshBlockTree(Mesh *pmesh, const RootGrid &root_grid);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();
  MeshBlockTree(const MeshBlockTree &) = delete;
  MeshBlockTree &operator=(const MeshBlockTree &) = delete;

  // accessors
  MeshBlockTree* GetLeaf(int ox1, int ox2, int ox3)
    { return pleaf_[(ox1 + (ox2<<1) + (ox3<<2))]; }
  int GetGID() {return gid_;}
  std::int32_t GetLevel() {return lloc_.level;}
  const LogicalLocation &GetLocation() const {return lloc_;}
  const RootGrid &GetRootGrid() const {return *root_grid_;}
  bool IsLeaf() const {return pleaf_ == nullptr;}

  // Mesh retains ownership of every root. This registry only dispatches native
  // tree operations and concatenates chart/Morton leaf lists; it owns no nodes.
  // Call on chart zero, passing roots in chart-id order, before AMR/lookup.
  void SetChartRoots(const std::vector<MeshBlockTree *> &roots,
                     bool cubed_sphere_seams = false);
  MeshBlockTree *GetChartRoot(int chart_id) const;
  // Native leaves touching an oriented cubed-sphere seam footprint, including
  // edge/corner contacts. This is refinement topology, not a conforming Bvals
  // neighbor: mapped field interpolation remains a separate operation.
  std::vector<MeshBlockTree *> FindSeamNeighbors(const LogicalLocation &loc,
                                                int axis, int side) const;
  // Restart insertion cannot create extra blocks without corresponding saved
  // state. Validate its reconstructed seam levels instead of refining it.
  void ValidateSeamBalance() const;

  // functions
  void CreateRootGrid();
  void AddNode(LogicalLocation rloc, int &nnew);
  void AddNodeWithoutRefinement(LogicalLocation rloc);
  void Refine(int &nnew);
  void Derefine(int &ndel);
  MeshBlockTree* FindMeshBlock(LogicalLocation tloc);
  void CountMeshBlocks(int& count);
  void CreateZOrderedLLList(LogicalLocation *list, int *pglist, int& count);
  MeshBlockTree* FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
                              bool amrflag=false);

 private:
  // data: note private variable names have trailing underscore for this class
  MeshBlockTree **pleaf_;  // 1D vector of pointers to leafs
  int gid_;                // grid ID
  LogicalLocation lloc_;   // stores logical x1/x2/x3 location, level for node in tree

  std::shared_ptr<const RootGrid> root_grid_;  // immutable, chart-specific context
  MeshBlockTree *proot_;                      // root of this chart
  int nleaf_;                                // 2/4/8 for 1D/2D/3D
  MeshBlockTree *forest_root_;                // used only on chart roots
  std::vector<MeshBlockTree *> chart_roots_;  // nonowning; only on forest root
  bool cubed_sphere_seams_ = false;           // used only on forest root

  void CountLeaves(int &count);
  void AppendZOrderedLLList(LogicalLocation *list, int *pglist, int &count);
  bool ContainsLocation(const LogicalLocation &loc) const;
  void CollectSeamLeaves(const LogicalLocation &source, const RootGrid &source_grid,
                         const chart_atlas::Seam &seam,
                         std::vector<MeshBlockTree *> &neighbors);
  void BalanceSeams(int &nnew);
  bool CanDerefineAcrossSeams() const;
  void ValidateLeafSeams() const;
};

#endif // MESH_MESHBLOCK_TREE_HPP_
