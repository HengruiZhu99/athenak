//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_tree.cpp
//  \brief implementation of constructor and functions in the MeshBlockTree class

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "meshblock_tree.hpp"
#include "chart_atlas.hpp"

MeshBlockTree::RootGrid::RootGrid() {
  bcs.fill(BoundaryFlag::outflow);
}

// The ordinary mesh keeps its historical single-chart behavior. Snapshot its
// root properties so another Mesh/tree cannot overwrite this tree's context.
MeshBlockTree::MeshBlockTree(Mesh *pmesh) : MeshBlockTree([pmesh]() {
  RootGrid grid;
  grid.root_level = pmesh->root_level;
  grid.ndim = pmesh->three_d ? 3 : (pmesh->multi_d ? 2 : 1);
  grid.nmb_root = {pmesh->nmb_rootx1, pmesh->nmb_rootx2, pmesh->nmb_rootx3};
  for (int f=0; f<6; ++f) grid.bcs[f] = pmesh->mesh_bcs[f];
  return grid;
}()) {}

MeshBlockTree::MeshBlockTree(Mesh *, const RootGrid &root_grid)
    : MeshBlockTree(root_grid) {}

MeshBlockTree::MeshBlockTree(const RootGrid &root_grid)
    : pleaf_(nullptr), gid_(-1),
      root_grid_(std::make_shared<const RootGrid>(root_grid)), proot_(this),
      nleaf_(0), forest_root_(this) {
  if (root_grid.chart_id < 0 || root_grid.root_level < 0 || root_grid.root_level > 31
      || root_grid.ndim < 1 || root_grid.ndim > 3) {
    throw std::invalid_argument("invalid MeshBlockTree chart/root level/dimension");
  }
  nleaf_ = 1 << root_grid.ndim;
  for (int a=0; a<3; ++a) {
    if (root_grid.nmb_root[a] < 1
        || root_grid.nmb_root[a] > (1LL << root_grid.root_level)
        || (a >= root_grid.ndim && root_grid.nmb_root[a] != 1)) {
      throw std::invalid_argument("invalid MeshBlockTree chart root dimensions");
    }
  }
  lloc_ = {0, 0, 0, 0, root_grid.chart_id};
}

void MeshBlockTree::SetChartRoots(const std::vector<MeshBlockTree *> &roots,
                                 bool cubed_sphere_seams) {
  if (this != proot_ || lloc_.chart_id != 0 || roots.empty() || roots[0] != this) {
    throw std::invalid_argument("forest registry must start with its chart-zero root");
  }
  for (std::size_t c=0; c<roots.size(); ++c) {
    if (roots[c] == nullptr || roots[c]->proot_ != roots[c]
        || roots[c]->lloc_.chart_id != static_cast<int>(c)
        || roots[c]->root_grid_->root_level != root_grid_->root_level) {
      throw std::invalid_argument("forest requires ordered charts and a common root level");
    }
  }
  if (cubed_sphere_seams) {
    if (roots.size() != 7) {
      throw std::invalid_argument("cubed-sphere seams require one core and six shell charts");
    }
    for (int chart=1; chart<7; ++chart) {
      const auto &grid = roots[chart]->GetRootGrid();
      if (grid.ndim != 3 || grid.nmb_root[0] != roots[1]->GetRootGrid().nmb_root[0]) {
        throw std::invalid_argument("shell seam roots require common radial topology");
      }
      for (int f=2; f<6; ++f) {
        if (grid.bcs[f] == BoundaryFlag::periodic || grid.bcs[f] == BoundaryFlag::shear_periodic) {
          throw std::invalid_argument("cubed-sphere angular seams cannot also be periodic");
        }
      }
    }
  }
  chart_roots_ = roots;
  cubed_sphere_seams_ = cubed_sphere_seams;
  for (auto *root : roots) root->forest_root_ = this;
}

MeshBlockTree *MeshBlockTree::GetChartRoot(int chart_id) const {
  if (chart_id == lloc_.chart_id) return proot_;
  const auto &roots = proot_->forest_root_->chart_roots_;
  return chart_id >= 0 && chart_id < static_cast<int>(roots.size())
      ? roots[chart_id] : nullptr;
}

bool MeshBlockTree::ContainsLocation(const LogicalLocation &loc) const {
  if (loc.chart_id != lloc_.chart_id || loc.level < 0 || loc.level > 31) return false;
  const std::int32_t index[3] = {loc.lx1, loc.lx2, loc.lx3};
  const int shift = loc.level - root_grid_->root_level;
  for (int a=0; a<3; ++a) {
    const std::int64_t n = a >= root_grid_->ndim ? 1 : (shift >= 0
        ? static_cast<std::int64_t>(root_grid_->nmb_root[a]) << shift
        : (root_grid_->nmb_root[a] + (1LL << -shift) - 1) >> -shift);
    if (index[a] < 0 || index[a] >= n) return false;
  }
  return true;
}

std::vector<MeshBlockTree *> MeshBlockTree::FindSeamNeighbors(
    const LogicalLocation &loc, int axis, int side) const {
  std::vector<MeshBlockTree *> neighbors;
  if (!proot_->forest_root_->cubed_sphere_seams_ || loc.chart_id == 0) return neighbors;
  auto *source = GetChartRoot(loc.chart_id);
  if (source == nullptr || !source->ContainsLocation(loc)
      || loc.level < source->root_grid_->root_level) return neighbors;
  const auto seam = chart_atlas::Atlas::AngularSeam(loc.chart_id, axis, side);
  const std::int32_t index[3] = {loc.lx1, loc.lx2, loc.lx3};
  const auto &grid = source->GetRootGrid();
  const std::int64_t extent = std::int64_t(grid.nmb_root[axis])
      << (loc.level-grid.root_level);
  if ((side < 0 && index[axis] != 0)
      || (side > 0 && std::int64_t(index[axis])+1 != extent)) return neighbors;
  auto *target = GetChartRoot(seam.neighbor_chart);
  target->CollectSeamLeaves(loc, grid, seam, neighbors);
  return neighbors;
}

void MeshBlockTree::CollectSeamLeaves(const LogicalLocation &source,
                                     const RootGrid &source_grid,
                                     const chart_atlas::Seam &seam,
                                     std::vector<MeshBlockTree *> &neighbors) {
  const std::int32_t src[3] = {source.lx1, source.lx2, source.lx3};
  const std::int32_t dst[3] = {lloc_.lx1, lloc_.lx2, lloc_.lx3};
  const int source_shift = 31-source.level, target_shift = 31-lloc_.level;
  const auto lower = [=](int a) {return std::int64_t(dst[a]) << target_shift;};
  const auto upper = [=](int a) {return (std::int64_t(dst[a])+1) << target_shift;};
  const std::int64_t edge = std::int64_t(root_grid_->nmb_root[seam.neighbor_axis])
      << (31-root_grid_->root_level);
  if (seam.neighbor_side < 0 ? lower(seam.neighbor_axis) != 0
      : upper(seam.neighbor_axis) < edge || lower(seam.neighbor_axis) >= edge) return;

  // Radial root indices agree across shell faces. Angular indices can have
  // different root counts after swapping u/v: compare exact rational intervals
  // using cross products. At most 2^31 times (2^31-1), which fits int64.
  const std::int64_t radial_lo = std::int64_t(src[0]) << source_shift;
  const std::int64_t radial_hi = (std::int64_t(src[0])+1) << source_shift;
  if (upper(0) < radial_lo || lower(0) > radial_hi) return;
  const std::int64_t source_lo = (std::int64_t(src[seam.tangent_axis]) << source_shift)
      *root_grid_->nmb_root[seam.neighbor_tangent_axis];
  const std::int64_t source_hi = ((std::int64_t(src[seam.tangent_axis])+1) << source_shift)
      *root_grid_->nmb_root[seam.neighbor_tangent_axis];
  const std::int64_t target_lo = lower(seam.neighbor_tangent_axis)
      *source_grid.nmb_root[seam.tangent_axis];
  const std::int64_t target_hi = upper(seam.neighbor_tangent_axis)
      *source_grid.nmb_root[seam.tangent_axis];
  if (target_hi < source_lo || target_lo > source_hi) return;
  if (IsLeaf()) {
    neighbors.push_back(this);
  } else {
    for (int n=0; n<nleaf_; ++n) {
      if (pleaf_[n] != nullptr) pleaf_[n]->CollectSeamLeaves(source, source_grid, seam, neighbors);
    }
  }
}

void MeshBlockTree::BalanceSeams(int &nnew) {
  if (!proot_->forest_root_->cubed_sphere_seams_ || lloc_.chart_id == 0) return;
  bool changed;
  do {
    changed = false;
    for (int axis=1; axis<3; ++axis) {
      for (int side : {-1, 1}) {
        for (auto *neighbor : FindSeamNeighbors(lloc_, axis, side)) {
          // This node has just produced children at level+1. Its seam neighbors
          // must therefore reach at least this node's level.
          if (neighbor->lloc_.level < lloc_.level && neighbor->IsLeaf()) {
            neighbor->Refine(nnew);
            changed = true;
          }
        }
      }
    }
  } while (changed);
}

bool MeshBlockTree::CanDerefineAcrossSeams() const {
  for (int axis=1; axis<3; ++axis) {
    for (int side : {-1, 1}) {
      for (const auto *neighbor : FindSeamNeighbors(lloc_, axis, side)) {
        if (neighbor->lloc_.level > lloc_.level+1) return false;
      }
    }
  }
  return true;
}

void MeshBlockTree::ValidateSeamBalance() const {
  const auto *forest = proot_->forest_root_;
  if (!forest->cubed_sphere_seams_) return;
  for (std::size_t chart=1; chart<forest->chart_roots_.size(); ++chart) {
    forest->chart_roots_[chart]->ValidateLeafSeams();
  }
}

void MeshBlockTree::ValidateLeafSeams() const {
  if (!IsLeaf()) {
    for (int n=0; n<nleaf_; ++n) {
      if (pleaf_[n] != nullptr) pleaf_[n]->ValidateLeafSeams();
    }
    return;
  }
  for (int axis=1; axis<3; ++axis) {
    for (int side : {-1, 1}) {
      for (const auto *neighbor : FindSeamNeighbors(lloc_, axis, side)) {
        if (lloc_.level > neighbor->lloc_.level+1 || neighbor->lloc_.level > lloc_.level+1) {
          throw std::runtime_error("native chart forest violates 2:1 shell seam balance");
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::MeshBlockTree(int gid, int ox1, int ox2, int ox3)
//! \brief constructor for a leaf

MeshBlockTree::MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3) :
  pleaf_(nullptr), gid_(parent->gid_), root_grid_(parent->root_grid_),
  proot_(parent->proot_), nleaf_(parent->nleaf_), forest_root_(nullptr) {
  lloc_.lx1 = (parent->lloc_.lx1<<1) + ox1;
  lloc_.lx2 = (parent->lloc_.lx2<<1) + ox2;
  lloc_.lx3 = (parent->lloc_.lx3<<1) + ox3;
  lloc_.level = parent->lloc_.level + 1;
  lloc_.chart_id = parent->lloc_.chart_id;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::~MeshBlockTree()
//! \brief destructor (for both root and leaves)

MeshBlockTree::~MeshBlockTree() {
  if (pleaf_ != nullptr) {
    for (int i=0; i<nleaf_; i++) { delete pleaf_[i]; }
    delete [] pleaf_;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CreateRootGrid()
//! \brief create the root grid; note the root grid can be incomplete (less than 8 leaves)

void MeshBlockTree::CreateRootGrid() {
  if (pleaf_ != nullptr) return;
  // do not create any nodes beyond the logical level of root grid (which corresponds to
  // logical level = 0 only in the case of a grid containing a single MeshBlock)
  if (lloc_.level == root_grid_->root_level) return;

  // Otherwise create vector of leaf pointers
  pleaf_ = new MeshBlockTree*[nleaf_];
  for (int n=0; n<nleaf_; n++) {pleaf_[n] = nullptr;}

  // test if any of the leaves need to be refined further
  std::int32_t levfac = 1<<(root_grid_->root_level - lloc_.level-1);
  for (int n=0; n<nleaf_; n++) {
    // i,j,k values are 1st/2nd/3rd bit from end in binary representation of n=0,..,7
    int i = n&1, j = (n>>1)&1, k = (n>>2)&1;
    if ((lloc_.lx3*2 + k)*levfac < root_grid_->nmb_root[2]
     && (lloc_.lx2*2 + j)*levfac < root_grid_->nmb_root[1]
     && (lloc_.lx1*2 + i)*levfac < root_grid_->nmb_root[0]) {
      pleaf_[n] = new MeshBlockTree(this, i, j, k); // call leaf constructor
      pleaf_[n]->CreateRootGrid();
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::AddNode(LogicalLocation rloc, int &nnew)
//! \brief add a new node for a MeshBlock to the tree, also creates neighboring blocks

void MeshBlockTree::AddNode(LogicalLocation rloc, int &nnew) {
  if (rloc.chart_id != lloc_.chart_id) {
    auto *root = GetChartRoot(rloc.chart_id);
    if (root == nullptr) throw std::out_of_range("unknown chart in AddNode");
    root->AddNode(rloc, nnew);
    return;
  }
  if (!ContainsLocation(rloc) || rloc.level < lloc_.level) {
    throw std::out_of_range("invalid logical location in AddNode");
  }
  if (lloc_.level == rloc.level) return; // done

  if (pleaf_ == nullptr) {
    Refine(nnew);             // leaf -> create the finer level
  }

  // get leaf index
  int sh = rloc.level-lloc_.level-1;
  int mx, my, mz;
  mx = ((rloc.lx1>>sh) & 1) == 1;
  my = ((rloc.lx2>>sh) & 1) == 1;
  mz = ((rloc.lx3>>sh) & 1) == 1;
  int n = mx + (my<<1) + (mz<<2);
  pleaf_[n]->AddNode(rloc, nnew);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::AddNodeWithoutRefinement(LogicalLocation rloc)
//! \brief add a node for a MeshBlock to the tree without refinement, used in restarting.
//! MeshBlockTree::CreateRootGrid must be called before this method

void MeshBlockTree::AddNodeWithoutRefinement(LogicalLocation rloc) {
  if (rloc.chart_id != lloc_.chart_id) {
    auto *root = GetChartRoot(rloc.chart_id);
    if (root == nullptr) throw std::out_of_range("unknown chart in restart tree");
    root->AddNodeWithoutRefinement(rloc);
    return;
  }
  if (!ContainsLocation(rloc) || rloc.level < lloc_.level) {
    throw std::out_of_range("invalid logical location in restart tree");
  }
  if (lloc_.level == rloc.level) // done
    return;

  if (pleaf_ == nullptr) {
    pleaf_ = new MeshBlockTree*[nleaf_];
    for (int n=0; n<nleaf_; n++) {
      pleaf_[n] = nullptr;
    }
  }

  // get leaf index
  int sh = rloc.level-lloc_.level-1;
  int mx, my, mz;
  mx = ((rloc.lx1>>sh) & 1) == 1;
  my = ((rloc.lx2>>sh) & 1) == 1;
  mz = ((rloc.lx3>>sh) & 1) == 1;
  int n = mx + (my<<1) + (mz<<2);
  if (pleaf_[n] == nullptr) {
    pleaf_[n] = new MeshBlockTree(this, mx, my, mz);
  }
  pleaf_[n]->AddNodeWithoutRefinement(rloc);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::Refine(int &nnew)
//! \brief make finer leaves

void MeshBlockTree::Refine(int &nnew) {
  if (pleaf_ != nullptr) return;
  if (lloc_.level < root_grid_->root_level || lloc_.level >= 31) {
    throw std::out_of_range("cannot refine outside physical/logical level limits");
  }

  pleaf_ = new MeshBlockTree*[nleaf_];
  for (int n=0; n<nleaf_; n++) {pleaf_[n] = nullptr;}

  for (int n=0; n<nleaf_; n++) {
    int i = n&1, j = (n>>1)&1, k = (n>>2)&1;
    pleaf_[n] = new MeshBlockTree(this, i, j, k);
  }

  std::int32_t nxmax,nymax,nzmax;
  std::int32_t oxmin, oxmax, oymin, oymax, ozmin, ozmax;
  LogicalLocation nloc = lloc_;

  oxmin = -1;
  oxmax = 1;
  nxmax = (root_grid_->nmb_root[0]<<(lloc_.level - root_grid_->root_level));
  oymin = 0;
  oymax = 0;
  nymax = 1;
  ozmin = 0;
  ozmax = 0;
  nzmax = 1;
  if ((root_grid_->ndim > 1)) {
    oymin = -1;
    oymax = 1;
    nymax = (root_grid_->nmb_root[1]<<(lloc_.level - root_grid_->root_level));
  }
  if ((root_grid_->ndim > 2)) { // 3D
    ozmin = -1;
    ozmax = 1;
    nzmax = (root_grid_->nmb_root[2]<<(lloc_.level - root_grid_->root_level));
  }

  for (std::int32_t oz=ozmin; oz<=ozmax; oz++) {
    nloc.lx3 = lloc_.lx3 + oz;
    if (nloc.lx3<0) {
      if (root_grid_->bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
        continue;
      } else {
        nloc.lx3 = nzmax - 1;
      }
    }
    if (nloc.lx3>=nzmax) {
      if (root_grid_->bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic) {
        continue;
      } else {
        nloc.lx3 = 0;
      }
    }

    for (std::int32_t oy=oymin; oy<=oymax; oy++) {
      nloc.lx2=lloc_.lx2+oy;
      if (nloc.lx2<0) {
        if (root_grid_->bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
          continue;
        } else {
          nloc.lx2 = nymax - 1;
        }
      }
      if (nloc.lx2>=nymax) {
        if (root_grid_->bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) {
          continue;
        } else {
          nloc.lx2=0;
        }
      }

      for (std::int32_t ox=oxmin; ox<=oxmax; ox++) {
        if (ox==0 && oy==0 && oz==0) continue;
        nloc.lx1 = lloc_.lx1 + ox;
        if (nloc.lx1<0) {
          if (root_grid_->bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic
          && root_grid_->bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) {
            continue;
          } else {
            nloc.lx1 = nxmax - 1;
          }
        }
        if (nloc.lx1>=nxmax) {
          if (root_grid_->bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic
          && root_grid_->bcs[BoundaryFace::outer_x1] != BoundaryFlag::shear_periodic) {
            continue;
          } else {
            nloc.lx1 = 0;
          }
        }
        proot_->AddNode(nloc, nnew);
      }
    }
  }
  BalanceSeams(nnew);
  // this block is not a leaf anymore
  gid_ = -1;

  nnew += nleaf_-1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::Derefine(int &ndel)
//! \brief destroy leaves and make this block a leaf

void MeshBlockTree::Derefine(int &ndel) {
  if (pleaf_ == nullptr || lloc_.level < root_grid_->root_level) return;
  for (int n=0; n<nleaf_; ++n) {
    if (pleaf_[n] == nullptr || !pleaf_[n]->IsLeaf()) return;
  }
  if (!CanDerefineAcrossSeams()) return;
  int s2=0, e2=0, s3=0, e3=0;
  if ((root_grid_->ndim > 1)) s2=-1, e2=1;
  if ((root_grid_->ndim > 2)) s3=-1, e3=1;
  for (int ox3=s3; ox3<=e3; ox3++) {
    for (int ox2=s2; ox2<=e2; ox2++) {
      for (int ox1=-1; ox1<=1; ox1++) {
        MeshBlockTree *bt = proot_->FindNeighbor(lloc_, ox1, ox2, ox3, true);
        if (bt != nullptr) {
          if (bt->pleaf_ != nullptr) {
            int lis, lie, ljs, lje, lks, lke;
            if (ox1==-1)       lis=lie=1;
            else if (ox1==1)   lis=lie=0;
            else              lis=0, lie=1;
            if ((root_grid_->ndim > 1)) {
              if (ox2==-1)     ljs=lje=1;
              else if (ox2==1) ljs=lje=0;
              else            ljs=0, lje=1;
            } else {
              ljs=lje=0;
            }
            if ((root_grid_->ndim > 2)) {
              if (ox3==-1)     lks=lke=1;
              else if (ox3==1) lks=lke=0;
              else            lks=0, lke=1;
            } else {
              lks=lke=0;
            }
            for (int lk=lks; lk<=lke; lk++) {
              for (int lj=ljs; lj<=lje; lj++) {
                for (int li=lis; li<=lie; li++) {
                  int n = li + (lj<<1) + (lk<<2);
                  if (bt->pleaf_[n]->pleaf_ != nullptr) return;
                }
              }
            }
          }
        }
      }
    }
  }

  gid_ = pleaf_[0]->gid_; // now this is a leaf; inherit the first leaf's GID
  for (int n=0; n<nleaf_; n++) {
    delete pleaf_[n];
  }
  delete [] pleaf_;
  pleaf_ = nullptr;
  ndel+=nleaf_-1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CountMeshBlocks(int& count)
//! \brief Returns the number of MeshBlocks in Tree

void MeshBlockTree::CountMeshBlocks(int& count) {
  if (lloc_.level == 0) count = 0;
  if (this == proot_ && this == forest_root_ && !chart_roots_.empty()) {
    for (auto *root : chart_roots_) root->CountLeaves(count);
  } else {
    CountLeaves(count);
  }
}

void MeshBlockTree::CountLeaves(int &count) {
  if (pleaf_ == nullptr) {
    count++;
  } else {
    for (int n=0; n<nleaf_; n++) {
      if (pleaf_[n] != nullptr) {pleaf_[n]->CountLeaves(count);}
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CreateZOrderedLLList(LogicalLocation *list, int *pg, int& cnt)
//! \brief Creates the Location list for tree sorted by Z-ordering, and creates new MB ids
//! based on this order.  Should be called from root of tree. Called in BuildTreeXXX()
//! functions when tree is constructed for first time, in which case second argument is
//! 'nullptr' and this function creates gids for all MBs based on Z-ordering. Also called
//! by ResdistributeAndRefineMeshBlocks() function with AMR with second argument pointing
//! to an integer array that is used to store old gid of node on old tree, before creating
//! a new gid based on Z-ordering in the new tree. Thus pglist[n] is a mapping of
//! (new gid n) --> (old gid). Also returns total number of MBs in tree in third argument.

void MeshBlockTree::CreateZOrderedLLList(LogicalLocation *list, int *pglist, int& count) {
  if (lloc_.level == 0) count = 0;
  if (this == proot_ && this == forest_root_ && !chart_roots_.empty()) {
    for (auto *root : chart_roots_) root->AppendZOrderedLLList(list, pglist, count);
  } else {
    AppendZOrderedLLList(list, pglist, count);
  }
}

void MeshBlockTree::AppendZOrderedLLList(LogicalLocation *list, int *pglist, int &count) {
  if (pleaf_ == nullptr) {
    list[count]=lloc_;
    if (pglist != nullptr) {pglist[count]=gid_;}
    gid_=count;
    count++;
  } else {
    for (int n=0; n<nleaf_; n++) {
      if (pleaf_[n] != nullptr) {pleaf_[n]->AppendZOrderedLLList(list, pglist, count);}
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree* MeshBlockTree::FindNeighbor(LogicalLocation myloc,
//!                                   int ox1, int ox2, int ox3, bool amrflag)
//! \brief find a neighboring block, called from the root of the tree
//!        If it is coarser or same level, return the pointer to that block.
//!        If it is a finer block, return the pointer to its parent.
//!        Note that this function must be called on a completed tree only

MeshBlockTree* MeshBlockTree::FindNeighbor(LogicalLocation myloc,
                                           int ox1, int ox2, int ox3, bool amrflag) {
  if (myloc.chart_id != lloc_.chart_id) {
    auto *root = GetChartRoot(myloc.chart_id);
    return root == nullptr ? nullptr : root->FindNeighbor(myloc, ox1, ox2, ox3, amrflag);
  }
  if (!ContainsLocation(myloc) || myloc.level < root_grid_->root_level) return nullptr;
  std::stringstream msg;
  std::int64_t lx, ly, lz;
  int ll;
  int ox, oy, oz;
  MeshBlockTree *bt = proot_;
  lx=myloc.lx1, ly=myloc.lx2, lz=myloc.lx3, ll=myloc.level;

  lx+=ox1; ly+=ox2; lz+=ox3;
  if (lx<0) {
    if (root_grid_->bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic
        || root_grid_->bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
      lx = (std::int64_t(root_grid_->nmb_root[0])<<(ll - root_grid_->root_level)) - 1;
    } else {
      return nullptr;
    }
  }
  if (lx>=std::int64_t(root_grid_->nmb_root[0])<<(ll-root_grid_->root_level)) {
    if (root_grid_->bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic
        || root_grid_->bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic) {
      lx = 0;
    } else {
      return nullptr;
    }
  }
  if (ly<0) {
    if (root_grid_->bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic) {
      ly = (std::int64_t(root_grid_->nmb_root[1])<<(ll - root_grid_->root_level)) - 1;
    } else {
      return nullptr;
    }
  }
  if (ly>=std::int64_t(root_grid_->nmb_root[1])<<(ll-root_grid_->root_level)) {
    if (root_grid_->bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic) {
      ly = 0;
    } else {
      return nullptr;
    }
  }
  std::int64_t num_x3 = std::int64_t(root_grid_->nmb_root[2])<<(ll - root_grid_->root_level);
  if (lz<0) {
    if (root_grid_->bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) {
      lz = num_x3 - 1;
    } else {
      return nullptr;
    }
  }
  if (lz>=num_x3) {
    if (root_grid_->bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic) {
      lz = 0;
    } else {
      return nullptr;
    }
  }

  if (ll<1) return proot_; // single grid; return root

  for (int level=0; level<ll; level++) {
    if (bt->pleaf_ == nullptr) { // leaf
      if (level == ll-1) {
        return bt;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Neighbor search failed; MeshBlockTree broken." << std::endl;
        exit(EXIT_FAILURE);
        return nullptr;
      }
    }
    // find a leaf in the next level
    int sh=ll-level-1;
    ox = ((lx>>sh) & 1) == 1;
    oy = ((ly>>sh) & 1) == 1;
    oz = ((lz>>sh) & 1) == 1;
    bt=bt->GetLeaf(ox, oy, oz);
    if (bt == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Neighbor search failed; MeshBlockTree broken." << std::endl;
      exit(EXIT_FAILURE);
      return nullptr;
    }
  }
  if (bt->pleaf_ == nullptr) { // leaf on the same level
    return bt;
  }
  // one level finer: check if it is a leaf
  ox = oy = oz = 0;
  if (ox1 < 0) ox = 1;
  if (ox2 < 0) oy = 1;
  if (ox3 < 0) oz = 1;
  MeshBlockTree *btleaf = bt->GetLeaf(ox, oy, oz);
  if (btleaf->pleaf_ == nullptr) {
    return bt;  // return this block
  }
  if (!amrflag) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Neighbor search failed. The Block Tree is broken." << std::endl;
    exit(EXIT_FAILURE);
  }
  return bt;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree* MeshBlockTree::FindMeshBlock(LogicalLocation tloc)
//! \brief find MeshBlock with LogicalLocation tloc and return a pointer

MeshBlockTree* MeshBlockTree::FindMeshBlock(LogicalLocation tloc) {
  if (tloc.chart_id != lloc_.chart_id) {
    auto *root = GetChartRoot(tloc.chart_id);
    return root == nullptr ? nullptr : root->FindMeshBlock(tloc);
  }
  if (!ContainsLocation(tloc) || tloc.level < lloc_.level) return nullptr;
  if (tloc.level == lloc_.level) {
    return tloc.lx1 == lloc_.lx1 && tloc.lx2 == lloc_.lx2 && tloc.lx3 == lloc_.lx3
        ? this : nullptr;
  }
  if (pleaf_ == nullptr) return nullptr;
  // get leaf index
  int sh = tloc.level - lloc_.level - 1;
  int mx = (((tloc.lx1>>sh) & 1) == 1);
  int my = (((tloc.lx2>>sh) & 1) == 1);
  int mz = (((tloc.lx3>>sh) & 1) == 1);
  int n = mx + (my<<1) + (mz<<2);
  if (pleaf_[n] == nullptr) {
    return nullptr;
  }
  return pleaf_[n]->FindMeshBlock(tloc);
}
