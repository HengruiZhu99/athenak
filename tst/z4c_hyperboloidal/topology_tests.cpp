// Native forest and chart-geometry regression tests. These exercise the actual
// MeshBlockTree implementation; no alternate block owner/tree is used.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "mesh/mesh.hpp"
#include "mesh/chart_atlas.hpp"

namespace {

void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void Near(Real actual, Real expected, Real tolerance, const std::string &message) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance*(1+std::abs(expected))) {
    throw std::runtime_error(message + ": " + std::to_string(actual)
                             + " versus " + std::to_string(expected));
  }
}

bool SameLocation(const LogicalLocation &a, const LogicalLocation &b) {
  return a.chart_id == b.chart_id && a.level == b.level && a.lx1 == b.lx1
      && a.lx2 == b.lx2 && a.lx3 == b.lx3;
}

using Roots = std::vector<std::unique_ptr<MeshBlockTree>>;

Roots MakeForest(const chart_atlas::Atlas &atlas, bool balance_seams = true) {
  Roots roots;
  std::vector<MeshBlockTree *> registry;
  for (int chart=0; chart<atlas.NumCharts(); ++chart) {
    MeshBlockTree::RootGrid grid;
    grid.chart_id = chart;
    grid.root_level = atlas.RootLevel();
    grid.nmb_root = atlas.RootBlocks(chart);
    roots.push_back(std::make_unique<MeshBlockTree>(grid));
    roots.back()->CreateRootGrid();
    registry.push_back(roots.back().get());
  }
  roots.front()->SetChartRoots(registry, balance_seams);
  return roots;
}

std::vector<LogicalLocation> Leaves(MeshBlockTree &tree, std::vector<int> *old_gids = nullptr) {
  int count = -1;
  tree.CountMeshBlocks(count);
  Require(count > 0, "nonempty native leaf list");
  std::vector<LogicalLocation> locations(count);
  if (old_gids != nullptr) old_gids->resize(count);
  tree.CreateZOrderedLLList(locations.data(), old_gids == nullptr ? nullptr : old_gids->data(), count);
  Require(count == static_cast<int>(locations.size()), "native count/list agreement");
  for (int gid=0; gid<count; ++gid) {
    const auto *node = tree.FindMeshBlock(locations[gid]);
    Require(node != nullptr && node->IsLeaf(), "leaf lookup through forest");
    Require(tree.FindMeshBlock(locations[gid])->GetGID() == gid, "one global GID space");
  }
  return locations;
}

std::uint64_t Morton(const LogicalLocation &loc, int fine_level) {
  const int shift = fine_level-loc.level;
  const std::uint64_t coord[3] = {std::uint64_t(loc.lx1)<<shift,
                                 std::uint64_t(loc.lx2)<<shift,
                                 std::uint64_t(loc.lx3)<<shift};
  std::uint64_t key = 0;
  for (int bit=0; bit<fine_level; ++bit) {
    for (int axis=0; axis<3; ++axis) key |= ((coord[axis]>>bit)&1) << (3*bit+axis);
  }
  return key;
}

void CheckOrderAndBalance(const std::vector<LogicalLocation> &locations) {
  int fine_level = 0;
  for (const auto &loc : locations) fine_level = std::max(fine_level, int(loc.level));
  for (std::size_t m=1; m<locations.size(); ++m) {
    const auto &a = locations[m-1], &b = locations[m];
    Require(a.chart_id < b.chart_id || (a.chart_id == b.chart_id
        && Morton(a, fine_level) < Morton(b, fine_level)), "chart/Morton leaf ordering");
  }
  // Independently check the native same-chart 2:1 condition on touching leaf
  // boxes, including edges/corners. Cross-chart closure belongs to the atlas
  // interface layer and is deliberately not asserted by this foundation test.
  for (std::size_t m=0; m<locations.size(); ++m) {
    for (std::size_t n=m+1; n<locations.size(); ++n) {
      const auto &a = locations[m], &b = locations[n];
      if (a.chart_id != b.chart_id) continue;
      const int ia[3] = {a.lx1, a.lx2, a.lx3}, ib[3] = {b.lx1, b.lx2, b.lx3};
      bool touches = true;
      for (int axis=0; axis<3; ++axis) {
        const int alo = ia[axis] << (fine_level-a.level);
        const int ahi = (ia[axis]+1) << (fine_level-a.level);
        const int blo = ib[axis] << (fine_level-b.level);
        const int bhi = (ib[axis]+1) << (fine_level-b.level);
        if (ahi < blo || bhi < alo) touches = false;
      }
      if (touches) Require(std::abs(a.level-b.level) <= 1, "same-chart 2:1 closure");
    }
  }
}

void TestForest() {
  const std::vector<Real> radii{.16, .24, .32, .40, .48, .56, .64, .72, .80, .88, .94, 1};
  chart_atlas::Atlas atlas(.1, radii, {3, 3, 3}, {1, 1}, .08, true, 3, true);
  Require(atlas.RootLevel() == 4, "common root level accommodates twelve radial roots");
  auto roots = MakeForest(atlas);
  auto &forest = *roots.front();
  auto locations = Leaves(forest);
  Require(locations.size() == 99, "27 core plus 72 shell root blocks");
  CheckOrderAndBalance(locations);

  LogicalLocation core{1, 1, 1, atlas.RootLevel(), 0};
  const int old_core_gid = forest.FindMeshBlock(core)->GetGID();
  int added = 0;
  forest.FindMeshBlock(core)->Refine(added);
  Require(added == 7, "refine core parent into eight native leaves");
  std::vector<int> old_gids;
  locations = Leaves(forest, &old_gids);
  Require(locations.size() == 106, "N12 topology has 34 core and 72 shell leaves");
  Require(std::count(old_gids.begin(), old_gids.end(), old_core_gid) == 8,
          "children inherit old native GID for redistribution");

  LogicalLocation shell{5, 0, 0, atlas.RootLevel(), 1};
  auto *shell_node = forest.FindMeshBlock(shell);
  Require(shell_node != nullptr && shell_node != forest.FindMeshBlock(core),
          "chart id separates otherwise shared logical coordinates");
  Require(forest.FindNeighbor(shell, 1, 0, 0)->GetLocation().chart_id == 1,
          "same-chart neighbor dispatch from chart zero");
  Require(forest.FindNeighbor(shell, 0, 1, 0) == nullptr,
          "chart seam is not a Cartesian wraparound neighbor");
  Require(roots[6]->FindMeshBlock(shell) == shell_node, "dispatch from any chart root");
  added = 0;
  shell_node->Refine(added);
  Require(added == 7, "shell roots use native refinement");
  LogicalLocation child{10, 0, 0, atlas.RootLevel()+1, 1};
  auto *child_node = forest.FindMeshBlock(child);
  Require(child_node != nullptr && &child_node->GetRootGrid() == &shell_node->GetRootGrid(),
          "children inherit immutable chart context");
  added = 0;
  child_node->Refine(added);
  Require(added > 7, "native refinement closes a coarser radial neighbor");
  locations = Leaves(forest);
  CheckOrderAndBalance(locations);

  int removed = 0;
  shell_node->Derefine(removed);
  Require(removed == 0, "cannot discard a refined grandchild");
  child_node->Derefine(removed);
  Require(removed == 7, "native child derefinement");
  removed = 0;
  shell_node->Derefine(removed);
  Require(removed == 7 && shell_node->IsLeaf(), "native shell parent derefinement");
  locations = Leaves(forest);
  CheckOrderAndBalance(locations);

  // Reconstruct the exact forest using the native restart insertion path.
  auto restored = MakeForest(atlas);
  for (const auto &loc : locations) restored[0]->AddNodeWithoutRefinement(loc);
  const auto restart_locations = Leaves(*restored[0]);
  Require(restart_locations.size() == locations.size(), "restart forest leaf count");
  for (std::size_t m=0; m<locations.size(); ++m) {
    Require(SameLocation(locations[m], restart_locations[m]), "restart forest identity/order");
  }

  LogicalLocation invalid{0, 0, 0, atlas.RootLevel(), 7};
  Require(forest.FindMeshBlock(invalid) == nullptr, "unknown chart cannot alias chart zero");
  bool caught = false;
  try {forest.AddNode(invalid, added);} catch (const std::out_of_range &) {caught = true;}
  Require(caught, "unknown chart rejected during insertion");
  invalid = {12, 0, 0, atlas.RootLevel(), 1};
  Require(forest.FindMeshBlock(invalid) == nullptr, "chart-specific root bounds");

  // Concurrent ordinary 1D/2D trees formerly shared the static root/nleaf
  // variables. Their destruction/refinement must leave this forest untouched.
  {
    MeshBlockTree::RootGrid grid;
    grid.ndim = 1;
    grid.bcs[0] = grid.bcs[1] = BoundaryFlag::periodic;
    MeshBlockTree line(grid);
    line.CreateRootGrid();
    Require(line.FindNeighbor({0, 0, 0, 0}, -1, 0, 0) == &line,
            "single ordinary periodic block");
    added = 0;
    line.Refine(added);
    Require(added == 1 && Leaves(line).size() == 2, "independent binary refinement");
    grid.ndim = 2;
    grid.root_level = 2;
    grid.nmb_root = {3, 2, 1};
    MeshBlockTree plane(grid);
    plane.CreateRootGrid();
    Require(Leaves(plane).size() == 6, "incomplete ordinary quad root grid");
    Require(plane.FindNeighbor({0, 0, 0, 2}, -1, 0, 0)->GetLocation().lx1 == 2,
            "periodic wrap uses this tree's root dimensions");
    added = 0;
    plane.FindMeshBlock({2, 1, 0, 2})->Refine(added);
    Require(added == 3 && Leaves(plane).size() == 9, "independent quad refinement");
  }
  Require(Leaves(forest).size() == locations.size(), "ordinary tree lifetime isolation");

  // Keep AthenaK's supported logical level 31. A graded binary chain is small,
  // while the right endpoint probes the exact INT32_MAX / 2^31 boundary.
  MeshBlockTree::RootGrid deep_grid;
  deep_grid.ndim = 1;
  deep_grid.bcs[0] = deep_grid.bcs[1] = BoundaryFlag::periodic;
  MeshBlockTree deep(deep_grid);
  deep.CreateRootGrid();
  const LogicalLocation last{std::numeric_limits<std::int32_t>::max(), 0, 0, 31};
  added = 0;
  deep.AddNode(last, added);
  Require(added < 200, "level-31 regression remains a bounded graded chain");
  Require(deep.FindMeshBlock(last) != nullptr && deep.FindMeshBlock(last)->IsLeaf(),
          "level-31 insertion and exact lookup");
  const auto *wrapped = deep.FindNeighbor(last, 1, 0, 0);
  Require(wrapped != nullptr && wrapped->GetLocation().lx1 == 0,
          "level-31 periodic wrap uses a 64-bit extent");
  const auto deep_locations = Leaves(deep);
  Require(deep_locations.size() == std::size_t(added+1), "level-31 chain block count");
  MeshBlockTree restored_deep(deep_grid);
  restored_deep.CreateRootGrid();
  for (const auto &loc : deep_locations) restored_deep.AddNodeWithoutRefinement(loc);
  Require(restored_deep.FindMeshBlock(last) != nullptr, "level-31 restart insertion");

  MeshBlockTree::RootGrid high_root_grid;
  high_root_grid.ndim = 1;
  high_root_grid.root_level = 31;
  MeshBlockTree high_root(high_root_grid);
  high_root.CreateRootGrid();
  Require(Leaves(high_root).size() == 1
          && high_root.FindMeshBlock({0, 0, 0, 31}) != nullptr,
          "a common root level of 31 remains representable");
}

void TestMap(const chart_atlas::Atlas &atlas, const LogicalLocation &loc) {
  const auto map = atlas.GetBlockMap(loc);
  const auto bounds = atlas.PhysicalBounds(loc);
  const Real samples[3][3] = {{.17, .53, .81}, {.71, .22, .37}, {.43, .66, .58}};
  for (const auto &local : samples) {
    Real x[3], inverse[3];
    map.Forward(local, x);
    map.Inverse(x, inverse);
    const auto point = map.Geometry(local);
    for (int a=0; a<3; ++a) {
      Near(inverse[a], local[a], 3e-13, "map inverse roundtrip");
      Near(point.x[a], x[a], 1e-15, "geometry point agreement");
      Require(x[a] >= bounds.lo[a]-1e-14 && x[a] <= bounds.hi[a]+1e-14,
              "physical AABB contains mapped points");
    }
    const Real h = 2e-6*std::max(Real(.1), std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]));
    for (int i=0; i<3; ++i) {
      Real xp[3]{x[0], x[1], x[2]}, xm[3]{x[0], x[1], x[2]};
      xp[i] += h;
      xm[i] -= h;
      Real qp[3], qm[3];
      map.Inverse(xp, qp);
      map.Inverse(xm, qm);
      const auto plus = map.Geometry(qp), minus = map.Geometry(qm);
      for (int A=0; A<3; ++A) {
        Near((qp[A]-qm[A])/(2*h), point.J[A][i], 2e-8,
             "analytic inverse Jacobian matches finite differences");
        for (int j=0; j<3; ++j) {
          Near((plus.J[A][j]-minus.J[A][j])/(2*h), point.H[A][j][i], 2e-7,
               "analytic inverse Hessian matches finite differences");
          Near(point.H[A][i][j], point.H[A][j][i], 1e-14, "symmetric inverse Hessian");
        }
      }
    }
  }
  // Dense samples catch spherical-face extrema missed by corner-only AABBs.
  for (int k=0; k<=4; ++k) for (int j=0; j<=4; ++j) for (int i=0; i<=4; ++i) {
    const Real local[3] = {Real(i)/4, Real(j)/4, Real(k)/4};
    Real x[3];
    map.Forward(local, x);
    for (int a=0; a<3; ++a) {
      Require(x[a] >= bounds.lo[a]-1e-14 && x[a] <= bounds.hi[a]+1e-14,
              "AABB contains face interiors and edges");
    }
  }
}

void TestGeometry() {
  static_assert(std::is_trivially_copyable<chart_atlas::BlockMap>::value,
                "block map must be device-copyable");
  const chart_atlas::Atlas cube(.1, {.2, .4, 1}, {3, 3, 3}, {2, 3}, 0, false, 3, true);
  const chart_atlas::Atlas sphere(.1, {.16, .24, .94, 1}, {3, 3, 3}, {1, 1}, .08, true, 3, true);
  const chart_atlas::Atlas sphere_copy(.1, {.16, .24, .94, 1}, {3, 3, 3}, {1, 1}, .08, true, 3, true);
  const chart_atlas::Atlas changed_knots(.1, {.16, .24, .95, 1}, {3, 3, 3}, {1, 1}, .08, true, 3, true);
  const chart_atlas::Atlas changed_overlap(.1, {.16, .24, .94, 1}, {3, 3, 3}, {1, 1}, .08, true, 2, true);
  const chart_atlas::Atlas changed_offset(.1, {.16, .24, .94, 1}, {3, 3, 3}, {1, 1}, .08, true, 3, false);
  Require(sphere.Identity() == sphere_copy.Identity(), "stable exact atlas identity");
  for (const auto *changed : {&changed_knots, &changed_overlap, &changed_offset}) {
    Require(sphere.Identity() != changed->Identity(), "restart detects changed chart geometry/convention");
  }
  for (const auto *atlas : {&cube, &sphere}) {
    for (int chart=0; chart<atlas->NumCharts(); ++chart) {
      const auto dims = atlas->RootBlocks(chart);
      for (int level=0; level<=2; ++level) {
        const int factor = 1 << level;
        for (int radial=0; radial<dims[0]; ++radial) {
          TestMap(*atlas, {radial*factor, 0, 0, atlas->RootLevel()+level, chart});
          TestMap(*atlas, {(radial+1)*factor-1, dims[1]*factor-1, dims[2]*factor-1,
                           atlas->RootLevel()+level, chart});
        }
      }
    }
  }
  const int level = sphere.RootLevel();
  for (int chart=1; chart<7; ++chart) {
    const auto outer = sphere.GetBlockMap({3, 0, 0, level, chart});
    Require(outer.outer_surface && outer.node_offset[0] == 0, "outer endpoint radial grid");
    const Real local[3] = {1, .27, .63};
    Real x[3];
    outer.Forward(local, x);
    Near(std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]), 1, 2e-14,
         "every chart terminates at the common scri sphere");
    const auto inner = sphere.GetBlockMap({0, 0, 0, level, chart});
    Require(inner.overlap_sides[0] == 0 && inner.overlap_sides[1] == 3,
            "spherical inner radial overlap contract");
    for (int f=2; f<6; ++f) Require(inner.overlap_sides[f] == 3, "angular evolved overlap");
  }
  const auto center = sphere.GetBlockMap({1, 1, 1, level, 0});
  for (int f=0; f<6; ++f) Require(center.overlap_sides[f] == 0, "no core internal overlap");
  for (int octant=0;octant<8;++octant) {
    const auto child=sphere.GetBlockMap({2+(octant&1),2+((octant>>1)&1),
                                         2+((octant>>2)&1),level+1,0});
    for (int a=0;a<3;++a) {
      const Real middle=(center.lo[a]+center.hi[a])/2;
      Require(child.lo[a]==((octant&(1<<a)) ? middle : center.lo[a]) &&
              child.hi[a]==((octant&(1<<a)) ? center.hi[a] : middle),
              "refined core coordinates preserve parent midpoint bits");
    }
  }
  const auto corner = sphere.GetBlockMap({0, 0, 0, level, 0});
  for (int a=0; a<3; ++a) {
    Require(corner.overlap_sides[2*a] == 3 && corner.overlap_sides[2*a+1] == 0,
            "only outer core faces carry evolved overlap");
  }

  // Face orientation, exact root radial knots, and neighboring chart seam maps.
  for (int chart=1; chart<7; ++chart) {
    const auto map = sphere.GetBlockMap({1, 0, 0, level, chart});
    const Real axis[3] = {.5, .5, .5};
    Real x[3];
    map.Forward(axis, x);
    const int dominant = (chart-1)/2;
    Near(x[dominant], chart%2 ? .20 : -.20, 1e-14, "shell face sign and radial knot map");
    for (int a=0; a<3; ++a) if (a != dominant) Near(x[a], 0, 1e-14, "shell face axis");
  }
  const auto xp = sphere.GetBlockMap({1, 0, 0, level, 1});
  const auto yp = sphere.GetBlockMap({1, 0, 0, level, 3});
  const Real on_x[3] = {.37, 1, .69}, on_y[3] = {.37, .69, 1};
  Real xx[3], xy[3];
  xp.Forward(on_x, xx);
  yp.Forward(on_y, xy);
  for (int a=0; a<3; ++a) Near(xx[a], xy[a], 1e-14, "shared chart edge orientation");

  const chart_atlas::Atlas deep_atlas(.1, {.2});
  const auto deep = deep_atlas.GetBlockMap({std::numeric_limits<std::int32_t>::max(),
                                           0, 0, 31, 1});
  Require(deep.outer_surface && deep.hi[0] == 1 && deep.lo[0] < deep.hi[0],
          "level-31 shell bounds preserve the outer radial endpoint");
}

void CheckSeamBalance(MeshBlockTree &forest) {
  forest.ValidateSeamBalance();
  const auto locations = Leaves(forest);
  CheckOrderAndBalance(locations);
  for (const auto &loc : locations) {
    if (loc.chart_id == 0) continue;
    for (int axis=1; axis<3; ++axis) for (int side : {-1, 1}) {
      const auto seam = chart_atlas::Atlas::AngularSeam(loc.chart_id, axis, side);
      for (const auto *neighbor : forest.FindSeamNeighbors(loc, axis, side)) {
        Require(neighbor->IsLeaf(), "seam topology returns native leaves");
        const auto &other = neighbor->GetLocation();
        Require(std::abs(other.level-loc.level) <= 1, "cross-chart 2:1 closure");
        const auto reverse = forest.FindSeamNeighbors(other, seam.neighbor_axis, seam.neighbor_side);
        Require(std::find(reverse.begin(), reverse.end(), forest.FindMeshBlock(loc)) != reverse.end(),
                "reciprocal seam leaf intersection");
      }
    }
  }
}

void TestSeams() {
  const chart_atlas::Atlas atlas(.1, {.2, .4, 1}, {1, 1, 1}, {1, 1}, .08, true);
  const int level = atlas.RootLevel();
  for (int chart=1; chart<=6; ++chart) {
    for (int axis=1; axis<3; ++axis) for (int side : {-1, 1}) {
      const auto seam = chart_atlas::Atlas::AngularSeam(chart, axis, side);
      const auto reverse = chart_atlas::Atlas::AngularSeam(
          seam.neighbor_chart, seam.neighbor_axis, seam.neighbor_side);
      Require(reverse.neighbor_chart == chart && reverse.neighbor_axis == axis
              && reverse.neighbor_side == side, "all 24 oriented seams are reciprocal");
      const auto source_map = atlas.GetBlockMap({1, 0, 0, level, chart});
      const auto target_map = atlas.GetBlockMap({1, 0, 0, level, seam.neighbor_chart});
      Real qs[3]{.37, 0, 0}, qt[3]{.37, 0, 0}, xs[3], xt[3];
      qs[axis] = side < 0 ? 0 : 1;
      qs[seam.tangent_axis] = .21;
      qt[seam.neighbor_axis] = seam.neighbor_side < 0 ? 0 : 1;
      qt[seam.neighbor_tangent_axis] = .21;
      source_map.Forward(qs, xs);
      target_map.Forward(qt, xt);
      for (int a=0; a<3; ++a) Near(xs[a], xt[a], 2e-14, "seam orientation agrees with physical maps");

      auto roots = MakeForest(atlas);
      auto &forest = *roots[0];
      const LogicalLocation source{1, 0, 0, level, chart};
      auto *source_node = forest.FindMeshBlock(source);
      int added = 0;
      source_node->Refine(added);
      Require(added == 7, "first shell refinement needs no same-level seam closure");
      LogicalLocation child{2, 0, 0, level+1, chart};
      if (axis == 1) child.lx2 = side > 0 ? 1 : 0;
      else child.lx3 = side > 0 ? 1 : 0;
      auto *fine_parent = forest.FindMeshBlock(child);
      fine_parent->Refine(added);
      auto *target_root = forest.FindMeshBlock({1, 0, 0, level, seam.neighbor_chart});
      Require(!target_root->IsLeaf(), "refinement propagates across each oriented shell seam");
      CheckSeamBalance(forest);
      Require(forest.FindNeighbor(child, 0, axis == 1 ? side : 0, axis == 2 ? side : 0) == nullptr,
              "seams remain outside the conforming Cartesian Bvals graph");
      int removed = 0;
      target_root->Derefine(removed);
      Require(removed == 0, "derefine cannot create a two-level cross-seam jump");
      fine_parent->Derefine(removed);
      Require(removed == 7, "fine seam parent can safely coarsen first");
      removed = 0;
      target_root->Derefine(removed);
      Require(removed == 7, "neighbor can coarsen once the seam level difference allows it");
      CheckSeamBalance(forest);
    }
  }

  // A u/v swap need not map one root edge cell to one edge cell when root
  // angular counts differ. Exact overlaps must include both intersecting roots
  // and their radial edge contacts, rather than rounding a face-center lookup.
  const chart_atlas::Atlas rectangular(.1, {.2, .4, 1}, {1, 1, 1}, {2, 3}, .08, true);
  auto roots = MakeForest(rectangular);
  auto &forest = *roots[0];
  const LogicalLocation source{1, 1, 1, rectangular.RootLevel(), 1};
  const auto neighbors = forest.FindSeamNeighbors(source, 1, 1);
  Require(neighbors.size() == 6, "unequal angular roots include every intersecting seam leaf");
  for (int radial=0; radial<3; ++radial) for (int tangent=0; tangent<2; ++tangent) {
    const LogicalLocation expected{radial, tangent, 2, rectangular.RootLevel(), 3};
    Require(std::find(neighbors.begin(), neighbors.end(), forest.FindMeshBlock(expected))
            != neighbors.end(), "exact angular/radial seam footprint");
  }
  int added = 0;
  forest.FindMeshBlock(source)->Refine(added);
  forest.FindMeshBlock({2, 3, 2, rectangular.RootLevel()+1, 1})->Refine(added);
  CheckSeamBalance(forest);

  // Old/incomplete chart restart records can describe a forest that is balanced
  // within each chart but not across faces. Reconstruct without mutation and
  // reject the incompatible seam levels before any state is assigned to leaves.
  auto unbalanced = MakeForest(atlas, false);
  added = 0;
  unbalanced[0]->AddNode({4, 0, 0, level+2, 1}, added);
  auto restored = MakeForest(atlas);
  for (const auto &loc : Leaves(*unbalanced[0])) restored[0]->AddNodeWithoutRefinement(loc);
  bool caught = false;
  try {restored[0]->ValidateSeamBalance();} catch (const std::runtime_error &) {caught = true;}
  Require(caught, "restart rejects an unbalanced chart seam without inventing saved state");
}

void TestShellSMR() {
  const chart_atlas::Atlas atlas(.1, {.16, .24, .32, .40, .48, .56, .64, .72, .80, .88, .94, 1},
                                {3, 3, 3}, {1, 1}, .08, true);
  const chart_atlas::Bounds region{{.24, -.5, -1}, {.32, .25, 0}};
  const auto range = atlas.ShellRefinementParents(1, atlas.RootLevel()+2, region);
  Require(range.first[0] == 4 && range.last[0] == 6, "SMR physical radial knots select exact parents");
  Require(range.first[1] == 0 && range.last[1] == 2
          && range.first[2] == 0 && range.last[2] == 1, "SMR normalized angular ranges");
  auto roots = MakeForest(atlas);
  auto &forest = *roots[0];
  int added = 0;
  for (auto k=range.first[2]; k<range.last[2]; ++k) {
    for (auto j=range.first[1]; j<range.last[1]; ++j) {
      for (auto i=range.first[0]; i<range.last[0]; ++i) {
        forest.AddNode({int(2*i), int(2*j), int(2*k), atlas.RootLevel()+2, 1}, added);
      }
    }
  }
  CheckSeamBalance(forest);
  Require(forest.FindMeshBlock({8, 0, 0, atlas.RootLevel()+2, 1})->IsLeaf(),
          "shell SMR inserts selected child locations through native tree");
  const chart_atlas::Bounds endpoint{{1, 1, 1}, {1, 1, 1}};
  const auto point = atlas.ShellRefinementParents(6, atlas.RootLevel()+2, endpoint);
  Require(point.first[0] == 23 && point.last[0] == 24 && point.first[1] == 1
          && point.last[1] == 2, "SMR point on outer boundary is clamped to last parent");
  const chart_atlas::Atlas cube(.1, {.2, .4, 1});
  bool caught = false;
  try {
    cube.ShellRefinementParents(1, cube.RootLevel()+1, {{.15, -1, -1}, {.3, 1, 1}});
  } catch (const std::invalid_argument &) {caught = true;}
  Require(caught, "physical-radius SMR rejects the non-spherical transition layer");
  const auto outer = cube.ShellRefinementParents(1, cube.RootLevel()+1, {{.2, -1, -1}, {.4, 1, 1}});
  Require(outer.first[0] == 1 && outer.last[0] == 2, "cube atlas spherical layers remain selectable");
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    TestForest();
    TestGeometry();
    TestSeams();
    TestShellSMR();
    std::cout << "native forest and chart geometry tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << "topology test failure: " << error.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
