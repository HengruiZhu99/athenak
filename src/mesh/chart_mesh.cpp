//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chart_mesh.cpp
//! \brief Connect chart geometry to the native MeshBlock trees and ownership lists.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh.hpp"
#include "chart_atlas.hpp"

void Mesh::ConfigureChartAtlas(ParameterInput *pin, bool restarting) {
  if (!pin->DoesBlockExist("mesh_atlas")) return;
  if (!three_d || pin->GetString("mesh_atlas", "type") != "cubed_sphere") {
    throw std::invalid_argument("mesh_atlas requires a three-dimensional cubed_sphere");
  }
  const Real half_width = mesh_size.x1max;
  if (half_width <= 0 || mesh_size.x1min != -half_width ||
      mesh_size.x2min != -half_width || mesh_size.x3min != -half_width ||
      mesh_size.x2max != half_width || mesh_size.x3max != half_width) {
    throw std::invalid_argument("cubed_sphere core must be a cube centered at the origin");
  }
  for (auto bc : mesh_bcs) {
    if (bc != BoundaryFlag::mapped) {
      throw std::invalid_argument("cubed_sphere core faces must use mapped boundaries");
    }
  }
  const bool explicit_radii = pin->DoesParameterExist("mesh_atlas", "shell_radii");
  const bool has_rmin = pin->DoesParameterExist("mesh_atlas", "rmin");
  if (has_rmin && pin->DoesParameterExist("mesh_atlas", "spherical_inner_radius"))
    throw std::invalid_argument("Specify only one inner radius in mesh_atlas");
  const Real inner = has_rmin ? pin->GetReal("mesh_atlas", "rmin")
      : pin->GetOrAddReal("mesh_atlas", "spherical_inner_radius", 0.0);
  std::vector<Real> radii;
  std::string token;
  if (explicit_radii) {
    if (pin->DoesParameterExist("mesh_atlas", "nshell") ||
        pin->DoesParameterExist("mesh_atlas", "rmax"))
      throw std::invalid_argument("shell_radii conflicts with nshell or rmax");
    std::istringstream input(pin->GetString("mesh_atlas", "shell_radii"));
    while (std::getline(input, token, ',')) {
      std::istringstream value(token);
      Real radius;
      if (!(value >> radius) || !std::isfinite(radius)) {
        throw std::invalid_argument("mesh_atlas shell_radii contains an invalid radius");
      }
      value >> std::ws;
      if (!value.eof()) {
        throw std::invalid_argument("shell_radii must be comma-separated reals");
      }
      radii.push_back(radius);
    }
  } else {
    const int shells = pin->GetInteger("mesh_atlas", "nshell");
    const Real outer = pin->GetReal("mesh_atlas", "rmax");
    if (!(shells > 0 && inner > 0 && std::isfinite(outer) && outer > inner))
      throw std::invalid_argument("Uniform shells need nshell > 0 and 0 < rmin < rmax");
    radii.resize(shells);
    for (int i = 0; i < shells; ++i)
      radii[i] = inner + (outer - inner) * (Real(i + 1) / shells);
    radii.back() = outer;
  }
  const std::array<int, 3> core{nmb_rootx1, nmb_rootx2, nmb_rootx3};
  std::vector<int> radial_blocks;
  if (pin->DoesParameterExist("mesh_atlas", "shell_radial_blocks")) {
    const auto specification = pin->GetString("mesh_atlas", "shell_radial_blocks");
    std::istringstream counts(specification);
    while (std::getline(counts, token, ',')) {
      std::istringstream value(token);
      int count;
      if (!(value >> count) || count <= 0)
        throw std::invalid_argument("shell_radial_blocks requires positive integers");
      value >> std::ws;
      if (!value.eof())
        throw std::invalid_argument("shell_radial_blocks must be comma-separated integers");
      radial_blocks.push_back(count);
    }
    if (radial_blocks.empty() || specification.back() == ',')
      throw std::invalid_argument("shell_radial_blocks must not contain empty entries");
  }
  std::vector<int> levels(radii.size(), 0);
  bool refine_shells = false;
  for (const auto &block : pin->block) {
    if (block.block_name.compare(0, 13, "refined_shell") != 0) continue;
    if (!radial_blocks.empty() || inner <= 0 || radii.empty())
      throw std::invalid_argument(
          "refined_shell requires spherical layers without shell_radial_blocks");
    refine_shells = true;
    const auto &name = block.block_name;
    const Real lo = pin->GetReal(name, "rmin"), hi = pin->GetReal(name, "rmax");
    const int level = pin->GetInteger(name, "level");
    if (!(inner <= lo && lo < hi && hi <= radii.back() && level > 0 && level < 31))
      throw std::invalid_argument("Invalid radius range or dyadic level in " + name);
    bool selected = false;
    for (std::size_t i = 0; i < radii.size(); ++i) {
      const Real midpoint = 0.5 * ((i == 0 ? inner : radii[i-1]) + radii[i]);
      if (lo <= midpoint && midpoint < hi) {
        levels[i] = std::max(levels[i], level);
        selected = true;
      }
    }
    if (!selected) throw std::invalid_argument(name + " contains no base shell midpoint");
  }
  if (refine_shells) {
    // Extend the requested levels by the minimum collars needed for 2:1 grading.
    for (std::size_t i = 1; i < levels.size(); ++i)
      levels[i] = std::max(levels[i], levels[i-1] - 1);
    for (std::size_t i = levels.size() - 1; i > 0; --i)
      levels[i-1] = std::max(levels[i-1], levels[i] - 1);
    for (int level : levels) radial_blocks.push_back(1 << level);
  }
  const std::array<int, 2> angular{
      pin->GetOrAddInteger("mesh_atlas", "angular_blocks_2", 1),
      pin->GetOrAddInteger("mesh_atlas", "angular_blocks_3", 1)};
  atlas = std::make_unique<chart_atlas::Atlas>(half_width, radii, core, angular,
      inner, pin->GetOrAddBoolean("mesh_atlas", "equiangular", false),
      pin->GetOrAddInteger("mesh_atlas", "evolved_overlap", 0),
      pin->GetOrAddBoolean("mesh_atlas", "outer_endpoint_grid", true), radial_blocks);
  if (restarting && root_level != atlas->RootLevel()) {
    throw std::invalid_argument("restart root level differs from its chart atlas");
  }
  root_level = atlas->RootLevel();
}

void Mesh::CreateMeshBlockTrees() {
  chart_trees.clear();
  ptree = std::make_unique<MeshBlockTree>(this);
  ptree->CreateRootGrid();
  if (!atlas) return;

  std::vector<MeshBlockTree *> roots{ptree.get()};
  for (int chart=1; chart<=6; ++chart) {
    MeshBlockTree::RootGrid grid;
    grid.chart_id = chart;
    grid.root_level = root_level;
    grid.ndim = 3;
    grid.nmb_root = atlas->RootBlocks(chart);
    grid.bcs.fill(BoundaryFlag::mapped);
    chart_trees.push_back(std::make_unique<MeshBlockTree>(this, grid));
    chart_trees.back()->CreateRootGrid();
    roots.push_back(chart_trees.back().get());
  }
  ptree->SetChartRoots(roots, true);
}
