//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chart_mesh.cpp
//! \brief Connect chart geometry to the native MeshBlock trees and ownership lists.

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
  std::vector<Real> radii;
  std::istringstream input(pin->GetString("mesh_atlas", "shell_radii"));
  std::string token;
  while (std::getline(input, token, ',')) {
    std::istringstream value(token);
    Real radius;
    if (!(value >> radius) || !std::isfinite(radius)) {
      throw std::invalid_argument("mesh_atlas shell_radii contains an invalid radius");
    }
    value >> std::ws;
    if (!value.eof()) {
      throw std::invalid_argument("mesh_atlas shell_radii must be comma-separated reals");
    }
    radii.push_back(radius);
  }
  const std::array<int, 3> core{nmb_rootx1, nmb_rootx2, nmb_rootx3};
  const std::array<int, 2> angular{
      pin->GetOrAddInteger("mesh_atlas", "angular_blocks_2", 1),
      pin->GetOrAddInteger("mesh_atlas", "angular_blocks_3", 1)};
  atlas = std::make_unique<chart_atlas::Atlas>(half_width, radii, core, angular,
      pin->GetOrAddReal("mesh_atlas", "spherical_inner_radius", 0.0),
      pin->GetOrAddBoolean("mesh_atlas", "equiangular", false),
      pin->GetOrAddInteger("mesh_atlas", "evolved_overlap", 0),
      pin->GetOrAddBoolean("mesh_atlas", "outer_endpoint_grid", true));
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
