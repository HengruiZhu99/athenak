// Regression for ownership before AddCoordinatesAndPhysics, including -m exits.
#include <cstring>
#include <iostream>
#include <new>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"

int main() {
  alignas(MeshBlockPack) unsigned char storage[sizeof(MeshBlockPack)];
  std::memset(storage, 0xa5, sizeof(storage));
  auto *pack = new (storage) MeshBlockPack(nullptr, 0, 0);
  MeshBlock *null_mb = nullptr;
  Coordinates *null_coord = nullptr;
  // Inspect object representations without evaluating an indeterminate pointer.
  bool mb_ok = std::memcmp(&pack->pmb, &null_mb, sizeof(null_mb)) == 0;
  bool coord_ok = std::memcmp(&pack->pcoord, &null_coord, sizeof(null_coord)) == 0;
  std::cout << "meshblocks_initialized=" << mb_ok
            << " coordinates_initialized=" << coord_ok << '\n';
  // Clean up safely even when running the regression against the unfixed object.
  if (!mb_ok) pack->pmb = nullptr;
  if (!coord_ok) pack->pcoord = nullptr;
  pack->~MeshBlockPack();
  return (mb_ok && coord_ok) ? 0 : 1;
}
