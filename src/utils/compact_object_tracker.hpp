//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef UTILS_COMPACT_OBJECT_TRACKER_HPP_
#define UTILS_COMPACT_OBJECT_TRACKER_HPP_

#include <cstdio>
#include <fstream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"

// Forward declaration
class Mesh;
class ParameterInput;

//! \class CompactObjectTracker
//! \brief Tracks a single puncture
class CompactObjectTracker {
  static constexpr int ndim = 3;
  enum CompactObjectType { BlackHole, NeutronStar };
  enum TrackerMode { ODE, Walk };

 public:
  //! Initialize a tracker
  CompactObjectTracker(Mesh *pmesh, ParameterInput *pin, int n,
                       const std::string &input_block);
  //! Destructor (will close output file)
  ~CompactObjectTracker();
  //! Interpolate the shift vector to the puncture position
  void InterpolateVelocity(MeshBlockPack *pmbp);
  //! Update and broadcast the puncture position
  void EvolveTracker(MeshBlockPack *pmbp);
  //! Write data to file
  void WriteTracker();
  //! Get position array
  inline Real * GetPos() {
    return &pos[0];
  }
  //! Get position
  inline Real GetPos(int a) const {
    return pos[a];
  }
  //! Set the position of the CO
  inline void SetPos(Real npos[ndim]) {
    std::memcpy(pos, npos, ndim*sizeof(Real));
  }
  //! Get wanted refinement level
  inline int GetReflevel() const {
    return reflevel;
  }
  //! Get radius
  inline Real GetRadius() const {
    return radius;
  }

 private:
  bool owns_compact_object;
  CompactObjectType type;
  TrackerMode mode;
  Real vel[ndim];
  int reflevel;         // requested minimum refinement level (-1 for infinity)
  Real radius;          // nominal radius of the object (for the AMR driver)
  Mesh const *pmesh;
  int out_every;
  std::ofstream ofile;
  Real pos[ndim];
};

#endif  // UTILS_COMPACT_OBJECT_TRACKER_HPP_
