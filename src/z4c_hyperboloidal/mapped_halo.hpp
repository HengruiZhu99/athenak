#ifndef Z4C_HYPERBOLOIDAL_MAPPED_HALO_HPP_
#define Z4C_HYPERBOLOIDAL_MAPPED_HALO_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <array>
#include <map>
#include <vector>

#include "bvals/mapped_exchange.hpp"
#include "mesh/chart_atlas.hpp"
#include "z4c_hyperboloidal/activity_policy.hpp"

class Mesh;
class MeshBlockPack;
class PhaseTimer;

namespace z4c_hyperboloidal {

struct WeightedDonor {bvals::DonorPoint point; Real weight;};
struct HaloPoint {int m, k, j, i;};

// Geometry/topology metadata only. Native leaf GIDs remain the sole ownership
// authority; no field array or second mesh tree is held here.
class NativeDonorSelector {
 public:
  NativeDonorSelector(const Mesh &mesh, int origin, ActivityPolicy activity={});
  std::vector<WeightedDonor> Cartesian(const Real x[3], int width=6) const;
  std::vector<WeightedDonor> Spherical(const Real x[3], int width=6,
                                     bool force_shell=false) const;
  std::vector<WeightedDonor> Interpolate(const Real x[3], int width=6) const;
  const chart_atlas::BlockMap &Map(int gid) const {return maps_.at(gid);}
  int Cells() const {return n_;}
  int Origin() const {return origin_;}
  int Lower(int gid, int axis) const;
  int Upper(int gid, int axis) const;
  int Owner(const Real x[3], bool shell=false) const;
  bool IsEvolved(const bvals::DonorPoint &point) const;

 private:
  ActivityPolicy activity_;
  int n_, origin_;
  Real half_width_, outer_radius_, first_radius_;
  bool spherical_;
  std::vector<Real> radii_;
  std::vector<chart_atlas::BlockMap> maps_;
  std::vector<int> levels_, order_;
  // Candidate subsets of order_, each in order_ order. A search over a subset
  // visits the leaves that can pass its filter in the same sequence as a search
  // over order_, so every selection and every tie is unchanged.
  std::vector<int> core_, first_shell_;
  std::vector<std::vector<int>> shell_by_face_;
  std::map<std::pair<int,int>,std::vector<int>> by_face_layer_;
  // Uniform buckets over the Cartesian leaves, each extended by its evolved
  // overlap and one further cell. A bucket lists every leaf whose extended box
  // meets it.
  struct CoreGrid {
    Real lower[3]{}, upper[3]{}, inverse[3]{};
    int cells[3]{};
    bool active=false;
    std::vector<int> offsets, gids;
  } grid_;
  int Bucket(Real x,int axis) const;
  std::pair<const int*,const int*> CoreCandidates(const Real x[3]) const;
  // Radial chain of the leaves sharing one angular patch. The nodes are sorted
  // once when no member has a (u,v)-dependent radius.
  struct RadialNode {Real radius; bvals::DonorPoint point;};
  struct RadialChain {
    std::vector<int> gids;
    bool mixed=false, fixed_radii=false;
    std::vector<RadialNode> nodes;
  };
  mutable std::vector<int> chain_of_;
  mutable std::vector<RadialChain> chains_;
  const RadialChain &Chain(int owner) const;
  static void SortRadial(std::vector<RadialNode> &nodes);
  std::vector<WeightedDonor> Overlap(const Real x[3], int width) const;
  std::vector<WeightedDonor> Shell(int owner, const Real x[3],
                                  const Real local[3], int width) const;
  std::vector<WeightedDonor> Tensor(int owner, const Real local[3],
                                   const int first[3], const int length[3]) const;
};

class MappedHalo {
 public:
  struct Row {
    HaloPoint target;
    std::vector<WeightedDonor> donors;
    bool remote=false;
  };
  struct ExteriorTarget {HaloPoint target; Real x[3];};
  MappedHalo() = default;
  void Rebuild(MeshBlockPack *pack, int origin, ActivityPolicy activity={});
  TaskStatus StartReceives();
  TaskStatus PackAndSend(const DvceArray5D<Real> &residual,
                         const DvceArray5D<Real> &physical);
  TaskStatus Receive();
  TaskStatus FillLocal(const DvceArray5D<Real> &residual,
                       const DvceArray5D<Real> &physical);
  TaskStatus FillRemote(const DvceArray5D<Real> &residual,
                        const DvceArray5D<Real> &physical);
  TaskStatus ClearSend();
  TaskStatus ClearRecv();

  void FillExternal(const DvceArray5D<Real> &residual,
      const DvceArray5D<Real> &physical,
      bvals::MappedExchange::ConstValues residual_values,
      bvals::MappedExchange::ConstValues physical_values,
      DvceArray1D<int> active, const std::vector<int> *fixed_slots=nullptr);
  const std::vector<int> &DonorOrdinalsForBatch(const std::vector<int> &slots);
  const std::vector<Row> &Rows() const {return rows_;}
  const std::vector<ExteriorTarget> &ExteriorTargets() const {return exterior_;}
  const std::vector<HaloPoint> &InteriorPoints() const {return interior_points_;}
  const std::vector<HaloPoint> &LocalHaloPoints() const {return local_points_;}
  const std::vector<HaloPoint> &RemoteHaloPoints() const {return remote_points_;}
  const std::vector<HaloPoint> &SurfaceDependentPoints() const {return surface_points_;}
  const bvals::MappedExchange &ResidualExchange() const {return residual_exchange_;}
  const bvals::MappedExchange &PhysicalExchange() const {return physical_exchange_;}

 private:
  ActivityPolicy activity_;
 public:  // nvcc requires public access for device-lambda types and functions
  struct DeviceRow {HaloPoint target; int begin, end;};
  struct DeviceEntry {int ordinal; Real weight;};
 private:
  struct SlotOffset {int row=0, entry=0;};
  using AxisWeights = std::array<std::array<Real, 6>, 3>;
 public:  // nvcc requires public access for device-lambda types and functions
  struct TensorLine { int block, ordinal[6]; Real weight[6]; };
  struct TensorRow {
    HaloPoint target;
    int line[36];
    Real y[6], z[6], sum_x;
  };
  struct BatchPlan {
    DvceArray1D<TensorLine> lines;
    DvceArray1D<TensorRow> tensors;
    DvceArray1D<DeviceRow> rows;
    std::vector<int> donor_ordinals;
  };
 private:
  std::map<int, AxisWeights> tensor_weights_;
  DvceArray2D<Real> line_values_;
  bool AppendTensorBlock(const SlotOffset &begin, const SlotOffset &end,
                         std::vector<TensorLine> &lines,
                         std::vector<TensorRow> &tensors) const;
 public:  // nvcc requires public access for device-lambda types and functions
  void FillTensors(const BatchPlan &plan, const DvceArray5D<Real> &residual,
                   const DvceArray5D<Real> &physical,
                   bvals::MappedExchange::ConstValues rv,
                   bvals::MappedExchange::ConstValues pv, DvceArray1D<int> active) const;
 private:
  std::vector<int> all_slots_;
  std::vector<SlotOffset> slot_offsets_;
  std::map<std::vector<int>,BatchPlan> batches_;
  const BatchPlan &PlanForBatch(const std::vector<int> &slots);
  bvals::MappedExchange residual_exchange_, physical_exchange_;
  std::vector<Row> rows_;
  std::vector<ExteriorTarget> exterior_;
  std::vector<HaloPoint> interior_points_, local_points_, remote_points_, surface_points_;
  DvceArray1D<DeviceRow> local_rows_, remote_rows_;
  DvceArray1D<DeviceEntry> entries_;
  bool ready_=false, packed_=false, received_=false;
  void BuildRows(MeshBlockPack *pack, int origin,
                 std::vector<bvals::DonorPoint> &requests, PhaseTimer &phases);
 public:  // nvcc requires public access for device-lambda types and functions
  void Fill(const DvceArray1D<DeviceRow> &rows, const DvceArray5D<Real> &residual,
            const DvceArray5D<Real> &physical,
            bvals::MappedExchange::ConstValues residual_values={},
            bvals::MappedExchange::ConstValues physical_values={},
            DvceArray1D<int> active={}) const;
};

}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_MAPPED_HALO_HPP_
