#ifndef Z4C_Z4C_HPP_
#define Z4C_Z4C_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.hpp
//! \brief definitions for Z4c class

#include <map>
#include <memory>    // make_unique, unique_ptr
#include <list>
#include <string>
#include <vector>
#include "athena.hpp"
#include "utils/finite_diff.hpp"
#include "utils/cart_grid.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "athena_tensor.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "z4c/amr_jump_diagnostic.hpp"
#include "z4c/chi_parent_provenance.hpp"
#include "z4c/telegraph_damping.hpp"
#include "z4c/z4c_grid.hpp"

// forward declarations
class Coordinates;
class Driver;
class CompactObjectTracker;
class FastFlow;
class HorizonDump;

namespace z4c {
class Z4c_AMR;

// Default-off AMR transfer ablation.  This changes only Z4c interlevel transfer;
// the bulk finite-difference stencil and time integrator remain configured separately.
enum class Z4cAMRTransfer {
  high_order,
  limited_o2,
};

enum class Z4cShiftMode {
  gamma_driver,
  prescribed_zero,
};

enum class Z4cShiftAdvectionOrder {
  spatial,
  o2,
};

// The name is deliberately a writer/checkpoint label, rather than an inferred
// root cause.  A failure record can therefore distinguish a bad RK update from
// a later AMR, communication, or boundary write without retaining a large
// per-cell provenance array.
enum class Z4cStateCheckpoint {
  pre_rhs,
  post_rk_update,
  post_restriction,
  post_receive,
  post_physical_bc,
  post_prolongation,
  pre_algconstr,
  post_algconstr,
  post_amr_transfer,
};

const char *Z4cStateCheckpointName(Z4cStateCheckpoint checkpoint);

inline const char *Z4cAMRTransferName(const Z4cAMRTransfer transfer) {
  switch (transfer) {
    case Z4cAMRTransfer::high_order: return "high_order";
    case Z4cAMRTransfer::limited_o2: return "limited_o2";
  }
  return "unknown";
}

// Shift needed for derivatives
//----------------------------------------------------------------------------------------
//! \class Z4c

class Z4c {
 public:
  Z4c(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4c();

  template <typename Centering>
  void AllocateNativeStorage(int nmb);
  void ValidateNativeStorageExtents() const;

  // Indices of evolved variables
  enum {
    I_Z4C_CHI,
    I_Z4C_GXX, I_Z4C_GXY, I_Z4C_GXZ, I_Z4C_GYY, I_Z4C_GYZ, I_Z4C_GZZ,
    I_Z4C_KHAT,
    I_Z4C_AXX, I_Z4C_AXY, I_Z4C_AXZ, I_Z4C_AYY, I_Z4C_AYZ, I_Z4C_AZZ,
    I_Z4C_GAMX, I_Z4C_GAMY, I_Z4C_GAMZ,
    I_Z4C_THETA,
    I_Z4C_ALPHA,
    I_Z4C_BETAX, I_Z4C_BETAY, I_Z4C_BETAZ,
    I_Z4C_BX, I_Z4C_BY, I_Z4C_BZ,
    nz4c
  };
  // Names of Z4c variables
  static char const * const Z4c_names[nz4c];
  // Indices of Constraint variables
  enum {
    I_CON_C,
    I_CON_H,
    I_CON_M,
    I_CON_Z,
    I_CON_MX, I_CON_MY, I_CON_MZ,
    ncon,
  };
  // Names of constraint variables
  static char const * const Constraint_names[ncon];
  // Indices of matter fields
  /*enum {
    I_MAT_RHO,
    I_MAT_SX, I_MAT_SY, I_MAT_SZ,
    I_MAT_SXX, I_MAT_SXY, I_MAT_SXZ, I_MAT_SYY, I_MAT_SYZ, I_MAT_SZZ,
    nmat
  };
  // Names of matter variables
  static char const * const Matter_names[nmat];*/

  // data
  Z4cGridLayout layout;         // authoritative Z4c active/stored index geometry
  // flags to denote relativistic dynamics
  DvceArray5D<Real> u_con;     // constraints fields
  DvceArray5D<Real> u_mat;
  DvceArray5D<Real> u0;        // z4c solution
  DvceArray5D<Real> u1;        // z4c solution at intermediate timestep
  DvceArray5D<Real> u_rhs;     // z4c rhs storage
  DvceArray5D<Real> chi_provenance_terms; // default-off exact chi RHS terms
  DvceArray5D<Real> u_telegraph_mu; // physical inverse-length damping profile
  DvceArray5D<Real> coarse_u0; // coarse representation of z4c solution
  DvceArray5D<Real> u_weyl; // weyl scalars
  DvceArray5D<Real> coarse_u_weyl; // coarse representation of weyl scalars

  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };
  ADM_vars adm;

  struct ADMhost_vars {
    AthenaHostTensor<Real, TensorSymm::NONE, 3, 0> psi4;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;
    AthenaHostTensor<Real, TensorSymm::SYM2, 3, 2> vK_dd;
  };

  struct Wave_Extr_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rpsi4;
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> ipsi4;
  };
  Wave_Extr_vars weyl;

  struct Z4c_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;     // conf. factor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vKhat;   // trace extr. curvature
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> vTheta;  // Theta var in Z4c
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;   // lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vGam_u;  // Gamma functions (BSSN)
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;  // shift
    // advective derivative of shift or heat flux for telegraph lapse
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vB_d;
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;    // conf. 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vA_dd;   // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;

  // aliases for the constraints
  struct Constraint_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> C;         // Z constraint monitor
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> M;         // norm squared of M_d
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> Z;         // Z constraint violation
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  /*struct Matter_vars {
    AthenaTensor<Real, TensorSymm::NONE, 3, 0> rho;       // matter energy density
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> vS_d;       // matter momentum density
    AthenaTensor<Real, TensorSymm::SYM2, 3, 2> vS_dd;      // matter stress tensor
  };
  Matter_vars mat;*/

  struct Options {
    Real chi_psi_power;   // chi = psi^N, N = chi_psi_power
    // puncture's floor value for chi, use max(chi, chi_div_floor)
    // in non-differentiated chi
    Real chi_div_floor;
    Real chi_min_floor;   // minimum of chi, only used in slow-start-lapse
    bool floor_chi;       // used as a safe guard after RK update
    // where a square root is necessary.
    Real diss;            // amount of numerical dissipation
    Real eps_floor;       // a small number O(10^-12)
    // Safety factor applied only to the explicit local-source stability ceiling.
    // This is deliberately distinct from <time>/cfl_number.
    Real timestep_source_safety;
    // Constraint damping parameters
    Real damp_kappa1;
    Real damp_kappa2;
    // If true, damp_kappa1 and target_kappa1 are dimensionless multipliers of max|K|.
    bool damp_kappa1_max_K;
    // Compute the volume-global maximum Kretschmann scalar in Z4c history output.
    bool history_kretschmann;
    // Default-off, fail-visible stage diagnostics for bounded instability audits.
    bool rhs_stage_diagnostics;
    Real rhs_stage_diagnostics_start_time;
    Real rhs_stage_diagnostics_rho_max;
    Real rhs_stage_diagnostics_abs_z_max;
    AMRJumpDiagnosticConfig amr_jump_diagnostic;
    ChiParentProvenanceConfig chi_parent_provenance;
    // Gauge conditions for the lapse
    Real lapse_oplog;
    Real lapse_harmonicf;
    Real lapse_harmonic;
    Real lapse_advect;
    // Alcubierre shock-avoiding Bona-Masso slicing:
    // (d_t - beta^i d_i) alpha = -(alpha^2 + kappa) K.
    bool lapse_shock_avoiding;
    Real lapse_shock_avoiding_kappa;
    // slow start lapse condition
    bool slow_start_lapse;
    Real ssl_damping_amp;
    Real ssl_damping_time;
    Real ssl_damping_index;
    // telegrapher lapse condition
    bool telegraph_lapse;
    bool telegraph_max_K;
    TelegraphDampingPrescription telegraph_damping_prescription;
    Real telegraph_tau;
    Real telegraph_kappa;

    // Gauge condition for the shift
    Real shift_ggamma;
    Real shift_alpha2ggamma;
    Real shift_hh;
    Real shift_advect;
    Real shift_eta;
    Z4cShiftMode shift_mode;
    Z4cShiftAdvectionOrder shift_advection_order;
    bool shift_invariant_diagnostic;
    // If true, shift_eta is a dimensionless multiplier of max|K|.
    bool shift_eta_max_K;
    // slow start shift condition
    Real sss_damping_amp;
    Real sss_damping_time;

    // Enable BSSN if false (disable theta)
    bool use_z4c;
    // Apply the Sommerfeld condition for user BCs.
    bool user_Sbc;
    // Boundary extrapolation order
    int extrap_order;
    // Spatial finite-difference order for Z4c derivatives
    int spatial_order;
    // Internal finite-difference stencil selector: 2, 3, 4 -> 2nd, 4th, 6th order
    int fd_stencil;
    // Interlevel transfer only; does not change the bulk spatial order.
    Z4cAMRTransfer amr_transfer;
    // Value of chi to specify the excision region for constraint evaluation
    Real excise_chi;

    // Time dependent constraint damping
    bool roll_kappa;
    Real kappa_roll_start_time;
    Real roll_window;
    Real target_kappa1;
  };
  Options opt;
  Real diss;              // Dissipation parameter
  std::unique_ptr<AMRJumpDiagnosticRuntime> amr_jump_diagnostic;
  std::unique_ptr<ChiParentProvenanceRuntime> chi_parent_provenance;

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Boundary communication buffers for the weyl scalar
  MeshBoundaryValuesCC *pbval_weyl;

  // Z4c timestep contracts. dt_spatial receives the ordinary mesh CFL multiplier;
  // dt_source is an already-final hard source ceiling and must not receive it again.
  Real dtnew;
  Real dt_spatial;
  Real dt_source;
  Real max_source_rate;
  Real max_coordinate_speed;
  Real negative_real_stability_radius;

  // geodesic grid for wave extr
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;
  // array storing waveform at each radii
  Real * psi_out;
  Real waveform_dt;
  Real last_output_time;
  int nrad; // number of radii to perform wave extraction

  // CCE
  Real cce_dump_dt;
  Real cce_dump_last_output_time;
  // dump data cube at horizon

  // functions
  void QueueZ4cTasks();
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus InitRecvWeyl(Driver *d, int stage);
  TaskStatus ClearRecvWeyl(Driver *d, int stage);
  TaskStatus ClearSendWeyl(Driver *d, int stage);
  TaskStatus CopyU(Driver *d, int stage);
  TaskStatus FillAxisParityGhosts(Driver *d, int stage);
  void ReconstructAxisParityGhosts();
  void ReconstructConstraintAxisParityGhosts();
  void ReconstructConstraintAxisParityGhosts(
      DvceArray5D<Real> &constraint_state);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendWeyl(Driver *d, int stage);
  TaskStatus RecvWeyl(Driver *d, int stage);
  TaskStatus Prolongate(Driver *pdrive, int stage);
  TaskStatus ProlongateWeyl(Driver *pdrive, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  void InitializePrescribedZeroShift();
  void CheckPrescribedZeroShiftInvariant(Driver *d, int stage);
  TaskStatus Z4cFloorChi(Driver *pdrive, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  void WriteTimestepContractRecord(Real final_dt) const;
  void FillBuiltInPhysicalBoundaryGhosts();
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);
  TaskStatus EnforceAlgConstr(Driver *d, int stage);

  TaskStatus ConvertZ4cToADM(Driver *d, int stage);
  TaskStatus UpdateExcisionMasks(Driver *d, int stage);
  TaskStatus ADMConstraints_(Driver *d, int stage);
  TaskStatus Z4cBoundaryRHS(Driver *d, int stage);
  template <typename Centering, typename Symmetry>
  TaskStatus Z4cBoundaryRHSImpl(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus RestrictWeyl(Driver *d, int stage);
  TaskStatus CCEDump(Driver *pdrive, int stage);
  TaskStatus TrackCompactObjects(Driver *d, int stage);
  TaskStatus FindHorizon(Driver *d, int stage);
  TaskStatus CalcWeylScalar(Driver *d, int stage);
  TaskStatus CalcWaveForm(Driver *d, int stage);
  TaskStatus DumpHorizons(Driver *d, int stage);

  template <int NGHOST>
  TaskStatus CalcRHS(Driver *d, int stage);
  template <typename Centering, typename Symmetry, int NGHOST>
  TaskStatus CalcRHSImpl(Driver *d, int stage);
  template <int NGHOST>
  void ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin);
  void GaugePreCollapsedLapse(MeshBlockPack *pmbp, ParameterInput *pin);
  void Z4cToADM(MeshBlockPack *pmbp);
  void EvaluateDiagnosticConstraints(DvceArray5D<Real> &scratch_adm,
                                     DvceArray5D<Real> &scratch_constraints,
                                     int diagnostic_stencil = -1);
  template <int NGHOST>
  void ADMConstraints(MeshBlockPack *pmbp);
  template <int NGHOST>
  void Z4cWeyl(MeshBlockPack *pmbp);
  template <typename Centering, typename Symmetry, int NGHOST>
  void Z4cWeylImpl(MeshBlockPack *pmbp);
  void WaveExtr(MeshBlockPack *pmbp);
  void AlgConstr(MeshBlockPack *pmbp, Driver *driver = nullptr, int stage = 0);
  void CheckStateAdmissibility(Driver *driver, int stage,
                               Z4cStateCheckpoint checkpoint,
                               bool include_ghosts = false);
#if defined(ATHENA_Z4C_KERNEL_TESTS)
  void InjectStateAdmissibilityExtractionTestFailure(Driver *driver);
#endif

  Z4c_AMR *pamr;
  std::vector<std::unique_ptr<CompactObjectTracker>> ptracker;
  std::vector<std::unique_ptr<FastFlow>> pfastflow;
  std::vector<std::unique_ptr<HorizonDump>> phorizon_dump;

  // TODO(@hzhu): think about how to automatically trigger common horizon
  // maybe have a horizon dump object to save all the space here
  // same for the waveform.
 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
};

} // namespace z4c
#endif //Z4C_Z4C_HPP_
