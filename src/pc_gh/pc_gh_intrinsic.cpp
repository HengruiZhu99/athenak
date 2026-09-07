// AthenaK astrophysical plasma code, 3-clause BSD License (LICENSE).
// Initial uniform periodic mesh integration of the independent 50-field system.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "pc_gh/pc_gh.hpp"
#include "pc_gh/intrinsic_finite_difference.hpp"
#include "pc_gh/intrinsic_physical_constraints.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"

namespace pc_gh {
namespace {
void IntrinsicError(const std::string &message) {
  std::cerr << "### FATAL ERROR: intrinsic_clean: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}
double MinimumEigenvalue(const double g[3][3]) {
  double mean=(g[0][0]+g[1][1]+g[2][2])/3;
  double b[3][3], sum=0;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    b[i][j]=g[i][j]-(i==j ? mean : 0); sum+=b[i][j]*b[i][j];
  }
  if (sum==0) return mean;
  double scale=std::sqrt(sum/6);
  for (auto &row : b) for (double &v : row) v/=scale;
  double det=b[0][0]*(b[1][1]*b[2][2]-b[1][2]*b[2][1])
    -b[0][1]*(b[1][0]*b[2][2]-b[1][2]*b[2][0])
    +b[0][2]*(b[1][0]*b[2][1]-b[1][1]*b[2][0]);
  // Bound the acos argument against roundoff; no metric/state is altered.
  double angle=std::acos(std::max(-1.0,std::min(1.0,det/2)))/3;
  return mean+2*scale*std::cos(angle+2.0943951023931954923);
}
const char *const intrinsic_names[50] = {
  "pcghi_w", "pcghi_rho", "pcghi_a", "pcghi_c", "pcghi_b", "pcghi_d", "pcghi_e",
  "pcghi_betax", "pcghi_betay", "pcghi_betaz", "pcghi_K",
  "pcghi_Ahatxx", "pcghi_Ahatxy", "pcghi_Ahatxz", "pcghi_Ahatyy", "pcghi_Ahatyz",
  "pcghi_Zx", "pcghi_Zy", "pcghi_Zz", "pcghi_C",
  "pcghi_px", "pcghi_py", "pcghi_pz", "pcghi_lx", "pcghi_ly", "pcghi_lz",
  "pcghi_Sxa", "pcghi_Sxc", "pcghi_Sxb", "pcghi_Sxd", "pcghi_Sxe",
  "pcghi_Sya", "pcghi_Syc", "pcghi_Syb", "pcghi_Syd", "pcghi_Sye",
  "pcghi_Sza", "pcghi_Szc", "pcghi_Szb", "pcghi_Szd", "pcghi_Sze",
  "pcghi_Bxx", "pcghi_Bxy", "pcghi_Bxz", "pcghi_Byx", "pcghi_Byy", "pcghi_Byz",
  "pcghi_Bzx", "pcghi_Bzy", "pcghi_Bzz"};
}
const char *PcGh::StateName(int v) const {
  return IsIntrinsic() ? intrinsic_names[v] : PcGhNames[v];
}

void PcGh::InitializeIntrinsic(ParameterInput *pin) {
  // Reject unimplemented paths rather than accepting an ignored legacy setting.
  const std::set<std::string> allowed = {"formulation", "spatial_order", "shift_eta",
    "kappa", "reduction_rate", "reduction_profile", "dissipation", "research_dt_ceiling",
    "intrinsic_stage_dump", "intrinsic_diagnostics", "intrinsic_diagnostic_dcycle",
    "restart_layout", "restart_layout_version", "restart_layout_fields",
    "restart_untagged_layout", "restart_tracker_state", "project_gauge_constraints", "project_reduction_constraints"};
  for (const auto &block : pin->block) {
    if (block.block_name != "pc_gh") continue;
    for (const auto &line : block.line)
      if (!allowed.count(line.param_name)) IntrinsicError("unsupported option " + line.param_name);
  }
  if (pin->DoesParameterExist("pc_gh", "restart_tracker_state")
      && pin->GetBoolean("pc_gh", "restart_tracker_state"))
    IntrinsicError("tracker restart state is not supported");
  if (pmy_pack->pmesh->multilevel || !pmy_pack->pmesh->strictly_periodic)
    IntrinsicError("mesh integration currently requires uniform periodic boundaries");
  for (const auto &block : pin->block) {
    if (block.block_name.rfind("output", 0) != 0) continue;
    const auto type = pin->GetString(block.block_name, "file_type");
    if (type == "hst") IntrinsicError("physical history diagnostics are not integrated yet");
    if (type != "rst" && pin->GetString(block.block_name, "variable") != "pcgh")
      IntrinsicError("only complete intrinsic state output and restart are enabled yet");
  }
  if (pin->GetOrAddString("time", "integrator", "rk3") != "rk3")
    IntrinsicError("initial mesh qualification requires rk3");
  opt = {};
  intrinsic_stage_dump = pin->GetOrAddBoolean("pc_gh", "intrinsic_stage_dump", false);
  intrinsic_diagnostics = pin->GetOrAddBoolean("pc_gh", "intrinsic_diagnostics", false);
  intrinsic_diagnostic_dcycle = pin->GetOrAddInteger("pc_gh", "intrinsic_diagnostic_dcycle", 1);
  if (intrinsic_diagnostic_dcycle < 1) IntrinsicError("diagnostic cadence must be positive");
  opt.coherent_transfer = "none";
  opt.spatial_order = pin->GetOrAddInteger("pc_gh", "spatial_order", 6);
  opt.fd_stencil = opt.spatial_order/2+1;
  if ((opt.spatial_order != 2 && opt.spatial_order != 4 && opt.spatial_order != 6)
      || pmy_pack->pmesh->mb_indcs.ng < opt.fd_stencil)
    IntrinsicError("unsupported order or insufficient ghost cells");
  opt.shift_eta = pin->GetOrAddReal("pc_gh", "shift_eta", 2);
  opt.kappa = pin->GetOrAddReal("pc_gh", "kappa", 1);
  opt.reduction_rate = pin->GetOrAddReal("pc_gh", "reduction_rate", 1);
  opt.reduction_profile = pin->GetOrAddString("pc_gh", "reduction_profile", "lapse_scaled");
  if (opt.reduction_profile != "lapse_scaled" && opt.reduction_profile != "constant")
    IntrinsicError("reduction_profile must be lapse_scaled or constant");
  opt.dissipation = pin->GetOrAddReal("pc_gh", "dissipation", 0.3);
  opt.research_dt_ceiling = pin->GetOrAddReal("pc_gh", "research_dt_ceiling", 0);
  for (double value : {opt.shift_eta, opt.kappa, opt.reduction_rate,
                       opt.dissipation, opt.research_dt_ceiling})
    if (!std::isfinite(value) || value < 0) IntrinsicError("rates must be finite and nonnegative");
  for (const char *key : {"project_gauge_constraints", "project_reduction_constraints"})
    if (pin->GetOrAddBoolean("pc_gh", key, false))
      IntrinsicError("free intrinsic evolution requires both projections off");
  const auto &a = pmy_pack->pmesh->mb_indcs;
  int nm = std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);
  int ni = a.nx1+2*a.ng, nj = a.nx2 > 1 ? a.nx2+2*a.ng : 1;
  int nk = a.nx3 > 1 ? a.nx3+2*a.ng : 1;
  Kokkos::realloc(u0, nm, EvolvedVariables(), nk, nj, ni);
  Kokkos::realloc(u1, nm, EvolvedVariables(), nk, nj, ni);
  Kokkos::realloc(u_rhs, nm, EvolvedVariables(), nk, nj, ni);
  // No legacy tensor views or legacy diagnostic storage are bound to these arrays.
  pbval_u = new MeshBoundaryValuesCC(pmy_pack, pin, true);
  pbval_u->InitializeBuffers(EvolvedVariables());
  pbval_weyl = new MeshBoundaryValuesCC(pmy_pack, pin, true);
  pbval_weyl->InitializeBuffers(2);
}

void PcGh::IntrinsicInitialData(ParameterInput *pin, bool restart) {
  if (restart) return;
  const auto name = pin->GetString("problem", "pgen_name");
  if (name != "intrinsic_minkowski" && name != "intrinsic_smooth")
    IntrinsicError("initial data must be intrinsic_minkowski or intrinsic_smooth");
  double amplitude = name == "intrinsic_smooth" ?
    pin->GetOrAddReal("problem", "amplitude", 0.001) : 0;
  if (!std::isfinite(amplitude)) IntrinsicError("nonfinite initial amplitude");
  auto h = Kokkos::create_mirror_view(u0);
  auto &a = pmy_pack->pmesh->mb_indcs;
  auto &sz = pmy_pack->pmb->mb_size; sz.template sync<HostMemSpace>();
  const auto &domain = pmy_pack->pmesh->mesh_size;
  for (int m=0;m<pmy_pack->nmb_thispack;++m)
    for (int k=0;k<h.extent_int(2);++k) for (int j=0;j<h.extent_int(3);++j)
      for (int i=0;i<h.extent_int(4);++i) {
        const auto &s=sz.h_view(m);
        double x=CellCenterX(i-a.is,a.nx1,s.x1min,s.x1max);
        double y=CellCenterX(j-a.js,a.nx2,s.x2min,s.x2max);
        double z=CellCenterX(k-a.ks,a.nx3,s.x3min,s.x3max);
        double phase=6.283185307179586*(x/(domain.x1max-domain.x1min)
          +(a.nx2>1 ? y/(domain.x2max-domain.x2min) : 0)
          +(a.nx3>1 ? z/(domain.x3max-domain.x3min) : 0));
        for (int n=0;n<50;++n)
          h(m,n,k,j,i)=(n<2 ? 1.0 : 0.0)
            +amplitude*std::sin(phase+0.17*n)/(1+0.03*n);
      }
  Kokkos::deep_copy(u0,h);
}

void PcGh::ValidateIntrinsic(const char *stage, bool check_rhs) {
  auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u0);
  auto r=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u_rhs);
  const auto &a=pmy_pack->pmesh->mb_indcs;
  int bad=0;
  double min_w=INFINITY, min_rho=INFINITY, min_alpha=INFINITY, max_alpha=0;
  double min_eigen=INFINITY, max_condition=0, min_margin=INFINITY;
  double maxima[50]={};
  long long count=0;
  for (int m=0;m<pmy_pack->nmb_thispack;++m)
    for (int k=0;k<h.extent_int(2);++k) for (int j=0;j<h.extent_int(3);++j)
      for (int i=0;i<h.extent_int(4);++i) {
        double u[50]; bool valid=true;
        bool active=i>=a.is && i<=a.ie && j>=a.js && j<=a.je && k>=a.ks && k<=a.ke;
        for (int n=0;n<50;++n) {
          u[n]=h(m,n,k,j,i); valid=valid && std::isfinite(u[n]);
          if (check_rhs && active) valid=valid && std::isfinite(r(m,n,k,j,i));
        }
        double alpha=u[0]*u[1];
        valid=valid && u[0]>0 && u[1]>0 && alpha>0 && alpha<2
          && alpha*alpha*u[0]*u[0]<4;
        intrinsic::BaseGeometry<double> g; intrinsic::BuildBaseGeometry(u,g);
        for (int b=0;b<3;++b) for (int c=0;c<3;++c)
          valid=valid && std::isfinite(g.metric[b][c])
            && std::isfinite(g.inverse_metric[b][c]);
        for (int b=0;b<3;++b) valid=valid && g.tri[b][b]>0;
        double eigen=MinimumEigenvalue(g.metric), norm=0, inverse_norm=0;
        for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
          norm+=g.metric[b][c]*g.metric[b][c];
          inverse_norm+=g.inverse_metric[b][c]*g.inverse_metric[b][c];
        }
        double condition=std::sqrt(norm)*std::sqrt(inverse_norm);
        valid=valid && std::isfinite(eigen) && eigen>0 && std::isfinite(condition);
        if (active) {
          ++count;
          min_w=std::min(min_w,u[0]); min_rho=std::min(min_rho,u[1]);
          min_alpha=std::min(min_alpha,alpha); max_alpha=std::max(max_alpha,alpha);
          min_eigen=std::min(min_eigen,eigen); max_condition=std::max(max_condition,condition);
          min_margin=std::min(min_margin,4-alpha*alpha*u[0]*u[0]);
          for (int n=0;n<50;++n) maxima[n]=std::max(maxima[n],std::abs(u[n]));
        }
        if (!valid && !bad) {
          std::ofstream f("intrinsic-first-bad-rank"+std::to_string(global_variable::my_rank)+".txt");
          f << std::setprecision(17) << stage << " time " << pmy_pack->pmesh->time
            << " block " << m << " kji " << k << ' ' << j << ' ' << i << '\n';
          f << "# stencil k j i field state rhs (RHS valid only on active cells)\n";
          for (int kk=std::max(0,k-a.ng);kk<=std::min(h.extent_int(2)-1,k+a.ng);++kk)
            for (int jj=std::max(0,j-a.ng);jj<=std::min(h.extent_int(3)-1,j+a.ng);++jj)
              for (int ii=std::max(0,i-a.ng);ii<=std::min(h.extent_int(4)-1,i+a.ng);++ii)
                for (int n=0;n<50;++n)
                  f << kk << ' ' << jj << ' ' << ii << ' ' << n << ' '
                    << h(m,n,kk,jj,ii) << ' ' << r(m,n,kk,jj,ii) << '\n';
          bad=1;
        }
      }
#if MPI_PARALLEL_ENABLED
  int all_bad=0; MPI_Allreduce(&bad,&all_bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); bad=all_bad;
#endif
  if (bad) IntrinsicError(std::string("invalid state/RHS or hyperbolicity domain at ")+stage);
  // Active-cell local bounds; no ghost values enter these reported bounds.
  std::string path="intrinsic-health-rank"+std::to_string(global_variable::my_rank)+".csv";
  bool header=!std::ifstream(path).good();
  std::ofstream log(path,std::ios::app); log << std::setprecision(17);
  if (header) {
    log << "cycle,time,stage,active_cells,min_w,min_rho,min_alpha,max_alpha,"
           "min_metric_eigenvalue,max_frobenius_condition,min_4_alpha2w2,margin_2_alpha";
    for (int n=0;n<50;++n) log << ",max_abs_" << intrinsic_names[n];
    log << '\n';
  }
  log << pmy_pack->pmesh->ncycle << ',' << pmy_pack->pmesh->time << ',' << stage << ','
      << count << ',' << min_w << ',' << min_rho << ',' << min_alpha << ',' << max_alpha
      << ',' << min_eigen << ',' << max_condition << ',' << min_margin << ',' << 2-max_alpha;
  for (double value : maxima) log << ',' << value;
  log << '\n';
}

template<int Stencil> TaskStatus PcGh::IntrinsicRHS() {
  ValidateIntrinsic("pre-RHS including stencil ghosts", false);
  auto state=u0, result=u_rhs;
  const auto &a=pmy_pack->pmesh->mb_indcs;
  auto size=pmy_pack->pmb->mb_size.d_view;
  int dimensions=pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  bool lapse_scaled=opt.reduction_profile == "lapse_scaled";
  double rate=opt.reduction_rate, eta=opt.shift_eta, kappa=opt.kappa, ko=opt.dissipation;
  par_for("intrinsic clean RHS",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
    a.ks,a.ke,a.js,a.je,a.is,a.ie,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real idx[3]={1/size(m).dx1,1/size(m).dx2,1/size(m).dx3}; double rhs[50];
      double lambda=rate*(lapse_scaled ? state(m,0,k,j,i)*state(m,1,k,j,i) : 1.0);
      intrinsic::FiniteDifferenceRHS<Stencil>(state,m,k,j,i,idx,dimensions,lambda,eta,kappa,ko,rhs);
      for (int n=0;n<50;++n) result(m,n,k,j,i)=rhs[n];
    });
  ValidateIntrinsic("post-RHS",true);
  return TaskStatus::complete;
}

// Diagnostic-only native-endian float64 stream, independent of Kokkos layout.
// RHS and RK accumulator payloads contain active cells only: their ghosts are
// not synchronized. State ghosts are retained and explicitly labelled.
void PcGh::DumpIntrinsicStage(Driver *driver, int stage, const char *operation,
                             bool ghosts_valid, bool include_rhs) {
  if (!intrinsic_stage_dump || stage < 1) return;  // stage zero is initialization
  static_assert(sizeof(Real) == sizeof(double), "stage dumps require float64 Real");
  auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u0);
  auto &size=pmy_pack->pmb->mb_size;
  auto &gid=pmy_pack->pmb->mb_gid;
  size.template sync<HostMemSpace>(); gid.template sync<HostMemSpace>();
  const auto &a=pmy_pack->pmesh->mb_indcs;
  const int nm=pmy_pack->nmb_thispack;
  std::ostringstream name;
  name << "intrinsic-stage-r" << global_variable::my_rank << "-g" << gid.h_view(0)
       << "-c" << pmy_pack->pmesh->ncycle << "-s" << stage << "-" << operation << ".dat";
  // Exclusive creation prevents silently overwriting an earlier restart epoch.
  FILE *file=std::fopen(name.str().c_str(), "wbx");
  if (!file) IntrinsicError("cannot exclusively create stage dump " + name.str());
  std::ostringstream header; header << std::setprecision(17);
  header << "{\"version\":1,\"dtype\":\"native_float64\",\"operation\":\"" << operation
    << "\",\"rank\":" << global_variable::my_rank << ",\"cycle\":" << pmy_pack->pmesh->ncycle
    << ",\"stage\":" << stage << ",\"step_time\":" << pmy_pack->pmesh->time
    << ",\"dt\":" << pmy_pack->pmesh->dt << ",\"ghosts_valid\":" << (ghosts_valid?"true":"false")
    << ",\"rhs_active_only\":" << (include_rhs?"true":"false")
    << ",\"order\":" << opt.spatial_order << ",\"gam0\":" << driver->gam0[stage-1]
    << ",\"reduction_rate\":" << opt.reduction_rate << ",\"reduction_profile\":\""
    << opt.reduction_profile << "\",\"dissipation\":" << opt.dissipation
    << ",\"gam1\":" << driver->gam1[stage-1] << ",\"beta_dt\":"
    << driver->beta[stage-1]*pmy_pack->pmesh->dt
    << ",\"shape\":[" << nm << ",50," << h.extent_int(2) << "," << h.extent_int(3)
    << "," << h.extent_int(4) << "],\"active_kji\":[" << a.ks << "," << a.ke << ","
    << a.js << "," << a.je << "," << a.is << "," << a.ie << "],\"blocks\":[";
  for (int m=0;m<nm;++m) {
    const auto &b=size.h_view(m);
    if (m) header << ",";
    header << "{\"gid\":" << gid.h_view(m) << ",\"origin\":[" << b.x1min << ","
      << b.x2min << "," << b.x3min << "],\"spacing\":[" << b.dx1 << "," << b.dx2
      << "," << b.dx3 << "]}";
  }
  header << "]}\n";
  const std::string text=header.str();
  bool ok=std::fwrite(text.data(),1,text.size(),file)==text.size();
  auto write=[&](const auto &v, bool active) {
    for (int m=0;m<nm;++m) for (int n=0;n<50;++n)
      for (int k=active?a.ks:0;k<=(active?a.ke:h.extent_int(2)-1);++k)
        for (int j=active?a.js:0;j<=(active?a.je:h.extent_int(3)-1);++j)
          for (int i=active?a.is:0;i<=(active?a.ie:h.extent_int(4)-1);++i) {
            double value=v(m,n,k,j,i);
            if (std::fwrite(&value,sizeof(double),1,file)!=1) ok=false;
          }
  };
  write(h,false);
  if (include_rhs) {
    auto rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u_rhs);
    auto reg=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u1);
    write(rhs,true); write(reg,true);
  }
  if (std::fclose(file)!=0) ok=false;
  if (!ok) IntrinsicError("failed writing stage dump " + name.str());
}

template<int Stencil>
void PcGh::WriteIntrinsicDiagnostics(Driver *driver, int stage) {
  if (!intrinsic_diagnostics || (driver && stage != driver->nexp_stages)) return;
  const int cycle=pmy_pack->pmesh->ncycle+(driver ? 1 : 0);
  if (driver && cycle%intrinsic_diagnostic_dcycle != 0) return;
  const double time=pmy_pack->pmesh->time+(driver ? pmy_pack->pmesh->dt : 0);
  const auto &a=pmy_pack->pmesh->mb_indcs;
  const int nm=pmy_pack->nmb_thispack;
  const int nk=u0.extent_int(2),nj=u0.extent_int(3),ni=u0.extent_int(4);
  const int dimensions=pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  // Materialize on already synchronized state ghosts. Direct Dxx/Dxy reaches
  // at most three cells in each direction, including mixed-derivative corners.
  DvceArray5D<Real> material("intrinsic diagnostic primary geometry",nm,58,nk,nj,ni);
  DvceArray5D<Real> values("intrinsic diagnostics",nm,89,a.nx3,a.nx2,a.nx1);
  auto state=u0; auto size=pmy_pack->pmb->mb_size.d_view;
  par_for("intrinsic diagnostic materialization",DevExeSpace(),0,nm-1,
    0,nk-1,0,nj-1,0,ni-1,KOKKOS_LAMBDA(int m,int k,int j,int i) {
      double u[50]; for (int n=0;n<50;++n) u[n]=state(m,n,k,j,i);
      intrinsic::BaseGeometry<double> g; intrinsic::BuildBaseGeometry(u,g);
      material(m,0,k,j,i)=u[0]; material(m,1,k,j,i)=u[1]; material(m,2,k,j,i)=u[10];
      for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
        material(m,3+3*b+c,k,j,i)=g.metric[b][c];
        material(m,12+3*b+c,k,j,i)=g.curvature[b][c];
        for (int d=0;d<3;++d) material(m,21+9*d+3*b+c,k,j,i)=g.gradient[d][b][c];
      }
      material(m,48,k,j,i)=u[0]; material(m,49,k,j,i)=u[0]*u[1];
      for (int f=2;f<10;++f) material(m,48+f,k,j,i)=u[f];
    });
  int is=a.is,js=a.js,ks=a.ks;
  par_for("intrinsic primary physical and reduction diagnostics",DevExeSpace(),0,nm-1,
    a.ks,a.ke,a.js,a.je,a.is,a.ie,KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real idx[3]={1/size(m).dx1,1/size(m).dx2,1/size(m).dx3}; double physical[7];
      intrinsic::PhysicalConstraints<Stencil>(material,m,k,j,i,idx,dimensions,physical);
      for (int n=0;n<7;++n) values(m,n,k-ks,j-js,i-is)=physical[n];
      values(m,7,k-ks,j-js,i-is)=state(m,19,k,j,i);
      for (int d=0;d<3;++d) values(m,8+d,k-ks,j-js,i-is)=state(m,16+d,k,j,i);
      const int pairs[3][2]={{0,1},{0,2},{1,2}};
      const int symmetric[6]={0,1,2,4,5,8};
      for (int d=0;d<3;++d) for (int f=0;f<10;++f) {
        int n=f==0 ? 20+d : (f==1 ? 23+d : (f<7 ? 26+5*d+f-2 : 41+3*d+f-7));
        values(m,11+10*d+f,k-ks,j-js,i-is)=state(m,n,k,j,i)
          -(d<dimensions ? Dx<Stencil>(d,idx,material,m,48+f,k,j,i) : 0);
      }
      for (int pair=0;pair<3;++pair) {
        int d=pairs[pair][0],e=pairs[pair][1];
        for (int f=0;f<10;++f) {
          int nd=f==0 ? 20+d : (f==1 ? 23+d : (f<7 ? 26+5*d+f-2 : 41+3*d+f-7));
          int ne=f==0 ? 20+e : (f==1 ? 23+e : (f<7 ? 26+5*e+f-2 : 41+3*e+f-7));
          values(m,41+10*pair+f,k-ks,j-js,i-is)=
            (d<dimensions ? Dx<Stencil>(d,idx,state,m,ne,k,j,i) : 0)
            -(e<dimensions ? Dx<Stencil>(e,idx,state,m,nd,k,j,i) : 0);
        }
        for (int c=0;c<6;++c) values(m,71+6*pair+c,k-ks,j-js,i-is)=
          (d<dimensions ? Dx<Stencil>(d,idx,material,m,21+9*e+symmetric[c],k,j,i) : 0)
          -(e<dimensions ? Dx<Stencil>(e,idx,material,m,21+9*d+symmetric[c],k,j,i) : 0);
      }
    });
  auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),values);
  auto &sizes=pmy_pack->pmb->mb_size; sizes.template sync<HostMemSpace>();
  auto &gids=pmy_pack->pmb->mb_gid; gids.template sync<HostMemSpace>();
  // Per component: L1 integral, squared L2 integral, max absolute, signed
  // winner, gid,k,j,i,x,y,z. Final entries carry actual volume and cell count.
  constexpr int stride=11,count=89*stride+2;
  std::vector<double> local(count,0),all(count*global_variable::nranks);
  for (int n=0;n<89;++n) local[n*stride+2]=-1;
  int bad=0;
  for (int m=0;m<nm;++m) {
    const auto &b=sizes.h_view(m); double dv=b.dx1*b.dx2*b.dx3;
    for (int k=0;k<a.nx3;++k) for (int j=0;j<a.nx2;++j) for (int i=0;i<a.nx1;++i) {
      local[count-2]+=dv; local[count-1]+=1;
      for (int n=0;n<89;++n) {
        double v=h(m,n,k,j,i),av=std::abs(v); int p=n*stride;
        if (!std::isfinite(v)) {
          if (!bad) std::cerr << "Nonfinite intrinsic diagnostic component " << n
            << " gid=" << gids.h_view(m) << " kji=" << k << "," << j << "," << i << std::endl;
          bad=1;
        }
        local[p]+=av*dv; local[p+1]+=v*v*dv;
        if (av>local[p+2] || (av==local[p+2] && gids.h_view(m)<local[p+4])) {
          local[p+2]=av; local[p+3]=v; local[p+4]=gids.h_view(m);
          local[p+5]=k; local[p+6]=j; local[p+7]=i;
          local[p+8]=CellCenterX(i,a.nx1,b.x1min,b.x1max);
          local[p+9]=CellCenterX(j,a.nx2,b.x2min,b.x2max);
          local[p+10]=CellCenterX(k,a.nx3,b.x3min,b.x3max);
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Gather(local.data(),count,MPI_DOUBLE,all.data(),count,MPI_DOUBLE,0,MPI_COMM_WORLD);
#else
  all=local;
#endif
  if (bad) IntrinsicError("nonfinite physical/reduction diagnostic");
  if (global_variable::my_rank!=0) return;
  double volume=0,cells=0;
  for (int rank=0;rank<global_variable::nranks;++rank) {
    volume+=all[rank*count+count-2]; cells+=all[rank*count+count-1];
  }
  std::vector<std::string> names={"H","Mx","My","Mz","alpha_Mx","alpha_My","alpha_Mz",
    "C","Zx","Zy","Zz"};
  const char *families[10]={"w","alpha","a","c","b","d","e","betax","betay","betaz"};
  for (const char *dir : {"x","y","z"}) for (auto family : families)
    names.push_back(std::string("E_")+dir+"_"+family);
  for (const char *dir : {"xy","xz","yz"}) for (auto family : families)
    names.push_back(std::string("Omega_")+dir+"_"+family);
  for (const char *dir : {"xy","xz","yz"}) for (const char *ij : {"xx","xy","xz","yy","yz","zz"})
    names.push_back(std::string("OmegaQ_")+dir+"_"+ij);
  std::string filename="intrinsic-diagnostics-c"+std::to_string(cycle)+"-s"+std::to_string(stage)+".csv";
  FILE *file=std::fopen(filename.c_str(),"wx");
  if (!file) IntrinsicError("cannot exclusively create " + filename);
  std::fprintf(file,"cycle,time,stage,region,component,volume,cells,L1_integral,L2_integral,RMS,maximum,signed_at_max,gid,logical_level,k,j,i,x,y,z\n");
  for (int n=0;n<89;++n) {
    double l1=0,l2=0; int best=n*stride;
    for (int rank=0;rank<global_variable::nranks;++rank) {
      int p=rank*count+n*stride; l1+=all[p]; l2+=all[p+1];
      if (all[p+2]>all[best+2] || (all[p+2]==all[best+2] && all[p+4]<all[best+4])) best=p;
    }
    int gid=static_cast<int>(all[best+4]);
    if (!std::isfinite(l1) || !std::isfinite(l2)) IntrinsicError("nonfinite diagnostic norm");
    std::fprintf(file,"%d,%.17g,%d,full,%s,%.17g,%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.0f,%.0f,%.0f,%.17g,%.17g,%.17g\n",
      cycle,time,stage,names[n].c_str(),volume,cells,l1,std::sqrt(l2),std::sqrt(l2/volume),
      all[best+2],all[best+3],gid,pmy_pack->pmesh->lloc_eachmb[gid].level,
      all[best+5],all[best+6],all[best+7],all[best+8],all[best+9],all[best+10]);
  }
  bool failed=std::ferror(file); if (std::fclose(file)!=0) failed=true;
  if (failed) IntrinsicError("failed writing " + filename);
}

template void PcGh::WriteIntrinsicDiagnostics<2>(Driver *,int);
template void PcGh::WriteIntrinsicDiagnostics<3>(Driver *,int);
template void PcGh::WriteIntrinsicDiagnostics<4>(Driver *,int);

void PcGh::IntrinsicToADM() {
  auto state=u0;
  auto adm=pmy_pack->padm->adm;
  par_for("intrinsic physical ADM output",DevExeSpace(),0,pmy_pack->nmb_thispack-1,
    0,u0.extent_int(2)-1,0,u0.extent_int(3)-1,0,u0.extent_int(4)-1,
    KOKKOS_LAMBDA(int m,int k,int j,int i) {
      double u[50]; for (int n=0;n<50;++n) u[n]=state(m,n,k,j,i);
      intrinsic::BaseGeometry<double> g; intrinsic::BuildBaseGeometry(u,g);
      double iw2=1/(u[0]*u[0]);
      adm.alpha(m,k,j,i)=u[0]*u[1]; adm.psi4(m,k,j,i)=iw2;
      for (int b=0;b<3;++b) {
        adm.beta_u(m,b,k,j,i)=u[intrinsic::BETA+b];
        for (int c=b;c<3;++c) {
          adm.g_dd(m,b,c,k,j,i)=g.metric[b][c]*iw2;
          adm.vK_dd(m,b,c,k,j,i)=(g.curvature[b][c]+g.metric[b][c]*u[intrinsic::K]/3)*iw2;
        }
      }
    });
}

TaskStatus PcGh::IntrinsicTimeStep() {
  ValidateIntrinsic("pre-timestep",false);
  auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),u0);
  auto &size=pmy_pack->pmb->mb_size; size.template sync<HostMemSpace>();
  const auto &a=pmy_pack->pmesh->mb_indcs;
  int dim=pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  double dt=std::numeric_limits<double>::max();
  for (int m=0;m<pmy_pack->nmb_thispack;++m)
    for (int k=a.ks;k<=a.ke;++k) for (int j=a.js;j<=a.je;++j) for (int i=a.is;i<=a.ie;++i) {
      double u[50]; for (int n=0;n<50;++n) u[n]=h(m,n,k,j,i);
      intrinsic::BaseGeometry<double> g; intrinsic::BuildBaseGeometry(u,g);
      double alpha=u[0]*u[1], z=alpha*alpha*u[0]*u[0];
      double speed=std::max({1.0,alpha*u[0],u[0]*std::sqrt(2*alpha),
        std::sqrt((4-intrinsic::GaugePlateau(alpha*u[0]*u[0])*z)/3)});
      double dx[3]={size.h_view(m).dx1,size.h_view(m).dx2,size.h_view(m).dx3};
      double frequency=opt.reduction_rate*(opt.reduction_profile == "lapse_scaled" ? alpha : 1)
        +opt.shift_eta+2*opt.kappa;
      for (int d=0;d<dim;++d) frequency+=(std::abs(u[intrinsic::BETA+d])
        +speed*std::sqrt(g.inverse_metric[d][d])+opt.dissipation)/dx[d];
      if (frequency>0) dt=std::min(dt,1/frequency);
    }
  // A conservative scalar candidate, not a coupled nonlinear stability theorem.
  if (opt.research_dt_ceiling>0) dt=std::min(dt,opt.research_dt_ceiling/pmy_pack->pmesh->cfl_no);
  dtnew=dt;
  return TaskStatus::complete;
}
template TaskStatus PcGh::IntrinsicRHS<2>();
template TaskStatus PcGh::IntrinsicRHS<3>();
template TaskStatus PcGh::IntrinsicRHS<4>();
}  // namespace pc_gh
