// IrisK conformally flat, maximal photon data on the native Cartoon vertices.
#include "pgen/photon_torus.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "boltzmann/boltzmann.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/z4c_irisk_coordinate_map.hpp"
#include "z4c/z4c.hpp"

namespace {
double Bump(double x) {
  return x < 1 ? std::exp(1-1/(1-x)) : 0;
}

struct Torus {
  int nr, nt;
  double inner, outer, center, width, norm, b, db, nu;
  std::vector<double> coeff;
  explicit Torus(const std::string& path) {
    std::ifstream in(path);
    std::string magic;
    if (!(in >> magic >> nr >> nt >> inner >> outer >> center >> width >> norm
             >> b >> db >> nu) || magic != "IRISK_PHOTON_TORUS_V1" ||
        nr < 8 || nr > 512 || nt < 6 || nt > 512 || !(outer > inner && inner > 0 &&
        center > width && width > 0 && norm >= 0 && db > 0 && nu > 0 && nu < 1 &&
        (norm == 0 || std::abs(b) > db)))
      throw std::runtime_error("invalid photon torus header: " + path);
    coeff.resize(21*nr*nt);
    for (auto& v : coeff)
      if (!(in >> v) || !std::isfinite(v)) throw std::runtime_error("bad torus coefficient");
    if (in >> magic) throw std::runtime_error("trailing torus data");
  }

  std::array<double,7> Eval(double rho, double z) const {
    double r = std::hypot(rho,z), theta = std::atan2(std::abs(rho),z);
    int d = r < inner ? 0 : (r < outer ? 1 : 2);
    double x = d == 0 ? 2*r/inner-1 : (d == 1 ? 2*(r-inner)/(outer-inner)-1 : 1-2*outer/r);
    std::vector<double> even(nt), mixed(nt);
    for (int k=0; k<nt; ++k) {
      even[k] = std::cos(2*k*theta);
      mixed[k] = std::sin(2*(k+1)*theta)*(rho < 0 ? -1 : 1);
    }
    std::array<double,7> field{};
    for (int f=0; f<7; ++f) {
      const auto& angular = (f == 3 || f == 5) ? mixed : even;
      double b1=0, b2=0;
      for (int n=nr-1; n>=0; --n) {
        double value=0;
        auto row = coeff.data()+((d*7+f)*nr+n)*nt;
        for (int k=0; k<nt; ++k) value += row[k]*angular[k];
        double next = value+(n ? 2*x : x)*b1-b2;
        b2=b1;
        b1=next;
      }
      field[f]=b1;
    }
    if (!(field[0]>0)) throw std::runtime_error("nonpositive imported psi");
    // SO(2) isotropy fixes the tensor limit at the degenerate cylindrical frame.
    if (rho == 0) {
      field[1]=field[4]=0.5*(field[1]+field[4]);
      field[2]=field[3]=field[5]=0;
    }
    return field;
  }
};

// Integral of P(b)P(nu) db dnu over the preimage of one inward sky cell.
// Clip both integrals to the support before applying eight-point Gauss quadrature.
constexpr std::array<double,8> q = {-0.9602898564975363,-0.7966664774136267,
  -0.525532409916329,-0.1834346424956498,0.1834346424956498,0.525532409916329,
  0.7966664774136267,0.9602898564975363};
constexpr std::array<double,8> wt = {0.1012285362903763,0.2223810344533745,
  0.3137066458778873,0.362683783378362,0.362683783378362,0.3137066458778873,
  0.2223810344533745,0.1012285362903763};

double BinMass(const Torus& t, double radius, double m0, double dm, double a0, double da) {
  double lo=std::max(t.b-t.db,radius*m0), hi=std::min(t.b+t.db,radius*(m0+dm));
  double a=std::max(M_PI/2,a0), end=std::min(1.5*M_PI,a0+da);
  if (hi <= lo || end <= a) return 0;
  double sum=0;
  for (int i=0; i<8; ++i) {
    double b=(lo+hi+(hi-lo)*q[i])/2, s=std::sqrt(1-std::pow(b/radius,2));
    double v0=std::max(-t.nu,s*std::sin(end)), v1=std::min(t.nu,s*std::sin(a));
    if (v1 <= v0) continue;
    double vertical=0;
    for (int j=0; j<8; ++j) {
      double v=(v0+v1+(v1-v0)*q[j])/2;
      vertical += wt[j]*Bump(std::pow(v/t.nu,2));
    }
    sum += wt[i]*Bump(std::pow((b-t.b)/t.db,2))*vertical*(v1-v0);
  }
  return sum*(hi-lo)/4;
}
}  // namespace

void InitializePhotonTorus(MeshBlockPack* pack, ParameterInput* pin) {
  Torus t(pin->GetString("problem","torus_file"));
  auto* photons=pack->pboltzmann;
  const auto& v=pack->pz4c->layout;
  if (v.centering != z4c::Z4cGridCentering::vertex ||
      z4c_irisk::SelectAdmMap(pack->z4c_symmetry) != z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2)
    throw std::runtime_error("torus requires native vertex Cartoon");
  if (t.norm && ((t.b>0 && photons->mu_min+photons->nmu*photons->dmu < 0.5) ||
                 (t.b<0 && photons->mu_min >= 0)))
    throw std::runtime_error("Boltzmann hemisphere excludes launch");
  const auto b=pack->pmesh->mb_indcs;
  pack->pmb->mb_size.sync_host();
  auto size=pack->pmb->mb_size.h_view;
  auto state=create_mirror(pack->pz4c->u0);
  Kokkos::deep_copy(state,0.);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    for (int j=0; j<v.n2; ++j) for (int i=0; i<v.n1; ++i) {
      const auto f=t.Eval(size(m).x1min+(i-v.is)*size(m).dx1,
                          size(m).x2min+(j-v.js)*size(m).dx2);
      const auto a=z4c_irisk::SymmetricTensorFromPhysicalCartesian<
        z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2>(
          std::array<double,6>{f[1],f[2],f[3],f[4],f[5],f[6]});
      for (int k=0; k<v.n3; ++k) {
        using Z=z4c::Z4c;
        state(m,Z::I_Z4C_CHI,k,j,i)=std::pow(f[0],pack->pz4c->opt.chi_psi_power);
        state(m,Z::I_Z4C_ALPHA,k,j,i)=1;
        state(m,Z::I_Z4C_GXX,k,j,i)=state(m,Z::I_Z4C_GYY,k,j,i)=state(m,Z::I_Z4C_GZZ,k,j,i)=1;
        for (int n=0; n<6; ++n) state(m,Z::I_Z4C_AXX+n,k,j,i)=a[n]*std::pow(f[0],-6);
      }
    }
  }
  Kokkos::deep_copy(pack->pz4c->u0,state);
  pack->pz4c->ApplyVertexAxisRegularity(pack->pz4c->u0,0,"initial_data");
  pack->pz4c->ReconstructAxisParityGhosts();
  pack->pz4c->Z4cToADM(pack);
  auto metric=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->padm->u_adm);
  auto u=create_mirror(photons->u);
  Kokkos::deep_copy(u,0.);
  std::vector<double> mass(photons->nang);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    for (int j=b.js-1; j<=b.je+1; ++j) for (int i=b.is-1; i<=b.ie+1; ++i) {
      double rho=size(m).x1min+(i-b.is+0.5)*size(m).dx1;
      double z=size(m).x2min+(j-b.js+0.5)*size(m).dx2;
      double e=t.norm*Bump((std::pow(rho-t.center,2)+z*z)/(t.width*t.width));
      if (!e) continue;
      double psi4=metric(m,adm::ADM::I_ADM_GZZ,0,j,i);
      double radius=rho*std::sqrt(psi4);
      if (!(std::pow((std::abs(t.b)+t.db)/radius,2)+t.nu*t.nu < 1))
        throw std::runtime_error("infeasible torus on cell metric");
      double sum=0, first=0, second=0;
      for (int n=0; n<photons->nang; ++n) {
        double lo=photons->mu_min+n/photons->npsi*photons->dmu;
        double center=radius*(lo+0.5*photons->dmu);
        mass[n]=BinMass(t,radius,lo,photons->dmu,n%photons->npsi*photons->dpsi,photons->dpsi);
        sum+=mass[n];
        first+=center*mass[n];
        second+=center*center*mass[n];
      }
      if (!(sum>0 && second/sum>std::pow(first/sum,2)))
        throw std::runtime_error("unresolved launch impact band");
      double mean=first/sum, slope=(t.b-mean)/(second/sum-mean*mean);
      for (int n=0; n<photons->nang; ++n) {
        double impact=radius*(photons->mu_min+(n/photons->npsi+0.5)*photons->dmu);
        double projected=mass[n]*(1+slope*(impact-mean))/sum;
        if (projected<0) throw std::runtime_error("negative moment-fit bin; refine the sky");
        u(m,n,0,j,i)=e*rho*std::pow(psi4,1.5)*projected/(photons->dmu*photons->dpsi);
      }
    }
  }
  Kokkos::deep_copy(photons->u,u);
}

void AuditPhotonTorus(MeshBlockPack* pack, ParameterInput* pin) {
  std::string prefix=pin->GetOrAddString("problem","torus_audit_prefix","");
  if (prefix.empty()) return;
  auto sz=pack->pmb->mb_size.h_view;
  auto a=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u_adm_native);
  auto c=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pz4c->u_con);
  auto mom=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pboltzmann->moments);
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pboltzmann->u);
  const auto& v=pack->pz4c->layout;
  const auto& b=pack->pmesh->mb_indcs;
  std::string rank=std::to_string(global_variable::my_rank);
  std::ofstream vertex(prefix+".vertices."+rank), cell(prefix+".cells."+rank);
  vertex << "# rho z psi4 Krr Krz Krp Kzz Kzp Kpp H M_r M_z M_p\n" << std::setprecision(17);
  cell << "# rho z E S_r S_z S_p occupied_mu effective_mu\n" << std::setprecision(17);
  auto* p=pack->pboltzmann;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    for (int j=v.js; j<=v.je; ++j) for (int i=v.is; i<=v.ie; ++i) {
      vertex << sz(m).x1min+(i-v.is)*sz(m).dx1 << ' ' << sz(m).x2min+(j-v.js)*sz(m).dx2
             << ' ' << a(m,adm::ADM::I_ADM_PSI4,v.ks,j,i);
      for (int n=0; n<6; ++n) vertex << ' ' << a(m,adm::ADM::I_ADM_KXX+n,v.ks,j,i);
      vertex << ' ' << c(m,z4c::Z4c::I_CON_H,v.ks,j,i);
      for (int n=0;n<3;++n) vertex << ' ' << c(m,z4c::Z4c::I_CON_MX+n,v.ks,j,i);
      vertex << '\n';
    }
    for (int j=b.js; j<=b.je; ++j) for (int i=b.is; i<=b.ie; ++i) {
      cell << sz(m).x1min+(i-b.is+0.5)*sz(m).dx1 << ' ' << sz(m).x2min+(j-b.js+0.5)*sz(m).dx2;
      for (int n=6;n<10;++n) cell << ' ' << mom(m,n,0,j,i);
      int occupied=0;
      double sum=0, square=0;
      for (int h=0; h<p->nmu; ++h) {
        double band=0;
        for (int k=0; k<p->npsi; ++k) band+=u(m,h*p->npsi+k,0,j,i);
        occupied+=(band>0);
        sum+=band;
        square+=band*band;
      }
      cell << ' ' << occupied << ' ' << (square ? sum*sum/square : 0) << '\n';
    }
  }
  if (!vertex || !cell) throw std::runtime_error("writing torus audit failed");
}
