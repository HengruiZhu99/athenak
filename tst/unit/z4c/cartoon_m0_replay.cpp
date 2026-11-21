// Replay the production direct generating-curve solve on archived meridional data.
// native_replay.py supplies a regular grid of physical ADM tensors and derivatives.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "z4c/cartoon_m0_fastflow.hpp"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  const bool folded = std::string(argv[2]) == "folded";
  std::ifstream file(argv[1], std::ios::binary);
  std::int32_t extent = 0, degree = 0;
  double dx = 0;
  file.read(reinterpret_cast<char*>(&extent), sizeof(extent));
  file.read(reinterpret_cast<char*>(&degree), sizeof(degree));
  file.read(reinterpret_cast<char*>(&dx), sizeof(dx));
  if (!file || extent < 4 || extent > 4096 || degree < 2 || degree > 96 || !(dx > 0))
    throw std::runtime_error("invalid archived meridian");
  std::vector<double> rows(static_cast<size_t>(2*extent+1)*(extent+1)*30);
  file.read(reinterpret_cast<char*>(rows.data()), rows.size()*sizeof(double));
  std::vector<Real> outer(degree+1), inner(degree+1);
  for (auto *shape : {&outer, &inner}) {
    for (int l=0; l<=degree; ++l) {
      double coeff = 0;
      file.read(reinterpret_cast<char*>(&coeff), sizeof(coeff));
      (*shape)[l] = coeff*std::sqrt(4*std::acos(-1.)/(2*l+1));
    }
  }
  if (!file) throw std::runtime_error("incomplete archived meridian");
  {
    z4c::M0Sampler sample = [&](const z4c::M0Locations &locations) {
      std::vector<z4c::M0AdmSample> result(locations.size());
      for (size_t n=0; n<locations.size(); ++n) {
        const auto &p = locations[n];
        const double r = p[0]/dx, z = (folded ? std::abs(p[1]) : p[1])/dx+extent;
        if (r < 0 || r >= extent || z < 0 || z >= 2*extent) continue;
        // Use a cubic stencil of active vertices within the owning 24-cell
        // block, including one-sided stencils at the axis and the equator.
        auto weights = [](double q, int lo, int hi, int origin, double w[4]) {
          const int edge = origin + 24*static_cast<int>(std::floor((q-origin)/24));
          const int first = std::clamp(static_cast<int>(q)-1, std::max(lo,edge),
                                       std::min(hi,edge+24)-3);
          for (int k=0; k<4; ++k) {
            w[k] = 1.;
            for (int l=0; l<4; ++l) if (k!=l) w[k] *= (q-first-l)/(k-l);
          }
          return first;
        };
        double wr[4], wz[4];
        const int i = weights(r,0,extent,0,wr);
        const int j = weights(z,folded ? extent : 0,2*extent,extent,wz);
        auto &m = result[n];
        m.valid = true;
        m.spacing = dx;
        for (int y=0; y<4; ++y) for (int x=0; x<4; ++x) {
          const double weight = wz[y]*wr[x];
          const double *v = rows.data()+30*((j+y)*(extent+1)+i+x);
          for (int k=0; k<6; ++k) {
            m.metric[k] += weight*v[k];
            m.curvature[k] += weight*v[6+k];
          }
          for (int k=0; k<18; ++k) m.metric_derivative[k] += weight*v[12+k];
        }
        if (folded && p[1]<0) z4c::ReflectM0AdmSample(m);
      }
      return result;
    };
    for (double target : {0., -.0005}) for (const std::string guess : {"shape","scaled","sphere"}) {
      auto seed = target == 0 ? outer : inner;
      if (guess == "scaled") for (auto &c : seed) c *= .9;
      if (guess == "sphere") std::fill(seed.begin()+1, seed.end(), 0.);
      const auto start = std::chrono::steady_clock::now();
      const auto fit = z4c::RefineM0Surface(seed, 0, std::max(56,degree*2), 20,
          target, target < 0 ? -.25*target : .006, .006, 1.e-4, sample, folded);
      const double ms = std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-start).count();
      std::printf("%d,%g,%s,%d,%.12g,%.12g,%.12g,%.12g,%.6f,%s,", folded,
          target,guess.c_str(),fit.converged,fit.direct_residual,fit.area,fit.mass,
          fit.mass>0 ? fit.spin_z/(fit.mass*fit.mass) : 0., ms, fit.failure.c_str());
      for (size_t l=0; l<fit.coefficients.size(); ++l) {
        std::printf("%s%.17g", l ? " " : "",
            fit.coefficients[l]*std::sqrt((2*l+1)/(4*std::acos(-1.))));
      }
      std::putchar('\n');
    }
  }
}
