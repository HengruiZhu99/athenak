#ifndef RESEARCH_FINITE_RB_CONSTRAINT_RATE_API_HPP_
#define RESEARCH_FINITE_RB_CONSTRAINT_RATE_API_HPP_
// Private point-action API, included AFTER actual_bridge.cpp.  No radial
// assembly or reconstruction occurs here.  q=(H,M_x,M_y,M_z,Z_x,Z_y,Z_z,Theta)
// consists of physical Cartesian components; in particular Theta is NOT /Omega.
#include <sstream>
#include "../../continuum/constraint-propagation/immutable-constraint-propagation-20261009/subsidiary.hpp"

namespace rateapi {
inline void Finite(double v) {
  if (!std::isfinite(v)) throw std::runtime_error("nonfinite constraint-rate API input/output");
}
inline void End(std::istringstream &s) {
  s >> std::ws;
  if (!s.eof()) throw std::runtime_error("extra constraint-rate API input");
}
template<class T> inline void Read(std::istringstream &s,T &v) {
  if (!(s >> v)) throw std::runtime_error("truncated constraint-rate API input");
}
inline X3 Point(std::istringstream &s) {
  X3 x{};for(auto &v:x) {Read(s,v);Finite(v);}return x;
}
inline hyp::LayerPoint<double> At(const X3 &x) {
  auto p=reference.At(x[0],x[1],x[2]);
  if (!(p.omega>0)) throw std::runtime_error("noninterior constraint-rate API point");
  return p;
}
inline TJ ReadJet(std::istringstream &s) {
  TJ v{};Read(s,v.value);Finite(v.value);
  for(int i=0;i<3;++i) {Read(s,v.d[i]);Finite(v.d[i]);}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    Read(s,v.dd[i][j]);Finite(v.dd[i][j]);
  }
  return v;
}
template<class T> inline void Print(const T &a) {
  for(const auto v:a) {Finite(v);std::cout<<v<<' ';}
}
inline totalj::WJet<double> Envelope(int id,const X3 &x) {
  double rho=0;for(const auto v:x)rho+=v*v;
  totalj::WJet<double> w{};
  switch(id) {
    case 0:w={1,0,0};break;
    case 1:w={rho,1,0};break;
    case 2:w={rho*rho,2*rho,2};break;
    case 3:w={rho*rho*rho,3*rho*rho,6*rho};break;
    case 4:{const double e=std::exp(-8*rho);w={e,-8*e,64*e};break;}
    case 5:{const double t=(rho-.49)/.16,e=std::exp(-t*t);
      w={e,-2*t*e/.16,(4*t*t-2)*e/(.16*.16)};break;}
    case 6:w={1+rho/3-rho*rho/5+rho*rho*rho/7,
      1./3-2*rho/5+3*rho*rho/7,-2./5+6*rho/7};break;
    default:throw std::runtime_error("unknown manufactured envelope");
  }
  Finite(w.value);Finite(w.rho_d);Finite(w.rho_dd);return w;
}
}  // namespace rateapi

inline Physical MakeManufactured(int J,int m,int channel,int phase,int envelope,
                                 const X3 &x) {
  if(J<0||J>2||m<0||m>J||channel<0||channel>=(J==0?8:J==1?16:20)
      ||phase<0||phase>1)throw std::runtime_error("invalid manufactured basis index");
  return SeedPhysical(J,m,channel,phase,x,rateapi::Envelope(envelope,x));
}

// One input line: J m channel phase envelope x y z.
// One output line: raw22 tangent RHS, then physical eight initial constraints.
inline void ManufacturedRateBatch() {
  std::cout<<std::setprecision(17);std::string line;
  while(std::getline(std::cin,line)) {
    if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
    std::istringstream s(line);int J,m,c,phase,env;
    rateapi::Read(s,J);rateapi::Read(s,m);rateapi::Read(s,c);
    rateapi::Read(s,phase);rateapi::Read(s,env);
    const auto x=rateapi::Point(s);rateapi::End(s);const auto p=rateapi::At(x);
    const auto u=LiftPhysical(p,MakeManufactured(J,m,c,phase,env,x));
    rateapi::Print(Derivative(ActualDual(p,u)));
    rateapi::Print(Constraints(u,p));std::cout<<'\n';
  }
}

// One input line: x y z, then 22 jets each (v,d0,d1,d2,dd00,...,dd22).
// Full raw tangent: no algebraic projection is silently inserted before DC.
inline void ConstraintRateBatch() {
  std::cout<<std::setprecision(17);std::string line;
  while(std::getline(std::cin,line)) {
    if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
    std::istringstream s(line);const auto x=rateapi::Point(s);
    std::array<TJ,22> f{};for(auto &v:f)v=rateapi::ReadJet(s);
    rateapi::End(s);const auto p=rateapi::At(x);
    rateapi::Print(Constraints(RawTangent(p,f),p));std::cout<<'\n';
  }
}

// One input line: x y z, then physical eight jets in the same packing.
inline void SubsidiaryBatch() {
  std::cout<<std::setprecision(17);std::string line;
  while(std::getline(std::cin,line)) {
    if(line.find_first_not_of(" \t\r")==std::string::npos)continue;
    std::istringstream s(line);const auto x=rateapi::Point(s);ConstraintJet q{};
    for(int f=0;f<8;++f) {
      const auto v=rateapi::ReadJet(s);q.q[f]=C(v.value,0);
      for(int i=0;i<3;++i) {q.d[i][f]=C(v.d[i],0);
        for(int j=0;j<3;++j)q.dd[i][j][f]=C(v.dd[i][j],0);}
    }
    rateapi::End(s);const auto p=rateapi::At(x);
    const auto out=Subsidiary(p,10,q);std::array<double,8> real{};
    for(int f=0;f<8;++f) {
      if(out[f].imag()!=0)throw std::runtime_error("unexpected complex real-field subsidiary output");
      real[f]=out[f].real();
    }
    rateapi::Print(real);std::cout<<'\n';
  }
}
#endif
