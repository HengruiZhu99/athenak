// Actual Kokkos execution-space comparison on the production GL nodes.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "athena.hpp"
#include "z4c/fastflow_harmonics.hpp"
#include "z4c/fastflow_harmonic_cache.hpp"
#include "utils/legendre_roots.hpp"

KOKKOS_INLINE_FUNCTION
Real Component(FastFlowBasis kind,const FastFlowHarmonic &h) {
  const Real sqrt2=Kokkos::sqrt(2.0);
  switch (kind) {
    case FF_Y0:return h.real;
    case FF_dY0dth:return h.th_real;
    case FF_dY0dth2:return h.th2_real;
    case FF_Yc:return sqrt2*h.real;
    case FF_Ys:return sqrt2*h.imag;
    case FF_dYcdth:return sqrt2*h.th_real;
    case FF_dYsdth:return sqrt2*h.th_imag;
    case FF_dYcdph:return sqrt2*h.ph_real;
    case FF_dYsdph:return sqrt2*h.ph_imag;
    case FF_dYcdth2:return sqrt2*h.th2_real;
    case FF_dYsdth2:return sqrt2*h.th2_imag;
    case FF_dYcdph2:return sqrt2*h.ph2_real;
    case FF_dYsdph2:return sqrt2*h.ph2_imag;
    case FF_dYcdthdph:return sqrt2*h.thph_real;
    default:return sqrt2*h.thph_imag;
  }
}

void Check(int lmax,int nt,bool exhaustive) {
  const int lm1=lmax+1,lmp=lm1*lm1,nang=2*nt*nt;
  DvceArray2D<Real> positions("positions",nang,2);
  auto host=Kokkos::create_mirror_view(positions);
  const auto roots=RootsAndWeights(nt);
  for (int p=0;p<nang;++p) {
    host(p,0)=acos(roots[0][p%nt]);host(p,1)=2*M_PI/(2*nt)*(p/nt);
  }
  Kokkos::deep_copy(positions,host);
  FastFlowHarmonicCache<DvceArray2D<Real>,DvceArray3D<Real>> cache;
  cache.ntheta=nt;cache.lmax1=lm1;cache.factorized=true;cache.sqrt2=Kokkos::sqrt(2.0);
  cache.polar=DvceArray3D<Real>("polar",3,nt,lmp);
  cache.phase=DvceArray3D<Real>("phase",2,2*nt,lm1);
  Kokkos::parallel_for("polar",Kokkos::RangePolicy<>(0,nt),KOKKOS_LAMBDA(int t) {
    for (int l=0;l<=lmax;++l) for (int m=0;m<=l;++m) {
      const auto h=StableFastFlowHarmonic(l,m,positions(t,0),0.0);
      const int q=l*lm1+m;
      cache.polar(0,t,q)=h.real;cache.polar(1,t,q)=h.th_real;cache.polar(2,t,q)=h.th2_real;
    }
  });
  Kokkos::parallel_for("phase",Kokkos::RangePolicy<>(0,2*nt),KOKKOS_LAMBDA(int ph) {
    const Real phi=positions(ph*nt,1);
    for (int m=0;m<=lmax;++m) {
      cache.phase(0,ph,m)=std::cos(m*phi);cache.phase(1,ph,m)=std::sin(m*phi);
    }
  });
  auto dense=cache;dense.factorized=false;
  if (exhaustive) {
    for (int kind=0;kind<FF_basis_count;++kind)
      dense.dense[kind]=DvceArray2D<Real>("dense",nang,lmp);
    Kokkos::parallel_for("dense",Kokkos::RangePolicy<>(0,nang),KOKKOS_LAMBDA(int p) {
      for (int l=0;l<=lmax;++l) for (int m=0;m<=l;++m) {
        const auto h=StableFastFlowHarmonic(l,m,positions(p,0),positions(p,1));
        for (int kind=0;kind<FF_basis_count;++kind) {
          const auto k=static_cast<FastFlowBasis>(kind);
          const bool axis=k==FF_Y0 || k==FF_dY0dth || k==FF_dY0dth2;
          if (axis==(m==0)) dense.dense[k](p,axis?l:l*lm1+m)=Component(k,h);
        }
      }
    });
  }
  int failures=0;
  Kokkos::parallel_reduce("comparison",Kokkos::RangePolicy<>(0,nang),
  KOKKOS_LAMBDA(int p,int &bad) {
    const int t=p%nt,ph=p/nt;
    if (!exhaustive && !(t==0 || t==1 || t==nt/2 || t==nt-2 || t==nt-1)) return;
    if (!exhaustive && !(ph==0 || ph==1 || ph==nt/2 || ph==nt || ph==2*nt-1)) return;
    for (int l=0;l<=lmax;++l) for (int m=0;m<=l;++m) {
      if (!exhaustive && !(l<3 || l==16 || l==64 || l==lmax-1 || l==lmax)) continue;
      const auto h=StableFastFlowHarmonic(l,m,positions(p,0),positions(p,1));
      for (int kind=0;kind<FF_basis_count;++kind) {
        const auto k=static_cast<FastFlowBasis>(kind);
        const bool axis=k==FF_Y0 || k==FF_dY0dth || k==FF_dY0dth2;
        if (axis!=(m==0)) continue;
        const int q=axis?l:l*lm1+m;
        const Real expected=Component(k,h),value=cache(k,p,q);
        if (!Kokkos::isfinite(value) || value!=expected || std::signbit(value)!=std::signbit(expected)) ++bad;
        if (exhaustive && (value!=dense(k,p,q) || std::signbit(value)!=std::signbit(dense(k,p,q)))) ++bad;
      }
    }
  },failures);
  std::cout << "lmax=" << lmax << " ntheta=" << nt << " exhaustive=" << exhaustive
            << " execution=" << DevExeSpace::name() << " mismatches=" << failures << '\n';
  if (failures) throw std::runtime_error("Dense/factorized harmonic components differ");
}

int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  try {Check(8,16,true);Check(16,18,true);Check(160,162,false);}
  catch (...) {Kokkos::finalize();throw;}
  Kokkos::finalize();
}
