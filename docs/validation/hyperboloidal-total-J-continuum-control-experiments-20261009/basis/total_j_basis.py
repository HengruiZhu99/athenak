"""Exact Cartesian total-J harmonics. Mathematical prototype; no PDE kernel."""
from functools import lru_cache
import sympy as s
from sympy.physics.wigner import clebsch_gordan

x, y, z = s.symbols('x y z', real=True)
xyz = (x, y, z)
rho = x*x + y*y + z*z
I = s.I


@lru_cache(None)
def solid(l, m):
    """r**l Y_lm with Condon--Shortley phase and unit sphere norm."""
    if abs(m) > l:
        return s.S.Zero
    if m < 0:
        return s.expand((-1)**(-m) * s.conjugate(solid(l, -m)))
    t = s.symbols('t', real=True)
    p = s.Poly(s.diff(s.legendre(l, t), t, m), t)
    radial = 0
    for (k,), c in p.terms():
        power = l - m - k
        assert power >= 0 and power % 2 == 0
        radial += c * z**k * rho**(power//2)
    norm = s.sqrt(s.Rational(2*l+1, 4) / s.pi * s.factorial(l-m)/s.factorial(l+m))
    return s.expand(norm * (-1)**m * (x+I*y)**m * radial)


@lru_cache(None)
def spin(spin, m):
    if spin == 0:
        assert m == 0
        return (s.S.One,)
    es = {1: (-1/s.sqrt(2), -I/s.sqrt(2), 0),
          0: (0, 0, 1), -1: (1/s.sqrt(2), -I/s.sqrt(2), 0)}
    if spin == 1:
        return tuple(s.sympify(v) for v in es[m])
    assert spin == 2
    ans = [s.S.Zero]*9
    for p in range(-1, 2):
        for q in range(-1, 2):
            c = clebsch_gordan(1, 1, 2, p, q, m)
            for i in range(3):
                for j in range(3):
                    ans[3*i+j] += c * es[p][i] * es[q][j]
    return tuple(s.simplify(v) for v in ans)


@lru_cache(None)
def basis(j, m, spin_rank, orbital):
    """Real-phase Cartesian coupled solid polynomial, rho radial factor omitted.

    i**(L+s-J) makes m=0 real and B_J,-m=(-1)**m conjugate(B_J,m).
    Each Cartesian component is homogeneous of degree L. Multiplication by an
    arbitrary smooth W_L(rho) supplies the regular radial amplitude.
    """
    assert abs(m) <= j and abs(j-spin_rank) <= orbital <= j+spin_rank
    size = (1, 3, 9)[spin_rank]
    ans = [s.S.Zero]*size
    for q in range(-spin_rank, spin_rank+1):
        p = m-q
        if abs(p) <= orbital:
            c = clebsch_gordan(orbital, spin_rank, j, p, q, m)
            es = spin(spin_rank, q)
            for i in range(size):
                ans[i] += c * solid(orbital, p) * es[i]
    phase = I**(orbital+spin_rank-j)
    return tuple(s.simplify(s.expand(phase*v)) for v in ans)


def allowed_l(j, spin_rank):
    return range(abs(j-spin_rank), j+spin_rank+1)


def eps(i, j, k):
    return s.LeviCivita(i, j, k)


def rotation(v, spin_rank, axis):
    """J_axis = -i (x cross grad)_axis + Cartesian tensor spin generator."""
    out = []
    for comp, field in enumerate(v):
        orb = -I*sum(eps(axis, i, k)*xyz[i]*s.diff(field, xyz[k])
                     for i in range(3) for k in range(3))
        intrinsic = 0
        if spin_rank == 1:
            intrinsic = -I*sum(eps(axis, comp, q)*v[q] for q in range(3))
        elif spin_rank == 2:
            i, k = divmod(comp, 3)
            intrinsic = -I*sum(eps(axis, i, q)*v[3*q+k]
                               + eps(axis, k, q)*v[3*i+q] for q in range(3))
        out.append(s.expand(orb+intrinsic))
    return tuple(out)


def sphere_integral(expr):
    """Exact integral of a Cartesian polynomial on the unit sphere."""
    ans = 0
    for powers, coefficient in s.Poly(s.expand(expr), *xyz).terms():
        if any(p % 2 for p in powers):
            continue
        ans += coefficient * 4*s.pi * s.prod(s.factorial2(p-1) for p in powers) / s.factorial2(sum(powers)+1)
    return s.simplify(ans)


def inner(v, w):
    return sphere_integral(sum(s.conjugate(a)*b for a, b in zip(v, w)))


def channel_layout(j):
    channels = []
    for name in ('alpha', 'metric_trace', 'P', 'Theta_phys'):
        channels.append({'name': name, 'spin': 0, 'L': j})
    for name in ('beta', 'Lambda'):
        for l in allowed_l(j, 1):
            channels.append({'name': name, 'spin': 1, 'L': l})
    for name in ('metric_STF', 'independent_A'):
        for l in allowed_l(j, 2):
            channels.append({'name': name, 'spin': 2, 'L': l})
    return channels


def coefficient_records(v):
    terms = []
    for component, field in enumerate(v):
        if field == 0:
            continue
        for powers, coefficient in s.Poly(field, *xyz).terms():
            terms.append({'component': component, 'powers': list(powers),
                          'coefficient_exact': str(coefficient),
                          'coefficient_real': float(s.re(coefficient)),
                          'coefficient_imag': float(s.im(coefficient))})
    return terms


def cpp_header():
    """Generate the m=0 real standalone value/gradient/Hessian evaluator."""
    records, terms = [], []
    for j in range(3):
        for sr in range(3):
            for l in allowed_l(j, sr):
                start = len(terms)
                for t in coefficient_records(basis(j, 0, sr, l)):
                    assert t['coefficient_imag'] == 0
                    terms.append(t)
                records.append((j, sr, l, start, len(terms)-start))
    head = '''// Generated mathematical prototype; no evolution or reference source.
#ifndef RESEARCH_TOTAL_J_SOLID_BASIS_HPP_
#define RESEARCH_TOTAL_J_SOLID_BASIS_HPP_
#include <array>
#include <stdexcept>
namespace totalj {
// Each component: value; d[Cartesian derivative]; dd[symmetric/full].
template<class T> struct Jet { T value{}; std::array<T,3> d{}; std::array<std::array<T,3>,3> dd{}; };
template<class T> struct WJet { T value{}, rho_d{}, rho_dd{}; };
template<class T> struct Field { int components=0; std::array<Jet<T>,9> component{}; };
struct Term { int component,px,py,pz; double coefficient; };
struct Record { int J,spin,L,start,count; };
inline constexpr Term terms[] = {
'''
    for t in terms:
        px, py, pz = t['powers']
        head += f"  {{{t['component']},{px},{py},{pz},{t['coefficient_real']:.17g}}},\n"
    head += '};\ninline constexpr Record records[] = {\n'
    for row in records:
        head += '  {' + ','.join(str(v) for v in row) + '},\n'
    head += '''};
template<class T> T Power(T x,int n) { T value(1); for(int k=0;k<n;++k) value=value*x; return value; }
template<class T> T MonomialDerivative(const Term&t,const std::array<T,3>&x,const std::array<int,3>&orders) {
  const int powers[3]={t.px,t.py,t.pz}; T value(t.coefficient);
  for(int i=0;i<3;++i) { if(orders[i]>powers[i]) return T(0);
    for(int k=0;k<orders[i];++k) value=value*T(powers[i]-k);
    value=value*Power(x[i],powers[i]-orders[i]); }
  return value;
}
// Real m=0 coupled solid harmonics times W(rho), rho=x.x. Smooth W gives
// origin-regular Cartesian fields without divisions by r or origin branches.
template<class T> Field<T> EvaluateBasis(int J,int spin,int L,const std::array<T,3>&x,const WJet<T>&w) {
  const Record*record=nullptr; for(const auto&r:records) if(r.J==J&&r.spin==spin&&r.L==L) record=&r;
  if(!record) throw std::invalid_argument("Unsupported J,spin,L in total-J basis");
  Field<T> out; out.components=spin==0?1:(spin==1?3:9);
  std::array<Jet<T>,9> p{};
  for(int k=record->start;k<record->start+record->count;++k) {
    const auto&t=terms[k]; auto&v=p[t.component]; v.value+=MonomialDerivative(t,x,{0,0,0});
    for(int i=0;i<3;++i) { std::array<int,3> o{}; ++o[i]; v.d[i]+=MonomialDerivative(t,x,o);
      for(int j=0;j<3;++j) { ++o[j]; v.dd[i][j]+=MonomialDerivative(t,x,o); --o[j]; } }
  }
  for(int c=0;c<out.components;++c) { auto&v=out.component[c]; v.value=p[c].value*w.value;
    for(int i=0;i<3;++i) { const T wi=T(2)*x[i]*w.rho_d;
      v.d[i]=p[c].d[i]*w.value+p[c].value*wi;
      for(int j=0;j<3;++j) { const T wj=T(2)*x[j]*w.rho_d;
        const T wij=T(i==j?2:0)*w.rho_d+T(4)*x[i]*x[j]*w.rho_dd;
        v.dd[i][j]=p[c].dd[i][j]*w.value+p[c].d[i]*wj+p[c].d[j]*wi+p[c].value*wij; }
    }
  }
  return out;
}
} // namespace totalj
#endif
'''
    return head
