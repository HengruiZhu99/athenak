#ifndef RESEARCH_EXACT_SIGNED_PRODUCTS_HPP
#define RESEARCH_EXACT_SIGNED_PRODUCTS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Standalone CPU proposal. No gauge implementation includes this header.
namespace signed_products {
static_assert(sizeof(double) == sizeof(std::uint64_t), "binary64 storage required");
static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::radix == 2 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::max_exponent == 1024 &&
              std::numeric_limits<double>::min_exponent == -1021,
              "IEEE binary64 required");
#if !defined(__SIZEOF_INT128__)
#error "This CPU prototype requires unsigned __int128"
#endif

constexpr std::size_t kPrimalTerms = 32;
constexpr std::size_t kFactors = 4;
constexpr std::size_t kDualTerms = kPrimalTerms * kFactors;
constexpr std::size_t kWords = 136;
constexpr int kBaseExponent = -4352;
using Wide = unsigned __int128;
using Big = std::array<std::uint64_t, kWords>;

enum class Status {
  ok, invalid_sign, invalid_shift, factor_cap, term_cap, invalid_atom,
  null_input, accumulator_overflow, exact_overflow
};

inline const char* Name(Status s) {
  switch (s) {
    case Status::ok: return "ok";
    case Status::invalid_sign: return "invalid_sign";
    case Status::invalid_shift: return "invalid_shift";
    case Status::factor_cap: return "factor_cap";
    case Status::term_cap: return "term_cap";
    case Status::invalid_atom: return "invalid_atom";
    case Status::null_input: return "null_input";
    case Status::accumulator_overflow: return "accumulator_overflow";
    case Status::exact_overflow: return "exact_overflow";
  }
  return "unknown";
}

inline std::uint64_t Bits(double x) {
  std::uint64_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}
inline double FromBits(std::uint64_t bits) {
  double x;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

struct Atom {
  std::uint64_t significand = 0;
  int exponent = -1074;
  bool negative = false;
  bool finite = true;
};
inline Atom Decode(double x) {
  const std::uint64_t bits = Bits(x);
  const unsigned exp = static_cast<unsigned>((bits >> 52) & 0x7ffU);
  Atom out;
  out.negative = (bits >> 63) != 0;
  out.finite = exp != 0x7ffU;
  out.significand = bits & UINT64_C(0x000fffffffffffff);
  if (exp != 0 && out.finite) {
    out.significand |= UINT64_C(0x0010000000000000);
    out.exponent = static_cast<int>(exp) - 1075;
  }
  return out;
}

struct Term {
  int sign = 1;
  int binary_shift = 0;
  std::size_t arity = 0;
  std::array<double, kFactors> factors{};
};
struct DualAtom { double value = 0; double tangent = 0; };
struct DualTerm {
  int sign = 1;
  int binary_shift = 0;
  std::size_t arity = 0;
  std::array<DualAtom, kFactors> factors{};
};

struct Audit {
  std::size_t monomials = 0;
  std::size_t nonzero_monomials = 0;
  std::size_t zero_monomials = 0;
  bool exact_zero = false;
  bool inexact = false;
  bool rounded_to_zero = false;
  bool negative_rounded_zero = false;
  bool subnormal_result = false;
  // Inexact rounding error is <= 2^half_grid_exponent in absolute value.
  // This integer exponent remains meaningful below binary64's range.
  int half_grid_exponent = 0;
};
struct Result {
  Status status = Status::ok;
  std::uint64_t bits = 0;
  Audit audit{};
  double value() const { return FromBits(bits); }
};
struct DualResult {
  Status validation = Status::ok;
  Result primal{}, tangent{};
  std::size_t generated_tangent_terms = 0;
};

namespace detail {
inline Status Validate(int sign, int shift, std::size_t arity) {
  if (sign != 1 && sign != -1) return Status::invalid_sign;
  if (shift != -1 && shift != 0) return Status::invalid_shift;
  if (arity > kFactors) return Status::factor_cap;
  return Status::ok;
}
inline int Compare(const Big& a, const Big& b) {
  for (std::size_t i = kWords; i-- > 0;) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}
inline Big Subtract(const Big& a, const Big& b) {
  // Caller has established a >= b. One exact subtraction after accumulation.
  Big out{};
  std::uint64_t borrow = 0;
  for (std::size_t i = 0; i < kWords; ++i) {
    const Wide rhs = static_cast<Wide>(b[i]) + borrow;
    out[i] = a[i] - static_cast<std::uint64_t>(rhs);
    borrow = static_cast<Wide>(a[i]) < rhs ? 1 : 0;
  }
  return out;
}
inline bool AddWord(Big& dest, std::size_t index, std::uint64_t word) {
  while (word != 0) {
    if (index >= kWords) return false;
    const Wide sum = static_cast<Wide>(dest[index]) + word;
    dest[index] = static_cast<std::uint64_t>(sum);
    word = static_cast<std::uint64_t>(sum >> 64);
    ++index;
  }
  return true;
}
inline bool AddProduct(Big& dest, const std::array<std::uint64_t, 4>& product,
                       int shift) {
  if (shift < 0) return false;
  const std::size_t first = static_cast<std::size_t>(shift / 64);
  const unsigned offset = static_cast<unsigned>(shift % 64);
  for (std::size_t i = 0; i < product.size(); ++i) {
    if (!AddWord(dest, first + i, product[i] << offset)) return false;
    if (offset != 0 && !AddWord(dest, first + i + 1, product[i] >> (64 - offset)))
      return false;
  }
  return true;
}
inline bool Multiply(std::array<std::uint64_t, 4>& product, std::uint64_t sig) {
  Wide carry = 0;
  for (auto& word : product) {
    const Wide next = static_cast<Wide>(word) * sig + carry;
    word = static_cast<std::uint64_t>(next);
    carry = next >> 64;
  }
  return carry == 0;
}
inline int Highest(const Big& n) {
  for (std::size_t i = kWords; i-- > 0;) {
    if (n[i] != 0) {
      unsigned high = 0;
      for (std::uint64_t w = n[i]; w >>= 1;) ++high;
      return static_cast<int>(64 * i + high);
    }
  }
  return -1;
}
inline bool Bit(const Big& n, int index) {
  return index >= 0 && index < static_cast<int>(64 * kWords) &&
         ((n[static_cast<std::size_t>(index / 64)] >> (index % 64)) & 1U) != 0;
}
inline bool AnyBelow(const Big& n, int count) {
  if (count <= 0) return false;
  const std::size_t full = static_cast<std::size_t>(count / 64);
  const unsigned partial = static_cast<unsigned>(count % 64);
  for (std::size_t i = 0; i < full; ++i) if (n[i] != 0) return true;
  return partial != 0 &&
         (n[full] & ((UINT64_C(1) << partial) - 1)) != 0;
}
inline std::uint64_t Quotient(const Big& n, int shift) {
  // Only the low 53 quotient bits can be nonzero at either rounding branch.
  std::uint64_t q = 0;
  for (int i = 0; i < 53; ++i) if (Bit(n, shift + i)) q |= UINT64_C(1) << i;
  return q;
}
inline Big MaxFinite() {
  Big out{};
  std::array<std::uint64_t, 4> sig{{UINT64_C(0x001fffffffffffff), 0, 0, 0}};
  (void)AddProduct(out, sig, 971 - kBaseExponent);
  return out;
}
inline Result Round(const Big& magnitude, bool negative, Audit audit) {
  Result out;
  out.audit = audit;
  const int highest = Highest(magnitude);
  if (highest < 0) {
    out.audit.exact_zero = true;
    return out;  // Exact cancellation is canonical +0.
  }
  if (Compare(magnitude, MaxFinite()) > 0) {
    out.status = Status::exact_overflow;
    return out;  // Deliberately stronger than only RN-overflow rejection.
  }
  int exponent = kBaseExponent + highest;
  const bool normal = exponent >= -1022;
  const int trim = normal ? highest - 52 : -1074 - kBaseExponent;
  std::uint64_t q = Quotient(magnitude, trim);
  const bool guard = Bit(magnitude, trim - 1);
  const bool sticky = AnyBelow(magnitude, trim - 1);
  out.audit.inexact = guard || sticky;
  out.audit.half_grid_exponent = normal ? exponent - 53 : -1075;
  if (guard && (sticky || (q & 1U) != 0)) ++q;
  std::uint64_t bits;
  if (normal) {
    if (q == (UINT64_C(1) << 53)) { q >>= 1; ++exponent; }
    bits = (static_cast<std::uint64_t>(exponent + 1023) << 52) |
           (q & UINT64_C(0x000fffffffffffff));
  } else {
    bits = q;  // q == 2^52 carries exactly into minimum normal.
  }
  if (negative) bits |= UINT64_C(0x8000000000000000);
  out.bits = bits;
  out.audit.rounded_to_zero = q == 0;
  out.audit.negative_rounded_zero = negative && q == 0;
  out.audit.subnormal_result = (bits & UINT64_C(0x7ff0000000000000)) == 0 && q != 0;
  return out;
}
inline Result Sum(const Term* terms, std::size_t count, std::size_t limit) {
  Result out;
  if (count > limit) { out.status = Status::term_cap; return out; }
  if (count != 0 && terms == nullptr) { out.status = Status::null_input; return out; }
  // Validate all consumed atoms before zero-product shortcuts or accumulation.
  for (std::size_t i = 0; i < count; ++i) {
    const auto& t = terms[i];
    out.status = Validate(t.sign, t.binary_shift, t.arity);
    if (out.status != Status::ok) return out;
    for (std::size_t j = 0; j < t.arity; ++j)
      if (!Decode(t.factors[j]).finite) { out.status = Status::invalid_atom; return out; }
  }
  Big positive{}, negative{};
  Audit audit;
  audit.monomials = count;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& t = terms[i];
    std::array<std::uint64_t, 4> product{{1, 0, 0, 0}};
    int exponent = t.binary_shift;
    bool neg = t.sign < 0;
    bool zero = false;
    for (std::size_t j = 0; j < t.arity; ++j) {
      const Atom a = Decode(t.factors[j]);
      neg = neg != a.negative;
      exponent += a.exponent;
      zero = zero || a.significand == 0;
      if (!Multiply(product, a.significand)) {
        out.status = Status::accumulator_overflow;
        return out;
      }
    }
    if (zero) { ++audit.zero_monomials; continue; }
    ++audit.nonzero_monomials;
    if (!AddProduct(neg ? negative : positive, product, exponent - kBaseExponent)) {
      out.status = Status::accumulator_overflow;
      return out;
    }
  }
  const int cmp = Compare(positive, negative);
  return Round(cmp < 0 ? Subtract(negative, positive) : Subtract(positive, negative),
               cmp < 0, audit);
}
}  // namespace detail

inline Result Evaluate(const Term* terms, std::size_t count) {
  return detail::Sum(terms, count, kPrimalTerms);
}
inline DualResult EvaluateDual(const DualTerm* terms, std::size_t count) {
  DualResult out;
  if (count > kPrimalTerms) { out.validation = Status::term_cap; return out; }
  if (count != 0 && terms == nullptr) { out.validation = Status::null_input; return out; }
  for (std::size_t i = 0; i < count; ++i) {
    const auto& t = terms[i];
    out.validation = detail::Validate(t.sign, t.binary_shift, t.arity);
    if (out.validation != Status::ok) return out;
    for (std::size_t j = 0; j < t.arity; ++j)
      if (!Decode(t.factors[j].value).finite || !Decode(t.factors[j].tangent).finite) {
        out.validation = Status::invalid_atom;
        return out;
      }
  }
  std::array<Term, kPrimalTerms> primal{};
  std::array<Term, kDualTerms> tangent{};
  std::size_t generated = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& t = terms[i];
    primal[i].sign = t.sign;
    primal[i].binary_shift = t.binary_shift;
    primal[i].arity = t.arity;
    for (std::size_t j = 0; j < t.arity; ++j) primal[i].factors[j] = t.factors[j].value;
    // Complete polynomial product rule: no live-factor division and no seed test.
    for (std::size_t differentiated = 0; differentiated < t.arity; ++differentiated) {
      Term& dt = tangent[generated++];
      dt.sign = t.sign;
      dt.binary_shift = t.binary_shift;
      dt.arity = t.arity;
      for (std::size_t j = 0; j < t.arity; ++j)
        dt.factors[j] = j == differentiated ? t.factors[j].tangent : t.factors[j].value;
    }
  }
  out.generated_tangent_terms = generated;
  out.primal = detail::Sum(primal.data(), count, kPrimalTerms);
  out.tangent = detail::Sum(tangent.data(), generated, kDualTerms);
  return out;
}
}  // namespace signed_products
#endif
