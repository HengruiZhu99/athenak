// Private CPU-only arithmetic work in progress. No native gauge adoption.
// Every operation is integer-only; finite binary64 atoms are decoded by bits.
#ifndef RESEARCH_EXACT_DYADIC_RATIO_HPP_
#define RESEARCH_EXACT_DYADIC_RATIO_HPP_
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exact_ratio {

// Explicit resource bound, independently of any later gauge degree bound.
// Active lengths are used; this is not a stack array or a GPU facility.
class UInt {
 public:
  static constexpr std::size_t kMaxLimbs = 4096;  // 131072 bits
  UInt() = default;
  explicit UInt(std::uint64_t value) {
    if (value) {
      limb_.push_back(static_cast<std::uint32_t>(value));
      if (value >> 32) limb_.push_back(static_cast<std::uint32_t>(value >> 32));
    }
  }
  bool zero() const { return limb_.empty(); }
  std::size_t bits() const {
    if (zero()) return 0;
    std::uint32_t top = limb_.back();
    std::size_t n = 0;
    while (top) { ++n; top >>= 1; }
    return 32 * (limb_.size() - 1) + n;
  }
  std::size_t trailing() const {
    if (zero()) throw std::logic_error("zero has no finite trailing count");
    std::size_t result = 0;
    for (std::uint32_t value : limb_) {
      if (!value) { result += 32; continue; }
      while (!(value & 1)) { ++result; value >>= 1; }
      break;
    }
    return result;
  }
  bool bit(std::size_t i) const {
    return i / 32 < limb_.size() && ((limb_[i / 32] >> (i % 32)) & 1);
  }
  static int compare(const UInt& a, const UInt& b) {
    if (a.limb_.size() != b.limb_.size()) return a.limb_.size() < b.limb_.size() ? -1 : 1;
    for (std::size_t i = a.limb_.size(); i-- > 0;) {
      if (a.limb_[i] != b.limb_[i]) return a.limb_[i] < b.limb_[i] ? -1 : 1;
    }
    return 0;
  }
  static UInt add(const UInt& a, const UInt& b) {
    UInt result;
    const std::size_t n = std::max(a.limb_.size(), b.limb_.size());
    check(n);
    result.limb_.resize(n);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t sum = carry + (i < a.limb_.size() ? a.limb_[i] : 0) +
                                (i < b.limb_.size() ? b.limb_[i] : 0);
      result.limb_[i] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32;
    }
    if (carry) { check(n + 1); result.limb_.push_back(static_cast<std::uint32_t>(carry)); }
    return result;
  }
  static UInt subtract(const UInt& a, const UInt& b) {
    if (compare(a, b) < 0) throw std::logic_error("unsigned subtraction is negative");
    UInt result;
    result.limb_.resize(a.limb_.size());
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < a.limb_.size(); ++i) {
      const std::uint64_t rhs = (i < b.limb_.size() ? b.limb_[i] : 0) + borrow;
      const std::uint64_t lhs = a.limb_[i];
      result.limb_[i] = static_cast<std::uint32_t>(lhs - rhs);
      borrow = lhs < rhs;
    }
    if (borrow) throw std::logic_error("unsigned subtraction borrow escaped");
    result.trim();
    return result;
  }
  static UInt multiply(const UInt& a, const UInt& b) {
    if (a.zero() || b.zero()) return UInt();
    UInt result;
    // This conservative capacity check is an explicit resource failure.
    // It need not admit every mathematical result near the chosen limit.
    check(a.limb_.size() + b.limb_.size());
    result.limb_.assign(a.limb_.size() + b.limb_.size(), 0);
    for (std::size_t i = 0; i < a.limb_.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < b.limb_.size(); ++j) {
        // (B-1)^2 + (B-1) + (B-1) = B^2-1, B=2^32.
        const std::uint64_t sum = std::uint64_t(a.limb_[i]) * b.limb_[j] +
                                  result.limb_[i + j] + carry;
        result.limb_[i + j] = static_cast<std::uint32_t>(sum);
        carry = sum >> 32;
      }
      // No previous row can have reached this index.
      result.limb_[i + b.limb_.size()] = static_cast<std::uint32_t>(carry);
    }
    result.trim();
    return result;
  }
  UInt left(std::size_t count) const {
    if (zero()) return UInt();
    const std::size_t words = count / 32;
    const unsigned shift = static_cast<unsigned>(count % 32);
    if (words > kMaxLimbs) throw std::length_error("exact integer shift capacity");
    check(limb_.size() + words + (shift ? 1 : 0));
    UInt result;
    result.limb_.assign(limb_.size() + words + (shift ? 1 : 0), 0);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < limb_.size(); ++i) {
      const std::uint64_t value = (std::uint64_t(limb_[i]) << shift) | carry;
      result.limb_[i + words] = static_cast<std::uint32_t>(value);
      carry = value >> 32;
    }
    if (shift) result.limb_[limb_.size() + words] = static_cast<std::uint32_t>(carry);
    result.trim();
    return result;
  }
  UInt right_exact(std::size_t count) const {
    if (zero()) return UInt();
    if (count > trailing()) throw std::logic_error("inexact integer right shift");
    const std::size_t words = count / 32;
    const unsigned shift = static_cast<unsigned>(count % 32);
    UInt result;
    result.limb_.resize(limb_.size() - words);
    for (std::size_t i = 0; i < result.limb_.size(); ++i) {
      std::uint64_t value = limb_[i + words];
      if (shift && i + words + 1 < limb_.size()) value |= std::uint64_t(limb_[i + words + 1]) << 32;
      result.limb_[i] = static_cast<std::uint32_t>(value >> shift);
    }
    result.trim();
    return result;
  }
 private:
  std::vector<std::uint32_t> limb_;
  static void check(std::size_t n) {
    if (n > kMaxLimbs) throw std::length_error("exact integer limb capacity");
  }
  void trim() { while (!limb_.empty() && limb_.back() == 0) limb_.pop_back(); }
};

// Compare a*2^ae and b*2^be without materializing either shift.
// All exponents accepted by this backend are bounded well inside int64.
inline int compare_scaled(const UInt& a, std::int64_t ae, const UInt& b, std::int64_t be) {
  if (a.zero() || b.zero()) return a.zero() ? (b.zero() ? 0 : -1) : 1;
  const std::int64_t atop = static_cast<std::int64_t>(a.bits()) + ae;
  const std::int64_t btop = static_cast<std::int64_t>(b.bits()) + be;
  if (atop != btop) return atop < btop ? -1 : 1;
  const std::int64_t low = std::min(ae, be);
  for (std::int64_t place = atop; place-- > low;) {
    const bool av = place >= ae && a.bit(static_cast<std::size_t>(place - ae));
    const bool bv = place >= be && b.bit(static_cast<std::size_t>(place - be));
    if (av != bv) return av ? 1 : -1;
  }
  return 0;
}

struct Dyadic {
  int sign = 0;
  UInt magnitude;
  std::int64_t exponent = 0;
  Dyadic() = default;
  Dyadic(int s, UInt m, std::int64_t e) : sign(s), magnitude(std::move(m)), exponent(e) {
    normalize();
  }
  void normalize() {
    if (magnitude.zero()) { sign = 0; exponent = 0; return; }
    if (sign != 1 && sign != -1) throw std::invalid_argument("invalid nonzero dyadic sign");
    const std::size_t n = magnitude.trailing();
    magnitude = magnitude.right_exact(n);
    exponent += static_cast<std::int64_t>(n);
    // Explicit arithmetic resource contract; subsequent additions cannot
    // overflow signed exponent arithmetic or size_t conversion.
    if (exponent < -131072 || exponent > 131072) throw std::length_error("exact dyadic exponent capacity");
  }
  static Dyadic atom(std::uint64_t bits) {
    const std::uint64_t eb = (bits >> 52) & 2047;
    if (eb == 2047) throw std::invalid_argument("nonfinite binary64 atom");
    const std::uint64_t mantissa = (bits & UINT64_C(0xfffffffffffff)) | (eb ? UINT64_C(0x10000000000000) : 0);
    return Dyadic((bits >> 63) ? -1 : 1, UInt(mantissa), eb ? static_cast<std::int64_t>(eb) - 1075 : -1074);
  }
  static Dyadic integer(std::int64_t n) {
    const std::uint64_t magnitude = n < 0 ? std::uint64_t(-(n + 1)) + 1 : std::uint64_t(n);
    return Dyadic(n < 0 ? -1 : 1, UInt(magnitude), 0);
  }
};

inline Dyadic negate(Dyadic a) { a.sign = -a.sign; return a; }
inline Dyadic add(const Dyadic& a, const Dyadic& b) {
  if (!a.sign) return b;
  if (!b.sign) return a;
  const std::int64_t exponent = std::min(a.exponent, b.exponent);
  const UInt av = a.magnitude.left(static_cast<std::size_t>(a.exponent - exponent));
  const UInt bv = b.magnitude.left(static_cast<std::size_t>(b.exponent - exponent));
  if (a.sign == b.sign) return Dyadic(a.sign, UInt::add(av, bv), exponent);
  const int cmp = UInt::compare(av, bv);
  return cmp >= 0 ? Dyadic(a.sign, UInt::subtract(av, bv), exponent) :
                    Dyadic(b.sign, UInt::subtract(bv, av), exponent);
}
inline Dyadic subtract(const Dyadic& a, const Dyadic& b) { return add(a, negate(b)); }
inline Dyadic multiply(const Dyadic& a, const Dyadic& b) {
  if (!a.sign || !b.sign) return Dyadic();
  return Dyadic(a.sign * b.sign, UInt::multiply(a.magnitude, b.magnitude), a.exponent + b.exponent);
}

struct Quotient {
  std::uint64_t floor = 0;
  UInt remainder;
  UInt denominator;
};

// Only used after the final output exponent is known; quotient has <=53bits.
inline Quotient small_quotient(const UInt& numerator, const UInt& denominator, std::int64_t shift) {
  Quotient q;
  q.remainder = shift >= 0 ? numerator.left(static_cast<std::size_t>(shift)) : numerator;
  q.denominator = shift < 0 ? denominator.left(static_cast<std::size_t>(-shift)) : denominator;
  if (q.denominator.zero()) throw std::invalid_argument("zero rational denominator");
  const std::int64_t top = static_cast<std::int64_t>(q.remainder.bits()) - static_cast<std::int64_t>(q.denominator.bits());
  if (top > 53) throw std::logic_error("final rational quotient exceeded53bits");
  for (std::int64_t i = top; i >= 0; --i) {
    const UInt candidate = q.denominator.left(static_cast<std::size_t>(i));
    if (UInt::compare(q.remainder, candidate) >= 0) {
      q.remainder = UInt::subtract(q.remainder, candidate);
      q.floor |= UINT64_C(1) << i;
    }
  }
  return q;
}

enum class Status { finite, exact_overflow };
struct Rounded { Status status; std::uint64_t bits; };

// Strong range contract: abs(exact quotient) <= maxfinite is required.
// Values just above maxfinite return explicit overflow even if IEEE rounding
// would return maxfinite. Exact zero is +0; nonzero negative underflow is -0.
inline Rounded round_ratio(const Dyadic& numerator, const Dyadic& denominator) {
  if (!denominator.sign) throw std::invalid_argument("zero rational denominator");
  if (!numerator.sign) return {Status::finite, 0};
  const std::uint64_t sign = numerator.sign == denominator.sign ? 0 : UINT64_C(0x8000000000000000);
  const UInt& u = numerator.magnitude;
  const UInt& v = denominator.magnitude;
  const std::int64_t shift = numerator.exponent - denominator.exponent;
  const UInt vmax = UInt::multiply(v, UInt(UINT64_C(0x1fffffffffffff)));
  if (compare_scaled(u, shift, vmax, 971) > 0) return {Status::exact_overflow, 0};
  std::int64_t exponent = static_cast<std::int64_t>(u.bits()) - static_cast<std::int64_t>(v.bits()) + shift;
  if (compare_scaled(u, shift, v, exponent) < 0) --exponent;
  // A value strictly below half a minimum subnormal always rounds to zero.
  // This also avoids unnecessary enormous shifts for tiny exact quotients.
  if (exponent < -1075) return {Status::finite, sign};
  const std::int64_t unit = exponent >= -1022 ? exponent - 52 : -1074;
  Quotient q = small_quotient(u, v, shift - unit);
  const int halfway = compare_scaled(q.remainder, 1, q.denominator, 0);
  if (halfway > 0 || (halfway == 0 && (q.floor & 1))) ++q.floor;
  if (exponent < -1022) return {Status::finite, sign | q.floor};
  if (q.floor == UINT64_C(0x20000000000000)) { q.floor >>= 1; ++exponent; }
  if (exponent > 1023 || q.floor < UINT64_C(0x10000000000000) || q.floor >= UINT64_C(0x20000000000000))
    throw std::logic_error("normal rational rounding invariant");
  return {Status::finite, sign | (static_cast<std::uint64_t>(exponent + 1023) << 52) |
                          (q.floor - UINT64_C(0x10000000000000))};
}
}  // namespace exact_ratio
#endif
