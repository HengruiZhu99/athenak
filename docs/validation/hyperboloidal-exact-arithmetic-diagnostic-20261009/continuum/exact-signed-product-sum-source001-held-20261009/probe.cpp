#include "signed_products.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sp = signed_products;
std::string Hex(std::uint64_t bits) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << bits;
  return out.str();
}
std::uint64_t ParseHex(const std::string& word) {
  if (word.size() != 16 || word.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error("expected exactly16 lowercase hex digits");
  std::size_t end = 0;
  auto value = std::stoull(word, &end, 16);
  if (end != word.size()) throw std::runtime_error("bad hex atom");
  return value;
}
void PrintResult(const sp::Result& r) {
  const auto& a = r.audit;
  std::cout << "{\"status\":\"" << sp::Name(r.status) << "\",\"bits\":\"" << Hex(r.bits)
            << "\",\"audit\":{\"monomials\":" << a.monomials
            << ",\"nonzero_monomials\":" << a.nonzero_monomials
            << ",\"zero_monomials\":" << a.zero_monomials
            << ",\"exact_zero\":" << a.exact_zero
            << ",\"inexact\":" << a.inexact
            << ",\"rounded_to_zero\":" << a.rounded_to_zero
            << ",\"negative_rounded_zero\":" << a.negative_rounded_zero
            << ",\"subnormal_result\":" << a.subnormal_result
            << ",\"half_grid_exponent\":" << a.half_grid_exponent << "}}";
}
int main() {
  try {
    std::cout << std::boolalpha;
    std::string marker, id, mode;
    std::size_t count;
    int null_input;
    while (std::cin >> marker) {
      if (marker != "CASE" || !(std::cin >> id >> mode >> count >> null_input) ||
          id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos ||
          (mode != "scalar" && mode != "dual") || count > 33 ||
          (null_input != 0 && null_input != 1))
        throw std::runtime_error("invalid case header");
      std::vector<sp::DualTerm> terms(count);
      std::vector<sp::Term> scalar(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& t = terms[i];
        if (!(std::cin >> marker >> t.sign >> t.binary_shift >> t.arity) || marker != "TERM")
          throw std::runtime_error("invalid term header");
        scalar[i].sign = t.sign;
        scalar[i].binary_shift = t.binary_shift;
        scalar[i].arity = t.arity;
        for (std::size_t j = 0; j < sp::kFactors; ++j) {
          std::string value, tangent;
          if (!(std::cin >> value >> tangent)) throw std::runtime_error("truncated atoms");
          t.factors[j] = {sp::FromBits(ParseHex(value)), sp::FromBits(ParseHex(tangent))};
          scalar[i].factors[j] = t.factors[j].value;
        }
      }
      std::cout << "{\"id\":\"" << id << "\",\"mode\":\"" << mode
                << "\",\"null_input\":" << (null_input != 0) << ",\"terms\":[";
      for (std::size_t i = 0; i < count; ++i) {
        const auto& t = terms[i];
        if (i != 0) std::cout << ',';
        std::cout << "{\"sign\":" << t.sign << ",\"shift\":" << t.binary_shift
                  << ",\"arity\":" << t.arity << ",\"atoms\":[";
        for (std::size_t j = 0; j < sp::kFactors; ++j) {
          if (j != 0) std::cout << ',';
          std::cout << "[\"" << Hex(sp::Bits(t.factors[j].value)) << "\",\""
                    << Hex(sp::Bits(t.factors[j].tangent)) << "\"]";
        }
        std::cout << "]}";
      }
      std::cout << ']';
      if (mode == "scalar") {
        std::cout << ",\"result\":";
        PrintResult(sp::Evaluate(null_input ? nullptr : scalar.data(), count));
      } else {
        auto r = sp::EvaluateDual(null_input ? nullptr : terms.data(), count);
        std::cout << ",\"validation\":\"" << sp::Name(r.validation)
                  << "\",\"generated_tangent_terms\":" << r.generated_tangent_terms
                  << ",\"primal\":";
        PrintResult(r.primal);
        std::cout << ",\"tangent\":";
        PrintResult(r.tangent);
      }
      std::cout << "}\n";
    }
    if (!std::cin.eof()) throw std::runtime_error("input parser failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
