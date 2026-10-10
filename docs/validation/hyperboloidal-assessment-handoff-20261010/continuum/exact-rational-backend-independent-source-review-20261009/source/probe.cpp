// Uncompiled CPU-private unit probe for exact dyadic arithmetic.
#include "exact_dyadic_ratio.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
std::uint64_t parse_bits(const std::string& s) {
  std::size_t used = 0;
  const std::uint64_t value = std::stoull(s, &used, 16);
  if (used != s.size() || s.empty() || s.size() > 16)
    throw std::invalid_argument("malformed atom bits");
  return value;
}

std::string exact_hex(const exact_ratio::UInt& value) {
  if (value.zero()) return "0";
  const char* digit = "0123456789abcdef";
  const std::size_t nibbles = (value.bits() + 3) / 4;
  std::string result(nibbles, '0');
  for (std::size_t n = 0; n < nibbles; ++n) {
    unsigned x = 0;
    for (unsigned j = 0; j < 4; ++j) if (value.bit(4*n+j)) x |= 1u << j;
    result[nibbles - 1 - n] = digit[x];
  }
  return result;
}

void write_dyadic(const exact_ratio::Dyadic& value) {
  std::cout << '\t' << value.sign << '\t' << value.exponent << '\t' << exact_hex(value.magnitude);
}
}  // namespace

// One complete case per line: id, atom_count, operation_count, numerator_index,
// denominator_index, atom hex bits, then triples(op,left_index,right_index).
// L's right token is an unsigned integer shift, solely for resource controls.
int main() {
  std::string line;
  std::size_t cases = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream input(line);
    std::string id;
    std::size_t atoms = 0, operations = 0, numerator = 0, denominator = 0;
    if (!(input >> id >> atoms >> operations >> numerator >> denominator) ||
        atoms == 0 || atoms > 256 || operations > 512 || ++cases > 4096) {
      std::cerr << "invalid unit case framing\n";
      return 2;
    }
    try {
      std::vector<exact_ratio::Dyadic> value;
      value.reserve(atoms + operations);
      for (std::size_t i = 0; i < atoms; ++i) {
        std::string text;
        if (!(input >> text)) throw std::invalid_argument("missing atom");
        value.push_back(exact_ratio::Dyadic::atom(parse_bits(text)));
      }
      for (std::size_t i = 0; i < operations; ++i) {
        char op;
        std::size_t a, b;
        if (!(input >> op >> a >> b) || a >= value.size() || (op != 'L' && b >= value.size()))
          throw std::invalid_argument("malformed operation");
        if (op == '+') value.push_back(exact_ratio::add(value[a], value[b]));
        else if (op == '-') value.push_back(exact_ratio::subtract(value[a], value[b]));
        else if (op == '*') value.push_back(exact_ratio::multiply(value[a], value[b]));
        else if (op == 'N') value.push_back(exact_ratio::negate(value[a]));
        else if (op == 'L') value.emplace_back(value[a].sign, value[a].magnitude.left(b), value[a].exponent);
        else throw std::invalid_argument("unknown operation");
      }
      std::string excess;
      if (input >> excess) throw std::invalid_argument("extra unit tokens");
      if (numerator >= value.size() || denominator >= value.size())
        throw std::invalid_argument("final operand index out of range");
      const auto rounded = exact_ratio::round_ratio(value[numerator], value[denominator]);
      std::cout << id << '\t' << (rounded.status == exact_ratio::Status::finite ? "finite" : "exact_overflow")
                << '\t' << std::hex << std::setfill('0') << std::setw(16) << rounded.bits << std::dec;
      write_dyadic(value[numerator]);
      write_dyadic(value[denominator]);
      std::cout << '\n';
    } catch (const std::length_error& error) {
      std::cout << id << "\tresource_failure\t" << error.what() << '\n';
    } catch (const std::invalid_argument& error) {
      std::cout << id << "\tdomain_failure\t" << error.what() << '\n';
    } catch (const std::exception& error) {
      std::cerr << id << ": unexpected exception: " << error.what() << '\n';
      return 3;
    }
  }
  if (!cases) { std::cerr << "empty unit input\n"; return 2; }
  return 0;
}
