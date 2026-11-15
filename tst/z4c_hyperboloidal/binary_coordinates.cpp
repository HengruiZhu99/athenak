#include <cassert>
#include <cmath>
#include <limits>
#include "z4c_hyperboloidal/binary_coordinates.hpp"
int main() {
  using z4c_hyperboloidal::BinaryCoordinateMatches;
  for(double x:{0.,-.20233727083953934,.94,1.}) {
    assert(BinaryCoordinateMatches(x,x));
    assert(BinaryCoordinateMatches(std::nextafter(x,2.),x));
    assert(!BinaryCoordinateMatches(x+1e-8,x));
    assert(!BinaryCoordinateMatches(x+0.6/36/32,x));
    assert(!BinaryCoordinateMatches(std::numeric_limits<double>::quiet_NaN(),x));
    assert(!BinaryCoordinateMatches(std::numeric_limits<double>::infinity(),x));
  }
}
