#ifndef Z4C_HYPERBOLOIDAL_RESTART_HPP_
#define Z4C_HYPERBOLOIDAL_RESTART_HPP_

#include <string>
#include "athena.hpp"
#include "outputs/io_wrapper.hpp"

class Mesh;
namespace z4c_hyperboloidal {
class Z4cHyperboloidal;
namespace restart {
// Native restart block records use fixed strides indexed by native leaf GID.
// Only evolved cells (including evolved overlap) and owned scri points are packed.
IOWrapperSizeT RecordBytes(const Z4cHyperboloidal &p);
std::string Identity(const Z4cHyperboloidal &p);
IOWrapperSizeT WriteMetadata(IOWrapper &file, const Mesh &mesh, bool per_rank);
void ReadMetadata(IOWrapper &file, const Mesh &mesh, bool per_rank);
IOWrapperSizeT ReadStride(IOWrapper &file, bool per_rank);
void Pack(const Z4cHyperboloidal &p, HostArray2D<Real> &host);
void Unpack(Z4cHyperboloidal &p, const HostArray2D<Real> &host);
void WriteRecords(IOWrapper &file, const Mesh &mesh, const HostArray2D<Real> &host,
                  IOWrapperSizeT offset, IOWrapperSizeT stride, bool per_rank);
void ReadRecords(IOWrapper &file, Mesh &mesh, IOWrapperSizeT offset,
                 IOWrapperSizeT stride, bool per_rank);
// Checked arithmetic also guards offsets before the original generic I/O loops.
IOWrapperSizeT Add(IOWrapperSizeT a, IOWrapperSizeT b);
IOWrapperSizeT Multiply(IOWrapperSizeT a, IOWrapperSizeT b);
void CheckFileExtent(IOWrapper &file, IOWrapperSizeT offset, IOWrapperSizeT stride,
                     IOWrapperSizeT expected_stride, int records, bool per_rank);
}  // namespace restart
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_RESTART_HPP_
