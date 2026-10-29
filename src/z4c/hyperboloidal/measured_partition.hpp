// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_MEASURED_PARTITION_HPP_
#define Z4C_HYPERBOLOIDAL_MEASURED_PARTITION_HPP_
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>
namespace z4c::hyperboloidal {
struct MeasuredPartition {
  std::vector<int> cuts;
  std::string source_report_sha256;
};
// A fixed ownership choice only: no field interpolation or mesh mutation.
inline MeasuredPartition ReadMeasuredPartition(std::istream&input,int n,int cells,int ranks){
  std::string magic,tail;int file_n=0,file_cells=0,file_ranks=0;MeasuredPartition out;
  if(!(input>>magic>>file_n>>file_cells>>file_ranks>>out.source_report_sha256)
      ||magic!="HYP3D_COST_PARTITION_V1"||file_n!=n||file_cells!=cells||file_ranks!=ranks
      ||ranks<1||cells<ranks||out.source_report_sha256.size()!=64
      ||out.source_report_sha256.find_first_not_of("0123456789abcdef")!=std::string::npos)
    throw std::invalid_argument("measured partition descriptor/grid mismatch");
  out.cuts.resize(ranks+1);
  for(int&cut:out.cuts)if(!(input>>cut))throw std::invalid_argument("truncated measured partition cuts");
  if(out.cuts.front()!=0||out.cuts.back()!=cells||input>>tail)
    throw std::invalid_argument("measured partition coverage/trailing data");
  for(int r=0;r<ranks;++r)if(out.cuts[r]>=out.cuts[r+1])
    throw std::invalid_argument("measured partition has empty or reversed ownership");
  return out;
}
}
#endif
