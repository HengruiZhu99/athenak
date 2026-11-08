// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "config.hpp"
#include "mesh/logical_location.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
void Check(bool ok) {
  if (!ok) throw std::runtime_error("LogicalLocation record or MPI roundtrip failed");
}
bool Same(const LogicalLocation &a, const LogicalLocation &b) {
  return a.lx1 == b.lx1 && a.lx2 == b.lx2 && a.lx3 == b.lx3 &&
         a.level == b.level && a.chart_id == b.chart_id;
}
}  // namespace

int main(int argc, char **argv) {
  int rank=0, nranks=1;
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
  std::array<char, 3*5*sizeof(std::int32_t)> bytes{};
  for (int version=1; version<=2; ++version) {
    const auto size = logical_location::RecordSize(version);
    const LogicalLocation input[3] = {{1,2,3,4}, {2147483647,0,0,31},
                                      {12,7,2,7,version == 1 ? 0 : 6}};
    for (int i=0; i<3; ++i) logical_location::Encode(bytes.data()+i*size,input[i],version);
    for (int i=0; i<3; ++i) {
      Check(Same(input[i], logical_location::Decode(bytes.data()+i*size,version)));
    }
  }
  bool rejected=false;
  try {logical_location::Encode(bytes.data(),{1,2,3,4,1},1);}
  catch (const std::runtime_error &) {rejected=true;}
  Check(rejected);
  for (int version : {-1,0,3}) {
    rejected=false;
    try {logical_location::RecordSize(version);}
    catch (const std::runtime_error &) {rejected=true;}
    Check(rejected);
  }
#if MPI_PARALLEL_ENABLED
  // Transport several adjacent records so a stale four-integer stride fails.
  LogicalLocation send[3] = {{rank,2,3,5,rank%7}, {0,rank,4,7,6}, {1,2,rank,9,1}};
  std::vector<LogicalLocation> recv(3*nranks);
  MPI_Datatype location_type;
  MPI_Type_contiguous(LogicalLocation::nint, MPI_INT32_T, &location_type);
  MPI_Type_commit(&location_type);
  MPI_Allgather(send,3,location_type,recv.data(),3,location_type,MPI_COMM_WORLD);
  MPI_Type_free(&location_type);
  for (int r=0; r<nranks; ++r) {
    Check(Same(recv[3*r], {r,2,3,5,r%7}));
    Check(Same(recv[3*r+1], {0,r,4,7,6}));
    Check(Same(recv[3*r+2], {1,2,r,9,1}));
  }
  MPI_Finalize();
#endif
  if (rank == 0) std::cout << "location records and " << nranks << "-rank MPI passed\n";
}
