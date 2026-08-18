// Standalone Perlmutter preflight: exchange CUDA device buffers across an MPI ring.
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__global__ void Fill(int *data, int n, int value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value;
}

void CheckCuda(cudaError_t status, const char *what, int rank) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "rank=%d %s failed: %s\n", rank, what,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank);
  if (device_count != 1) {
    std::fprintf(stderr, "rank=%d expected one visible GPU, found %d\n", rank,
                 device_count);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  CheckCuda(cudaSetDevice(0), "cudaSetDevice", rank);

  constexpr int n = 4096;
  int *send = nullptr, *recv = nullptr;
  CheckCuda(cudaMalloc(&send, n * sizeof(int)), "cudaMalloc(send)", rank);
  CheckCuda(cudaMalloc(&recv, n * sizeof(int)), "cudaMalloc(recv)", rank);
  Fill<<<(n + 255) / 256, 256>>>(send, n, rank);
  CheckCuda(cudaGetLastError(), "Fill launch", rank);
  CheckCuda(cudaDeviceSynchronize(), "Fill synchronize", rank);

  const int source = (rank + size - 1) % size;
  const int destination = (rank + 1) % size;
  MPI_Request requests[2];
  MPI_Irecv(recv, n, MPI_INT, source, 17, MPI_COMM_WORLD, &requests[0]);
  MPI_Isend(send, n, MPI_INT, destination, 17, MPI_COMM_WORLD, &requests[1]);
  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
  CheckCuda(cudaDeviceSynchronize(), "MPI receive synchronize", rank);

  std::vector<int> host(n);
  CheckCuda(cudaMemcpy(host.data(), recv, n * sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(receive)", rank);
  int local_bad = 0;
  for (int value : host) local_bad += (value != source);
  int global_bad = 0;
  MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties", rank);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_length = 0;
  MPI_Get_processor_name(hostname, &hostname_length);
  std::printf("rank=%d host=%.*s gpu=%s source=%d bad=%d\n", rank,
              hostname_length, hostname, properties.name, source, local_bad);

  cudaFree(recv);
  cudaFree(send);
  MPI_Finalize();
  return global_bad == 0 ? 0 : 1;
}
