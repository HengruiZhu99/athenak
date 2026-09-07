// Diagnostic only: these pure point headers use Kokkos only for annotations.
// The real CPU/CUDA oracle targets continue to use actual Kokkos.
#ifndef PC_GH_HOST_ANNOTATION_SHIM_HPP_
#define PC_GH_HOST_ANNOTATION_SHIM_HPP_
#define KOKKOS_INLINE_FUNCTION inline
#endif
