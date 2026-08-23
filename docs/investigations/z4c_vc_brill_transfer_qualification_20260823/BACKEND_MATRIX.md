# Backend and restart matrix

Source authority: `278b63a740a947de55ad8bdd1c333095c68fedcd`

| Backend/gate | q4 | q6 | Classification |
|---|---|---|---|
| Portable host suite | passed | passed | `vc_host=QUALIFIED` |
| Host MPI controls | passed | passed | `vc_host=QUALIFIED` |
| Perlmutter CUDA, one A100 | passed | passed | `vc_cuda=QUALIFIED` |
| Perlmutter CUDA, two MPI ranks / one A100 | passed | passed | `vc_cuda=QUALIFIED` |
| Refined restart | bit-exact | bit-exact | `vc_restart=BIT_EXACT_QUALIFIED` |
| Post-derefinement restart | bit-exact | bit-exact | `vc_restart=BIT_EXACT_QUALIFIED` |
| Current-source Aurora/SYCL | not run | not run | `vc_sycl=PENDING` |

The clean CUDA/MPI evidence is in `artifacts/perlmutter_phase4_current_cuda_mpi/`, whose manifest hash is:

```text
90982dcee8eb69432aea841a5a787a504b09d40d95bf3be7e6417f913409d15a
```

Historical Aurora artifacts were not copied into this current-source qualification. They remain historical evidence only and cannot promote `vc_sycl`.
