# Link-only per-kernel split: negative gate

Aurora job `8791910` ran on `x4307c5s7b0n0` from exact commit
`de4b88f3e01a47a7ace95c5e705d892dff3fbce1`, with the frozen input and pinned
Kokkos/toolchain.

The compile database contained zero uses of
`-fsycl-device-code-split=per_kernel`. The verbose final executable link line
contained exactly the requested option. The resulting executable SHA-256 was
`d88f62ed9fc9e6199771556eadbdbf25fc29e42cdff17f7b82ea86c676bba16a`.

The twelve-rank cycle classified `FAIL_LEVEL_ZERO`, status 143, latest history
time zero, with 22 preserved `NotPresent` messages. Therefore final-link
splitting alone is insufficient. Together with job `8791895`, this narrows the
next candidate to target-scoped compile plus link options; it does not justify
a scientific stability or portability-restored claim.
