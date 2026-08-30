# Phase 5 explicit send completion

Aurora job `8791807` moved `ClearSend` from stage end to the evolved path
immediately after `RecvU` and before prolongation/physical BC. Receive clearing
remained at stage end. This is an equation-preserving communication-lifetime
discriminator.

The twelve-rank run reproduced 22 PDE-level Level Zero `NotPresent` writes,
ended with PBS status 143, and wrote no positive-time history. Executable
SHA-256 was
`2e6bde2a138f62ee1fe7e4c6744859d7f4f68c07f433027da562d2d4374c5dc0`.

Waiting for sends alone does not repair the regression.
