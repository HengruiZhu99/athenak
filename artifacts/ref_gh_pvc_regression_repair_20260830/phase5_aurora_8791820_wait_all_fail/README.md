# Phase 5 explicit send and receive completion

Aurora job `8791820` moved both `ClearSend` and `ClearRecv` from stage end to
the evolved path immediately after `RecvU` and before prolongation/physical
BC. This is an equation-preserving communication-lifetime discriminator.

The twelve-rank run reproduced four PDE-level Level Zero `NotPresent` writes,
ended with PBS status 143, and wrote no positive-time history. Executable
SHA-256 was
`b6f79f9448b847bf9778393c33b4e0c5bf9f18b9ddb13a67e7897a912c1786ae`.

Explicit completion of both directions does not repair the regression.
Together with the one-rank failure, this removes outstanding GPU-aware MPI
completion from the leading cause list.
