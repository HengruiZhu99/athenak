# Held root preparation and bounded launch

Neither new script has been executed. Root must read both scripts and their
launcher-source-index before running preparation, then inspect its authorization
and pinned manifest before launch. Existing root review files remain unchanged.

`prepare_release001.py` is stdlib metadata only. It requires the exact owner index
c8d1b760, recipe66b29ff7, independent review307340b0/3e0f8c6a, and all3,283 identical
root/independent source pins. It adds the complete independent review, root source
files, and a metadata inventory of installed SciPy, its distribution metadata and
any sibling bundled shared-library directories. Bytecode is excluded; scientific
payloads and binaries are hashed only, never decoded/copied. It requires both
fixed owner destinations to be absent and writes each root output only once.

`launch.py` uses a fresh root outer-invocation001, the recipe's exact Python
`-B -s` command, PYTHONPATH including the existing venv, one thread, optimize0,
and unchanged owner diagnostic001/outer-invocation001. It starts a new process
session and imposes a180-second process-group cap from child launch, excluding
pre/post hashing. On timeout/error it terminates the entire group, retaining
exact stdout/stderr, return codes and any partial outputs. A2-second SIGTERM
grace precedes SIGKILL cleanup. No existing output is reused or rewritten.

All source/runtime/review/script pins and SciPy inventory membership are checked
before and after. Both owner wrapper and diagnostic receipts plus the compact
32-row/131,072-component result must pass for root completion. This classifies
only the observed E measure product; it does not accumulate E or load operator
matrices, perform SVD, query a source or qualify the failed radial readback.

Future root commands, after source review:

```text
/Library/Developer/CommandLineTools/usr/bin/python3 -I -B prepare_release001.py
/Library/Developer/CommandLineTools/usr/bin/python3 -I -B launch.py
```

The separate loads product is the same positive scalar measure times a completed
finite binary64 array, with shape64x33. Its product-only fallback contract is the
same as E/Ks/Kw/G; no occurrence is asserted and no v10 source is prepared here.
