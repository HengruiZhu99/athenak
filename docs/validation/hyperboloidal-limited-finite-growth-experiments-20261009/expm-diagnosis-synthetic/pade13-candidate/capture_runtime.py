"""Record library bytes without evaluating a matrix or suppressing warnings."""
import hashlib
import json
from pathlib import Path
import sys
import mpmath
import numpy
import numpy.linalg._umath_linalg as lapack
import numpy._core._multiarray_umath as core


def rec(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


out = {"python": sys.version, "executable": sys.executable,
       "numpy_version": numpy.__version__, "mpmath_version": mpmath.__version__,
       "files": [rec(p) for p in (sys.executable, numpy.__file__,
                 lapack.__file__, core.__file__, mpmath.__file__, __file__)]}
print(json.dumps(out, indent=2))
