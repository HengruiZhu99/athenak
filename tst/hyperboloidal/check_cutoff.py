"""High-precision independent oracle for compiled cutoff endpoint derivatives."""
import json
from pathlib import Path
import subprocess
import sys

import mpmath as mp

mp.mp.dps = 100
samples = json.loads(subprocess.check_output(
    [str(Path(sys.argv[1]).resolve()), '--cutoff'], text=True))
maximum = 0
for row in samples:
    r = mp.mpf(row[0])  # preserve the exact binary input
    if r <= 0 or r >= 1:
        expected = [int(r >= 1), 0, 0, 0]
    else:
        def weight(x):
            return 1 / (1 + mp.exp(1 / x - 1 / (1 - x)))
        # Evaluate derivatives of the small complementary expression near 1.
        function = weight if r <= mp.mpf('0.5') else lambda x: weight(1 - x)
        values = [mp.diff(function, r, k) for k in range(4)]
        expected = values if r <= mp.mpf(
            '0.5') else [1 - values[0]] + [-v for v in values[1:]]
    for k, (value, oracle) in enumerate(zip(row[1:], expected)):
        oracle = float(oracle)
        error = abs(value - oracle)
        assert error <= 1e-14 + 2e-10 * abs(oracle), (float(r), k, value, oracle)
        if abs(oracle) > 1e-280:
            maximum = max(maximum, error / max(1, abs(oracle)))
print(json.dumps({'cutoff_samples': len(samples), 'max_scaled_error': maximum,
                  'analytic_derivatives_through': 3, 'endpoint_finite': True}, indent=2))
