"""Fit the two SMR witness orders independently, preserving each order's seed chain."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import time

FITS = ((40, -.002), (48, -.0005))


def fit_pair(seed, directory, snapshots):
    directory = Path(directory)
    snapshots = tuple(map(str, snapshots))
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=directory, prefix=".fits-") as scratch:
        scratch = Path(scratch)

        def fit(settings):
            order, target = settings
            out = scratch / f"witness-l{order}.json"
            log = scratch / f"fits-l{order}.log"
            with log.open("w") as stream:
                subprocess.run([sys.executable, str(Path(__file__).with_name("witness.py")),
                                str(seed), str(out), *snapshots, f"--order={order}",
                                f"--target={target}"], stdout=stream, check=True)
            return out, log

        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(fit, FITS))
        # The caller publishes the segment only after both subprocesses succeeded.
        for pair in outputs:
            for path in pair:
                path.replace(directory / path.name)
    return time.monotonic() - start
