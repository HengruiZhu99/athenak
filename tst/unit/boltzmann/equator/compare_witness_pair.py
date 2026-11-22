"""Time serial and parallel SMR fits on identical slices and compare every output byte."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from witness_pair import FITS, fit_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--parallel-first", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        serial, parallel = root / "serial", root / "parallel"
        serial.mkdir()
        parallel.mkdir()
        for mode in ((parallel, serial) if args.parallel_first else (serial, parallel)):
            if mode == parallel:
                parallel_s = fit_pair(args.seed, parallel, args.snapshots)
            else:
                start = time.monotonic()
                for degree, target in FITS:
                    with (serial / f"fits-l{degree}.log").open("w") as log:
                        subprocess.run([sys.executable, str(Path(__file__).with_name("witness.py")),
                                        str(args.seed), str(serial / f"witness-l{degree}.json"),
                                        *map(str, args.snapshots), f"--order={degree}",
                                        f"--target={target}"], stdout=log, check=True)
                serial_s = time.monotonic() - start
        for path in serial.iterdir():
            if path.read_bytes() != (parallel / path.name).read_bytes():
                raise ValueError(f"Serial and parallel outputs differ in {path.name}")
        first = {}
        for degree, _ in FITS:
            rows = json.loads((serial / f"witness-l{degree}.json").read_text())
            if len(rows) != len(args.snapshots):
                raise ValueError("Incomplete fit output")
            first[degree] = min((r["cycle"] for r in rows if r["all_negative"]), default=None)
        print(json.dumps(dict(slices=len(args.snapshots), serial_s=serial_s, parallel_s=parallel_s,
                              first_verified_cycle=first)))


if __name__ == "__main__":
    main()
