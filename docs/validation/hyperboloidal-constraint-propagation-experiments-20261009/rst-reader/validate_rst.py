"""Validate exact RST/BIN pairing and quantization without changing run files."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import numpy as np
from restart_reader import read_rst, align_with_bin, VARIABLES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("bin_reader", ROOT/"vis/python/bin_convert.py")
bin_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bin_reader)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(rst_path, bin_path):
    before = {str(p): {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in (rst_path, bin_path)}
    restart = read_rst(rst_path)
    binary = bin_reader.read_binary(str(bin_path))
    if restart["time"] != binary["time"] or restart["cycle"] != binary["cycle"]:
        raise ValueError("RST/BIN exact time or cycle mismatch")
    view = align_with_bin(restart, binary)
    mask = np.asarray(binary["mb_data"]["z4c_active"], dtype=np.float32)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("invalid native spherical mask")
    selected = mask.astype(bool)
    if view[:, 0].shape != selected.shape or not selected.any():
        raise ValueError("empty or mismatched native mask")
    region, block = restart["mesh_size"], restart["mb_indcs"]
    coordinates = []
    for axis in range(3):
        start = int(binary["mb_index"][0][2*axis])
        step = region[f"dx{axis+1}"]
        coordinates.append(region[f"x{axis+1}min"]+
                           (start+.5+np.arange(selected.shape[3-axis]))*step)
    z, y, x = np.meshgrid(*coordinates[::-1], indexing="ij")
    if not np.array_equal(selected[0], x*x+y*y+z*z < 1):
        raise ValueError("BIN mask is not the current native unit sphere")
    fields = []
    for i, name in enumerate(VARIABLES):
        double = view[:, i][selected]
        if not np.isfinite(double).all():
            raise ValueError("nonfinite active restart field: "+name)
        rounded = double.astype(np.float32)
        recorded = np.asarray(binary["mb_data"][name], dtype=np.float32)[selected]
        # uint32 equality includes signed zero and is a genuine bitwise check.
        different = rounded.view(np.uint32) != recorded.view(np.uint32)
        count = int(different.sum())
        if count:
            raise ValueError(f"RST rounded to BIN differs: {name}, count={count}")
        fields.append({"name": name, "active_values": len(double), "bitwise_mismatches": count,
                       "double_min": float(double.min()), "double_max": float(double.max())})
    if any(sha(p) != record["sha256"] or p.stat().st_size != record["bytes"]
           for p, record in [(rst_path, before[str(rst_path)]), (bin_path, before[str(bin_path)])]):
        raise ValueError("run file changed during validation")
    # Independently recover exact scalar bytes at the offsets exposed by parser.
    with rst_path.open("rb") as stream:
        scalar_bits = {}
        for name, fmt, key in [("time", "d", "time"), ("dt", "d", "dt"),
                               ("ncycle", "i", "cycle")]:
            stream.seek(restart["offsets"][name])
            raw = stream.read(struct.calcsize("<"+fmt))
            if struct.unpack("<"+fmt, raw)[0] != restart[key]:
                raise ValueError("exact RST scalar bytes mismatch")
            scalar_bits[key] = raw.hex()
    if restart["u"].flags.writeable or restart["u"].shape != tuple(restart["shape"])[1:]:
        raise ValueError("root API shape/read-only contract failed")
    return {"rst": str(rst_path.relative_to(ROOT)), "bin": str(bin_path.relative_to(ROOT)),
            "files": before, "time": restart["time"], "dt": restart["dt"],
            "cycle": restart["cycle"], "exact_scalar_little_endian_hex": scalar_bits,
            "offsets": restart["offsets"], "shape": restart["shape"],
            "active_box_shape": restart["active_data"].shape,
            "active_spherical_cells": int(selected.sum()), "variables": fields,
            "all_25_float32_quantizations_bitwise_equal": True,
            "header_time_cycle_exactly_equal_to_bin": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=HERE/"checked-receipt.json")
    args = parser.parse_args()
    abi_path = HERE/"abi.json"
    critical = [HERE/"restart_reader.py", HERE/"validate_rst.py", HERE/"layout_probe.cpp",
                abi_path, ROOT/"vis/python/bin_convert.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in critical}
    prior = [ROOT/"build-layer-research/detached-wormhole"/name/"frozen-index.json"
             for name in ["gauge-gate", "spatialnorm-gate", "general-spatialnorm-gate"]]
    prior += [ROOT/"build-layer-research/inner-trumpet-gate/frozen-index.json"]

    def unchanged():
        return all(all(sha(ROOT/relative) == v["sha256"]
                       for relative, v in json.loads(index.read_text())["files"].items())
                   for index in prior)

    assert unchanged()
    rows, cases = [], []
    started = time.monotonic()
    # Inventory is frozen for this invocation; concurrent runs may add later pairs.
    for directory in args.directories:
        directory = directory.resolve()
        paths = sorted((directory/"rst").glob("*.rst"))
        if not paths:
            raise ValueError("no RST snapshots: "+str(directory))
        for path in paths:
            match = re.fullmatch(r"(.+)\.(\d{5})\.rst", path.name)
            if match is None:
                raise ValueError("unexpected RST filename")
            basename, index = match.groups()
            bin_path = directory/"bin"/f"{basename}.z4c.{index}.bin"
            if not bin_path.is_file():
                raise ValueError("same-index BIN is absent: "+str(path))
            rows.append(validate(path, bin_path))
        cases.append({"directory": str(directory.relative_to(ROOT)), "snapshots": len(paths),
                      "time_first": rows[-len(paths)]["time"], "time_last": rows[-1]["time"],
                      "cycle_last": rows[-1]["cycle"], "dt_last_header": rows[-1]["dt"]})
    assert unchanged()
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in critical}
    result = {"scope": "Read-only binary64 native RST parser, one uniform vacuum Z4c block; no evolution",
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "seconds": time.monotonic()-started, "snapshots": len(rows), "cases": cases,
              "source_sha256": hashes, "production_abi_source_sha256": json.loads(abi_path.read_text())["source_sha256"],
              "all_25_float32_quantizations_bitwise_equal": True,
              "sources_and_run_files_unchanged": True, "prior_69_34_27_40_files_unchanged": True,
              "binary_headers_have_no_independent_dt": True,
              "dt_semantics": "Exact pm->dt at output. Cadence writes before NewTimeStep; finalization writes after NewTimeStep, so final header dt need not equal the clipped last completed step.",
              "rows": rows}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("seconds", "snapshots", "cases",
                     "all_25_float32_quantizations_bitwise_equal")}, indent=2))


if __name__ == "__main__":
    main()
