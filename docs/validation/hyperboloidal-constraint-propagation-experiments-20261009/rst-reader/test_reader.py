"""Independent malformed-file rejection and full-precision reference drift audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import time

import numpy as np
from restart_reader import read_rst, VARIABLES
from validate_rst import validate, bin_reader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT/"build-layer-research/continuum/preferred/native-overlay/spatial-norm-family"
CONTROL = ROOT/"build-layer-research/time-projection-controls"
PATH = BASE/"native-long/finite-angular-long-N24/rst/hyp.00000.rst"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


started = time.monotonic()
original_hash = sha(PATH)
first = read_rst(PATH)
assert first["u"].dtype == np.dtype("<f8") and not first["u"].flags.writeable
try:
    first["u"][0, 0, 0, 0] = 1
    raise AssertionError("write to read-only RST unexpectedly allowed")
except ValueError:
    pass
assert sha(PATH) == original_hash
rejections = []
with tempfile.TemporaryDirectory(dir=HERE) as temporary:
    directory = Path(temporary)
    original = PATH.read_bytes()
    cases = [("truncated_payload", None, None), ("missing_terminator", None, None),
        ("multi_block", "nmb_total", ("i", 2)), ("refined_root", "root_level", ("i", 1)),
        ("nan_time", "time", ("d", float("nan"))), ("zero_dt", "dt", ("d", 0.)),
        ("negative_cycle", "ncycle", ("i", -1)), ("wrong_payload_size", "data_size", ("Q", first["data_size"]+8)),
        ("wrong_active_dimension", "mb_indcs", ("i", 22)), ("wrong_coarse_dimension", "mb_indcs", ("i", 13))]
    for name, key, value in cases:
        data = bytearray(original)
        if name == "truncated_payload":
            data = data[:-1]
        elif name == "missing_terminator":
            offset = data.index(b"<par_end>\n")
            data[offset] = ord("!")
        else:
            offset = first["offsets"][key]
            if name == "wrong_active_dimension":
                offset += 4
            if name == "wrong_coarse_dimension":
                offset += 40
            struct.pack_into("<"+value[0], data, offset, value[1])
        path = directory/(name+".rst")
        path.write_bytes(data)
        try:
            read_rst(path)
            raise AssertionError("corrupted RST unexpectedly accepted: "+name)
        except ValueError as error:
            rejections.append({"case": name, "rejected": True, "message": str(error)})
    # Quantization is independently sensitive to an altered evolved field.
    altered = bytearray(original)
    shape = first["shape"]
    cell = (0, 0, first["mb_indcs"]["ks"]+12, first["mb_indcs"]["js"]+12,
            first["mb_indcs"]["is"]+12)
    offset = first["offsets"]["data"]+8*np.ravel_multi_index(cell, shape)
    old = struct.unpack_from("<d", altered, offset)[0]
    struct.pack_into("<d", altered, offset, old+.001)
    path = directory/"altered.rst"
    path.write_bytes(altered)
    binary = PATH.parent.parent/"bin/hyp.z4c.00000.bin"
    try:
        validate(path, binary)
        raise AssertionError("altered active payload unexpectedly matched BIN")
    except ValueError as error:
        assert "rounded to BIN differs" in str(error)
        rejections.append({"case": "changed_active_payload", "rejected": True, "message": str(error)})

drifts = []
for name, directory in [("baseline_reference", BASE/"native-reference/reference-long-N24"),
                        ("stage_projection_reference", CONTROL/"stage-projection-reference/reference-long-N24")]:
    paths = sorted((directory/"rst").glob("*.rst"))
    initial = read_rst(paths[0])
    binary = bin_reader.read_binary(str(directory/"bin/hyp.z4c.00000.bin"))
    mask = np.asarray(binary["mb_data"]["z4c_active"])[0].astype(bool)
    rows = []
    maximum = 0.
    for path in paths:
        current = read_rst(path)
        # Both are full ghost-layout arrays; mask was independently aligned by
        # validate_rst.py before this supplemental drift audit.
        fields = {}
        for i, variable in enumerate(VARIABLES):
            delta = current["u"][i][mask]-initial["u"][i][mask]
            value = float(np.max(np.abs(delta)))
            fields[variable] = {"max_abs_drift": value, "rms_drift": float(np.sqrt(np.mean(delta*delta)))}
            maximum = max(maximum, value)
        rows.append({"time": current["time"], "cycle": current["cycle"], "fields": fields})
    assert maximum < 5e-12
    drifts.append({"name": name, "snapshots": len(paths), "max_binary64_active_drift": maximum,
                   "fields": rows})
assert sha(PATH) == original_hash
result = {"passed": True, "seconds": time.monotonic()-started,
          "negative_controls": rejections, "read_only_write_rejected": True,
          "original_rst_unchanged": True, "reference_drifts": drifts,
          "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
              (HERE/"restart_reader.py", HERE/"validate_rst.py", HERE/"test_reader.py", HERE/"abi.json")}}
(HERE/"supplemental-receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({"seconds": result["seconds"], "negative_controls": len(rejections),
    "reference_binary64_drifts": {v["name"]: v["max_binary64_active_drift"] for v in drifts}}, indent=2))
