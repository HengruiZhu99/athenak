"""Read-only catalog verification; no scientific reruns or scratch dependencies."""

import hashlib
import json
import math
from pathlib import Path
import sys


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


root = Path(sys.argv[1]).resolve()
catalog = json.loads((root / "catalog.json").read_text())
finite(catalog)
total = 0
json_count = 0
opaque_failure_count = 0
for name, record in catalog["files"].items():
    path = (root / name).resolve()
    assert path.is_relative_to(root)
    data = path.read_bytes()
    assert len(data) == record["bytes"], name
    assert hashlib.sha256(data).hexdigest() == record["sha256"], name
    if record.get("format") == "opaque_historical_invalid_json":
        assert name == "local-core/history/failed-fd-JSON-reader/full20.json"
        assert record["sha256"] == "d99e084eaee1ba5bf69382ec693d4137dfce2849b96b6e5cea55414685748326"
        try:
            json.loads(data)
        except json.JSONDecodeError as error:
            assert error.pos == 78321
        else:
            raise AssertionError("Historical failure unexpectedly parsed")
        opaque_failure_count += 1
    elif path.suffix == ".json":
        finite(json.loads(data))
        json_count += 1
    total += len(data)
actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
assert actual == set(catalog["files"]) | {"catalog.json"}
print(json.dumps({"status": "PASS", "files": len(catalog["files"]),
                  "bytes": total, "finite_json_files": json_count, "opaque_historical_failures": opaque_failure_count,
                  "catalog_sha256": hashlib.sha256((root / "catalog.json").read_bytes()).hexdigest()}))
