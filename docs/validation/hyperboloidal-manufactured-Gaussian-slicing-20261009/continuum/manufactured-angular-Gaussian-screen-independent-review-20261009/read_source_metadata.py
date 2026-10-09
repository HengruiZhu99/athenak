"""Saved source/metadata readback only. Never import the reviewed screen."""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "reviewer-inputs"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    recipe = json.loads((CAPTURE / "recipe.json").read_text())
    index = json.loads((CAPTURE / "source-index.json").read_text())
    capture_pins = json.loads((HERE / "reviewer-input-pins.json").read_text())
    checked = {}
    for item in capture_pins["files"]:
        assert digest(item["origin"]) == item["sha256"]
        assert digest(HERE / item["copy"]) == item["sha256"]
        assert (HERE / item["copy"]).stat().st_size == item["bytes"]
        checked[item["origin"]] = item["sha256"]
    for item in index["files"]:
        assert digest(item["path"]) == item["sha256"]
        assert Path(item["path"]).stat().st_size == item["bytes"]
        checked[item["path"]] = item["sha256"]
    for path, expected in recipe["pins"].items():
        assert digest(path) == expected, path
        if path in checked:
            assert checked[path] == expected
        checked[path] = expected

    syntax = []
    for name in ("screen.py", "prepare_metadata001.py"):
        ast.parse((CAPTURE / name).read_text(), filename=name)
        syntax.append(name)
    rows = []
    for radius in recipe["radius_over_sigma"]:
        r = Fraction(radius)
        ordinary = {Fraction(t) for t in recipe["time_over_sigma"]}
        retarded = {r + Fraction(u) for u in recipe["retarded_time_over_sigma"]
                    if r + Fraction(u) >= 0}
        times = ordinary | retarded
        rows.append({"radius_over_sigma": radius, "time_count": len(times),
                     "ordinary_times": len(ordinary),
                     "added_retarded_times": len(times - ordinary)})
    radial_time_events = sum(row["time_count"] for row in rows)
    records = radial_time_events * len(recipe["sigma"]) * len(recipe["epsilon"])
    assert records == recipe["anticipated_records_per_precision"] == 28032
    assert recipe["precision_and_terms"] == [[80, 40], [110, 60]]
    assert recipe["precision_tolerance"] == "1e-60"
    assert recipe["identity_tolerance"] == "1e-65"
    assert recipe["sigma"] == ["7/20", "1/2"]
    assert recipe["epsilon"] == ["0", "1/4", "1/2", "3/4"]
    assert recipe["a"] == "1/2"
    result = {
        "passed": True,
        "scope": "Standard-library source/hash/AST and exact rational grid-count readback only.",
        "reviewed_source_imported": False,
        "numerical_or_CAS_imports": False,
        "scientific_screen_executed": False,
        "original_source_and_recipe_unchanged": True,
        "source_index_file_count": len(index["files"]),
        "runtime_and_context_pin_count": len(recipe["pins"]),
        "unique_rehashed_inputs_including_index": len(checked),
        "AST_only_files": syntax,
        "radial_time_events_per_profile": radial_time_events,
        "records_per_precision": records,
        "total_fixed_records": records * len(recipe["precision_and_terms"]),
        "radial_time_counts": rows,
        "checked_pins": checked,
    }
    (HERE / "metadata-readback.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("checked_pins", "radial_time_counts")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
