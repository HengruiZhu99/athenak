"""Standard-library-only guard/source metadata review; never import screen."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "reviewer-inputs"
OLD = HERE.parent / "manufactured-angular-Gaussian-screen-independent-review-20261009"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    recipe = json.loads((CAPTURE / "recipe.json").read_text())
    old_recipe = json.loads((OLD / "reviewer-inputs/recipe.json").read_text())
    index = json.loads((CAPTURE / "source-index.json").read_text())
    checked = {}
    for row in json.loads((HERE / "reviewer-input-pins.json").read_text())["files"]:
        assert digest(row["origin"]) == row["sha256"]
        assert digest(HERE / row["copy"]) == row["sha256"]
        checked[row["origin"]] = row["sha256"]
    for row in index["files"]:
        assert digest(row["path"]) == row["sha256"]
        assert Path(row["path"]).stat().st_size == row["bytes"]
        checked[row["path"]] = row["sha256"]
    for path, expected in recipe["pins"].items():
        assert digest(path) == expected, path
        if path in checked:
            assert checked[path] == expected
        checked[path] = expected
    for key, value in old_recipe.items():
        if key not in ("status", "pins"):
            assert recipe[key] == value, key
    assert all(recipe["pins"][k] == v for k, v in old_recipe["pins"].items())
    assert recipe["python_runtime_sha256"] == recipe["pins"][recipe["python_runtime_path"]]
    assert recipe["mpmath_init"] in recipe["pins"]
    assert recipe["mpmath_parent"] == str(Path(recipe["mpmath_init"]).parent.parent)
    old_source = (OLD / "reviewer-inputs/screen.py").read_text()
    new_source = (CAPTURE / "screen.py").read_text()
    # The complete scientific body from def rat through the run return is byte-exact.
    assert old_source[old_source.index("    def rat(s):"):old_source.index("\ndef main():")] == new_source[new_source.index("    def rat(s):"):new_source.index("\ndef main():")]
    expected = old_source.replace(
        "def run(recipe,out):\n    import mpmath as mp\n",
        "def run(recipe,out):\n    sys.path.insert(0,recipe['mpmath_parent'])\n"
        "    import mpmath as mp\n"
        "    if str(Path(mp.__file__).resolve())!=recipe['mpmath_init']:\n"
        "        raise RuntimeError('unexpected mpmath import origin')\n")
    expected = expected.replace(
        "        if sys.flags.optimize!=0 or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('unoptimized bytecode-off runtime required')",
        "        if sys.flags.optimize!=0 or not sys.flags.isolated or not sys.dont_write_bytecode:\n"
        "            raise RuntimeError('isolated unoptimized bytecode-off runtime required')")
    expected = expected.replace(
        "        pins.update(recipe['pins'])",
        "        if sha(Path(sys.executable).resolve())!=recipe['python_runtime_sha256']:\n"
        "            raise RuntimeError('actual Python runtime differs')\n"
        "        for key,value in recipe['environment'].items():\n"
        "            if os.environ.get(key)!=value:raise RuntimeError('fixed environment differs '+key)\n"
        "        pins.update(recipe['pins'])")
    assert expected == new_source
    old_origin = next(row["origin"] for row in json.loads((OLD / "reviewer-input-pins.json").read_text())["files"] if row["origin"].endswith("/screen.py"))
    new_origin = next(row["origin"] for row in json.loads((HERE / "reviewer-input-pins.json").read_text())["files"] if row["origin"].endswith("/screen.py"))
    diff = "".join(difflib.unified_diff(old_source.splitlines(True), new_source.splitlines(True), fromfile=old_origin, tofile=new_origin))
    assert diff == (CAPTURE / "admission-only.diff").read_text()
    for name in ("screen.py", "prepare_source001.py"):
        ast.parse((CAPTURE / name).read_text(), filename=name)
    result = {"passed": True,
              "scope": "Standard-library metadata/AST/text comparison only; no numerical imports or reviewed-source execution.",
              "complete_scientific_body_byte_equal": True,
              "all_original_recipe_scientific_fields_equal": True,
              "three_expected_guard_only_replacements_equal": True,
              "supplied_diff_exact": True,
              "runtime_and_context_pins": len(recipe["pins"]),
              "unique_rehashed_inputs": len(checked),
              "v1_inputs_unchanged": True,
              "v2_inputs_unchanged": True,
              "reviewed_source_imported": False,
              "scientific_screen_executed": False,
              "checked_pins": checked}
    (HERE / "metadata-readback.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_pins"}, sort_keys=True))


if __name__ == "__main__":
    main()
