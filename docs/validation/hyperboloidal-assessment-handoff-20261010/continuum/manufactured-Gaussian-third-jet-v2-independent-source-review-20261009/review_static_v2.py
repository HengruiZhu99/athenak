"""Independent stdlib byte/AST/admission diff readback. No candidate import."""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE / "captured"
OLD = HERE.parent / "manufactured-Gaussian-third-jet-independent-source-review-20261009" / "captured"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def reverse_patch(new, lines):
    source = new.splitlines(keepends=True)
    out = []
    cursor = 0
    pos = 0
    while pos < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[pos])
        if not match:
            raise RuntimeError("unexpected diff line")
        old_start, old_count, new_start, new_count = match.groups()
        old_count = int(old_count or 1)
        new_count = int(new_count or 1)
        start = int(new_start)-1
        if start < cursor:
            raise RuntimeError("overlapping hunks")
        out.extend(source[cursor:start])
        cursor = start
        pos += 1
        seen_old = seen_new = 0
        while pos < len(lines) and not lines[pos].startswith("@@ "):
            line = lines[pos]
            marker, content = line[:1], line[1:]
            if marker in (" ", "+"):
                if cursor >= len(source) or source[cursor] != content:
                    raise RuntimeError("new context mismatch")
                cursor += 1
                seen_new += 1
            if marker in (" ", "-"):
                out.append(content)
                seen_old += 1
            if marker not in (" ", "+", "-"):
                raise RuntimeError("unexpected hunk marker")
            pos += 1
        if seen_old != old_count or seen_new != new_count:
            raise RuntimeError("hunk count mismatch")
    out.extend(source[cursor:])
    return "".join(out)

def main():
    output = HERE / "static-readback-v2.json"
    if output.exists():
        raise RuntimeError("fresh one-shot output required")
    inventory = load(HERE / "inputs-before-review.json")
    for row in inventory["inputs"]:
        for kind in ("original", "captured"):
            if sha(row[kind]) != row["sha256"] or Path(row[kind]).stat().st_size != row["bytes"]:
                raise RuntimeError("pre-read input drift")
    index, recipe = load(NEW/"source-index.json"), load(NEW/"recipe.json")
    if sha(NEW/"source-index.json") != "92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3":
        raise RuntimeError("wrong v2 index")
    pins = {**index["files"], **recipe["protected_inputs"]}
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError("protected drift: "+path)
    lines = (NEW/"v1-to-v2.diff").read_text().splitlines(keepends=True)
    patches = {}
    pos = 0
    while pos < len(lines):
        if not lines[pos].startswith("--- v1/") or not lines[pos+1].startswith("+++ v2/"):
            raise RuntimeError("unexpected diff header")
        name = lines[pos][len("--- v1/"):].strip()
        if lines[pos+1][len("+++ v2/"):].strip() != name:
            raise RuntimeError("renamed diff file")
        pos += 2
        start = pos
        while pos < len(lines) and not lines[pos].startswith("--- v1/"):
            pos += 1
        patches[name] = lines[start:pos]
    reconstructed = []
    for name, patch in patches.items():
        restored = reverse_patch((NEW/name).read_text(), patch)
        if restored.encode() != (OLD/name).read_bytes():
            raise RuntimeError("reverse byte mismatch: "+name)
        if name.endswith(".py") and ast.dump(ast.parse(restored),include_attributes=False) != ast.dump(ast.parse((OLD/name).read_text()),include_attributes=False):
            raise RuntimeError("reverse AST mismatch")
        reconstructed.append(name)
    unchanged = ["taylor3.py", "gaussian3.py", "reference3.py", "geometry.py", "oracle.py", "units.py", "values_context.py"]
    for name in unchanged:
        if (NEW/name).read_bytes() != (OLD/name).read_bytes():
            raise RuntimeError("mathematical module changed: "+name)
    old_recipe = load(OLD/"recipe.json")
    for key, value in old_recipe.items():
        if key not in ("status", "protected_inputs") and recipe[key] != value:
            raise RuntimeError("old science recipe field changed: "+key)
    for key, value in old_recipe["protected_inputs"].items():
        if recipe["protected_inputs"].get(key) != value:
            raise RuntimeError("old protected pin lost")
    parsed = []
    for path in sorted(NEW.rglob("*.py")):
        ast.parse(path.read_text(),filename=str(path))
        parsed.append(str(path.relative_to(NEW)))
    result = {"passed": True, "source_only": True, "candidate_imported_or_executed": False,
              "target_arithmetic_CAS_compile_queries_eigen_evolution": False,
              "reviewed_source_index_sha256": sha(NEW/"source-index.json"),
              "reverse_byte_and_AST_files": reconstructed,
              "seven_unchanged_mathematical_modules": unchanged,
              "all_old_science_recipe_fields_unchanged": True,
              "all_original_and_captured_inputs_unchanged": True,
              "source_files": len(index["files"]), "captured_including_index": len(inventory["inputs"]),
              "protected_external_pins": len(recipe["protected_inputs"]),
              "unique_protected_source_external_pins": len(pins), "AST_parsed": parsed,
              "argv": sys.argv, "reviewer_source_sha256": sha(__file__)}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"source_only_passed": True, "source_files": len(index["files"]),
                      "external_pins": len(recipe["protected_inputs"]), "result_sha256": sha(output)},sort_keys=True))

if __name__ == "__main__":
    main()
