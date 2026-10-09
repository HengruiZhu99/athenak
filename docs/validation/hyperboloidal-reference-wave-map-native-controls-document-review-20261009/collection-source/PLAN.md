# HELD compact saved-evidence collector

This is source-only preparation. The fixed scope-config.json names completed roots plus selected files; no shared live parent is traversed. recipe.json and source-index.json pin exact inventories/byte identities and explicit completion gates. All original outputs and old archives remain unchanged. Metadata preparation parses only saved JSON identity/admission fields and hashes bytes; it performs no scientific imports, queries or computations.

Root must supply one_shot_compact_collection_authorized=true binding exact recipe/source-index hashes and destination. Run Python -B collect_once.py --authorization ROOT_AUTH_PATH --authorization-sha256 ROOT_AUTH_SHA with PYTHONDONTWRITEBYTECODE=1 and outer stdout/stderr/returncode retained. Collection is not released by source preparation. A failed precheck or changed pin is preserved in invocation001; never retry into existing destinations. Final current documents and overview are external metadata only. Arrays, binaries, objects, JSONL and everyfile>1MiB remain metadata-only; all copied logs preserve whitespace.

Scope: Completed N24 document preparation/review side evidence only; final current prose and overview are external metadata, not duplicate frozen documents.
