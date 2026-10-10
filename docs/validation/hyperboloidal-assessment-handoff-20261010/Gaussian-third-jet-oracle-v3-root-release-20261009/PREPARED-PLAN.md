# Unexecuted root Gaussian v3 review and units scripts

All five scripts are SOURCE-ONLY. Neither root metadata review/preparation nor any unit launcher has been executed. The owner remains the immutable held v3 index `41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a`, recipe `847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf`, driver `de0b83ba44f1a837c2b477d6d60f5a8f2a947b8a885159a638bbbaeee94e7915`, and unchanged outer `68706de821c990be9302848aeea41d16d774c506bb8dea71221b1e2a675cea04`.

`prepare_review001.py` requires the exact future CLI hash of this frozen `root-scripts-source-index001.json`, then verifies all 437 distinct owner source/runtime/history pins (58 indexed files, 378 external pins, and the owner index). It independently checks nine numerical/outer modules byte-for-byte and by AST against v2, reverses the sole driver precision-label change exactly, and restores the recipe to exact v2 bytes after removing only approved precision/root/history changes. It checks the unchanged unit recipe, registry/counts, thresholds and child/outer caps. It writes metadata source pins and a preparation record, with mathematical review still pending. No candidate is imported.

`finalize_review001.py --root-source-math-reviewed` seals root's explicit manual source/math/admission decision only after the above static checks pass and all 437 pins still match. The CLI flag represents root's human/source review, not a mathematical conclusion inferred by the script. Both metadata phases write fresh invocation records and preserve failures.

`prepare_units001.py` requires `--review`, `--review-index-sha256`, and `--review-receipt-sha256`. These are intentionally supplied only after the independent v3 review exists. Its receipt must contain `passed: true` and `reviewed_source_index_sha256` equal to the exact v3 index. Its indexed files are resolved relative to its own review directory (absolute indexed paths are also admitted only within that directory); they are all hash-verified and protected. The successful root review and static-phase receipts are required. The only generated authorization is for exactly 318 fresh v3 units. Historical v2 units PASS and timing FAIL are provenance; they do not admit v3 timing or full execution.

`launch_units.py --release-sha256 <actual generated release hash>` requires the exact preparation-generated release and authorization binding, all 437 original pins plus root/independent review and launcher pins, and fresh child/outer/root invocation paths. It invokes the unchanged candidate outer with exact pinned Python, `-I -B`, `PYTHONOPTIMIZE=0`, bytecode off, BLAS/OMP/VECLIB one, and removes injected Python path/home/warning/startup/userbase variables. The outer is started in a new process group. A root 60-second hard process-group cap covers the outer and its analytic child, including cleanup on exceptions. The unchanged owner outer/child nominal limits remain 120/60 seconds; root imposes the stronger enclosing cap because the unit function's soft check is not reached internally.

Acceptance requires actual return code zero, exact same-source completed/passed/unchanged unit receipt, result PASS with exactly 318 checks and zero failures, successful unchanged outer receipt, verified child output hashes, and unchanged complete root pins afterward. All command, fixed environment, process IDs, true return code, stdout/stderr, failures and output metadata are retained. Missing/partial/drifted results cannot pass. The launcher creates no timing/full release, native query, inverse coverage, global slicing or BH adoption.

## Fixed fresh paths

- Root metadata: `review-preparation-invocation001`, `finalize-review-invocation001`, `units-preparation-invocation001`.
- Source records: `source-pins001.json`, `review-preparation001.json`, `source-review001.json`.
- Authorization/release: `units-authorization.json`, `units-release.json`.
- Actual units child: `continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009/attempts/units001`.
- Actual candidate outer/root invocation: this prefix's `units-outer001` and `units-invocation001`.

Every destination is single-use. No original v2 file, failed attempt, existing root script or frozen owner is changed. Every NPZ/NPY/JSONL and every output over 1 MiB is metadata-only in the root output inventory.

## Proposed commands, not run

Use the pinned Python runtime from the owner recipe, with `PYTHONOPTIMIZE=0` and `-I -B` for every root script. After root reads all scripts and the exact diff:

1. `prepare_review001.py --scripts-index-sha256 <frozen root scripts index SHA>`.
2. `finalize_review001.py --root-source-math-reviewed`.
3. After independent review PASS, `prepare_units001.py --review <independent review directory> --review-index-sha256 <exact SHA> --review-receipt-sha256 <exact SHA>`.
4. `launch_units.py --release-sha256 <actual units-release.json SHA printed by preparation>`.

The first three are metadata only. The fourth is a separately authorized scientific unit execution and is currently held. This prepared source does not itself supply a release or imply future scientific acceptance.
