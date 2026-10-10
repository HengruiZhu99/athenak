# Unexecuted root v10 release and launch source

Root must review these files and then run preparation separately from any launch. No script in this prefix has been executed by the author. Frozen v10 remains unchanged at source-index a67f587c07e18ee98e6aaaedfba1249fc334f51905a84c64aa3751df150408ff, recipe05ba2b41afdcb39aeecfc7dbf659670b3e0011d7a6635a25cf42106f617e3fb2, verifier ee7de62d0faa1f221a636d03d1ba43901cb4b28c0bfbc63ce946fe9c5fc10b11.

The NEW cap is exactly900 seconds per process group, explicitly supplied as --cap-seconds900 at preparation and bound in authorization/release. This is not a preserved historical cap: v9 launchers and recipes declared no timeout. A cap result stays failed with exact logs and available child outputs; there is no retry or cap increase.

Preparation uses only stdlib source/AST/JSON/hash checks. Root supplies exact completed independent review receipt and index paths/hashes after that review freezes. The receipt must declare passed=true, inputs_unchanged=true, source_review_only=true and reviewed_source_index_sha256 equal to the frozen v10 index. Preparation independently reverses the complete unified diff to v9 bytes/AST, checks five labels/one unshadowed dictionary, exact helper bytes, inherited cases/gates/runtime and completed E-only diagnosis plus preserved radial FAIL. All source/runtime/review pins are checked before and after. The actual SciPy1332-file manifest and files are already direct recipe pins, retaining bytecode exclusion and metadata-only binaries.

Each case has its own fresh root primary-invocation001/radial_pair-invocation001/angular_pair-invocation001 and frozen-suite attempts/independent-CASE001. All three may run independently in parallel after one completed unchanged-input release; they share read-only inputs only. No mutable case output is a prerequisite of another. No old pass substitutes for a fresh case.

Root launch runs the fresh stdlib child_review_gate with -I -B. That gate checks exact successful source review and all protected source/runtime inputs before exec of the unchanged verifier with -B -s. This supplies an explicit child review gate without modifying the frozen v10 verifier. Exact original NumPy/SciPy/backend paths, one-thread environment and optimize0 are retained; no numerical import occurs in root scripts. The gate and verifier stay in the same newly created process group and share the900-second cap. Root retains stdout/stderr byte-for-byte, true command/environment, source/runtime/review pins before/after and all five audit hashes.

Example preparation (root replaces review placeholders):

    /Library/Developer/CommandLineTools/usr/bin/python3 -I -B prepare_release001.py --independent-review-receipt REVIEW_RECEIPT --independent-review-receipt-sha256 RECEIPT_HASH --independent-review-index REVIEW_INDEX --independent-review-index-sha256 INDEX_HASH --cap-seconds 900

Example independent launches, only after actual preparation PASS:

    /Library/Developer/CommandLineTools/usr/bin/python3 -I -B launch.py primary --release-sha256 ROOT_RELEASE_HASH
    /Library/Developer/CommandLineTools/usr/bin/python3 -I -B launch.py radial_pair --release-sha256 ROOT_RELEASE_HASH
    /Library/Developer/CommandLineTools/usr/bin/python3 -I -B launch.py angular_pair --release-sha256 ROOT_RELEASE_HASH

This admits only the fixed saved-data reconstructions if root releases them. No source query, compile, new assembly, generator spectrum, propagation, native evolution or continuum/BH stability acceptance is granted. All three actual successful readbacks plus independent results/provenance review remain prerequisites of any later generator proposal. Original v8/v9 failures remain unchanged.
