#!/usr/bin/env python3
"""Allocation-free composition and Brill-manifest checks for the Iris importer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json_strict(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON token {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=reject_constant,
                       object_pairs_hook=unique_object)
    require(isinstance(value, dict), "manifest fixture must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_dir.resolve()

    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    source_cmake = (root / "src/CMakeLists.txt").read_text(encoding="utf-8")
    config = (root / "config.hpp.in").read_text(encoding="utf-8")
    dispatch = (root / "src/pgen/pgen.cpp").read_text(encoding="utf-8")
    header = (root / "src/pgen/pgen.hpp").read_text(encoding="utf-8")
    defaults = (root / "src/pgen/pgen_defaults.hpp").read_text(
        encoding="utf-8")
    mesh_validation = (root / "src/mesh/meshblock_pack.cpp").read_text(
        encoding="utf-8")
    importer = (root / "src/pgen/z4c_irisk_xcts.cpp").read_text(
        encoding="utf-8")
    coordinate_map = (
        root / "src/pgen/z4c_irisk_coordinate_map.hpp").read_text(
            encoding="utf-8")
    symmetry = (root / "src/z4c/z4c_symmetry.cpp").read_text(
        encoding="utf-8")

    require('option(Athena_ENABLE_IRISK_INTERPOLATOR' in cmake,
            "input-selected importer option disappeared")
    require('if (${PROBLEM} STREQUAL "z4c_irisk_xcts")' in cmake and
            'set(Athena_ENABLE_IRISK_INTERPOLATOR ON CACHE BOOL' in cmake,
            "historical custom PROBLEM compatibility disappeared")
    require('if (Athena_ENABLE_IRISK_INTERPOLATOR)' in cmake and
            'target_link_libraries(athena PUBLIC "${IRISK_INTERPOLATOR_LIBRARY}")'
            in cmake,
            "existing Iris interpolator linkage is not feature-bound")
    require('#define IRISK_INTERPOLATOR_ENABLED '
            '@IRISK_INTERPOLATOR_ENABLED@' in config,
            "configured dispatch feature macro disappeared")
    require('if (Athena_ENABLE_IRISK_INTERPOLATOR)' in source_cmake and
            'target_sources(athena PRIVATE pgen/z4c_irisk_xcts.cpp)'
            in source_cmake,
            "built-in importer source is not feature-bound")

    guarded_dispatch = re.search(
        r"#if IRISK_INTERPOLATOR_ENABLED\s+"
        r"} else if \(pgen_fun_name\.compare\(\"z4c_irisk_xcts\"\) == 0\) \{\s+"
        r"Z4cIrisXcts\(pin, is_restart\);\s+#endif", dispatch)
    require(guarded_dispatch is not None,
            "input-selected Iris dispatch is absent or unguarded")
    require('compiled_problem == "z4c_irisk_xcts" ? "z4c_irisk_xcts" : "none"'
            in defaults,
            "historical custom-build default inventory changed")
    resolver_call = "DefaultInputSelectedPgen(PROBLEM_GENERATOR)"
    require(dispatch.count(resolver_call) == 1,
            "runtime dispatch does not use the shared custom-build default")
    require(mesh_validation.count(resolver_call) == 1,
            "preallocation validation does not use the shared custom-build default")
    pgen_readers = {
        path.relative_to(root).as_posix()
        for path in (root / "src").rglob("*.cpp")
        if '"pgen_name"' in path.read_text(encoding="utf-8")
    }
    require(pgen_readers == {"src/pgen/pgen.cpp",
                             "src/mesh/meshblock_pack.cpp"},
            f"unexpected pgen_name reader inventory: {sorted(pgen_readers)}")
    require("void Z4cIrisXcts(ParameterInput *pin, const bool restart);" in
            header, "named importer entry declaration disappeared")
    require("void ProblemGenerator::Z4cIrisXcts" in importer and
            "ProblemGenerator::UserProblem" not in importer,
            "importer does not have one named shared entry point")
    require(importer.count("IrisAthenakSpectralInterpolateCartesian") == 1,
            "interpolation call was duplicated")
    require(importer.count("IrisAthenakSpectralOpen") == 1,
            "payload open call was duplicated")
    require(importer.count("IrisAthenakSpectralClose") == 1 and
            importer.index("FillAdmFromIrisSpectral(pmbp, interpolator)") <
            importer.index("IrisAthenakSpectralClose(interpolator)"),
            "Iris importer ownership is not one open/fill/close lifetime")
    for token in (
            "CartoonIrisInterpolationCoordinates",
            "IrisTensorProductDimensions<Map>",
            "IrisPointIndex<Map>",
            "ScalarFromPhysicalCartesian<Map>",
            "SymmetricTensorFromPhysicalCartesian<Map>",
            "VectorFromPhysicalCartesian<Map>"):
        require(token in importer,
                f"importer does not apply reviewed Iris map seam {token}")
    require("template <z4c_irisk::AdmMap Map>" in importer and
            "SelectAdmMap(pmbp->z4c_symmetry)" in importer,
            "Iris map is not selected once before compile-time mapped loops")
    require("AdmMap::half_rho_z_suppressed_y_v2" in coordinate_map and
            "signed_rho_z_suppressed_y_v1" not in coordinate_map,
            "Iris map still advertises an independently stored signed plane")
    require("z4c::MakeStoredDomainBounds(indcs)" in importer,
            "Iris import does not use allocated stored-domain bounds")
    require("const int interpolation_is = axis_block ? indcs.is : isg;" in importer,
            "axis blocks still interpolate independently into negative-rho ghosts")
    require("FillAdmAxisGhostLine" in importer and
            "FillZ4cAxisGhostLine" in importer,
            "Iris half-plane ghosts are not parity-derived")
    require(importer.index("FillAdmAxisGhostLine") <
            importer.index("Kokkos::deep_copy(u_adm, host_u_adm)"),
            "Iris ADM axis parity is not established before device import")
    require(importer.index("ReconstructAxisParityGhosts();") <
            importer.index("pmbp->pz4c->Z4cToADM(pmbp);"),
            "Iris derived Z4c parity is not established before constraints")
    for unsafe_bound in ("indcs.ks - indcs.ng", "indcs.ke + indcs.ng",
                         "indcs.js - indcs.ng", "indcs.je + indcs.ng"):
        require(unsafe_bound not in importer,
                f"Iris import fabricates collapsed ghost storage: {unsafe_bound}")
    require("return {code_x1, Scalar{0}, code_x2};" in coordinate_map,
            "Cartoon Iris point is not physical (X,Y,Z)=(x1,0,x2)")
    require("return {physical[0], physical[2], physical[1]};" in
            coordinate_map,
            "Cartoon Iris vector map is not (X,Y,Z)->(X,Z,Y)")
    require("physical[5], physical[4], physical[3]" in coordinate_map,
            "Cartoon Iris symmetric-tensor map changed")
    for token in ("std::filesystem::absolute", "weakly_canonical",
                  "is_regular_file", "irisk_adm_spectral_file"):
        require(token in importer, f"fail-closed payload path check lost {token}")
    require('input.problem_generator == "z4c_irisk_xcts"' in symmetry,
            "Cartoon importer admission disappeared")

    manifest_path = root / (
        "tst/unit/z4c/fixtures/"
        "brill_A-0.047000000_rho0_5.000000000.adm_spectral.manifest.json")
    manifest = load_json_strict(manifest_path)
    require(set(manifest) == {"schema_version", "artifact_format", "family",
                              "branch", "amplitude", "source_faithful",
                              "solver", "residuals", "export", "artifacts"},
            "manifest fixture inventory changed")
    require(manifest["schema_version"] == 1 and
            manifest["artifact_format"] ==
            "IRIS_ATHENAK_SPECTRAL_ADM_PROVENANCE_V1",
            "manifest fixture schema changed")
    require(manifest["family"] == "brill_gaussian" and
            manifest["branch"] == "unique_time_symmetric" and
            manifest["amplitude"] == -0.047 and
            manifest["source_faithful"] is True,
            "manifest fixture is not the Figure-3 Brill family")
    artifacts = manifest["artifacts"]
    require(isinstance(artifacts, dict) and
            set(artifacts) == {"source", "executable", "input",
                               "coefficients", "adm_payload"},
            "manifest artifact inventory changed")
    payload = artifacts["adm_payload"]
    require(isinstance(payload, dict) and
            set(payload) == {"path", "sha256"},
            "ADM payload binding is malformed")
    payload_path = payload["path"]
    payload_hash = payload["sha256"]
    require(isinstance(payload_path, str) and
            manifest_path.name == Path(payload_path + ".manifest.json").name,
            "adjacent payload-manifest naming changed")
    require(isinstance(payload_hash, str) and
            re.fullmatch(r"[0-9a-f]{64}", payload_hash) is not None,
            "ADM payload hash is malformed")
    print("Z4c Iris input dispatch and Brill manifest static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
