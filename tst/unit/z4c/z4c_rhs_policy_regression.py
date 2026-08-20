#!/usr/bin/env python3
"""Enforced numeric goldens for the lightweight Cartesian Z4c production run."""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import struct
import subprocess
import sys


# These metadata-independent hashes freeze the reviewed accepted-state ordering
# in which final-stage active-cell algebraic projection precedes restriction and
# boundary exchange.  That ordering intentionally changes the one-step payload
# from the historical post-prolongation projection golden.
STATE_FINAL = {
    2: "08d50c2ae7dee9878bdcedf47f22e9c559134e4c20768d634aeca50dff979dea",
    4: "6e50cec9a5de98806333a12071ea3aef50700a5c8bad4f67a21ef62486053c56",
    6: "818410481a9815179d07b91f8b0dee02bfada33516bbbd4d6e7d1b6c3c0657bb",
}
DIAGNOSTIC_FINAL = {
    2: "44d695ec905b52e8498fc89f800855c2f97892b74197566c70938daf35d604b7",
    4: "a19039ed148f046ff846b6c68e69563e59fbc879284bbf150ccb95289d0a9df8",
    6: "1426c94a1c05f5609658e548afc10fe1d9fab3a61b912fd74d9213e49a5daccf",
}
HISTORY_NUMERIC = {
    2: "808f7e6654fb6b19ff053ca65559f80b79b7cd1e81903eed350e1ed32b8f4212",
    4: "4db4d626420eb0b36c681a5eef5c4fc7993f20a71b18f0bacbcfe8a3a766a4e6",
    6: "2291e58806948617cfcb1e998661e28a6bf023ce7b90a49082ffc15d0368873f",
}
WEYL_REAL_NUMERIC = {
    2: "9f5e2f4d1a5421d43d486b3ddcb25217600691a52ca9869ca4a17a48b333a974",
    4: "b2cc918b1f080090af108d93c46c7e2e22fe4f9f7b8d1128f1d5b5afb831137a",
    6: "47474bfbee0309a7fe15f372225670c0b3aeb17d667078376c98fd53a73cea5b",
}
WEYL_IMAG_NUMERIC = {
    2: "7fb7ef90ad2c3f12ea34cb1f719e0689d23cb4481f2b4c93264a9a37c13da088",
    4: "cdae17fbba9c1b7bf704610fa5e03d3e8bab7d7a60c7f0a7ec5d96a41a20ced7",
    6: "e0d4b0c3683ba47ae1fee732ae1c290998fa0c06a785e2ff8e4b601b970af27d",
}

# Full-file hashes for the same reviewed ordering.  The comparison decks use the
# stable ``cmp_oN`` basename, so these values enforce output layout and
# serialization metadata in addition to decoded numerical values.
STATE_FILE = {
    2: "3faf8e4c1056db653448f51ae29afabe5511797160c96b111ffbf4b536d6f988",
    4: "1db3d058799605fe0b970a016dbcadbdbeac3b37ff5bca46a438b1ef48553fbd",
    6: "c40b9921bfea76277ca7c7f518451a72c706cebcf3d3d8473e4a023d1f0ed50f",
}
HISTORY_FILE = {
    2: "7327ee50dc7373b689651c34b156d02767c58b10f2f7710995075f35def1e6ed",
    4: "9125d9cb4bc53ecf7dc89c82d9c35777da5f5d1b437fca881b489ca4618bb8a8",
    6: "2e0b337a5a67c1fc56d7eae5f29107f17d89d3df4f868fb5ef5c03684d40907e",
}
WEYL_REAL_FILE = {
    2: "065b89cf1ce7da82c1d64c6b83c92d95d3df7bcb4ed6b8ce2bbb7fdeced4a056",
    4: "3e5a0b09ef584532f726a92198ce2a9cc2eeb6b17237f7c0750983e183c0b82b",
    6: "477d167b14ce016f9b76516ed896c208204389a62fb84b1b50d8a484fdf97a2a",
}
WEYL_IMAG_FILE = {
    2: "31884bda4eb16526f926ca848652b8479e4cb9b7aabe18d43a29b1ed11bcb745",
    4: "6dbfc708bc97b0ddc729ce938ab96c0d9d2663bd16464d0ddcca441ebdbca374",
    6: "d8d493b2e9072c3b5962748d36c957d29b53ca39a150438af59c178e5efa667e",
}
DIAGNOSTIC_ORDER6_FILE = (
    "499ba391b9ee2afea96a1c8fdfde0de8d6b7dd0e942892f80ac4f7711f6987d7"
)

HISTORY_GOLDEN = {
    2: [
        [0.0, 0.008838834733581843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0],
        [0.008838834733581843, 0.008838834733581843,
         4.8609168882403355e-17, 3.330288587137061e-20,
         4.857192506502791e-17, 9.852328760425326e-22,
         3.9654474720608265e-30, 2.4285962549106002e-17,
         2.428596251591791e-17, 8.112424845191525e-34, 8.0,
         3.309016113771575e-17, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    ],
    4: [
        [0.0, 0.008838834733581843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, math.inf, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0],
        [0.008838834733581843, 0.008838834733581843,
         7.696113993855242e-17, 6.801747717066916e-20,
         7.688689444416744e-17, 1.5570043035524903e-21,
         6.036794710855561e-30, 3.844344732728763e-17,
         3.844344711687373e-17, 4.010089442110816e-33, 8.0,
         1.4552100569554952e-16, 1.0, 5.956620933777848e-13,
         0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    ],
    6: [
        [0.0, 0.008838834733581843, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, math.inf, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0],
        [0.008838834733581843, 0.008838834733581843,
         9.001830507144616e-17, 8.717389401832743e-20,
         8.992385090199482e-17, 1.820068858264116e-21,
         7.029999962891466e-30, 4.496192568858577e-17,
         4.496192521340186e-17, 4.466147136107364e-33, 8.0,
         1.1489308924980757e-16, 1.0, 6.181489003358232e-13,
         0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    ],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric_rows(path: pathlib.Path) -> list[list[float]]:
    return [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def numeric_hash(rows: list[list[float]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_payload_hash(data: dict) -> str:
    digest = hashlib.sha256()
    for name in data["var_names"]:
        digest.update(name.encode("utf-8") + b"\0")
        for block in data["mb_data"][name]:
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def flattened_binary(data: dict):
    for name in data["var_names"]:
        for block in data["mb_data"][name]:
            yield from block.flat


def check_history(order: int, rows: list[list[float]], backend: str,
                  with_kretschmann: bool = True) -> None:
    require(len(rows) == 3, f"order {order}: expected three history rows")
    require(rows[1][0] == rows[2][0] and rows[1][2:] == rows[2][2:],
            f"order {order}: repeated final history physics row changed")
    require(math.isfinite(rows[1][1]) and math.isfinite(rows[2][1]) and
            abs(rows[1][1] - rows[2][1]) <= 1.0e-9 * abs(rows[1][1]),
            f"order {order}: repeated final accepted timestep changed materially")
    if backend == "Serial":
        require(numeric_hash(rows) == HISTORY_NUMERIC[order],
                f"order {order}: exact-base Serial history hash changed")
        return

    # Parallel reductions may change the final few ulps. Compare against the reviewed
    # Serial base with a scale-aware tolerance while keeping state/diagnostic/Weyl exact.
    expected_rows = [HISTORY_GOLDEN[order][0], HISTORY_GOLDEN[order][1],
                     HISTORY_GOLDEN[order][1]]
    if order in (4, 6) and not with_kretschmann:
        expected_rows = [row[:13] + row[14:] for row in expected_rows]
    require([len(row) for row in rows] == [len(row) for row in expected_rows],
            f"order {order}: history shape changed")
    for row_index, (observed, expected) in enumerate(zip(rows, expected_rows)):
        for column, (value, golden) in enumerate(zip(observed, expected)):
            if math.isinf(golden):
                require(value == golden,
                        f"order {order}: history[{row_index},{column}] inf changed")
            else:
                if row_index == 2 and column == 1:
                    tolerance = 1.0e-9 * max(abs(golden), 1.0e-18)
                else:
                    tolerance = (512.0 * sys.float_info.epsilon *
                                 max(abs(golden), 1.0e-18) + 1.0e-31)
                require(math.isfinite(value) and abs(value - golden) <= tolerance,
                        f"order {order}: history[{row_index},{column}]={value} "
                        f"differs from {golden} by more than {tolerance}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=pathlib.Path, required=True)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--test-dir", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.source_dir / "vis/python"))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    import numpy as np  # pylint: disable=import-error,import-outside-toplevel

    config = subprocess.run(
        [str(arguments.athena), "-c"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    backend_lines = [line for line in config.splitlines()
                     if "Kokkos default execution space:" in line]
    require(len(backend_lines) == 1, "could not identify Kokkos execution space")
    backend = backend_lines[0].rsplit(":", 1)[1].strip()

    for order in (2, 4, 6):
        case = arguments.test_dir / f"order{order}"
        basename = f"z4c_rhs_policy_o{order}"
        state = bin_convert.read_binary(
            str(case / "bin" / f"{basename}.state.00002.bin"))
        require(binary_payload_hash(state) == STATE_FINAL[order],
                f"order {order}: exact-base final state payload changed")

        diagnostics = []
        for output in range(3):
            diagnostics.append(bin_convert.read_binary(
                str(case / "bin" /
                    f"{basename}.diagnostics.{output:05d}.bin")))
        initial = np.fromiter(flattened_binary(diagnostics[0]), dtype=float)
        require(initial.size == 8192 and np.isnan(initial).all(),
                f"order {order}: t=0 diagnostics must be exactly 8192 unavailable NaNs")
        for output in (1, 2):
            values = np.fromiter(flattened_binary(diagnostics[output]), dtype=float)
            require(values.size == 8192 and np.isfinite(values).all(),
                    f"order {order}: diagnostic output {output} is not wholly finite")
        require(binary_payload_hash(diagnostics[2]) == DIAGNOSTIC_FINAL[order],
                f"order {order}: reviewed final diagnostic payload changed")

        history = numeric_rows(case / f"{basename}.z4c.user.hst")
        check_history(order, history, backend)
        real_weyl = numeric_rows(case / "waveforms/rpsi4_real_0.25.txt")
        imaginary_weyl = numeric_rows(case / "waveforms/rpsi4_imag_0.25.txt")
        require(numeric_hash(real_weyl) == WEYL_REAL_NUMERIC[order],
                f"order {order}: exact-base real Weyl values changed")
        require(numeric_hash(imaginary_weyl) == WEYL_IMAG_NUMERIC[order],
                f"order {order}: exact-base imaginary Weyl values changed")

        exact_case = arguments.test_dir / f"exact_base_order{order}"
        exact_basename = f"cmp_o{order}"
        exact_state = exact_case / "bin" / f"{exact_basename}.state.00002.bin"
        exact_history = exact_case / f"{exact_basename}.z4c.user.hst"
        exact_real = exact_case / "waveforms/rpsi4_real_0.25.txt"
        exact_imaginary = exact_case / "waveforms/rpsi4_imag_0.25.txt"
        require(file_hash(exact_state) == STATE_FILE[order],
                f"order {order}: exact-base state file bytes changed")
        require(file_hash(exact_real) == WEYL_REAL_FILE[order],
                f"order {order}: exact-base real Weyl file bytes changed")
        require(file_hash(exact_imaginary) == WEYL_IMAG_FILE[order],
                f"order {order}: exact-base imaginary Weyl file bytes changed")
        if backend == "Serial":
            require(file_hash(exact_history) == HISTORY_FILE[order],
                    f"order {order}: exact-base Serial history file bytes changed")
        else:
            check_history(order, numeric_rows(exact_history), backend,
                          with_kretschmann=False)

    exact_diagnostic = (arguments.test_dir / "exact_base_diagnostic_order6" /
                        "bin/cmp_diag_o6.diagnostics.00002.bin")
    require(file_hash(exact_diagnostic) == DIAGNOSTIC_ORDER6_FILE,
            "order 6 exact-base diagnostic file bytes changed")

    print(f"Enforced Cartesian Z4c base/golden regression passed on {backend}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
