#!/usr/bin/env python3
"""Fail-closed source contract for the bounded Brill shift controls."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")

constructor = (ROOT / "src/z4c/z4c.cpp").read_text()
rhs = (ROOT / "src/z4c/z4c_calcrhs.cpp").read_text()
update = (ROOT / "src/z4c/z4c_update.cpp").read_text()
derivatives = (ROOT / "src/z4c/cartoon_derivatives.hpp").read_text()

require('GetOrAddString("z4c", "shift_mode", "gamma_driver")' in constructor,
        "default shift mode is not gamma_driver")
require('GetOrAddString("z4c", "shift_advection_order", "spatial")' in constructor,
        "default advection order is not spatial")
require("prescribed_zero_shift" in rhs and "rhs.beta_u" in rhs,
        "prescribed-zero RHS branch is missing")
require("InitializePrescribedZeroShift" in update,
        "initial prescribed state is missing")
require("CheckPrescribedZeroShiftInvariant" in update and "global_max != 0.0" in update,
        "exact runtime invariant is missing")
require("prescribed_component" in update and "I_Z4C_BETAX" in update,
        "RK update does not preserve exact prescribed state")
for token in ("ScalarAdvectiveO2", "VectorAdvectiveO2", "TensorAdvectiveO2"):
    require(token in derivatives, f"missing provider method {token}")
    require(token in rhs, f"production RHS does not use {token}")
require("VectorFirst" in rhs and "VectorSecond" in rhs,
        "geometric beta derivatives disappeared")
require("telegraph_lapse ? I_Z4C_BETAZ : I_Z4C_BZ" in update,
        "telegraph lapse auxiliary B was mistaken for Gamma-driver state")

print("PASS: Z4c prescribed-zero and isolated O2 shift-advection source contract")
