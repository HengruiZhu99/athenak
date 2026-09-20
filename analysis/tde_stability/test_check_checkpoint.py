#!/usr/bin/env python3
"""Small on-disk regressions of the real checkpoint parser and metric checker.

Synthetic files follow restart.cpp's restricted double-precision MHD+Z4c
layout; no evolution, large real fixture, or external test framework is needed.
Run: python -m unittest discover -s analysis/tde_stability -p test_check_checkpoint.py
"""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from check_checkpoint import validate


def write_fixture(run, *, ranks=1, blocks=1, states=None, parameters=None):
    """One or two uniform x-adjacent blocks, stored in actual rank-file layout."""
    assert blocks in (1, 2) and ranks in (1, blocks)
    ng, nx = 4, 4
    n = nx + 2*ng
    cells = n**3
    faces = 3*(n+1)*n*n
    mhd_fields = 5
    stride = 8*((mhd_fields+25)*cells+faces)
    params = {
        "problem": {"bh_background": "schwarzschild_trumpet",
                    "use_direct_z4c_background": "true", "bh_mass": "1",
                    "bh_spin": "0", "bh_center_x1": "0",
                    "bh_center_x2": "0", "bh_center_x3": "0"},
        "z4c": {"use_analytic_background": "true", "chi_psi_power": "-4"},
        "mesh_refinement": {"refinement": "none"},
        "mhd": {"nscalars": "0"},
        "mesh": {**{f"x{a}min": "-2" for a in (1, 2, 3)},
                 **{f"x{a}max": "2" for a in (1, 2, 3)},
                 "nx1": str(nx*blocks), "nx2": str(nx), "nx3": str(nx)},
    }
    for block, updates in (parameters or {}).items():
        params[block].update(updates)
    text = "".join(f"<{block}>\n"+"".join(f"{k} = {v}\n" for k, v in values.items())
                   for block, values in params.items())+"<par_end>\n"
    level = 0 if blocks == 1 else 1
    def indices(ns):
        active = tuple(x for extent in ns for x in (ng, ng+extent-1))
        return (ng, *ns, *active, *([0]*9))
    header = text.encode()+struct.pack("<ii", blocks, level)
    header += struct.pack("<9d", -2, -2, -2, 2, 2, 2, 1/blocks, 1, 1)
    header += struct.pack("<19i", *indices((nx*blocks, nx, nx)))
    header += struct.pack("<19i", *indices((nx, nx, nx)))
    header += struct.pack("<ddi", 3.75, .0375, 100)
    header += b"".join(struct.pack("<4i", b, 0, 0, level) for b in range(blocks))
    header += struct.pack(f"<{blocks}f", *([1]*blocks))
    header += struct.pack("<ddQ", 0, 0, stride)
    if states is None:
        states = [np.zeros((25, n, n, n), dtype="<f8") for _ in range(blocks)]
    prefix = np.zeros(mhd_fields*cells+faces, dtype="<f8").tobytes()
    payload = [prefix+np.asarray(state, dtype="<f8").tobytes() for state in states]
    files = []
    for rank in range(ranks):
        path = run/"rst"/f"rank_{rank:08d}"/"fixture.00001.rst"
        path.parent.mkdir(parents=True)
        blocks_here = payload if ranks == 1 else [payload[rank]]
        path.write_bytes(header+b"".join(blocks_here))
        files.append(path)
    return files, len(header)


class CheckCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_valid_serial_and_mpi_cohorts(self):
        for blocks, ranks in ((1, 1), (2, 1), (2, 2)):
            with self.subTest(blocks=blocks, ranks=ranks):
                run = self.root/f"valid_{blocks}_{ranks}"
                write_fixture(run, blocks=blocks, ranks=ranks)
                result = validate(run, ranks)
                self.assertTrue(result["passed"])
                self.assertEqual(result["invalid_metric_cells_including_ghosts"], 0)
                self.assertEqual((result["blocks"], result["ranks"]), (blocks, ranks))
                self.assertEqual(result["cycle"], 100)
                self.assertEqual(result["minimum"]["determinant"], 1.)

    def test_positive_determinant_indefinite_fourth_ghost_rejected(self):
        states = [np.zeros((25, 12, 12, 12)) for _ in range(2)]
        # diag(1,-1,-1) is finite with det=+1 and gxx>0, but is not SPD.
        states[1][4, 0, 0, 0] = -2
        states[1][6, 0, 0, 0] = -2
        run = self.root/"bad_ghost"
        write_fixture(run, blocks=2, ranks=2, states=states)
        result = validate(run, 2)
        self.assertFalse(result["passed"])
        self.assertTrue(result["all_payload_finite"])
        self.assertEqual(result["invalid_metric_cells_including_ghosts"], 1)
        sample = result["invalid_metric_samples"][0]
        self.assertEqual((sample["rank"], sample["global_block"]), (1, 1))
        self.assertEqual(sample["ghost_depth"], [4, 4, 4])
        self.assertEqual(sample["xyz_M"], [-1.75, -5.5, -5.5])
        self.assertEqual((sample["logical_level"], sample["relative_level"]), (1, 0))
        self.assertEqual(sample["values"]["determinant"], 1.)
        self.assertEqual(sample["values"]["second_minor"], -1.)

    def test_nonfinite_fluid_payload_rejected(self):
        run = self.root/"nan_fluid"
        files, offset = write_fixture(run)
        data = bytearray(files[0].read_bytes())
        struct.pack_into("<d", data, offset, np.nan)
        files[0].write_bytes(data)
        result = validate(run, 1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_payload_finite"])
        self.assertEqual(result["invalid_metric_cells_including_ghosts"], 0)

    def test_invalid_active_lapse_rejected(self):
        state = np.zeros((25, 12, 12, 12))
        state[18, 4, 4, 4] = -2
        run = self.root/"bad_lapse"
        write_fixture(run, states=[state])
        result = validate(run, 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_metric_samples"][0]["ghost_depth"], [0, 0, 0])

    def test_residual_lapse_default_chain_and_preservation(self):
        accepted = [
            {},  # analytic=true -> evolve_gauge=true -> evolve_lapse=true
            {"evolve_gauge_residual": "0", "evolve_lapse_residual": "1"},
            {"evolve_lapse_residual": "false", "preserve_lapse_residual": "TRUE"},
        ]
        for index, settings in enumerate(accepted):
            with self.subTest(settings=settings):
                run = self.root/f"lapse_allowed_{index}"
                write_fixture(run, parameters={"z4c": settings})
                self.assertTrue(validate(run, 1)["passed"])
        rejected = [
            {"evolve_gauge_residual": "false"},
            {"evolve_lapse_residual": "false", "preserve_lapse_residual": "0"},
            {"evolve_lapse_residual": "ambiguous"},
        ]
        for index, settings in enumerate(rejected):
            with self.subTest(settings=settings):
                run = self.root/f"lapse_rejected_{index}"
                write_fixture(run, parameters={"z4c": settings})
                with self.assertRaises(ValueError):
                    validate(run, 1)

    def test_unsupported_geometry_guards(self):
        cases = [{"z4c": {"chi_psi_power": "-2"}},
                 {"mesh_refinement": {"refinement": "adaptive"}},
                 {"problem": {"bh_center_x1": "1"}}]
        for index, parameters in enumerate(cases):
            with self.subTest(parameters=parameters):
                run = self.root/f"unsupported_{index}"
                write_fixture(run, parameters=parameters)
                with self.assertRaises(AssertionError):
                    validate(run, 1)

    def test_incomplete_or_mismatched_rank_cohort_rejected(self):
        run = self.root/"missing_rank"
        files, _ = write_fixture(run, blocks=2, ranks=2)
        files[1].unlink()
        with self.assertRaises(AssertionError):
            validate(run, 2)
        run = self.root/"mismatched_rank"
        files, _ = write_fixture(run, blocks=2, ranks=2)
        data = bytearray(files[1].read_bytes())
        marker_end = data.index(b"<par_end>\n")+len(b"<par_end>\n")
        # Retain payload shape but make rank1's saved time inconsistent.
        struct.pack_into("<d", data, marker_end+8+72+2*76, 9.)
        files[1].write_bytes(data)
        with self.assertRaises(AssertionError):
            validate(run, 2)

    def test_truncated_payload_rejected(self):
        run = self.root/"truncated"
        files, _ = write_fixture(run)
        files[0].write_bytes(files[0].read_bytes()[:-8])
        with self.assertRaises(AssertionError):
            validate(run, 1)


if __name__ == "__main__":
    unittest.main()
