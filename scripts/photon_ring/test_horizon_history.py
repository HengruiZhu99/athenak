"""Keep an independently resolved outer MOTS when its inner partner is lost."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from horizon_history import export


class HistoryTest(unittest.TestCase):
    def test_unresolved_partner_and_production_handoff(self):
        outer = dict(
            area=12.0,
            J=-0.4,
            zeta=-0.837758,
            chi=-0.9845,
            min_cells=30.0,
            lambda0=0.125,
            checks=[[0.001, -0.2]],
            coeff=[1.0],
        )
        inner = {**outer, "lambda0": -0.06, "coeff": [0.9]}
        pair = dict(
            t_M=1.0, cycle=10, outer=outer, inner=inner, witness={"accepted": True}
        )
        # Production repeats the end time with a different cycle and writes
        # duplicate endpoint geometry. There must be only one record per branch.
        dense = [(2.0, 10, Path("d1")), (2.1, 15, Path("d2"))]
        production = [
            (2.1, 17, Path("p0")),
            (2.2, 20, Path("p1")),
            (2.2, 20, Path("p2")),
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("horizon_history.qualify", return_value=None),
            patch("horizon_history.continued_outer", return_value=outer) as fallback,
        ):
            out = Path(tmp)
            result = export(
                out, 2.0, dense, [pair], pair, production, {}, [[1.0], [0.9]]
            )
            with (out / result["file"]).open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(
            [r["status"] for r in rows],
            [
                "resolved",
                "resolved",
                "resolved",
                "unresolved",
                "resolved",
                "unresolved",
            ],
        )
        self.assertEqual(rows[4]["source"], "production")
        self.assertEqual(float(rows[4]["zeta"]), outer["zeta"])
        self.assertEqual(float(rows[4]["lambda0_Madm2"]), 0.5)
        self.assertEqual(rows[5]["zeta"], "")
        self.assertEqual(rows[4]["strict_witness"], "")
        self.assertEqual(
            result["first_verified_strict"]["inner"]["lambda0_Madm2"], -0.24
        )
        self.assertIsNone(result["reconstructed_fold"])


if __name__ == "__main__":
    unittest.main()
