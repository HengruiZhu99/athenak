"""Exercise dispatch without a scheduler and compare the two portable launch profiles."""

from decimal import Decimal as D
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import advance
import campaign
from survey import propose

CODE = campaign.CODE


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((CODE / "inputs/spin.json").read_text())
        self.keys = tuple(map(D, self.config["coarse"]))
        self.outcomes = {w: "collapse" for w in self.keys}
        self.outcomes[self.keys[-1]] = "dispersal"

    def test_threshold_and_peak(self):
        lo, high, last, end = self.keys
        spins = {lo: (0.98, 0.0001), high: (0.999, 0.0001), last: (0.993, 0.0001)}
        plan, _ = propose(self.outcomes, spins, set(), self.config)
        self.assertEqual(plan[(last + end) / 2], "threshold")
        self.assertTrue(
            any(lo < x < high and r == "onset spin" for x, r in plan.items())
        )
        for invalid in ("needs_review", "indeterministic", "inadmissible"):
            self.outcomes[last] = invalid
            self.assertNotIn(
                "threshold",
                propose(self.outcomes, spins, set(), self.config)[0].values(),
            )

    def test_invalid_coarse_and_unranked_collapse(self):
        a, b, c, d = self.keys
        plan, _ = propose(
            {a: "collapse", c: "collapse", d: "dispersal"},
            {a: (0.999, 0.001)},
            {b},
            self.config,
        )
        self.assertEqual(plan, {(c + d) / 2: "threshold"})
        plan, _ = propose(self.outcomes, {}, set(), self.config)
        self.assertEqual(plan, {(c + d) / 2: "threshold"})


class ClusterTest(unittest.TestCase):
    def test_launch_argv(self):
        with tempfile.TemporaryDirectory(prefix="two words ") as tmp:
            root = Path(tmp)
            exe = root / "athena test"
            exe.touch(mode=0o755)
            with patch.object(campaign, "ROOT", root):
                for name in ("slurm", "stellarai"):
                    profile = json.loads((CODE / f"profiles/{name}.json").read_text())
                    profile["executable"] = exe.name
                    (root / "cluster.json").write_text(json.dumps(profile))
                    args = campaign.solver_command(["-r", "checkpoint with spaces"])
                    self.assertEqual(
                        args[-3:], [str(exe.resolve()), "-r", "checkpoint with spaces"]
                    )
                    self.assertNotIn("{source}", " ".join(args))
                    if name == "slurm":
                        self.assertFalse(
                            any(
                                "cuda" in s.lower() or "stellar" in s.lower()
                                for s in args
                            )
                        )

    def test_executable_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "athena"
            exe.write_bytes(b"first")
            exe.chmod(0o755)
            digest = campaign.executable_digest(
                exe, exe.stat().st_mtime_ns, exe.stat().st_size
            )
            (root / "cluster.json").write_text(
                json.dumps(
                    {
                        "launcher": [],
                        "executable": "athena",
                        "executable_sha256": digest,
                    }
                )
            )
            with patch.object(campaign, "ROOT", root):
                self.assertEqual(campaign.solver_command([]), [str(exe.resolve())])
                exe.write_bytes(b"replaced")
                with self.assertRaisesRegex(ValueError, "digest"):
                    campaign.solver_command([])

    def test_dispatch_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="campaign with spaces ") as tmp:
            root = Path(tmp)
            case = root / "cases/0.95"
            case.mkdir(parents=True)
            profile = json.loads((CODE / "profiles/slurm.json").read_text())
            scan = {"name": "test"}
            with (
                patch.object(advance, "ROOT", root),
                patch("advance.subprocess.run") as run,
                patch("advance.subprocess.check_output") as accounting,
            ):
                run.return_value = subprocess.CompletedProcess(
                    [], 1, "", "AssocMaxSubmitJobLimit"
                )
                advance.submit("0.95", "gpu", scan, profile, set())
                self.assertFalse((case / "jobs.txt").exists())
                run.return_value = subprocess.CompletedProcess(
                    [], 0, "101;cluster\n", ""
                )
                queued = set()
                advance.submit("0.95", "gpu", scan, profile, queued)
                argv = run.call_args.args[0]
                self.assertIn(str(root), argv)
                self.assertIn("--gpus-per-task=1", argv)
                advance.submit("0.95", "gpu", scan, profile, queued)
                self.assertEqual(run.call_count, 2)
                for state in ("", "RUNNING", "FAILED"):
                    accounting.return_value = state
                    advance.submit("0.95", "gpu", scan, profile, set())
                self.assertEqual(run.call_count, 2)
                accounting.return_value = "PREEMPTED"
                run.return_value = subprocess.CompletedProcess([], 0, "102\n", "")
                advance.submit("0.95", "gpu", scan, profile, set())
                self.assertEqual(
                    (case / "jobs.txt").read_text().splitlines(), ["gpu 101", "gpu 102"]
                )

    def test_stage_shell_and_dry_run(self):
        with tempfile.TemporaryDirectory(prefix="campaign with spaces ") as tmp:
            root = Path(tmp)
            (root / "site.sh").write_text('export ROLE="$3"\n')
            probe = root / "dummy replay.py"
            probe.write_text(
                "import os,sys; print(os.getcwd(),sys.argv[1:],os.environ['ROLE'])"
            )
            result = subprocess.check_output(
                [CODE / "stage.sh", root, probe, "0.95", "plan"],
                env={**os.environ, "PHOTON_PYTHON": sys.executable},
                text=True,
            )
            self.assertIn("['0.95', 'plan'] plan", result)
            for axis in ("spin", "width"):
                (root / "scan.json").write_bytes(
                    (CODE / f"inputs/{axis}.json").read_bytes()
                )
                (root / "cluster.json").write_bytes(
                    (CODE / "profiles/slurm.json").read_bytes()
                )
                out = subprocess.check_output(
                    [sys.executable, CODE / "advance.py", "--dry-run"],
                    cwd=root,
                    env={**os.environ, "PHOTON_CAMPAIGN": str(root)},
                    text=True,
                )
                state = json.loads(out)
                self.assertEqual(len(state["stages"]), len(state["targets"]))
                self.assertTrue(all(stage == "cal" for _, stage in state["stages"]))
                self.assertFalse((root / "targets.json").exists())


if __name__ == "__main__":
    unittest.main()
