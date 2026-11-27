"""Actual patched executable lifecycle checks. Set ATHENA_DIAGNOSTICS_EXE to run."""
from pathlib import Path
import os, re, subprocess, sys, tempfile, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
import athena_read
EXE=os.environ.get('ATHENA_DIAGNOSTICS_EXE')
INPUT=Path(__file__).with_name('smoke.athinput')

TRANSPORT_QUANTITIES = (
    "M0_in", "M0_out", "E_inertial_f", "E_inertial_em",
    "Jz_center_f", "Jz_center_em", "Jz_orbit_f", "Jz_orbit_em",
)


def transport_totals(face):
    return [f"{quantity}_qualified_cumulative_{face}" for quantity in TRANSPORT_QUANTITIES]


def transport_rates(h, face, x, omega, sample=0):
    # The sampled centers are on the x axis of the circular-orbit fixture.
    def rate(name):
        return h[f"{name}_in_moving_{face}"][sample]
    rates = {name: h[f"{name}_moving_{face}"][sample] for name in ("M0_in", "M0_out")}
    for part in ("f", "em"):
        jz = rate("Jz_" + part)
        orbital = x * rate("Py_" + part)
        rates["Jz_orbit_" + part] = orbital
        rates["Jz_center_" + part] = jz - orbital
        rates["E_inertial_" + part] = rate("Et_" + part) + omega * jz
    return rates


def inventory_from_dump(dump, center_x, axes, radius, subcells=24):
    """Cartesian voxel control for an aligned, boosted horizon at orbital phase zero."""
    import bin_convert

    data = bin_convert.read_binary(str(dump))
    offsets = (np.arange(subcells) + 0.5) / subcells - 0.5
    total = np.zeros(3)
    for block, bounds in enumerate(data["mb_geometry"]):
        rho, px, py = (data["mb_data"][field][block] for field in ("dens", "mom1", "mom2"))
        lower = bounds[::2].astype(float)
        upper = bounds[1::2].astype(float)
        center = np.array([center_x, 0, 0])
        if np.any(upper < center - radius) or np.any(lower > center + radius):
            continue
        nx, ny, nz = rho.shape[::-1]
        dx, dy, dz = spacing = (upper - lower) / [nx, ny, nz]
        grid = [lower[d] + (np.arange(n) + 0.5) * spacing[d]
                for d, n in enumerate((nx, ny, nz))]
        z, y, x = np.meshgrid(*grid[::-1], indexing="ij")
        keep = ((abs(x-center_x) <= radius+dx/2) & (abs(y) <= radius+dy/2) &
                (abs(z) <= radius+dz/2))
        x, y, z, rho, px, py = (v[keep] for v in (x, y, z, rho, px, py))
        if not x.size:
            continue
        zsquared = (z[:, None] + dz * offsets)**2
        for u in offsets:
            xs = x + dx * u
            xr2 = (xs-center_x)**2
            for v in offsets:
                ys = y + dy * v
                outside_horizon = (zsquared / axes[2]**2 +
                                   (xr2 / axes[0]**2 + ys**2 / axes[1]**2)[:, None]) >= 1
                in_sphere = zsquared + (xr2 + ys**2)[:, None] <= radius**2
                dv = (outside_horizon & in_sphere).sum(axis=1) * (dx*dy*dz/subcells**3)
                total += [dv.sum(), (rho*dv).sum(), ((xs*py-ys*px)*dv).sum()]
    return total


@unittest.skipUnless(EXE,'set ATHENA_DIAGNOSTICS_EXE to test patched executable')
class Lifecycle(unittest.TestCase):
    def run_case(self, directory, *args, ranks=1, restart=None, input_file=INPUT):
        directory.mkdir(parents=True)
        cmd=([os.environ.get('MPIEXEC','mpiexec'),'-n',str(ranks)] if ranks>1 else [])
        cmd += [EXE, '-r' if restart else '-i',str(restart or input_file),'-d',str(directory)]
        cmd += ['output1/data_format=%24.16e',*args]
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (directory/'run.log').write_text(result.stdout)
        self.assertEqual(result.returncode,0,result.stdout[-2000:])
        return athena_read.hst(directory/'matched_bondi.user.hst')
    def test_restart_cadence_ranks_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            full=self.run_case(root/'full','time/nlim=5')
            split=self.run_case(root/'split','time/nlim=2')
            rst=sorted((root/'split/rst').glob('*.rst'))[-1]
            resume=self.run_case(root/'resume','time/nlim=5',restart=rst)
            sparse=self.run_case(root/'sparse','time/nlim=5','output1/dcycle=3')
            key='M0_qualified_cumulative_BH1_Kerr'
            self.assertGreater(full[key][-1],0)
            self.assertGreater(full['M0_covered_duration_BH1_Kerr'][-1],0)
            np.testing.assert_allclose(resume[key][-1],full[key][-1],rtol=2e-12)
            np.testing.assert_allclose(sparse[key][-1],full[key][-1],rtol=2e-12)
            self.assertEqual(full['M0_committed_cycle_BH1_Kerr'][-1],5)
            missing=self.run_case(root/'missing','time/nlim=3','problem/flux_min_cells_per_radius=100')
            self.assertEqual(missing[key][-1],0)
            self.assertEqual(missing['M0_covered_duration_BH1_Kerr'][-1],0)
            self.assertEqual(missing['M0_unavailable_duration_BH1_Kerr'][-1],missing['time'][-1])
            self.assertTrue(np.isnan(missing['M0_net_in_moving_BH1_Kerr']).all())
            if os.environ.get('ATHENA_DIAGNOSTICS_MPI'):
                mpi=self.run_case(root/'mpi','time/nlim=5',ranks=2)
                changed=self.run_case(root/'changed','time/nlim=5',ranks=2,restart=rst)
                np.testing.assert_allclose(mpi[key][-1],full[key][-1],rtol=2e-12)
                np.testing.assert_allclose(changed[key][-1],full[key][-1],rtol=2e-12)

            face = "BH1_Kerr"
            for label in transport_totals(face):
                for run in (resume, sparse):
                    np.testing.assert_allclose(run[label][-1], full[label][-1], rtol=2e-11, atol=2e-13)
                np.testing.assert_array_equal(missing[label], 0)
                if os.environ.get("ATHENA_DIAGNOSTICS_MPI"):
                    np.testing.assert_allclose(changed[label][-1], full[label][-1], rtol=2e-11, atol=2e-13)

            # Adopt a checkpoint with the original M0 accumulator but without a
            # transport ledger. Earlier rest mass survives; new integrals start here.
            raw = rst.read_bytes()
            end = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
            header, count = re.subn(rb"(?m)^BH1_Kerr_transport_v1\s*=[^\n]*\n", b"", raw[:end])
            self.assertEqual(count, 1)
            old = root / "mass-only.rst"
            old.write_bytes(header + raw[end:])
            adopted = self.run_case(root / "adopted", "time/nlim=5", restart=old)
            self.assertEqual(adopted["transport_epoch_start_" + face][-1], split["time"][-1])
            np.testing.assert_allclose(adopted[key][-1], full[key][-1], rtol=2e-12)
            for label in transport_totals(face):
                np.testing.assert_allclose(adopted[label][-1], full[label][-1] - split[label][-1],
                                           rtol=2e-10, atol=2e-16, err_msg=label)
            np.testing.assert_allclose(adopted["transport_covered_duration_" + face][-1],
                                       full["time"][-1] - split["time"][-1])

    def test_horizon_transport_frame_and_stage(self):
        fixture = INPUT.with_name("binary_shell.athinput")
        args = ["time/nlim=1", "time/cfl_number=0.10", "time/tlim=0.001",
                "problem/b_inf=0.002", "problem/flux_shell_budget=false"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rot = self.run_case(root / "rot", *args, "time/integrator=rk1", input_file=fixture)
            inert = self.run_case(root / "inert", *args, "time/integrator=rk1",
                                  "problem/corotating=false", "problem/frame_omega=0",
                                  input_file=fixture)
            rk2 = self.run_case(root / "rk2", *args, "time/integrator=rk2", input_file=fixture)
            # r1 = q*a/(1+q) and r2 = -a/(1+q) in this unequal-mass test.
            for face, x in (("BH1_Kerr", 4 * 0.75 / 1.75), ("BH2_Kerr", -4 / 1.75)):
                for h, omega in ((rot, 0.125), (inert, 0)):
                    rates = transport_rates(h, face, x, omega)
                    for quantity, expected in rates.items():
                        key = f"{quantity}_qualified_cumulative_{face}"
                        np.testing.assert_allclose(h[key][-1] / h["time"][-1], expected,
                                                   rtol=2e-10, atol=2e-12, err_msg=key)
                        np.testing.assert_allclose(rot[key][-1], inert[key][-1],
                                                   rtol=3e-9, atol=1e-14, err_msg=key)
                # Independently contract the inertial RK1 currents with the
                # component boost at phase zero, where v_y = Omega_bin * x.
                omega_bin = 4.0**(-1.5)
                vy = omega_bin * x
                gamma = 1 / np.sqrt(1 - vy**2)
                for part in ("f", "em"):
                    energy_rate = inert[f"Et_{part}_in_moving_{face}"][0]
                    momentum_rate = inert[f"Py_{part}_in_moving_{face}"][0]
                    self.assertNotEqual(vy * momentum_rate, 0)
                    boosted_rate = gamma * (energy_rate - vy * momentum_rate)
                    for h in (inert, rot):
                        dt = h["time"][-1] - h["time"][0]
                        energy = np.diff(h[f"E_inertial_{part}_qualified_cumulative_{face}"])
                        orbital = np.diff(h[f"Jz_orbit_{part}_qualified_cumulative_{face}"])
                        np.testing.assert_allclose(gamma * (energy - omega_bin*orbital) / dt,
                                                   boosted_rate, rtol=3e-9, atol=2e-12,
                                                   err_msg=f"{face} {part} component boost")
                # SSPRK2 evaluates stages at the initial state and the forward
                # Euler state. In the stationary frame both sample the same surface.
                stage1 = transport_rates(rot, face, x, 0.125)
                stage2 = transport_rates(rot, face, x, 0.125, sample=-1)
                for quantity in TRANSPORT_QUANTITIES:
                    key = f"{quantity}_qualified_cumulative_{face}"
                    expected = rk2["time"][-1] * (stage1[quantity] + stage2[quantity]) / 2
                    np.testing.assert_allclose(rk2[key][-1], expected, rtol=3e-10,
                                               atol=2e-15, err_msg=key)

    @unittest.skipUnless(os.environ.get("ATHENA_DIAGNOSTICS_MPI"), "requires MPI")
    def test_inertial_magnetic_flux_decomposition(self):
        from check_decomposition import compare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = INPUT.with_name("binary_shell.athinput")
            args = ("problem/corotating=false", "problem/frame_omega=0",
                    "problem/b_inf=0.002", "time/cfl_number=0.04",
                    "time/nlim=20", "time/tlim=0.02")
            for ranks in (1, 2):
                self.run_case(root/str(ranks), *args, ranks=ranks, input_file=fixture)
            compare(root/"1/matched_bondi.user.hst", root/"2/matched_bondi.user.hst")

    def test_torque_output_selection(self):
        from bench_diagnostics_gpu import check_torque_output
        with tempfile.TemporaryDirectory() as temp:
            fixture = INPUT.with_name("binary_shell.athinput").read_text()
            fixture = fixture.replace("nlim = 2", "nlim = 8").replace("b_inf = 0", "b_inf = 0.002")
            launcher = [os.environ.get("MPIEXEC", "mpiexec"), "-n", "2"] if os.environ.get(
                "ATHENA_DIAGNOSTICS_MPI") else []
            check_torque_output(EXE, Path(temp)/"selection", fixture, launcher)

    def test_binary_shell_restart_and_rejected_state(self):
        import re
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = INPUT.with_name("binary_shell.athinput")
            full = self.run_case(root/"full", "time/nlim=3", input_file=fixture)
            # Problem initialization marks circular ADM stationary after task assembly.
            np.testing.assert_array_equal(full["diagnostic_metric_time"], 0.0)
            self.run_case(root/"split", "time/nlim=2", input_file=fixture)
            checkpoint = sorted((root/"split/rst").glob("*.rst"))[-1]
            ranks = 2 if os.environ.get("ATHENA_DIAGNOSTICS_MPI") else 1
            resumed = self.run_case(root/"resumed", "time/nlim=3", ranks=ranks, restart=checkpoint)
            for label in full:
                if label.startswith("shell_"):
                    np.testing.assert_allclose(resumed[label][-1], full[label][-1],
                                               rtol=5e-9, atol=2e-11, err_msg=label)
            data = checkpoint.read_bytes()
            end = data.index(b"<par_end>") + len(b"<par_end>\n")
            def flag_shell(match):
                fields = match[2].split(b",")
                fields[17] = b"1"
                return match[1] + b",".join(fields)
            header, count = re.subn(rb"(shell_R3\.75\s*=\s*)([^\s#]+)", flag_shell, data[:end])
            self.assertEqual(count, 1)
            injected = root/"rejected.rst"
            injected.write_bytes(header + data[end:])
            h = self.run_case(root/"stopped", "time/nlim=2", restart=injected)
            self.assertEqual(h["shell_R3.75_volume_valid"][-1], 0)
            self.assertEqual(h["shell_R3.75_sampled_volume"][-1], 0)
            self.assertGreater(h["shell_R3.75_rejected_volume"][-1], 0)
            retry = sorted((root/"stopped/rst").glob("*.rst"))[-1]
            h = self.run_case(root/"retry", "time/nlim=3", restart=retry)
            for q in ("Px","Py","Pz","Jz"):
                self.assertTrue(np.isnan(h["shell_R3.75_"+q+"_interval_residual"][-1]))
                self.assertTrue(np.isfinite(h["shell_R4.25_"+q+"_interval_residual"][-1]))

    def test_transfer_inventory_against_conserved_dump(self):
        """The larger spatial control uses this same inertial input at time/nlim=0.

        Use N=48,64,96,128 cells on each mesh axis, with 16-cell blocks for
        N=48 and 32-cell blocks for the others. Set flux_ntheta=N/2, flux_nphi=N,
        and shell_subcells=8. Keep only the history output and compare the initial
        inventory_H1 and inventory_H2 storage, sampled volumes and boundary rates.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = INPUT.with_name("binary_shell.athinput").read_text()
            fixture = fixture.replace("<problem>", "<problem>\nflux_transfer_inventory=true"
                                      "\nflux_horizon2=true\nflux_radius2=1.55")
            fixture += "\n<output3>\nfile_type=bin\nvariable=mhd_u\ndcycle=2\n"
            inp = root / "inventory.athinput"
            inp.write_text(fixture)
            args = ("problem/flux_horizon1=true", "problem/flux_radius1=1.55",
                    "problem/b_inf=0.002", "problem/flux_shell_budget=false",
                    "problem/corotating=false", "problem/frame_omega=0", "time/tlim=0.00007")
            full = self.run_case(root/"full", "time/nlim=4", *args, input_file=inp)
            coarse = self.run_case(root/"coarse", "time/nlim=0", "problem/shell_subcells=2",
                                   *args, input_file=inp)
            dump = root/"full/bin/matched_bondi.mhd_u.00000.bin"
            # Mass ratio 0.75, separation 4, aligned spins. A boost in y contracts
            # the equatorial y axis of the intrinsic Kerr-Schild ellipsoid.
            for h, mass, chi, x in ((1, 4/7, 0.3, 12/7), (2, 3/7, 0.2, -16/7)):
                polar = mass * (1 + np.sqrt(1-chi**2))
                equatorial = np.hypot(polar, mass*chi)
                axes = (equatorial, equatorial*np.sqrt(1-(x/8)**2), polar)
                labels = ("sampled_volume", "M0_stored", "Jz_stored")
                prefix = f"inventory_H{h}_"
                got = [full[prefix + field][0] for field in labels]
                exact = (4*np.pi/3) * (1.55**3 - np.prod(axes))
                np.testing.assert_allclose(got[0], exact, rtol=5e-4)
                control = inventory_from_dump(dump, x, axes, 1.55)
                np.testing.assert_allclose(got, control, rtol=7e-4)
                control[0] = exact
                low = [coarse[prefix + field][0] for field in labels]
                self.assertTrue((np.abs(got-control) < np.abs(low-control)).all())
                np.testing.assert_array_equal(full[prefix+"storage_valid"], 1)
                np.testing.assert_array_equal(full[prefix+"rejected_volume"], 0)
                for field in ("M0", "Jz"):
                    np.testing.assert_array_equal(full[prefix+field+"_partial_stored"],
                                                  full[prefix+field+"_stored"])
                for field, flux in (("M0", "M0_net"), ("Jz_f", "Jz_f"), ("Jz_em", "Jz_em")):
                    outer = full[f"{flux}_in_moving_H{h}"]
                    inner = full[f"{flux}_in_moving_BH{h}_Kerr"]
                    np.testing.assert_allclose(full[prefix+field+"_boundary_in"], outer-inner,
                                               rtol=1e-12, atol=1e-14)
                    self.assertGreater(np.max(np.abs(outer-inner)), 1e-7)
            checkpoint = root/"full/rst/matched_bondi.00001.rst"
            ranks = 2 if os.environ.get("ATHENA_DIAGNOSTICS_MPI") else 1
            if ranks > 1:
                parallel = self.run_case(root/"two_ranks", "time/nlim=0",
                                         *args, ranks=ranks, input_file=inp)
                for key in full:
                    if key.startswith("inventory_"):
                        np.testing.assert_allclose(full[key][0], parallel[key][0],
                                                   rtol=1e-12, atol=1e-14, err_msg=key)
            # Production enables the inventory with a problem-only -i overlay on
            # an existing -r checkpoint that already tracks the transfer surfaces.
            before = self.run_case(root/"before", "time/nlim=2", *args,
                                   "problem/flux_transfer_inventory=false", input_file=inp)
            self.assertFalse(any(key.startswith("inventory_") for key in before))
            overlay = root/"enable_inventory.athinput"
            overlay.write_text("<problem>\nflux_transfer_inventory=true\n")
            previous = root/"before/rst/matched_bondi.00001.rst"
            with previous.open("rb") as stream:
                header = stream.read(40 * 1024)
            self.assertIn(b"<par_end>\n", header)
            block = header.split(b"<cbd_diagnostics>\n", 1)[1].split(b"\n<", 1)[0]
            values = [line.split(b"=", 1)[1].lstrip().split(b"#", 1)[0]
                      for line in block.splitlines() if b"=" in line]
            self.assertGreater(len(values), 2)
            self.assertLessEqual(max(len(v) - len(v.rstrip()) for v in values), 2)
            resumed = self.run_case(root/"resumed", "-i", str(overlay), "time/nlim=4",
                                    ranks=ranks, restart=previous)
            control = self.run_case(root/"control", "time/nlim=4", ranks=ranks, restart=checkpoint)
            self.assertEqual(len(resumed["time"]), 2)
            np.testing.assert_array_equal(resumed["time"], control["time"])
            self.assertEqual(resumed["time"][-1], full["time"][-1])
            stopped = self.run_case(root/"stopped", "time/nlim=2", ranks=ranks, restart=checkpoint)
            # Output at the checkpoint has no restored per-cell reset provenance.
            # A subsequent accepted stage qualifies the new field.
            for h in (1, 2):
                prefix = f"inventory_H{h}_"
                self.assertEqual(stopped[prefix+"storage_valid"][0], 0)
                self.assertGreater(stopped[prefix+"rejected_volume"][0], 0)
                self.assertEqual(stopped[prefix+"sampled_volume"][0], 0)
                for field in ("M0_stored", "Jz_stored", "M0_partial_stored", "Jz_partial_stored"):
                    self.assertTrue(np.isnan(stopped[prefix+field][0]))
                for key in full:
                    if key.startswith(prefix):
                        np.testing.assert_allclose(control[key], resumed[key],
                                                   rtol=1e-12, atol=1e-14, err_msg=key)
                        np.testing.assert_allclose(full[key][-1], resumed[key][-1],
                                                   rtol=5e-9, atol=2e-11, err_msg=key)
                face = f"BH{h}_Kerr"
                keys = transport_totals(face) + [name + face for name in (
                    "M0_qualified_cumulative_", "M0_epoch_start_", "M0_covered_duration_",
                    "M0_unavailable_duration_", "M0_committed_cycle_", "transport_epoch_start_",
                    "transport_covered_duration_", "transport_unavailable_duration_",
                    "transport_interval_valid_")]
                self.assertGreater(before["M0_qualified_cumulative_" + face][-1], 0)
                for key in keys:
                    np.testing.assert_allclose(before[key], full[key][:len(before[key])],
                                               rtol=1e-12, atol=2e-15, err_msg=key)
                    np.testing.assert_allclose(control[key], resumed[key],
                                               rtol=1e-12, atol=2e-15, err_msg=key)

    def test_transfer_inventory_zoom_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root/"inventory_zoom.athinput"
            fixture = INPUT.with_name("zoom_smoke.athinput").read_text()
            inp.write_text(fixture.replace("<problem>", "<problem>\nflux_transfer_inventory=true"))
            h = self.run_case(root/"zoom", input_file=inp)
            prefix = "inventory_H1_"
            rejected = h[prefix+"rejected_volume"] > 0
            self.assertTrue(rejected.any())
            self.assertTrue((~rejected[1:] & rejected[:-1]).any())
            np.testing.assert_array_equal(h[prefix+"storage_valid"], ~rejected)
            sampled = h[prefix+"sampled_volume"] > 0
            self.assertTrue((sampled & rejected).any())
            for field in ("M0", "Jz"):
                whole = h[prefix+field+"_stored"]
                partial = h[prefix+field+"_partial_stored"]
                np.testing.assert_array_equal(np.isnan(whole), rejected)
                np.testing.assert_array_equal(np.isfinite(partial), sampled)
                np.testing.assert_array_equal(partial[~rejected], whole[~rejected])
            for field in ("M0", "Jz_f", "Jz_em"):
                self.assertTrue(np.isnan(h[prefix+field+"_boundary_in"][rejected]).all())

    def test_regional_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('budget_smoke.athinput')
            full=self.run_case(root/'full','time/nlim=5',input_file=inp)
            self.run_case(root/'split','time/nlim=2',input_file=inp)
            rst=sorted((root/'split/rst').glob('*.rst'))[-1]
            resumed=self.run_case(root/'resumed','time/nlim=5',restart=rst)
            for key in full:
                if key.startswith('budget_'):
                    np.testing.assert_allclose(resumed[key][-1],full[key][-1],atol=2e-12,rtol=2e-10)
            self.assertTrue(np.isnan(full['budget_near1_residual_dJ'][-1]))
            self.assertTrue(np.isfinite(full['budget_annulus_residual_dJ'][-1]))

    def test_schema_change_and_old_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            inp=root/'without-epoch-override'
            inp.write_text(INPUT.read_text().replace('flux_new_epoch = false\n',''))
            first=self.run_case(root/'first','time/nlim=2',input_file=inp)
            rst=sorted((root/'first/rst').glob('*.rst'))[-1]
            bad=subprocess.run([EXE,'-r',str(rst),'-d',str(root/'bad'),
                'problem/flux_min_cells_per_radius=2'],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn('Changed diagnostic configuration',bad.stdout+bad.stderr)
            fresh=self.run_case(root/'fresh','time/nlim=3',restart=rst,
                *['problem/flux_min_cells_per_radius=2','problem/flux_new_epoch=true'])
            self.assertEqual(fresh['M0_epoch_start_BH1_Kerr'][-1],first['time'][-1])
            old=os.environ.get('ATHENA_DIAGNOSTICS_BASE_EXE')
            if old:
                (root/'old').mkdir()
                run=subprocess.run([old,'-i',str(INPUT),'-d',str(root/'old'),'time/nlim=2'],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                checkpoint=sorted((root/'old/rst').glob('*.rst'))[-1]
                new=self.run_case(root/'new','time/nlim=3',restart=checkpoint)
                self.assertEqual(new['M0_epoch_start_BH1_Kerr'][-1],first['time'][-1])
                self.assertIn('earlier accretion is unavailable',(root/'new/run.log').read_text())

    def test_partial_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            h=self.run_case(Path(temp)/'partial','time/nlim=1',
                'problem/flux_rsurf_inner=9','problem/flux_rsurf_outer=9')
            self.assertTrue(np.isnan(h['mdot_R9']).all())
            self.assertTrue(np.isnan(h['area_R9']).all())
            self.assertTrue((h['ownership_fraction_R9']<1).all())
            self.assertTrue((h['partial_covered_area_R9']>0).all())

    def test_injected_source_stage_weights(self):
        import bin_convert
        def moment(dump):
            data=bin_convert.read_binary(str(dump));total=0.0
            for mb,g in enumerate(data['mb_geometry']):
                axes=[np.linspace(g[2*a],g[2*a+1],9)[:-1]+0.5 for a in range(3)]
                Z,Y,X=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
                for dz in (-0.25,0.25):
                    for dy in (-0.25,0.25):
                        for dx in (-0.25,0.25):
                            x,y,z=X+dx,Y+dy,Z+dz
                            inside=(x-2)**2+y*y+z*z<1.2**2
                            outside=(x-2)**2+y*y/(1-0.25**2)+z*z>=0.4**2
                            total+=np.sum((x*data['mb_data']['dens'][mb])[inside&outside])/8
            return total
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('source_smoke.athinput')
            rk1=self.run_case(root/'rk1',input_file=inp)
            start=moment(root/'rk1/bin/matched_bondi.mhd_w.00000.bin')
            intermediate=moment(root/'rk1/bin/matched_bondi.mhd_w.00001.bin')
            dt=rk1['time'][-1];source=0.2
            np.testing.assert_allclose(rk1['budget_near1_integral_physical_dJ'][-1],
                dt*source*start,rtol=1e-7,atol=1e-12)
            rk2=self.run_case(root/'rk2','time/integrator=rk2',input_file=inp)
            np.testing.assert_allclose(rk2['budget_near1_integral_physical_dJ'][-1],
                0.5*dt*source*(start+intermediate),rtol=1e-7,atol=1e-12)

    def test_amr_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('amr_smoke.athinput')
            h=self.run_case(root/'serial','time/nlim=2',input_file=inp)
            for surface in ('R3.5','R4','H1','BH1_Kerr'):
                np.testing.assert_equal(h['ownership_fraction_'+surface],1)
            if os.environ.get('ATHENA_DIAGNOSTICS_MPI'):
                mpi=self.run_case(root/'mpi','time/nlim=2',input_file=inp,ranks=2)
                np.testing.assert_allclose(mpi['M0_qualified_cumulative_BH1_Kerr'][-1],
                    h['M0_qualified_cumulative_BH1_Kerr'][-1],rtol=2e-11,atol=1e-12)

    def test_frozen_halos_match_active_donors(self):
        import bin_convert

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root/"frozen_halos.athinput"
            text = INPUT.with_name("zoom_smoke.athinput").read_text()
            text += "\n<output3>\nfile_type=bin\nvariable=mhd_u\nghost_zones=true\ndcycle=1\n"
            for hook in (False, True):
                case = root/("with_hook" if hook else "plain")
                inp.write_text(text.replace("<coord>", "<coord>\nsmooth_excision=true"
                                            "\nsmooth_excision_b_damping=true"
                                            "\nsmooth_excision_b_damping_eta=0.01") if hook else text)
                ranks = 2 if os.environ.get("ATHENA_DIAGNOSTICS_MPI") else 1
                self.run_case(case, "time/nlim=3", "problem/b_inf=0.002",
                              ranks=ranks, input_file=inp)
                # Cycle 3 is output after a frozen-zone RK step, before the next remap.
                data = bin_convert.read_binary(str(case/"bin/matched_bondi.mhd_u.00003.bin"))
                locs = data["mb_logical"]
                by_location = {tuple(location): block for block, location in enumerate(locs)}
                n = np.array([8, 8, 8])
                checked = 0
                for block, (lx, ly, lz, level) in enumerate(locs):
                    bounds = data["mb_geometry"][block].astype(float)
                    lower = bounds[::2]
                    dx = (bounds[1::2] - lower)/n
                    k, j, i = np.indices(data["mb_data"]["dens"][block].shape)
                    cells = np.column_stack((i.ravel(), j.ravel(), k.ravel())) - 4
                    position = lower + (cells + 0.5)*dx
                    r = np.linalg.norm(position, axis=1)
                    # Cover the frozen ball while excluding the physical puncture mask.
                    compare = (r > 0.8) & (r < 2) & np.any((cells < 0) | (cells >= n), axis=1)
                    for cell in cells[compare]:
                        donor_loc = tuple(np.array([lx, ly, lz]) + cell//n) + (level,)
                        donor = by_location[donor_loc]
                        ghost_index = tuple((cell+4)[::-1])
                        donor_index = tuple((cell%n+4)[::-1])
                        for field, blocks in data["mb_data"].items():
                            self.assertEqual(blocks[block][ghost_index], blocks[donor][donor_index],
                                             f"{field} donor {donor_loc} ghost {ghost_index}")
                        checked += 1
                self.assertEqual(checked, 224)

    def test_zoom_pending_and_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=INPUT.with_name('zoom_smoke.athinput')
            full=self.run_case(root/'full',input_file=inp)
            split=self.run_case(root/'split','time/nlim=2',input_file=inp)
            # The scheduled file precedes AMR; the final file at the same cycle follows it.
            pre=root/'split/rst/matched_bondi.00001.rst'
            post=root/'split/rst/matched_bondi.00002.rst'
            for label,rst in (('pending',pre),('completed',post)):
                resumed=self.run_case(root/label,'time/nlim=12',restart=rst)
                for key in ('M0_qualified_cumulative_BH1_Kerr','M0_covered_duration_BH1_Kerr',
                            'M0_unavailable_duration_BH1_Kerr'):
                    np.testing.assert_allclose(resumed[key][-1],full[key][-1],rtol=3e-10,atol=1e-12)
            self.assertGreater(full['M0_unavailable_duration_BH1_Kerr'][-1],0)
            self.assertTrue((full['reason_BH1_Kerr'].astype(int)&32).any())
            np.testing.assert_allclose(full['M0_covered_duration_BH1_Kerr']+
                full['M0_unavailable_duration_BH1_Kerr'],full['time'],atol=1e-13)

            face = "BH1_Kerr"
            missing = full["transport_interval_valid_" + face][1:] == 0
            self.assertTrue(missing.any())
            self.assertTrue((~missing).any())
            for label in transport_totals(face):
                np.testing.assert_array_equal(np.diff(full[label])[missing], 0)
                np.testing.assert_allclose(resumed[label][-1], full[label][-1],
                                           rtol=3e-10, atol=1e-12, err_msg=label)
            np.testing.assert_allclose(full["transport_covered_duration_" + face] +
                                       full["transport_unavailable_duration_" + face],
                                       full["time"], atol=1e-13)

    def test_zoom_budget_jump_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            text=INPUT.with_name('zoom_smoke.athinput').read_text().replace('<problem>',
                '<problem>\nflux_budget=true\nbudget_r_inner=3.5\nbudget_r_outer=4\nbudget_subcells=2',1)
            inp=root/'budget-zoom-input';inp.write_text(text)
            full=self.run_case(root/'full',input_file=inp)
            self.run_case(root/'split','time/nlim=2',input_file=inp)
            for number in (1,2):
                resumed=self.run_case(root/f'resumed{number}','time/nlim=12',
                    restart=root/f'split/rst/matched_bondi.{number:05d}.rst')
                for region in ('near1','cavity','annulus'):
                    for channel in ('cumulative_quantified_remap_jump_dJ','cumulative_quantified_zoom_jump_dJ'):
                        key=f'budget_{region}_{channel}'
                        np.testing.assert_allclose(resumed[key][-1],full[key][-1],rtol=1e-9,atol=1e-10)
                    # A genuine restart in the frozen zone performs one additional
                    # startup recovery measurement. Its unavailable jump is an event,
                    # never an invented finite contribution to the cumulative budget.
                    events=f'budget_{region}_unquantifiable_events'
                    extra=1 if number==2 and region=='near1' else 0
                    self.assertEqual(resumed[events][-1],full[events][-1]+extra)
            self.assertGreater(full['budget_near1_unquantifiable_events'][-1],0)
            self.assertTrue(np.isfinite(full['budget_annulus_cumulative_quantified_remap_jump_dJ'][-1]))

    def test_reset_provenance_survives_repeated_checkpoint(self):
        # Manufacture an unlocalized current reset-provenance flag in an otherwise
        # healthy checkpoint. Exercise the production decoder, sampler, commit and
        # repeated pre-checkpoint export without changing an evolved variable.
        import re
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=self.run_case(root/'first','time/nlim=2')
            original=sorted((root/'first/rst').glob('*.rst'))[-1]
            data=original.read_bytes();end=data.index(b'<par_end>')+len(b'<par_end>\n')
            header,count=re.subn(rb'(source_flags\s*=\s*)[^\s#]+',
                lambda match:match.group(1)+b'0,0,0,16',data[:end])
            self.assertEqual(count,1)
            injected=root/'injected.rst';injected.write_bytes(header+data[end:])
            direct=self.run_case(root/'direct','time/nlim=3',restart=injected)
            stopped=self.run_case(root/'stopped','time/nlim=2',restart=injected)
            checkpoint=sorted((root/'stopped/rst').glob('*.rst'))[-1]
            resumed=self.run_case(root/'resumed','time/nlim=3',restart=checkpoint)
            key='M0_qualified_cumulative_BH1_Kerr'
            self.assertEqual(direct[key][-1],first[key][-1])
            self.assertEqual(resumed[key][-1],first[key][-1])
            self.assertGreater(direct['M0_unavailable_duration_BH1_Kerr'][-1],0)
            self.assertEqual(resumed['M0_interval_reason_BH1_Kerr'][-1],16)
            self.assertEqual(stopped['valid_BH1_Kerr'][-1],0)

if __name__=='__main__':unittest.main()
