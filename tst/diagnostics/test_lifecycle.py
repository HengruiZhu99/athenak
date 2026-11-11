"""Actual patched executable lifecycle checks. Set ATHENA_DIAGNOSTICS_EXE to run."""
from pathlib import Path
import os, subprocess, sys, tempfile, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
import athena_read
EXE=os.environ.get('ATHENA_DIAGNOSTICS_EXE')
INPUT=Path(__file__).with_name('smoke.athinput')

@unittest.skipUnless(EXE,'set ATHENA_DIAGNOSTICS_EXE to test patched executable')
class Lifecycle(unittest.TestCase):
    def run_case(self, directory, *args, ranks=1, restart=None):
        directory.mkdir(parents=True)
        cmd=([os.environ.get('MPIEXEC','mpiexec'),'-n',str(ranks)] if ranks>1 else [])
        cmd += [EXE, '-r' if restart else '-i',str(restart or INPUT),'-d',str(directory)]
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

if __name__=='__main__':unittest.main()
