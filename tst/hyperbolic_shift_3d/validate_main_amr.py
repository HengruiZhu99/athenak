import pathlib,os,sys,pytest,shutil
import argparse
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
binary=pathlib.Path(a.binary).resolve();work=pathlib.Path(a.output).resolve()
os.chdir(root/'tst');sys.path[:0]=[str(root/'tst'),str(root/'vis/python')]
import test_suite.testutils as utils
work.mkdir(parents=True,exist_ok=False)
shutil.copy2(binary,work/'athena');shutil.copytree(root/'tst/inputs',work/'inputs');os.chdir(work)
# Preserve error tables/logs; the stock cleanup removes these diagnostics.
utils.cleanup=lambda:None
sys.exit(pytest.main(['-v',str(root/'tst/test_suite/z4c/test_z4c_lwave2d_amr_mpicpu.py'),str(root/'tst/test_suite/z4c/test_z4c_lwave_amr_bc_cpu.py')]))
