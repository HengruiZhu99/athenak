#!/usr/bin/env python3
"""Offline checkpoint and 12-process adapter tests; no MPI or GPU claims."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import control
import preflight
import storage


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def checkpoint(self,name='state',value=0.125,steps=4,tail=b''):
        path=self.root/name
        header=preflight.HEADER.pack(b'HYP3D_MPI_V1'.ljust(16,b'\0'),6,31,1,1,1,1,steps,0.0004,0.0001)
        path.write_bytes(header+struct.pack('<64d',*([value]*64))+tail)
        return path

    def test_native_header_and_exact_payload(self):
        self.assertEqual(preflight.HEADER.size,72)
        a,b=self.checkpoint('a'),self.checkpoint('b')
        result=preflight.compare_checkpoints(a,b)
        self.assertTrue(result['restart_equivalence'])
        self.assertTrue(result['bitwise_payload_equal'])
        self.assertEqual(result['compared_doubles'],64)

    def test_difference_and_tolerance(self):
        a,b=self.checkpoint('a'),self.checkpoint('b',0.125000001)
        self.assertFalse(preflight.compare_checkpoints(a,b)['restart_equivalence'])
        self.assertTrue(preflight.compare_checkpoints(a,b,atol=1e-8)['restart_equivalence'])

    def test_nonfinite_and_mismatched_metadata_rejected(self):
        a=self.checkpoint('a')
        for kwargs in [dict(value=float('nan')),dict(steps=5),dict(tail=b'x')]:
            b=self.checkpoint('b',**kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                preflight.compare_checkpoints(a,b)


class SocketIntegrationTests(unittest.TestCase):
    def exercise(self,bad_rank,component='device'):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            here=Path(__file__).parent
            (root/'source').mkdir();(root/'build').mkdir();(root/'runs').mkdir()
            for name in ['control.py','worker.py','storage.py','preflight.py','run_debug.pbs']:
                shutil.copyfile(here/name,root/'source'/name)
            binary=root/'build/component'
            binary.write_text('#!'+sys.executable+'\n'+
                'import os\n'+
                f'bad=int(os.environ["MPI_LOCALRANKID"])=={bad_rank}\n'+
                'print("backend="+("Serial" if bad else "SYCL"))\n'+
                'print("scaled_error=0 partition_error=0 diagnostic_scaled_error=0 partition_sum_error=0 worst_scaled_error=0")\n'+
                'print("CPU/device and partition parity passed on 1 cells; worst_scaled_error=0")\n'+
                'print("scri_jet_error=0 null_before=0.1 null_after=1e-15")\n'+
                'print("CPU/device halo, off-node null closure and sparse scri donor parity passed")\n'+
                'print("projection_error=0 rate_error=0 dependent_trace_error=0 null=1e-15 pole=0.003 clock=0.099 compatibility=0 max_cfl=0.01 D_input_and_order_independent=1")\n'+
                'print("D_input_and_order_independent=1")\n'+
                'print("CPU/device scri projection, native rates, D independence and partition parity passed on 216 rows")\n')
            binary.chmod(0o755)
            (root/'build/CMakeCache.txt').write_text('Kokkos_ENABLE_SYCL:BOOL=ON\n')
            plan=dict(schema=1,reviewed=True,phase='preflight',n=16,low_resolution_n=16,
                      resources=dict(control.FIXED,walltime='00:05:00'),worker_seconds=210,
                      storage=dict(limit_bytes=5_000_000_000,planned_new_bytes=1_000_000,atomic_checkpoint_reserve_bytes=0),
                      output_dir='runs/output',commands=[dict(name='component',kind='check',
                      executable='source/preflight.py',args=[],timeout_seconds=120,
                      inputs=['build/component','build/CMakeCache.txt'],
                      report='runs/output/component.report.json',checks=[dict(path=['status'],op='eq',value='completed')])])
            control.atomic(root/'runs/plan.json',plan)
            control.atomic(root/'manifest.json',{'source/preflight.py':control.digest(root/'source/preflight.py')})
            binding=control.bind(root,'runs/plan.json','manifest.json')
            (root/'runs/output').mkdir()
            control.atomic(root/'runs/worker-status.json',dict(plan_sha256=binding['plan_sha256'],
                source_manifest_sha256=binding['source_manifest_sha256'],cases=[dict(name='component',generated_inputs={})]))
            command=[sys.executable,str(root/'source/preflight.py'),'--root',str(root),'--plan','runs/plan.json',
                     '--output-prefix','runs/output/component','--seconds','30','component','--component',component,
                     '--test-binary','build/component','--cmake-cache','build/CMakeCache.txt']
            processes=[]
            try:
                for rank in range(12):
                    env=dict(os.environ,MPI_LOCALRANKID=str(rank),ZE_AFFINITY_MASK=f'{rank//2}.{rank%2}',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE')
                    processes.append(subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
                outputs=[p.communicate(timeout=40) for p in processes]
                expected=0 if bad_rank<0 else 1
                self.assertEqual([p.returncode for p in processes],[expected]*12,outputs)
                report=json.loads((root/'runs/output/component.report.json').read_text())
                self.assertEqual(report['status'],'completed' if expected==0 else 'failed')
                self.assertEqual(report['gpu_cpu_parity'],expected==0)
                self.assertEqual(len(report['ranks']),12)
                self.assertEqual(len(list((root/'runs/output').glob('*.log'))),12)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill();process.wait()

    def test_all_twelve_reports_required(self):
        self.exercise(-1)

    def test_exchange_metric_adapter(self):
        self.exercise(-1,'exchange')

    def test_scri_metric_adapter(self):
        self.exercise(-1,'scri')

    def test_one_wrong_backend_fails_all_ranks(self):
        self.exercise(7)


class StorageTests(unittest.TestCase):
    def test_reserve_enforced_without_deletion(self):
        plan=dict(commands=[dict(kind='minkowski')],storage=dict(limit_bytes=5_000_000_000,
                  planned_new_bytes=1_000_000_000,atomic_checkpoint_reserve_bytes=500_000_000))
        with patch.object(storage,'footprint',return_value=dict(bytes=4_000_000_000,entries=1)):
            with self.assertRaises(RuntimeError):
                storage.require_capacity(Path('/unused'),plan,prospective=True)
            self.assertEqual(storage.require_capacity(Path('/unused'),plan)['headroom_bytes'],500_000_000)
        with patch.object(storage,'footprint',return_value=dict(bytes=4_600_000_000,entries=1)):
            with self.assertRaises(RuntimeError):
                storage.require_capacity(Path('/unused'),plan)

    def test_sparse_bytes_and_symlinks_do_not_hide_or_follow_data(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            root=Path(a);outside=Path(b)
            (outside/'unrelated').write_bytes(b'x'*1_000_000)
            (root/'link').symlink_to(outside,target_is_directory=True)
            with (root/'sparse').open('wb') as stream:
                stream.seek(2_000_000);stream.write(b'x')
            measured=storage.footprint(root)
            self.assertGreaterEqual(measured['bytes'],2_000_001)
            self.assertLess(measured['bytes'],2_100_000)


if __name__=='__main__':
    unittest.main()
