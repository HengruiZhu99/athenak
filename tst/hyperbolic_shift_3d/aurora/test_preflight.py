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


def benchmark_fixture(rank=7,sampling='staged'):
    return (f'backend=SYCL\nsampling={sampling}\n'
            f'benchmark_repeats=20 n=16 rank={rank} ranks=12 '
            f'rank_source=MPI_LOCALRANKID size_source=argument first_row={128*rank} '
            f'owned_rows=128 sampled_fields=25 staged_jet_bytes={332800 if sampling=="staged" else 0} '
            f'evaluate_seconds_total=0.4 evaluate_seconds_mean=0.02 rhs_norm2={rank+1} '
            f'surface_norm2={rank+2} repeat_error=0\n'
            'Scri repeated-Evaluate benchmark completed\n')


ALGEBRAIC_FIXTURE = ('backend=SYCL\nrotation_covariance_max=8.88e-16\n'
    'scri_null_before_reclosure=0.001 after=2.22e-16 ghost=4.44e-16 '
    'D_compatibility=1.11e-16 null_time_derivative=6.66e-16\n'
    'algebraic projection: tensors, rejection, device, unchanged fields, compensation, scri passed\n')


class ComponentParserTests(unittest.TestCase):
    def test_algebraic_requires_sycl_success_and_finite_metrics(self):
        parsed=preflight.parse_component(ALGEBRAIC_FIXTURE,'algebraic',0)
        self.assertTrue(parsed['gpu_cpu_parity'])
        self.assertEqual(parsed['maxima']['after'],2.22e-16)
        for log,code in [(ALGEBRAIC_FIXTURE.replace('backend=SYCL','backend=Serial'),0),
                         (ALGEBRAIC_FIXTURE.replace('scri passed','scri failed'),0),
                         (ALGEBRAIC_FIXTURE,1)]:
            with self.subTest(log=log,code=code):
                self.assertFalse(preflight.parse_component(log,'algebraic',code)['passed'])
        for value in ['nan','inf']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                preflight.parse_component(ALGEBRAIC_FIXTURE.replace('after=2.22e-16','after='+value),'algebraic',0)
        with self.assertRaises(ValueError):
            preflight.parse_component(ALGEBRAIC_FIXTURE.replace('ghost=4.44e-16',''),'algebraic',0)

    def test_benchmark_accepts_both_variants_and_all_rank_partitions(self):
        for sampling in ['staged','inline']:
            for rank in range(12):
                with self.subTest(sampling=sampling,rank=rank):
                    parsed=preflight.parse_benchmark(benchmark_fixture(rank,sampling),rank,sampling,0)
                    self.assertTrue(parsed['benchmark_passed'])
                    self.assertEqual(parsed['metrics']['first_row'],128*rank)

    def test_benchmark_rejects_missing_duplicate_nonfinite_and_wrong_partition(self):
        original=benchmark_fixture()
        changes=[('n=16','n=32'),('ranks=12','ranks=1'),('rank=7','rank=0'),
                 ('first_row=896','first_row=0'),('owned_rows=128','owned_rows=1536'),
                 ('benchmark_repeats=20','benchmark_repeats=1'),('sampled_fields=25','sampled_fields=31'),
                 ('rank_source=MPI_LOCALRANKID','rank_source=serial'),('size_source=argument','size_source=serial'),
                 ('staged_jet_bytes=332800','staged_jet_bytes=0'),('repeat_error=0','repeat_error=1e-16'),
                 ('rhs_norm2=8','rhs_norm2=nan'),('surface_norm2=9','surface_norm2=-1'),
                 ('evaluate_seconds_total=0.4','evaluate_seconds_total=inf'),
                 ('evaluate_seconds_mean=0.02','evaluate_seconds_mean=0'),
                 ('evaluate_seconds_mean=0.02','evaluate_seconds_mean=0.03'),
                 ('repeat_error=0','repeat_error=0 repeat_error=0'),('rhs_norm2=8','')]
        for before,after in changes:
            with self.subTest(before=before,after=after),self.assertRaises(ValueError):
                preflight.parse_benchmark(original.replace(before,after),7,'staged',0)
        with self.assertRaises(ValueError):
            preflight.parse_benchmark(original+original,7,'staged',0)

    def test_benchmark_requires_backend_variant_marker_and_success(self):
        original=benchmark_fixture()
        for log,code in [(original.replace('backend=SYCL','backend=Serial'),0),
                         (original.replace('sampling=staged','sampling=inline'),0),
                         (original.replace('benchmark completed','benchmark failed'),0),
                         (original,1)]:
            with self.subTest(log=log,code=code):
                self.assertFalse(preflight.parse_benchmark(log,7,'staged',code)['passed'])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def checkpoint(self,name='state',value=0.125,steps=4,tail=b'',magic=b'HYP3D_MPI_V1'):
        path=self.root/name
        header=preflight.HEADER.pack(magic.ljust(16,b'\0'),6,31,1,1,1,1,steps,0.0004,0.0001)
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

    def test_projected_schema_cannot_mix_with_original(self):
        a=self.checkpoint('a',magic=b'HYP3D_MPI_V2')
        b=self.checkpoint('b',magic=b'HYP3D_MPI_V2')
        self.assertTrue(preflight.compare_checkpoints(a,b)['restart_equivalence'])
        b=self.checkpoint('b')
        with self.assertRaises(ValueError):
            preflight.compare_checkpoints(a,b)

    def test_nonfinite_and_mismatched_metadata_rejected(self):
        a=self.checkpoint('a')
        for kwargs in [dict(value=float('nan')),dict(steps=5),dict(tail=b'x')]:
            b=self.checkpoint('b',**kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                preflight.compare_checkpoints(a,b)


class SocketIntegrationTests(unittest.TestCase):
    def exercise(self,bad_rank,component='device',mode='component',sampling='staged',bad_layout_rank=-1):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            here=Path(__file__).parent
            (root/'source').mkdir();(root/'build').mkdir();(root/'runs').mkdir()
            for name in ['control.py','worker.py','storage.py','preflight.py','run_debug.pbs']:
                shutil.copyfile(here/name,root/'source'/name)
            binary=root/'build/component'
            component_script=('#!'+sys.executable+'\n'+
                'import os\n'+
                f'bad=int(os.environ["MPI_LOCALRANKID"])=={bad_rank}\n'+
                'print("backend="+("Serial" if bad else "SYCL"))\n'+
                'print("scaled_error=0 partition_error=0 diagnostic_scaled_error=0 partition_sum_error=0 worst_scaled_error=0")\n'+
                'print("CPU/device and partition parity passed on 1 cells; worst_scaled_error=0")\n'+
                'print("scri_jet_error=0 null_before=0.1 null_after=1e-15")\n'+
                'print("CPU/device halo, off-node null closure and sparse scri donor parity passed")\n'+
                'print("projection_error=0 rate_error=0 dependent_trace_error=0 null=1e-15 pole=0.003 clock=0.099 compatibility=0 max_cfl=0.01 D_input_and_order_independent=1")\n'+
                'print("D_input_and_order_independent=1")\n'+
                'print("CPU/device scri projection, native rates, D independence and partition parity passed on 216 rows")\n'+
                'print('+repr(ALGEBRAIC_FIXTURE.split('\n',1)[1])+')\n')
            if mode=='benchmark':
                logs={rank:benchmark_fixture(rank,sampling) for rank in range(12)}
                if bad_layout_rank>=0:
                    logs[bad_layout_rank]=logs[bad_layout_rank].replace('owned_rows=128','owned_rows=1536')
                binary.write_text('#!'+sys.executable+'\nimport os,sys\n'+
                    'assert sys.argv[1:]==["--benchmark-repeats","20","--benchmark-n","16","--benchmark-ranks","12"]\n'+
                    'rank=int(os.environ["MPI_LOCALRANKID"])\n'+
                    'logs='+repr(logs)+'\n'+
                    f'print(logs[rank].replace("backend=SYCL","backend=Serial") if rank=={bad_rank} else logs[rank])\n')
            else:
                binary.write_text(component_script)
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
            with patch.object(control,'__file__',str(root/'source/control.py')):
                binding=control.bind(root,'runs/plan.json','manifest.json')
            (root/'runs/output').mkdir()
            control.atomic(root/'runs/worker-status.json',dict(plan_sha256=binding['plan_sha256'],
                source_manifest_sha256=binding['source_manifest_sha256'],cases=[dict(name='component',generated_inputs={})]))
            command=[sys.executable,str(root/'source/preflight.py'),'--root',str(root),'--plan','runs/plan.json',
                     '--output-prefix','runs/output/component','--seconds','30',mode,
                     '--test-binary','build/component','--cmake-cache','build/CMakeCache.txt']
            command+=['--component',component] if mode=='component' else ['--sampling',sampling]
            processes=[]
            try:
                for rank in range(12):
                    env=dict(os.environ,MPI_LOCALRANKID=str(rank),ZE_AFFINITY_MASK=f'{rank//2}.{rank%2}',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE')
                    processes.append(subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
                outputs=[p.communicate(timeout=40) for p in processes]
                expected=0 if bad_rank<0 and bad_layout_rank<0 else 1
                self.assertEqual([p.returncode for p in processes],[expected]*12,outputs)
                report=json.loads((root/'runs/output/component.report.json').read_text())
                self.assertEqual(report['status'],'completed' if expected==0 else 'failed')
                self.assertEqual(report['gpu_cpu_parity' if mode=='component' else 'benchmark_passed'],expected==0)
                self.assertEqual(len(report['ranks']),12)
                self.assertEqual(len(list((root/'runs/output').glob('*.log'))),12)
                if mode=='benchmark' and expected==0:
                    self.assertEqual(report['sampling'],sampling)
                    self.assertEqual(report['norms'],dict(rhs_norm2=78,surface_norm2=90))
                    self.assertEqual(report['timing']['evaluate_seconds_mean_max'],.02)
                    self.assertEqual([r['result']['metrics']['first_row'] for r in report['ranks']],list(range(0,1536,128)))
                    self.assertNotIn('gpu_cpu_parity',report)
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

    def test_algebraic_metric_adapter(self):
        self.exercise(-1,'algebraic')

    def test_benchmark_both_sampling_variants(self):
        for sampling in ['staged','inline']:
            with self.subTest(sampling=sampling):
                self.exercise(-1,mode='benchmark',sampling=sampling)

    def test_benchmark_one_wrong_backend_fails_all_ranks(self):
        self.exercise(7,mode='benchmark')

    def test_benchmark_one_wrong_partition_fails_all_ranks(self):
        self.exercise(-1,mode='benchmark',bad_layout_rank=7)

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
