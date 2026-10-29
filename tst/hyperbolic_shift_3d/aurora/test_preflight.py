#!/usr/bin/env python3
"""Offline checkpoint and 12-process adapter tests; no MPI or GPU claims."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import control
import preflight
import storage

_export_spec=importlib.util.spec_from_file_location('checkpoint_export',Path(__file__).resolve().parents[1]/'checkpoint_export.py')
checkpoint_export=importlib.util.module_from_spec(_export_spec)
_export_spec.loader.exec_module(checkpoint_export)


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

ANGULAR_DISSIPATION_FIXTURE = 'backend=SYCL\n'+''.join(
    f'n={n} degree={degree} points={6*n*n} epsilon=1.25 '
    'coefficient_reference_error=1e-14 source_reference_error=1e-14 '
    'device_residual_error=1e-14 device_source_error=1e-14 partition_error=0 '
    'energy_error=1e-14 adjoint_error=1e-14 constant_error=0 max_moment_error=1e-14 '
    'max_condition_bound=1000 max_coefficient_l1=2 max_neighbors=70 spectral_upper_bound=3\n'
    for n,degree in [(6,7),(8,7),(6,5)]) + \
    'Angular dissipation CPU/device, energy, constants, partitions and reference parity passed\n'

SCRI_DISSIPATION_FIXTURE = ('backend=SYCL\n'+''.join(
    f'closure_case={name} additive_rate_error=1e-14 dependent_D_response_error=1e-14 '
    'compatibility=1e-15 cpu_rate_error=1e-14 cpu_trace_error=1e-14 '
    'partition_parity=1 zero_off=1 rejection_checks=1\n' for name in ['synthetic','plan'])+''.join(
    f'plan_perturb={perturb} host_rate_error=1e-14 angular_cfl=.1 null=1e-15 compatibility=1e-15\n'
    for perturb in [0,1])+
    'cfl_rejection=1 cross_face_guard=1 epsilon_validation=1 shared_plan=1\n'
    'Optional additive scri closure checks passed\n')


class ComponentParserTests(unittest.TestCase):
    def test_new_dissipation_components_require_exact_markers_backend_and_metrics(self):
        for kind,fixture in [('angular_dissipation',ANGULAR_DISSIPATION_FIXTURE),
                             ('scri_dissipation_closure',SCRI_DISSIPATION_FIXTURE)]:
            with self.subTest(kind=kind):
                parsed=preflight.parse_component(fixture,kind,0)
                self.assertTrue(parsed['passed']);self.assertTrue(parsed['gpu_cpu_parity'])
                self.assertTrue(parsed['maxima']);self.assertTrue(parsed['cases'])
                for content,code in [(fixture.replace('backend=SYCL','backend=Serial'),0),
                                     (fixture.replace('checks passed','checks failed').replace('parity passed','parity failed'),0),
                                     (fixture,2),(fixture+'backend=SYCL\n',0)]:
                    self.assertFalse(preflight.parse_component(content,kind,code)['passed'])

    def test_angular_dissipation_checks_moments_energy_reference_and_coverage(self):
        for before,after in [('source_reference_error=1e-14','source_reference_error=3e-11'),
                             ('device_source_error=1e-14','device_source_error=nan'),
                             ('energy_error=1e-14','energy_error=3e-12'),('adjoint_error=1e-14','adjoint_error=-1'),
                             ('constant_error=0','constant_error=1e-30'),
                             ('max_moment_error=1e-14','max_moment_error=1e-10'),
                             ('max_condition_bound=1000','max_condition_bound=1e9'),
                             ('max_coefficient_l1=2','max_coefficient_l1=20'),
                             ('spectral_upper_bound=3','spectral_upper_bound=0'),
                             ('max_neighbors=70','max_neighbors=70.0'),
                             ('n=6 degree=7','n=4 degree=7'),('n=6 degree=5','n=6 degree=7'),
                             ('partition_error=0',''),('energy_error=1e-14','energy_error=0 energy_error=0')]:
            with self.subTest(before=before,after=after),self.assertRaises(ValueError):
                preflight.parse_component(ANGULAR_DISSIPATION_FIXTURE.replace(before,after),'angular_dissipation',0)

    def test_scri_dissipation_checks_closure_parity_zero_off_and_rejections(self):
        for before,after in [('dependent_D_response_error=1e-14','dependent_D_response_error=4e-12'),
                             ('cpu_rate_error=1e-14','cpu_rate_error=3e-10'),
                             ('cpu_trace_error=1e-14','cpu_trace_error=nan'),
                             ('additive_rate_error=1e-14','additive_rate_error=3e-13'),
                             ('partition_parity=1','partition_parity=0'),('zero_off=1','zero_off=0'),
                             ('rejection_checks=1','rejection_checks=0'),('angular_cfl=.1','angular_cfl=0'),
                             ('angular_cfl=.1','angular_cfl=2.8'),('null=1e-15','null=4e-14'),
                             ('compatibility=1e-15','compatibility=inf'),
                             ('closure_case=plan','closure_case=synthetic'),('plan_perturb=1','plan_perturb=0'),
                             ('cross_face_guard=1','cross_face_guard=0'),('epsilon_validation=1','epsilon_validation=0'),
                             ('shared_plan=1','')]:
            with self.subTest(before=before,after=after),self.assertRaises(ValueError):
                preflight.parse_component(SCRI_DISSIPATION_FIXTURE.replace(before,after),'scri_dissipation_closure',0)

    def test_angular_requires_sycl_success_and_metrics(self):
        log=('backend=SYCL\nfine_gradient_order=4 fine_hessian_order=4 fine_mixed_order=4\n'
             'projection_error=0 rate_error=1e-14 dependent_trace_error=1e-14 null=1e-15 compatibility=1e-16\n'
             'Cross-face scri constants, active donors, nonspherical jets and native device parity passed\n')
        self.assertTrue(preflight.parse_component(log,'angular',0)['passed'])
        self.assertFalse(preflight.parse_component(log,'angular',1)['passed'])
        self.assertFalse(preflight.parse_component(log.replace('backend=SYCL','backend=Serial'),'angular',0)['passed'])
        with self.assertRaises(ValueError):
            preflight.parse_component(log.replace('rate_error=1e-14','rate_error=nan'),'angular',0)

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

    def checkpoint(self,name='state',value=0.125,steps=4,tail=b'',magic=b'HYP3D_MPI_V1',config=(2.0,7,0)):
        path=self.root/name
        header=preflight.HEADER.pack(magic.ljust(16,b'\0'),6,31,1,1,1,1,steps,0.0004,0.0001)
        if magic==b'HYP3D_MPI_V4':header+=preflight.V4_CONFIG.pack(*config)
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
        c=self.checkpoint('c',magic=b'HYP3D_MPI_V3')
        self.assertTrue(preflight.compare_checkpoints(c,c)['restart_equivalence'])
        with self.assertRaises(ValueError):
            preflight.compare_checkpoints(a,c)

    def test_nonfinite_and_mismatched_metadata_rejected(self):
        a=self.checkpoint('a')
        for kwargs in [dict(value=float('nan')),dict(steps=5),dict(tail=b'x')]:
            b=self.checkpoint('b',**kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                preflight.compare_checkpoints(a,b)

    def test_v4_config_and_payload_offsets(self):
        self.assertEqual(preflight.V4_CONFIG.size,16)
        for epsilon in [0.0,1.0,2.0]:
            a=self.checkpoint('a',magic=b'HYP3D_MPI_V4',config=(epsilon,7,0))
            b=self.checkpoint('b',magic=b'HYP3D_MPI_V4',config=(epsilon,7,0))
            result=preflight.compare_checkpoints(a,b)
            self.assertTrue(result['restart_equivalence'])
            self.assertTrue(result['bitwise_payload_equal'])
            self.assertEqual(result['compared_doubles'],64)
            self.assertEqual(result['header']['header_bytes'],88)
            self.assertEqual(result['header']['scri_dissipation_epsilon'],epsilon)
            self.assertEqual(result['header']['scri_dissipation_degree'],7)
        b=self.checkpoint('b',magic=b'HYP3D_MPI_V4',value=.125000001)
        self.assertFalse(preflight.compare_checkpoints(a,b)['restart_equivalence'])
        self.assertTrue(preflight.compare_checkpoints(a,b,atol=1e-8)['restart_equivalence'])

    def test_v4_configuration_cannot_be_tolerated_or_cross_schema(self):
        a=self.checkpoint('a',magic=b'HYP3D_MPI_V4')
        for epsilon in [0.0,1.0,2.0000000000000004]:
            b=self.checkpoint('b',magic=b'HYP3D_MPI_V4',config=(epsilon,7,0))
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):
                preflight.compare_checkpoints(a,b,atol=1e100,rtol=1e100)
        zero=self.checkpoint('zero',magic=b'HYP3D_MPI_V4',config=(0.0,7,0))
        for magic in [b'HYP3D_MPI_V1',b'HYP3D_MPI_V2',b'HYP3D_MPI_V3']:
            b=self.checkpoint('b',magic=magic)
            self.assertEqual(b.read_bytes()[72:],zero.read_bytes()[88:])
            self.assertEqual(preflight.read_checkpoint(b)['header_bytes'],72)
            self.assertEqual(preflight.read_checkpoint(b)['scri_dissipation_degree'],0)
            with self.subTest(magic=magic),self.assertRaises(ValueError):
                preflight.compare_checkpoints(zero,b)

    def test_v4_invalid_config_truncation_and_trailing_bytes(self):
        for config in [(-1,7,0),(float('nan'),7,0),(float('inf'),7,0),(2,6,0),(2,7,1)]:
            p=self.checkpoint(magic=b'HYP3D_MPI_V4',config=config)
            with self.subTest(config=config),self.assertRaises(ValueError):
                preflight.read_checkpoint(p)
        p=self.checkpoint(magic=b'HYP3D_MPI_V4');original=p.read_bytes()
        for data in [original[:71],original[:72],original[:87],original[:-1],original+b'x']:
            p.write_bytes(data)
            with self.subTest(bytes=len(data)),self.assertRaises(ValueError):
                preflight.read_checkpoint(p)

    def test_v4_restart_requires_matching_report_configuration(self):
        a=self.checkpoint('a',magic=b'HYP3D_MPI_V4');b=self.checkpoint('b',magic=b'HYP3D_MPI_V4')
        args=SimpleNamespace(reference='a',candidate='b',reference_report='a.json',candidate_report='b.json',atol=0,rtol=0)
        report=dict(status='completed',backend='SYCL',mpi_size=12,n=6,kind='pulse',t=.0004,
                    target_time=.0004,dt=.0001,steps=4,checkpoint_schema='HYP3D_MPI_V4',
                    algebraic_projection='final_stage_standard',scri_angular_stencil='cross_face_centered5',
                    scri_dissipation_epsilon=2.0,scri_dissipation_degree=7,checkpoint_header_bytes=88)
        (self.root/'a.json').write_text(json.dumps(report));(self.root/'b.json').write_text(json.dumps(report))
        self.assertTrue(preflight.restart(args,0,self.root,lambda p:self.root/p)['passed'])
        for key,bad in [('scri_dissipation_epsilon',0),('scri_dissipation_epsilon',True),
                        ('scri_dissipation_degree',0),('checkpoint_header_bytes',72),
                        ('checkpoint_schema','HYP3D_MPI_V3'),('scri_angular_stencil','face_local6')]:
            changed=dict(report);changed[key]=bad;(self.root/'b.json').write_text(json.dumps(changed))
            with self.subTest(key=key,bad=bad),self.assertRaises(ValueError):
                preflight.restart(args,0,self.root,lambda p:self.root/p)
        for key in ['scri_dissipation_epsilon','scri_dissipation_degree','checkpoint_header_bytes']:
            changed=dict(report);del changed[key];(self.root/'b.json').write_text(json.dumps(changed))
            with self.subTest(missing=key),self.assertRaises(ValueError):
                preflight.restart(args,0,self.root,lambda p:self.root/p)


class ExportCheckpointTests(unittest.TestCase):
    """Actual fixed-topology payloads catch byte-offset and streaming-hash bugs."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,magic=b'HYP3D_MPI_V4',config=(2.0,7,0)):
        n=6;cells=19*n**3;ns=31*6*n*n;nv=31*cells
        header=checkpoint_export.HEADER.pack(magic.ljust(16,b'\0'),n,31,0,1,cells,ns,4,.0004,.0001)
        if magic==b'HYP3D_MPI_V4':header+=checkpoint_export.V4_CONFIG.pack(*config)
        path=self.root/'state.checkpoint'
        payload=b''.join(struct.pack('<d',value)*count for value,count in [(1.25,nv),(-0.0,nv),(2.5,ns),(3.75,ns)])
        path.write_bytes(header+payload)
        return path

    def test_all_schema_offsets_hashes_and_export_rows(self):
        for version in range(1,5):
            path=self.fixture(f'HYP3D_MPI_V{version}'.encode())
            h,arrays,sha=checkpoint_export.validate_checkpoint(path)
            self.assertEqual(sha,checkpoint_export.digest(path))
            self.assertEqual(h['header_bytes'],88 if version==4 else 72)
            self.assertEqual(h['scri_dissipation_epsilon'],2.0 if version==4 else 0.0)
            self.assertEqual(h['scri_dissipation_degree'],7 if version==4 else 0)
            self.assertEqual(arrays[0]['offset'],h['header_bytes'])
            offset=h['header_bytes']
            with path.open('rb') as stream:
                for array,value in zip(arrays,[1.25,-0.0,2.5,3.75]):
                    self.assertEqual(array['offset'],offset)
                    stream.seek(offset);self.assertEqual(struct.unpack('<d',stream.read(8))[0],value)
                    target=self.root/(str(version)+'-'+array['name']+'.csv')
                    checkpoint_export.export_array(stream,dict(array,count=31),target)
                    self.assertEqual(target.read_text().strip().split(','),[format(value,'.17g')]*31)
                    offset+=8*array['count']
            self.assertEqual(offset,path.stat().st_size)

    def test_invalid_v4_configs_sizes_and_nonfinite_payload_rejected(self):
        for config in [(-1,7,0),(float('nan'),7,0),(float('inf'),7,0),(0,0,0),(2,7,-1)]:
            path=self.fixture(config=config)
            with self.subTest(config=config),self.assertRaises(ValueError):
                checkpoint_export.validate_checkpoint(path)
        path=self.fixture();original=path.read_bytes()
        for data in [original[:71],original[:72],original[:87],original[:-1],original+b'x',
                     original[:88]+struct.pack('<d',float('nan'))+original[96:]]:
            path.write_bytes(data)
            with self.subTest(bytes=len(data)),self.assertRaises(ValueError):
                checkpoint_export.validate_checkpoint(path)
        path=self.fixture(b'HYP3D_MPI_V3');path.write_bytes(path.read_bytes()+struct.pack('<dii',2,7,0))
        with self.assertRaises(ValueError):checkpoint_export.validate_checkpoint(path)

    def test_report_config_required_and_preserved(self):
        path=self.fixture(config=(0.0,7,0));h,_,_=checkpoint_export.validate_checkpoint(path)
        report=dict(status='completed',n=h['n'],cells=h['cells'],steps=h['steps'],t=h['t'],dt=h['dt'],
                    kind='minkowski',checkpoint_schema='HYP3D_MPI_V4',algebraic_projection='final_stage_standard',
                    scri_angular_stencil='cross_face_centered5',scri_dissipation_epsilon=0,
                    scri_dissipation_degree=7,checkpoint_header_bytes=88)
        target=self.root/'report.json';target.write_text(json.dumps(report))
        checkpoint_export.validate_report(target,h)
        exported=checkpoint_export.compatible_report(h,report,dict(execution_artifacts_verified=False))
        self.assertEqual(exported['checkpoint_schema'],'HYP3D_MPI_V4')
        self.assertEqual(exported['checkpoint_header_bytes'],88)
        self.assertEqual(exported['scri_dissipation_epsilon'],0)
        self.assertEqual(exported['scri_dissipation_degree'],7)
        for key in ['scri_dissipation_epsilon','scri_dissipation_degree','checkpoint_header_bytes']:
            changed=dict(report);del changed[key];target.write_text(json.dumps(changed))
            with self.subTest(missing=key),self.assertRaises(ValueError):
                checkpoint_export.validate_report(target,h)
        changed=dict(report,scri_dissipation_epsilon=2);target.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):checkpoint_export.validate_report(target,h)


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
                'print('+repr(ALGEBRAIC_FIXTURE.split('\n',1)[1])+')\n'+
                'print('+repr(ANGULAR_DISSIPATION_FIXTURE.split('\n',1)[1])+')\n'+
                'print('+repr(SCRI_DISSIPATION_FIXTURE.split('\n',1)[1])+')\n')
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

    def test_angular_dissipation_metric_adapter(self):
        self.exercise(-1,'angular_dissipation')

    def test_scri_dissipation_closure_metric_adapter(self):
        self.exercise(-1,'scri_dissipation_closure')

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
