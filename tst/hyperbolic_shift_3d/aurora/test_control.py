#!/usr/bin/env python3
"""Offline scheduler-safety tests. Never invoke SSH, qsub, qstat, or mpiexec."""
import copy
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import control
import worker


def example():
    return dict(schema=1, reviewed=True, phase='preflight', n=16, low_resolution_n=16,
                resources=dict(control.FIXED, walltime='00:05:00'), worker_seconds=210,
                output_dir='runs/one/output', storage=dict(limit_bytes=5_000_000_000,planned_new_bytes=1000000,atomic_checkpoint_reserve_bytes=1000000), commands=[dict(name='minkowski', kind='minkowski',
                executable='build/evolve', args=['16', 'runs/one/output/minkowski', 'minkowski',
                '0.01', '0.0001', 'inputs/initial', '-', '@WALL_SECONDS@'],
                timeout_seconds=120, report='runs/one/output/minkowski.report.json',
                inputs=['inputs/initial'], checks=[dict(path=['status'], op='eq', value='completed')])])


def resolution_example():
    plan = example()
    plan.update(phase='resolution_control', n=32, admission='admission.json', worker_seconds=3510)
    plan['resources']['walltime'] = '01:00:00'
    template = plan['commands'][0]
    plan['commands'] = []
    for n in [32, 24, 16]:
        command = copy.deepcopy(template)
        prefix = f'runs/one/output/minkowski-n{n}'
        command.update(name=f'minkowski-n{n}', n=n, continue_after_failure=True,
                       report=prefix+'.report.json', artifacts=[prefix+'.checkpoint'])
        command['args'][0:2] = [str(n), prefix]
        command['checks'].append(dict(path=['H_L2'], op='le', value=1e-7))
        plan['commands'].append(command)
    return plan


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.launcher_relative = Path('source-v4/tst/hyperbolic_shift_3d/aurora')
        launcher = self.root/self.launcher_relative
        launcher.mkdir(parents=True)
        for name in ['control.py', 'worker.py', 'storage.py', 'preflight.py',
                     'run_debug.pbs']:
            shutil.copyfile(Path(__file__).resolve().parent/name, launcher/name)
        launcher_file = patch.object(control, '__file__', str(launcher/'control.py'))
        launcher_file.start()
        self.addCleanup(launcher_file.stop)
        for path, content in [('build/evolve', 'executable'), ('source/file', 'science'),
                              ('inputs/initial', 'initial')]:
            target = self.root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.relative = 'runs/one/plan.json'
        target = self.root/self.relative
        target.parent.mkdir(parents=True)
        control.atomic(target, example())
        control.atomic(self.root/'source-manifest.json', {'source/file':control.digest(self.root/'source/file')})

    def bound(self):
        return control.bind(self.root, self.relative, 'source-manifest.json')

    def test_resource_policy(self):
        for key, wrong in [('queue','capacity'), ('nodes',2), ('ranks',1), ('ranks_per_node',6)]:
            plan = example()
            plan['resources'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)

    def test_low_resolution_and_review_required(self):
        for key, wrong in [('n',32), ('reviewed',False)]:
            plan = example()
            plan[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)
        for value in ['00:04:59','01:00:01','00:60:00']:
            with self.assertRaises(ValueError):
                control.wall_seconds(value)

    def test_actual_arguments_must_match_reviewed_metadata(self):
        for index, wrong in [(0,'32'), (1,'another/output'), (2,'puncture'),
                             (3,'nan'), (4,'0'), (5,'unbound/file'), (7,'5000')]:
            plan = example()
            plan['commands'][0]['args'][index] = wrong
            with self.subTest(index=index), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)

    def test_generated_inputs_require_prior_artifact(self):
        plan = example()
        first = plan['commands'][0]
        first['artifacts'] = ['runs/one/output/minkowski.checkpoint']
        second = copy.deepcopy(first)
        second.update(name='restart', report='runs/one/output/restart.report.json',
                      generated_inputs=['runs/one/output/minkowski.checkpoint'])
        second['args'][1] = 'runs/one/output/restart'
        second['args'][6] = 'runs/one/output/minkowski.checkpoint'
        plan['commands'].append(second)
        control.validate_plan(self.root, plan)
        first['artifacts'] = []
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)

    def test_optional_horizon_angular_arguments_preserve_old_plans(self):
        for suffix in [[], ['.05'], ['.05', '.001'], ['.05', '.001', '6'],
                       ['.05', '.001', '12', '24']]:
            plan = example()
            plan['commands'][0]['args'].extend(suffix)
            with self.subTest(suffix=suffix):
                control.validate_plan(self.root, plan)

    def test_invalid_optional_horizon_arguments_rejected(self):
        for suffix in [['nan'], ['0'], ['.05', '-1'], ['.05', 'inf'],
                       ['.05', '.001', '0', '24'], ['.05', '.001', '12'],
                       ['.05', '.001', '12', '12'], ['.05', '.001', '12.0', '24'],
                       ['.05', '.001', '12x', '24'], ['.05', '.001', '12', '32768'],
                       ['.05', '.001', '12', '24', 'unused']]:
            plan = example()
            plan['commands'][0]['args'].extend(suffix)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)

    def test_v4_optional_epsilon_and_old_target_argument_limit(self):
        for suffix,epsilon in [([],2.0),(['.05','.001','6','12'],2.0),
                               (['.05','.001','6','12','0'],0.0),
                               (['.05','.001','6','12','1'],1.0),
                               (['.05','.001','6','12','2.5'],2.5)]:
            plan=example();command=plan['commands'][0]
            command['executable']='build/'+control.DISSIPATIVE_TARGET
            command['args'].extend(suffix)
            control.validate_plan(self.root,plan)
            self.assertEqual(control.dissipation_config(command),dict(
                executable=command['executable'],checkpoint_schema='HYP3D_MPI_V4',
                scri_dissipation_epsilon=epsilon,scri_dissipation_degree=7,checkpoint_header_bytes=88))
        for executable in ['build/evolve','build/hyperbolic_shift_3d_mpi_projected_angular',
                           'build/'+control.DISSIPATIVE_TARGET+'-other']:
            plan=example();command=plan['commands'][0];command['executable']=executable
            command['args'].extend(['.05','.001','6','12','0'])
            with self.subTest(executable=executable),self.assertRaises(ValueError):
                control.validate_plan(self.root,plan)

    def test_v5_plan_and_exact_normal_admission(self):
        plan=example();plan.update(phase='stability',admission='admission.json')
        command=plan['commands'][0];command['executable']='build/'+control.SAMPLED_NORMAL_TARGET
        control.validate_plan(self.root,plan)
        config=control.normal_derivative_config(command)
        self.assertEqual(config['checkpoint_schema'],'HYP3D_MPI_V5')
        self.assertEqual(config['checkpoint_header_bytes'],104)
        self.assertEqual(config['scri_dissipation_epsilon'],2.)
        self.assertEqual(config['scri_normal_derivative'],'sampled_native_numerator6')
        command['args'].extend(['.05','.001','6','12','2'])
        self.assertEqual(config,control.normal_derivative_config(command))
        source='a'*64;binary='b'*64
        admission=dict(passed=True,phase='preflight',source_manifest_sha256=source,low_resolution_n=16,
                       binaries={command['executable']:binary},gpu_cpu_parity=True,restart_equivalence=True,
                       scri_dissipation_configs=[dict(config,checkpoint_schema='HYP3D_MPI_V4',checkpoint_header_bytes=88)])
        control.atomic(self.root/'admission.json',admission)
        with self.assertRaisesRegex(ValueError,'V5 admission must explicitly'):
            control.validate_admission(self.root,plan,source,{command['executable']:binary})
        admission['normal_derivative_configs']=[config];control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for key,value in [('checkpoint_schema','HYP3D_MPI_V4'),('checkpoint_header_bytes',88),
                          ('scri_normal_derivative','analytic_jets'),('scri_normal_radial_nodes',6.0),
                          ('scri_normal_bulk_derivative_order',True),('scri_dissipation_epsilon',1),
                          ('scri_dissipation_degree',5)]:
            admission['normal_derivative_configs']=[dict(config,**{key:value})]
            control.atomic(self.root/'admission.json',admission)
            with self.subTest(key=key),self.assertRaises(ValueError):
                control.validate_admission(self.root,plan,source,{command['executable']:binary})
        admission['normal_derivative_configs']=[config];control.atomic(self.root/'admission.json',admission)
        for epsilon in ['0','1','2.0000000000000004']:
            command['args'][12]=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaisesRegex(ValueError,'exact requested V5'):
                control.validate_admission(self.root,plan,source,{command['executable']:binary})

    def test_v5_rejects_invalid_epsilon_extra_args_and_failure_continuation(self):
        for epsilon in ['nan','inf','-1','invalid']:
            plan=example();c=plan['commands'][0];c['executable']='build/'+control.SAMPLED_NORMAL_TARGET
            c['args'].extend(['.05','.001','6','12',epsilon])
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=example();c=plan['commands'][0];c['executable']='build/'+control.SAMPLED_NORMAL_TARGET
        c['args'].extend(['.05','.001','6','12','2','1'])
        with self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=resolution_example()
        for c in plan['commands']:c['executable']='build/'+control.SAMPLED_NORMAL_TARGET
        with self.assertRaisesRegex(ValueError,'does not inherit'):control.validate_plan(self.root,plan)

    def test_v6_plan_and_distinct_admission(self):
        plan=example();plan.update(phase='stability',admission='admission.json')
        command=plan['commands'][0];command['executable']='build/'+control.DONOR_SEVEN_TARGET
        command['args'].extend(['.05','.001','6','12','2'])
        control.validate_plan(self.root,plan)
        config=control.normal_derivative_config(command)
        self.assertEqual(config['checkpoint_schema'],'HYP3D_MPI_V6')
        self.assertEqual(config['scri_normal_radial_nodes'],7)
        self.assertEqual(config['scri_radial_closure'],'donor_seven_last3')
        self.assertEqual(config['scri_radial_bulk_rows'],3)
        source='a'*64;binary='b'*64
        admission=dict(passed=True,phase='preflight',source_manifest_sha256=source,low_resolution_n=16,
                       binaries={command['executable']:binary},gpu_cpu_parity=True,restart_equivalence=True,
                       normal_derivative_configs=[config])
        control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for key,value in [('checkpoint_schema','HYP3D_MPI_V5'),('scri_normal_radial_nodes',6),
                          ('scri_normal_derivative','sampled_native_numerator6'),
                          ('scri_radial_closure','ghost_lift'),('scri_radial_bulk_rows',3.0),
                          ('scri_radial_bulk_rows',True),('scri_radial_bulk_rows',None)]:
            admission['normal_derivative_configs']=[dict(config,**{key:value})]
            control.atomic(self.root/'admission.json',admission)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for epsilon in ['nan','inf','-1','invalid']:
            command['args'][12]=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        command['args'][12]='2';command['args'].append('1')
        with self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=resolution_example()
        for command in plan['commands']:command['executable']='build/'+control.DONOR_SEVEN_TARGET
        with self.assertRaisesRegex(ValueError,'does not inherit'):control.validate_plan(self.root,plan)

    def test_v7_plan_and_distinct_admission(self):
        plan=example();plan.update(phase='stability',admission='admission.json')
        command=plan['commands'][0];command['executable']='build/'+control.INNER_TRANSITION_TARGET
        command['args'].extend(['.05','.001','6','12','2'])
        control.validate_plan(self.root,plan)
        config=control.normal_derivative_config(command)
        self.assertEqual(config['checkpoint_schema'],'HYP3D_MPI_V7')
        self.assertEqual(config['scri_normal_radial_nodes'],7)
        self.assertEqual(config['scri_radial_closure'],'donor_seven_last3')
        self.assertEqual(config['scri_radial_bulk_rows'],3)
        self.assertEqual(config['interpatch_interpolation'],'logical_inner_transition')
        source='a'*64;binary='b'*64
        admission=dict(passed=True,phase='preflight',source_manifest_sha256=source,low_resolution_n=16,
                       binaries={command['executable']:binary},gpu_cpu_parity=True,restart_equivalence=True,
                       normal_derivative_configs=[config])
        control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for key,value in [('interpatch_interpolation',None),('interpatch_interpolation','logical_all'),('checkpoint_schema','HYP3D_MPI_V6'),('scri_normal_radial_nodes',6),
                          ('scri_normal_derivative','sampled_native_numerator6'),
                          ('scri_radial_closure','ghost_lift'),('scri_radial_bulk_rows',3.0),
                          ('scri_radial_bulk_rows',True),('scri_radial_bulk_rows',None)]:
            admission['normal_derivative_configs']=[dict(config,**{key:value})]
            control.atomic(self.root/'admission.json',admission)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for epsilon in ['nan','inf','-1','invalid']:
            command['args'][12]=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        command['args'][12]='2';command['args'].append('1')
        with self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=resolution_example()
        for command in plan['commands']:command['executable']='build/'+control.INNER_TRANSITION_TARGET
        with self.assertRaisesRegex(ValueError,'does not inherit'):control.validate_plan(self.root,plan)

    def test_v8_requires_distinct_mesh_admission(self):
        plan=example();plan.update(phase='stability',admission='admission.json')
        command=plan['commands'][0];command['executable']='build/'+control.CORE2_TARGET
        command['args'].extend(['.05','.001','6','12','2'])
        control.validate_plan(self.root,plan)
        config=control.normal_derivative_config(command)
        self.assertEqual(config['checkpoint_schema'],'HYP3D_MPI_V8')
        self.assertEqual(config['scri_normal_radial_nodes'],7)
        self.assertEqual(config['scri_radial_closure'],'donor_seven_last3')
        self.assertEqual(config['scri_radial_bulk_rows'],3)
        self.assertEqual(config['interpatch_interpolation'],'logical_inner_transition')
        source='a'*64;binary='b'*64
        admission=dict(passed=True,phase='preflight',source_manifest_sha256=source,low_resolution_n=16,
                       binaries={command['executable']:binary},gpu_cpu_parity=True,restart_equivalence=True,
                       normal_derivative_configs=[config])
        control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for key,value in [('mesh_topology',None),('mesh_blocks',19),('mesh_blocks',26.0),('cartesian_core_subdivisions',1),('cartesian_core_subdivisions',2.0),('checkpoint_header_bytes',104),('checkpoint_schema','HYP3D_MPI_V7'),('interpatch_interpolation',None),('interpatch_interpolation','logical_all'),('checkpoint_schema','HYP3D_MPI_V6'),('scri_normal_radial_nodes',6),
                          ('scri_normal_derivative','sampled_native_numerator6'),
                          ('scri_radial_closure','ghost_lift'),('scri_radial_bulk_rows',3.0),
                          ('scri_radial_bulk_rows',True),('scri_radial_bulk_rows',None)]:
            admission['normal_derivative_configs']=[dict(config,**{key:value})]
            control.atomic(self.root/'admission.json',admission)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                control.validate_admission(self.root,plan,source,{command['executable']:binary})
        for epsilon in ['nan','inf','-1','invalid']:
            command['args'][12]=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        command['args'][12]='2';command['args'].append('1')
        with self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=resolution_example()
        for command in plan['commands']:command['executable']='build/'+control.CORE2_TARGET
        with self.assertRaisesRegex(ValueError,'does not inherit'):control.validate_plan(self.root,plan)

    def test_v4_invalid_epsilon_and_failure_continuation_rejected(self):
        for epsilon in ['-1','nan','inf','-inf','invalid','']:
            plan=example();command=plan['commands'][0]
            command['executable']='build/'+control.DISSIPATIVE_TARGET
            command['args'].extend(['.05','.001','6','12',epsilon])
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):
                control.validate_plan(self.root,plan)
        plan=example();command=plan['commands'][0]
        command['executable']='build/'+control.DISSIPATIVE_TARGET
        command['args'].extend(['.05','.001','6','12','2','extra'])
        with self.assertRaises(ValueError):control.validate_plan(self.root,plan)
        plan=resolution_example()
        for command in plan['commands']:command['executable']='build/'+control.DISSIPATIVE_TARGET
        with self.assertRaisesRegex(ValueError,'does not inherit'):
            control.validate_plan(self.root,plan)

    def test_v4_admission_binds_exact_epsilon_including_default(self):
        plan=example();plan.update(phase='stability',admission='admission.json')
        command=plan['commands'][0];command['executable']='build/'+control.DISSIPATIVE_TARGET
        hashes={command['executable']:'binary-sha'}
        admission=dict(passed=True,phase='preflight',source_manifest_sha256='source-sha',
                       low_resolution_n=16,binaries=hashes,gpu_cpu_parity=True,restart_equivalence=True)
        control.atomic(self.root/'admission.json',admission)
        with self.assertRaisesRegex(ValueError,'explicitly bind'):
            control.validate_admission(self.root,plan,'source-sha',hashes)
        config=control.dissipation_config(command)
        admission['scri_dissipation_configs']=[config]
        control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,'source-sha',hashes)
        command['args'].extend(['.05','.001','6','12','2'])
        control.validate_admission(self.root,plan,'source-sha',hashes)
        for epsilon in ['0','1','2.0000000000000004']:
            command['args'][12]=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaisesRegex(ValueError,'exact requested'):
                control.validate_admission(self.root,plan,'source-sha',hashes)
        admission['scri_dissipation_configs']=[dict(config,scri_dissipation_epsilon=x) for x in [0,1,2]]
        control.atomic(self.root/'admission.json',admission)
        for epsilon in ['0','1','2']:
            command['args'][12]=epsilon
            control.validate_admission(self.root,plan,'source-sha',hashes)
        for key,value in [('scri_dissipation_degree',0),('scri_dissipation_degree',7.0),
                          ('checkpoint_header_bytes',72),('checkpoint_schema','HYP3D_MPI_V3'),
                          ('scri_dissipation_epsilon',True),('executable','build/another')]:
            admission['scri_dissipation_configs']=[dict(config,**{key:value})]
            control.atomic(self.root/'admission.json',admission)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                control.validate_admission(self.root,plan,'source-sha',hashes)

    def test_every_bound_input_is_immutable(self):
        self.bound()
        pbs = (self.root/'runs/one/run_debug.pbs').read_text()
        self.assertIn(f'exec /usr/bin/python3.11 "{self.launcher_relative}/worker.py"', pbs)
        self.assertNotIn('@WORKER_PATH@', pbs)
        for name in ['build/evolve','source/file','inputs/initial','runs/one/plan.json',
                     'runs/one/run_debug.pbs']:
            target = self.root/name
            before = target.read_bytes()
            target.write_bytes(before+b' ')
            with self.subTest(name=name), self.assertRaises(ValueError):
                control.read_bound(self.root, self.relative)
            target.write_bytes(before)
        control.read_bound(self.root, self.relative)

    def test_rebinding_rejected(self):
        self.bound()
        with self.assertRaises(ValueError):
            self.bound()

    def test_ambiguous_qsub_never_repeated(self):
        self.bound()
        empty = dict(jobs={}, blocking=[])
        result = subprocess.CompletedProcess([], 0, 'connection lost after submission\n', '')
        with patch.object(control, 'scheduler_snapshot', return_value=empty), \
             patch.object(control.subprocess, 'run', return_value=result) as launch:
            with self.assertRaises(RuntimeError):
                control.submit(self.root, self.relative)
            self.assertTrue((self.root/'.hyperbolic-debug-allocation-lock').is_dir())
            self.assertTrue((self.root/'runs/one/submission-pending.json').exists())
            with self.assertRaises(RuntimeError):
                control.submit(self.root, self.relative)
            self.assertEqual(launch.call_count, 1)

    def test_global_lock_prevents_cross_plan_submission(self):
        self.bound()
        (self.root/'.hyperbolic-debug-allocation-lock').mkdir()
        with patch.object(control, 'scheduler_snapshot', return_value=dict(jobs={},blocking=[])), \
             patch.object(control.subprocess, 'run') as launch:
            with self.assertRaises(FileExistsError):
                control.submit(self.root, self.relative)
            launch.assert_not_called()

    def test_active_or_held_own_task_blocks(self):
        self.bound()
        with patch.object(control, 'scheduler_snapshot', return_value=dict(jobs={},blocking=['123.server'])), \
             patch.object(control.subprocess, 'run') as launch:
            with self.assertRaises(RuntimeError):
                control.submit(self.root, self.relative)
            launch.assert_not_called()

    def test_scheduler_selection_is_exact_task_name(self):
        result = subprocess.CompletedProcess([], 153, '', '')
        with patch.object(control.subprocess, 'run', return_value=result) as call:
            self.assertEqual(control.scheduler_snapshot('test')['blocking'], [])
            self.assertEqual(call.call_args.args[0], ['qselect','-u','test','-N',control.JOB_NAME])

    def test_terminal_proof_required_for_release(self):
        control.atomic(self.root/'runs/one/submission.json', dict(job_id='123.server'))
        result = subprocess.CompletedProcess([], 0, json.dumps({'Jobs':{'123.server':{'job_state':'R'}}}), '')
        with patch.object(control.subprocess, 'run', return_value=result), self.assertRaises(RuntimeError):
            control.collect(self.root, self.relative, release=True)

    def test_finite_checks_and_missing_reports(self):
        for value in [float('nan'), float('inf'), -float('inf'), None, True]:
            result = control.evaluate_check({'x':value}, dict(path=['x'], op='finite'))
            self.assertFalse(result['passed'])
            json.dumps(result, allow_nan=False)
        self.assertFalse(control.evaluate_check({}, dict(path=['x'],op='eq',value=0))['passed'])

    def test_checkpoint_is_not_scientific_pass(self):
        self.assertEqual(worker.classify(0,False,{'status':'checkpointed'},[{'passed':True}]), 'checkpointed')
        self.assertEqual(worker.classify(0,False,{'status':'checkpointed'},[{'passed':False}]), 'failed')
        self.assertEqual(worker.classify(0,False,{'status':'checkpointed'},
                         [{'passed':False,'check':{'path':['status']}}]), 'checkpointed')
        self.assertEqual(worker.classify(0,False,{'status':'completed'},[{'passed':False}]), 'failed')
        self.assertEqual(worker.classify(0,True,{'status':'completed'},[{'passed':True}]), 'external_timeout')
        self.assertEqual(worker.classify(0,False,{'status':'completed'},[{'passed':True}]), 'passed')

    def test_convergence_requires_stability_evidence(self):
        plan = example()
        plan.update(phase='convergence',n=24,admission='admission.json')
        admission = dict(passed=True,phase='stability',source_manifest_sha256='sha',
                         low_resolution_n=16,binaries={},minkowski_stable=True,pulse_stable=True,
                         puncture_stable=True,puncture_completed_t_over_M=99.9)
        control.atomic(self.root/'admission.json',admission)
        with self.assertRaises(ValueError):
            control.validate_admission(self.root,plan,'sha',{})
        admission['puncture_completed_t_over_M']=100
        control.atomic(self.root/'admission.json',admission)
        control.validate_admission(self.root,plan,'sha',{})

    def test_resolution_controls_allow_explicit_grids_in_reviewed_range(self):
        plan = resolution_example()
        control.validate_plan(self.root, plan)
        # The bounds come from the plan and its admission, not hardcoded grids.
        plan['low_resolution_n'], plan['n'] = 8, 40
        for command, n in zip(plan['commands'], [40, 20, 8]):
            command['n'], command['args'][0] = n, str(n)
        control.validate_plan(self.root, plan)

    def test_resolution_controls_reject_ambiguous_grid_or_binary(self):
        for value in [None, 14, 33, 34, True]:
            plan = resolution_example()
            command = plan['commands'][0]
            if value is None:
                command.pop('n')
            else:
                command['n'] = value
                command['args'][0] = str(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)
        plan = resolution_example()
        plan['commands'][0]['args'][0] = '24'
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)
        plan = resolution_example()
        plan['commands'][0]['executable'] = 'build/another-evolution'
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)

    def test_continuation_requires_resolution_phase_evolution_and_boolean(self):
        for phase in ['preflight', 'stability', 'convergence']:
            plan = example()
            plan['phase'] = phase
            plan['commands'][0]['continue_after_failure'] = True
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                control.validate_plan(self.root, plan)
        plan = resolution_example()
        plan['commands'][0]['continue_after_failure'] = 'true'
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)
        plan = resolution_example()
        plan['commands'][0]['kind'] = 'check'
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)
        plan = resolution_example()
        plan['commands'][0]['artifacts'] = []
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)
        plan = example()
        plan['commands'][0]['n'] = 24
        plan['commands'][0]['args'][0] = '24'
        with self.assertRaises(ValueError):
            control.validate_plan(self.root, plan)

    def resolution_admission(self):
        return dict(passed=True, phase='preflight', low_resolution_n=16,
                    source_manifest_sha256=control.digest(self.root/'source-manifest.json'),
                    binaries={'build/evolve': control.digest(self.root/'build/evolve')},
                    gpu_cpu_parity=True, restart_equivalence=True)

    def test_resolution_requires_exact_preflight_source_and_executable(self):
        plan = resolution_example()
        admission = self.resolution_admission()
        control.atomic(self.root/'admission.json', admission)
        control.validate_admission(self.root, plan, admission['source_manifest_sha256'], admission['binaries'])
        for key, value in [('phase', 'stability'), ('source_manifest_sha256', 'changed'),
                           ('binaries', {'build/evolve': 'changed'}), ('low_resolution_n', 8),
                           ('gpu_cpu_parity', False), ('restart_equivalence', False)]:
            changed = dict(admission, **{key: value})
            control.atomic(self.root/'admission.json', changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_admission(self.root, plan, admission['source_manifest_sha256'], admission['binaries'])

    def run_resolution_worker(self, outcomes, continue_first=True):
        plan = resolution_example()
        plan['commands'][0]['continue_after_failure'] = continue_first
        control.atomic(self.root/self.relative, plan)
        control.atomic(self.root/'admission.json', self.resolution_admission())
        self.bound()
        admission_before = (self.root/'admission.json').read_bytes()
        launched = []

        def simulated_evolution(argv, root, log, seconds, active_plan):
            command = active_plan['commands'][len(launched)]
            mode = outcomes[len(launched)]
            launched.append(command['name'])
            log.write_text('Offline simulated evolution result\n')
            native_status = ('completed' if mode in ['completed', 'completed_checks_fail']
                             else 'checkpointed' if mode == 'checkpointed' else 'failed')
            admitted = native_status != 'failed' or mode == 'wrong_admission'
            accepted_time = .01 if native_status == 'completed' else .0001
            steps = 100 if native_status == 'completed' else 1
            reason = 'history append failed' if mode == 'io_error' else 'nonpositive interpolated scri geometry'
            report = dict(status=native_status, error=reason if native_status == 'failed' else '',
                          kind='minkowski', n=command['n'], cells=1, backend='SYCL', mpi_size=12,
                          checkpoint_written=True, checkpoint_admitted=admitted,
                          t=accepted_time, dt=.0001, steps=steps,
                          H_L2=1 if mode == 'completed_checks_fail' else 0)
            report_path = root/command['report']
            if mode == 'bad_report':
                report_path.write_text('{')
            else:
                control.atomic(report_path, report)
            checkpoint = root/(command['args'][1]+'.checkpoint')
            header = struct.pack('<16s4i3Q2d', b'HYP3D_MPI_V1'.ljust(16,b'\0'),
                                 command['n'],31,0,int(admitted or mode == 'header_admitted'),
                                 1,1,steps,accepted_time,.0001)
            checkpoint.write_bytes(header+struct.pack('<64d',*([0.0]*64)))
            if mode == 'changed_source':
                (root/'source/file').write_text('changed source')
            if mode == 'changed_binary':
                (root/'build/evolve').write_text('changed executable')
            return dict(returncode=0 if native_status != 'failed' else 1 if mode == 'launcher_error' else 2,
                        timed_out=mode == 'timeout', storage_error='reserve exhausted' if mode == 'storage_error' else None,
                        elapsed_seconds=.01)

        environment = dict(PBS_JOBID='123.server', OMP_NUM_THREADS='1', ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
                           ZE_ENABLE_PCI_ID_DEVICE_ORDER='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                           MPIR_CVAR_ENABLE_GPU='1')
        with patch.object(worker, 'allocation_node', return_value=['test-node']), \
             patch.object(worker.Path, 'home', return_value=self.root), \
             patch.dict(os.environ, environment), \
             patch.object(worker, 'probe_affinity', return_value=dict(passed=True)), \
             patch.object(worker, 'run_bounded', side_effect=simulated_evolution):
            code = worker.run(self.root, self.relative)
        status = json.loads((self.root/'runs/one/worker-status.json').read_text())
        self.assertEqual((self.root/'admission.json').read_bytes(), admission_before)
        self.assertFalse(status['stability_admission_eligible'])
        self.assertEqual(status['phase'], 'resolution_control')
        return code, status, launched

    def test_numerical_failure_continues_but_overall_result_stays_failed(self):
        code, status, launched = self.run_resolution_worker(['failed','completed','completed'])
        self.assertEqual(code, 1)
        self.assertEqual(launched, ['minkowski-n32','minkowski-n24','minkowski-n16'])
        self.assertEqual(status['status'], 'failed')
        self.assertEqual([c['status'] for c in status['cases']], ['failed','passed','passed'])
        self.assertTrue(status['cases'][0]['continued_after_failure'])
        self.assertFalse(status['cases'][0]['continuation_evidence']['checkpoint_admitted'])
        self.assertEqual(status['continued_failures'], ['minkowski-n32'])

    def test_all_resolution_controls_can_pass_without_stability_admission(self):
        code, status, launched = self.run_resolution_worker(['completed']*3)
        self.assertEqual(code, 0)
        self.assertEqual(status['status'], 'passed')
        self.assertEqual(len(launched), 3)
        self.assertEqual(status['continued_failures'], [])

    def test_unreviewed_failure_continuation_stops(self):
        code, status, launched = self.run_resolution_worker(['failed'], continue_first=False)
        self.assertEqual(code, 1)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(len(launched), 1)

    def test_completed_evolution_failing_checks_does_not_continue(self):
        code, status, launched = self.run_resolution_worker(['completed_checks_fail'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['cases'][0]['report_status'], 'completed')

    def test_timeout_stops_even_with_valid_numerical_failure_record(self):
        code, status, launched = self.run_resolution_worker(['timeout'])
        self.assertEqual(code, 1)
        self.assertEqual(status['status'], 'external_timeout')
        self.assertEqual(len(launched), 1)

    def test_storage_error_stops_continuation(self):
        code, status, launched = self.run_resolution_worker(['storage_error'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)

    def test_report_io_error_stops_continuation(self):
        code, status, launched = self.run_resolution_worker(['io_error'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)

    def test_invalid_report_stops_continuation(self):
        code, status, launched = self.run_resolution_worker(['bad_report'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)

    def test_launcher_failure_stops_continuation(self):
        code, status, launched = self.run_resolution_worker(['launcher_error'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)

    def test_incorrectly_admitted_failure_checkpoint_stops(self):
        code, status, launched = self.run_resolution_worker(['header_admitted'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)
        self.assertIn('incorrectly admitted', status['reason'])

    def test_report_admits_failure_checkpoint_stops(self):
        code, status, launched = self.run_resolution_worker(['wrong_admission'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)

    def test_source_hash_change_stops_before_next_resolution(self):
        code, status, launched = self.run_resolution_worker(['changed_source'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)
        self.assertIn('Source changed', status['reason'])

    def test_binary_hash_change_stops_before_next_resolution(self):
        code, status, launched = self.run_resolution_worker(['changed_binary'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 1)
        self.assertIn('Executable changed', status['reason'])

    def test_checkpoint_after_prior_failure_stops_and_retains_failed_overall(self):
        code, status, launched = self.run_resolution_worker(['failed','checkpointed'])
        self.assertEqual(code, 1)
        self.assertEqual(len(launched), 2)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['cases'][1]['status'], 'checkpointed')
        self.assertEqual(status['stop_reason_status'], 'checkpointed')

    def test_checkpoint_without_prior_failure_retains_existing_semantics(self):
        code, status, launched = self.run_resolution_worker(['checkpointed'])
        self.assertEqual(code, 3)
        self.assertEqual(len(launched), 1)
        self.assertEqual(status['status'], 'checkpointed')


if __name__ == '__main__':
    unittest.main()
