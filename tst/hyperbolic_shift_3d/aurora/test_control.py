#!/usr/bin/env python3
"""Offline scheduler-safety tests. Never invoke SSH, qsub, qstat, or mpiexec."""
import copy
import json
import os
from pathlib import Path
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


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
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

    def test_every_bound_input_is_immutable(self):
        self.bound()
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


if __name__ == '__main__':
    unittest.main()
