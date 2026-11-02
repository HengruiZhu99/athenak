import copy
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import control
import worker
import native_transition_campaign as c


class NativeTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        launcher=self.root/Path(c.SCIENCE).parent;launcher.mkdir(parents=True)
        source=Path(__file__).parent
        for name in ['control.py','worker.py','storage.py','run_debug.pbs','native_transition_campaign.py']:
            shutil.copyfile(source/name,launcher/name)
        p=patch.object(control,'__file__',str(launcher/'control.py'));p.start();self.addCleanup(p.stop)
        # Synthetic tiny payloads exercise the exact layout without N16-sized test files.
        p=patch.object(c,'cells',return_value=2);p.start();self.addCleanup(p.stop)
        (self.root/'science.hpp').write_text('synthetic unit test source')
        control.atomic(self.root/c.MANIFEST,{'science.hpp':control.digest(self.root/'science.hpp')})
        frozen=dict(schema=1,source_commit=c.SOURCE_COMMIT,source_manifest=c.MANIFEST,
            source_manifest_sha256=control.digest(self.root/c.MANIFEST),binaries={})
        for v in c.VARIANTS:
            b=self.root/c.binary(v);b.parent.mkdir(parents=True,exist_ok=True);b.write_text('synthetic binary '+v)
            frozen['binaries'][c.binary(v)]=control.digest(b)
        control.atomic(self.root/c.SCIENCE,frozen)

    def checkpoint(self,variant='A',kind='puncture',time=.01,steps=20):
        ns=6*16*16*31
        h=struct.pack('<16s4i3Q2d',c.magic(variant).encode(),16,31,{'puncture':2,'minkowski':0}[kind],1,2,ns,steps,time,.0005)
        return bytearray(h+c.CONFIG+struct.pack('<d',1.)*(2*(31*2+ns)))

    def write_cp(self,data,name='input.checkpoint'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        return p,control.digest(p)

    def test_each_case_rejects_every_other_schema_and_corruption(self):
        for variant in c.VARIANTS:
            p,sha=self.write_cp(self.checkpoint(variant));self.assertEqual(c.inspect_checkpoint(p,variant,sha)['time'],.01)
            bads=[(0,'<16s',c.magic(other).encode()) for other in c.VARIANTS if other!=variant]
            bads += [(0,'<16s',b'HYP3D_MPI_SG1'),(0,'<16s',b'HYP3D_MPI_DT1'),(28,'<i',0),
                (16,'<i',24),(24,'<i',0),(32,'<Q',3),(40,'<Q',3),(48,'<Q',40),
                (56,'<d',float('nan')),(56,'<d',0),(56,'<d',4),(64,'<d',.001),
                (144,'<i',1),(152,'<d',0),(160,'<d',.04),(168,'<d',-1),
                (168+2*31*2*8,'<d',0),(176,'<d',float('nan'))]
            for offset,fmt,value in bads:
                with self.subTest(variant=variant,offset=offset,value=value):
                    data=self.checkpoint(variant);struct.pack_into(fmt,data,offset,value);p,sha=self.write_cp(data)
                    with self.assertRaises(ValueError):c.inspect_checkpoint(p,variant,sha)
            for data in [self.checkpoint(variant)[:-8],self.checkpoint(variant)+b'bad',b'truncated']:
                p,sha=self.write_cp(data)
                with self.assertRaises(ValueError):c.inspect_checkpoint(p,variant,sha)
            p,sha=self.write_cp(self.checkpoint(variant))
            with self.assertRaises(ValueError):c.inspect_checkpoint(p,variant,'f'*64)
            seed=Path(str(p)+'.horizon-seed.txt');seed.write_text('HYP3D_FASTFLOW_SEED_V1 .01 20 .005 .07\n')
            c.inspect_checkpoint(p,variant,sha)
            seed.write_text('HYP3D_FASTFLOW_SEED_V1 .02 20 .005 .07\n')
            with self.assertRaises(ValueError):c.inspect_checkpoint(p,variant,sha)
            seed.unlink()

    def test_fresh_cases_smokes_and_frozen_bindings(self):
        schemas={'A':'HYP3D_MPI_NI1','B':'HYP3D_MPI_NT1','C':'HYP3D_MPI_NI2','D':'HYP3D_MPI_NC2','E':'HYP3D_MPI_NS2'}
        for v in c.VARIANTS:
            for purpose in ['evolution','smoke']:
                made=c.make_plan(self.root,v,purpose);path=self.root/made['plan'];plan=json.loads(path.read_text())
                self.assertEqual(plan['start_verification']['schema'],schemas[v]);self.assertEqual(plan['start_verification']['time'],0)
                self.assertEqual(plan['commands'][0]['args'][5:7],['-','-'])
                binding=control.bind(self.root,made['plan'],c.MANIFEST)
                self.assertEqual(binding['inputs'],{c.SCIENCE:control.digest(self.root/c.SCIENCE)})
                self.assertIn('native_transition_campaign.py',binding['launcher'])
                control.read_bound(self.root,made['plan'])
                for index,value in [(0,'24'),(3,'20'),(4,'.001'),(5,'old-input'),(6,'old3M'),(12,'1')]:
                    bad=copy.deepcopy(plan);bad['commands'][0]['args'][index]=value
                    with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                for key,value in [('n',24),('variant','Z'),('segment',2),('purpose','unknown'),('native_exit_reserve_seconds',0)]:
                    bad=copy.deepcopy(plan);bad[key]=value
                    with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                bad=copy.deepcopy(plan);bad['commands'][0]['checks']=bad['commands'][0]['checks'][1:]
                with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                bad=copy.deepcopy(plan);bad['commands'][0]['executable']=c.binary('B' if v=='A' else 'A')
                with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                bad=copy.deepcopy(plan);bad['phase']='preflight'
                with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                bad['commands'][0]['kind']='check'
                with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
                with self.assertRaises(ValueError):c.make_plan(self.root,v,purpose,1,'input','a'*64)
                with self.assertRaises(ValueError):c.make_plan(self.root,v,purpose,2)
        made=c.make_plan(self.root,'A','evolution',2,*self.restart())
        plan=json.loads((self.root/made['plan']).read_text())
        self.assertEqual(plan['commands'][0]['inputs'],[c.SCIENCE,'input.checkpoint'])
        control.bind(self.root,made['plan'],c.MANIFEST);control.read_bound(self.root,made['plan'])
        p=self.root/c.SCIENCE;p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):control.read_bound(self.root,made['plan'])

    def restart(self):
        p,sha=self.write_cp(self.checkpoint());return str(p.relative_to(self.root)),sha

    def test_worker_applies_native_180_second_exit_reserve(self):
        made=c.make_plan(self.root,'A','evolution');relative=made['plan']
        plan=json.loads((self.root/relative).read_text());control.bind(self.root,relative,c.MANIFEST)
        launched=[]
        def execute(argv,root,log,seconds,active):
            launched.append((argv,seconds));command=active['commands'][0]
            log.write_text('Synthetic launcher argument test\n')
            r={x['path'][0]:x['value'] for x in command['checks'] if x['op']=='eq'}
            r.update(t=c.END,steps=6667,dt=c.DT,max_null_surface=0.,max_null_ghost=0.)
            control.atomic(root/command['report'],r)
            Path(str(root/command['args'][1])+'.checkpoint').write_bytes(self.checkpoint(time=c.END,steps=6667))
            return dict(returncode=0,timed_out=False,storage_error=None,elapsed_seconds=.01)
        env=dict(PBS_JOBID='123.server',OMP_NUM_THREADS='1',ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
                 ZE_ENABLE_PCI_ID_DEVICE_ORDER='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',MPIR_CVAR_ENABLE_GPU='1')
        with patch.object(worker,'allocation_node',return_value=['test-node']), \
             patch.object(Path,'home',return_value=self.root.resolve()),patch.dict(os.environ,env), \
             patch.object(worker,'probe_affinity',return_value=dict(passed=True)), \
             patch.object(worker,'run_bounded',side_effect=execute):
            self.assertEqual(worker.run(self.root.resolve(),relative),0)
        self.assertEqual(len(launched),1)
        argv,timeout=launched[0];self.assertEqual(timeout,3100)
        exe=argv.index(str(self.root.resolve()/c.binary('A')))
        self.assertEqual(argv[exe+1+7],'2920')
        self.assertEqual(worker.native_exit_reserve(dict(phase='preflight',native_exit_reserve_seconds=180)),30)

    def test_scaling_queue_keeps_one_node_and_twelve_ranks(self):
        made=c.make_plan(self.root,'A','evolution');path=self.root/made['plan'];plan=json.loads(path.read_text())
        plan['resources']['queue']='debug-scaling';control.atomic(path,plan)
        control.bind(self.root,made['plan'],c.MANIFEST)
        pbs=path.with_name('run_debug.pbs').read_text()
        self.assertIn('#PBS -q debug-scaling\n',pbs)
        self.assertIn('#PBS -l select=1\n',pbs)
        for key,value in [('queue','prod'),('nodes',5),('ranks',60),('account','other')]:
            bad=copy.deepcopy(plan);bad['resources'][key]=value
            with self.assertRaises(ValueError):control.validate_plan(self.root,bad)
        nodefile=self.root/'nodes';nodefile.write_text('node1\n')
        with patch.dict(os.environ,dict(PBS_JOBID='123.server',PBS_NODEFILE=str(nodefile),PBS_QUEUE='debug-scaling')):
            self.assertEqual(worker.allocation_node('debug-scaling'),['node1'])
            with self.assertRaises(RuntimeError):worker.allocation_node('debug')
            nodefile.write_text('node1\nnode2\n')
            with self.assertRaises(RuntimeError):worker.allocation_node('debug-scaling')

    def finish(self,variant='A',purpose='evolution',segment=1,restart=None,sha=None):
        made=c.make_plan(self.root,variant,purpose,segment,restart,sha);relative=made['plan'];path=self.root/relative
        plan=json.loads(path.read_text());binding=control.bind(self.root,relative,c.MANIFEST)
        cases=[];smoke=purpose=='smoke';state='passed' if smoke else 'checkpointed'
        for command in plan['commands']:
            prefix=self.root/command['args'][1];prefix.parent.mkdir(exist_ok=True)
            t=.001 if smoke else (.02 if restart else .01);steps=round(t/.0005)
            report={x['path'][0]:x['value'] for x in command['checks'] if x['op']=='eq'}
            report.update(status='completed' if smoke else 'checkpointed',t=t,dt=.0005,steps=steps,max_null_surface=1e-15,max_null_ghost=1e-15)
            control.atomic(self.root/command['report'],report)
            Path(str(prefix)+'.checkpoint').write_bytes(self.checkpoint(variant,command['kind'],t,steps))
            Path(str(prefix)+'.history.csv').write_text('synthetic history\n')
            artifacts={name:dict(bytes=(self.root/name).stat().st_size,sha256=control.digest(self.root/name))
                       for name in [command['report'],str(prefix.relative_to(self.root))+'.checkpoint',str(prefix.relative_to(self.root))+'.history.csv']}
            cases.append(dict(name=command['name'],status=state,timed_out=False,storage_error=None,returncode=0,artifacts=artifacts))
        worker_state=dict(job_id='123.aurora',status=state,cases=cases,plan_sha256=binding['plan_sha256'],source_manifest_sha256=binding['source_manifest_sha256'],binaries=binding['binaries'])
        control.atomic(path.with_name('worker-status.json'),worker_state)
        control.atomic(path.with_name('submission.json'),dict(job_id='123.aurora',plan=relative,plan_sha256=binding['plan_sha256'],source_manifest_sha256=binding['source_manifest_sha256']))
        control.atomic(path.with_name('collected.json'),dict(job_id='123.aurora',returncode=0,scheduler=dict(job_state='F',Exit_status=0 if smoke else 3)))
        return relative,plan,worker_state

    def test_complete_smokes_and_healthy_walltime_verification(self):
        for v in c.VARIANTS:
            for purpose in ['smoke','evolution']:
                relative,plan,w=self.finish(v,purpose)
                result=c.verify(self.root,relative);self.assertTrue(result['verified'])
                self.assertEqual(result['cases'][-1]['completed'],purpose=='smoke')
                if purpose=='evolution':
                    start=result['cases'][0]
                    nxt,_,_=self.finish(v,purpose,2,start['checkpoint'],start['sha256'])
                    self.assertGreater(c.verify(self.root,nxt)['cases'][0]['time'],start['time'])
                for report in [dict(status='checkpointed'),dict(status='completed')]:
                    checks=[dict(check=x,passed=x['path']!=['status']) for x in plan['commands'][0]['checks']]
                    self.assertEqual(worker.classify(0,False,report,checks),'checkpointed' if report['status']=='checkpointed' else 'failed')
                    checks[-1]['passed']=False;self.assertEqual(worker.classify(0,False,report,checks),'failed')

    def test_terminal_verifier_rejects_failure_corruption_and_changed_inputs(self):
        relative,plan,w=self.finish();p=self.root/relative
        for changed in [dict(w,status='failed'),dict(w,status='running'),dict(w,job_id='456.aurora'),dict(w,plan_sha256='f'*64),dict(w,cases=[])]:
            control.atomic(p.with_name('worker-status.json'),changed)
            with self.assertRaises(ValueError):c.verify(self.root,relative)
        control.atomic(p.with_name('worker-status.json'),w)
        for key,value in [('timed_out',True),('storage_error','quota'),('returncode',2),('status','failed'),('name','other')]:
            bad=copy.deepcopy(w);bad['cases'][0][key]=value;control.atomic(p.with_name('worker-status.json'),bad)
            with self.assertRaises(ValueError):c.verify(self.root,relative)
        control.atomic(p.with_name('worker-status.json'),w)
        control.atomic(p.with_name('collected.json'),dict(scheduler=dict(job_state='R')))
        with self.assertRaises(ValueError):c.verify(self.root,relative)
        control.atomic(p.with_name('collected.json'),dict(job_id='123.aurora',returncode=0,scheduler=dict(job_state='F',Exit_status=3)))
        history=self.root/(plan['commands'][0]['args'][1]+'.history.csv');saved=history.read_bytes();history.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):c.verify(self.root,relative)
        history.write_bytes(saved)
        b=self.root/c.binary('A');b.write_text('changed binary')
        with self.assertRaises(ValueError):c.verify(self.root,relative)

    def test_actual_ns2_native_smoke_checkpoint_when_available(self):
        # An actual production artifact, never fabricated into an evolution claim.
        root=Path(__file__).resolve().parents[3]
        path=root/'output/native-shared-transition/scri-first6-smoke/puncture/state.checkpoint'
        if not path.exists():self.skipTest('Optional preserved actual NS2 smoke unavailable')
        self.cells_patch_stop_for_real(path)

    def cells_patch_stop_for_real(self,path):
        with patch.object(c,'cells',return_value=12*16**3+8*19**3+6*19*22**2):
            r=c.inspect_checkpoint(path,'E',control.digest(path),end=.001)
        self.assertEqual(r['steps'],2);self.assertEqual(r['time'],.001)


if __name__=='__main__':unittest.main()
