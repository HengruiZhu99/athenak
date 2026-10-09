#!/usr/bin/env python3
"""Independent compact saved-check association; no scientific imports."""
import argparse
import ast
import collections
from decimal import Decimal
import hashlib
import json
import pathlib
import sys
import time
import traceback

LIMIT = 1 << 20

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(LIMIT),b''): h.update(b)
    return h.hexdigest()

def load(path):
    p=pathlib.Path(path)
    if p.stat().st_size>LIMIT: raise ValueError('Compact JSON limit exceeded: '+str(p))
    if p.suffix!='.json': raise ValueError('Only compact JSON may be decoded')
    return json.loads(p.read_text())

def write(path,value):
    pathlib.Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def merge(pins,path,digest):
    path=str(pathlib.Path(path).resolve())
    if path in pins and pins[path]!=digest: raise ValueError('Pin conflict: '+path)
    pins[path]=digest

def guard(pins):
    for p,d in pins.items():
        if sha(p)!=d: raise ValueError('Changed input: '+p)

def validate(rows):
    assert len(rows)==len({r['name'] for r in rows})
    for r in rows:
        e,t=Decimal(r['error']),Decimal(r['tolerance'])
        assert e.is_finite() and t.is_finite() and e>=0 and t>=0
        assert type(r['passed']) is bool and r['passed']==(e<=t)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',required=True);ap.add_argument('--recipe-sha256',required=True);args=ap.parse_args()
    assert sha(args.recipe)==args.recipe_sha256
    cfg=load(args.recipe);out=pathlib.Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();receipt={'passed':False,'returncode':1,'completed':False,'scope':'stdlib saved checks/source/provenance only; no numerical or jet recomputation'}
    pins=dict(cfg['pins']);merge(pins,args.recipe,args.recipe_sha256);merge(pins,__file__,cfg['source_sha256'])
    try:
        owner=pathlib.Path(cfg['owner']);root=pathlib.Path(cfg['root']);r=load(owner/'control-recipe.json');si=load(owner/'source-index.json')
        for row in si['files']:
            p=owner/row['path'];assert p.stat().st_size==row['bytes'];merge(pins,p,row['sha256'])
        for p,d in si['protected_inputs'].items(): merge(pins,p,d)
        for stage,dir in [('qualification',owner/'saved-qualification001'),('controls',pathlib.Path(r['fresh_control_output']))]:
            child=load(dir/'receipt.json');outer=load(root/(stage+'-invocation001')/'receipt.json')
            assert child['source_before']==child['source_after'] and child['sources_unchanged'] is True
            for p,d in child['source_before'].items():merge(pins,p,d)
            for p,d in child['output_pins'].items():merge(pins,p,d)
            assert outer['completed'] is True and outer['accepted_stage'] is True and outer['returncode']==0 and outer['inputs_unchanged'] is True
            assert outer['original_full_gate_remains_failed'] is True and outer['stage']==stage
            assert outer['child_receipt_sha256']==sha(dir/'receipt.json')
            for item in outer['output_inventory']:
                assert pathlib.Path(item['path']).stat().st_size==item['bytes'];merge(pins,item['path'],item['sha256'])
            assert (root/(stage+'-invocation001')/'stderr.log').stat().st_size==0
        guard(pins);write(out/'pins-before.json',pins)
        old=load(r['original_full_receipt']);original=load(r['original_checks']);retained=load(owner/'saved-qualification001/retained-saved-checks.json')
        new=load(pathlib.Path(r['fresh_control_output'])/'checks.json');result=load(pathlib.Path(r['fresh_control_output'])/'result.json')
        qual=load(owner/'saved-qualification001/report.json');validate(original);validate(retained);validate(new)
        assert len(original)==2360 and len([c for c in original if not c['passed']])==156
        assert old['passed_analytic_scalar_derivative_gate'] is False and old['failed_checks']==[c for c in original if not c['passed']]
        assert old['source_before']==old['source_after'] and old['sources_unchanged'] is True
        for p,d in old['source_before'].items():merge(pins,p,d)
        is_control=lambda c:c['name'].startswith(('control/','exact/','wave/control/'))
        assert retained==[c for c in original if not is_control(c)] and len(retained)==1480 and all(c['passed'] for c in retained)
        assert len(new)==880 and all(c['passed'] for c in new)
        assert {c['name'] for c in new}=={c['name'] for c in original if is_control(c)}
        settings=r['scientific_settings'];expected=set()
        for dps in settings['precisions']:
            for level in settings['levels']:
                for c in settings['controls']:
                    for boost in settings['control_boosts']:
                        pre='control/%s/%s/%s/%s/'%(dps,level['name'],c['name'],boost)
                        expected.update(pre+k for k in ['root','null','first_graph','second_graph','factor_quotient','K_jet','D_source_jet','Lorentz_matrix'])
                        if level['name']==settings['final_level']:
                            expected.update('exact/%s/%s/%s/%s'%(dps,c['name'],boost,k)for k in ['u','gradient','hessian'])
                            expected.add('wave/control/%s/%s/%s/%s'%(dps,level['name'],c['name'],boost))
        assert len(expected)==880 and {c['name']for c in new}==expected
        assert result['passed_corrected_control_gate'] is True and result['failed_checks']==[]
        assert (result['checks'],result['control_rows'],result['completed_control_roots'])==(880,100,189440)
        assert result['original_full_gate_remains_failed'] is True
        assert qual['passed_saved_noncontrol_check_qualification'] is True and qual['retained_saved_checks']==1480 and qual['retained_saved_failures']==0 and qual['entire_control_slice_deferred']==880
        a=(owner/'source-history/derivative_core-original.py').read_bytes();b=(owner/'derivative_core.py').read_bytes()
        oldline=b'            b=omega*q/self.a\n';newline=b'            b=omega*(sumjet(t*t for t in Y)**mp.mpf(".5"))/self.a\n'
        assert a.count(oldline)==b.count(newline)==1 and b.replace(newline,oldline)==a
        assert ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b.replace(newline,oldline)),include_attributes=False)
        maxima={}
        for c in new:
            group=c['name'].split('/')[0]
            if group not in maxima or Decimal(c['error'])>Decimal(maxima[group]['error']):maxima[group]=c
        report={'passed_saved_association':True,'original_full_gate_remains_failed':True,'original_checks':2360,'original_failures':156,
            'retained_saved_checks':1480,'retained_saved_failures':0,'corrected_control_checks':880,'corrected_control_rows':100,'corrected_control_roots':189440,
            'retained_by_family':dict(collections.Counter(c['name'].split('/')[0]for c in retained)),
            'corrected_maximum_saved_error_by_family':maxima,'one_CMC_radius_jet_source_line_only':True,
            'source_and_output_pins_verified':len(pins),'large_control_jets_decoded':False,
            'limits':['No derivative/root/integral/coordinate/PDE recomputation.','Original full2360 gate remains FAIL; no combined full-PASS field is issued.','Corrected880 controls and saved1480 qualification have separate receipts and source identities.']}
        guard(pins);write(out/'report.json',report)
        receipt.update(passed=True,returncode=0,completed=True,inputs_unchanged=True,protected_pins=len(pins),report_sha256=sha(out/'report.json'))
    except BaseException as exc:
        (out/'failure.txt').write_text(traceback.format_exc());receipt['error']=repr(exc)
    receipt['seconds']=time.monotonic()-start;write(out/'receipt.json',receipt);print(json.dumps(receipt,sort_keys=True));return receipt['returncode']

if __name__=='__main__':sys.exit(main())
