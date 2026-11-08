#!/usr/bin/env python3
"""Local provenance/input controls; no solver execution or GPU claims."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from native_amr_admission_candidate import SAFE_DT, check_package, sha
from native_amr_candidate_cmake import check
from native_amr_gpu_compare import candidate_binding, checkpoint_audit, require
from native_amr_gpu_input import helper_text, restart_text


def rejected(call, fragment):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        require(fragment in str(error), 'unexpected rejection: '+str(error))
        return dict(rejected=True, reason=str(error))
    raise AssertionError('invalid candidate evidence was accepted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-build',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prototype-build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    candidate=json.loads(args.candidate_build.read_text())
    check(candidate)
    def bind(record, gpu=False):
        return candidate_binding(record,args.package,candidate['baseline_native_executable_sha256'],
                                 candidate['executable_sha256'],gpu=gpu)
    bind(candidate)
    checks={}
    checks['CPU_build_cannot_claim_GPU']=rejected(lambda:bind(candidate,True),'SYCL')
    checks['mixed_objects_cannot_qualify']=rejected(
        lambda:bind(json.loads(args.prototype_build.read_text())),'coherent reviewed')
    for label,key,value in [('frozen_executable_as_candidate','executable_sha256',candidate['baseline_native_executable_sha256']),
                            ('old_timestep','fixed_dt',.00046944203735617445),
                            ('wrong_patch','production_patch_sha256','0'*64)]:
        bad=copy.deepcopy(candidate);bad[key]=value
        checks[label]=rejected(lambda:bind(bad),'coherent reviewed')
    bad=copy.deepcopy(candidate);bad['baseline_objects_linked']=True
    checks['linked_baseline_objects']=rejected(lambda:bind(bad),'coherent reviewed')
    bad=copy.deepcopy(candidate)
    name=str(Path(bad['root'])/'src/z4c_hyperboloidal/z4c_hyperboloidal.hpp')
    bad['candidate_files'][name]='0'*64
    checks['wrong_class_definition']=rejected(lambda:bind(bad),'unbound coherent')
    with tempfile.TemporaryDirectory(dir=args.output) as name:
        altered=Path(name)/'package';shutil.copytree(args.package,altered)
        source=altered/'overlay/src/z4c_hyperboloidal/fixed_step_admission.hpp'
        source.write_text(source.read_text()+'\n// altered test copy\n')
        checks['changed_overlay_bytes']=rejected(lambda:check_package(altered),'package file changed')
        for key in ('compile_database','link_file','finalizer'):
            changed=Path(name)/key;changed.write_bytes(Path(candidate[key]).read_bytes()+b'\n# altered copy\n')
            bad=copy.deepcopy(candidate);bad[key]=str(changed)
            checks['changed_'+key]=rejected(lambda:check(bad),
                                           'changed finalizer' if key=='finalizer' else 'changed candidate '+key)
    before=sha(args.checkpoint)
    _,_,checkpoint=checkpoint_audit(args.checkpoint,2,72,SAFE_DT)
    checks['old_dt_against_safe_checkpoint']=rejected(
        lambda:checkpoint_audit(args.checkpoint,2,72,.00046944203735617445),'exact time changed')
    require(sha(args.checkpoint)==before,'original checkpoint changed')
    original=Path(__file__).with_name('n12_initialize.athinput').read_text()
    require(helper_text()==original+'\n<time>\ndt = .0002\n','helper changed beyond timestep')
    require(restart_text().startswith(helper_text()),'ordinary input changes original model')
    result=dict(passed=True,gpu_evidence=False,solver_launched=False,
        scope='isolated candidate packaging/link provenance and existing CPU restart parsing',
        controls=checks,safe_checkpoint=checkpoint,frozen_inputs_unchanged=True,
        candidate_build_sha256=sha(args.candidate_build),package_manifest_sha256=sha(args.package/'manifest.json'),
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('native_amr_candidate_controls.py','native_amr_admission_candidate.py','native_amr_candidate_cmake.py','native_amr_gpu.py',
             'native_amr_gpu_compare.py','native_amr_gpu_input.py','amr_restart.py','restart_roundtrip.py')})
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,controls=len(checks),gpu_evidence=False),indent=2))


if __name__=='__main__':
    main()
