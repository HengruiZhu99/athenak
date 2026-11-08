#!/usr/bin/env python3
"""Bind two verified terminal AMR collections for the existing science reader.

This offline finalizer never launches work, changes data, or declares the AMR
scientific gates passed. Raw files remain in their collected directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

SIDE=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
WORK=SIDE/'validation/amr-admission-candidate-v1'
BASELINE='f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
CANDIDATE='442594b847ff0eb56407513da48e96930ec41dbe537a69c202fd802da4ea6a31'
HELPER_INPUT='feb01f57769a884c2ddc87e9f1655a4f8b3e6d99d4a7bf11a82a54da6c8cc05a'
RESTART_INPUT='a4b3de1310649346ed4206be3a5cb93fe582330c3f7f1e2217cede61505dd621'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    assert not args.output.exists()
    all_files={}
    cases={}
    for nodes,folder in ((2,args.two.resolve()),(8,args.eight.resolve())):
        package=json.loads((folder/'package.json').read_text())
        assert package['schema']=='native-amr-candidate-collection-v1'
        assert (package['nodes'],package['ranks'],package['job_state'],package['exit_status'])==(nodes,12*nodes,'F',0)
        root=folder/'run-artifacts'
        manifest=json.loads((root/'files.json').read_text())
        assert len(manifest)==package['files']
        for name,record in manifest.items():
            path=(root/name).resolve()
            assert path.is_relative_to(root) and path.is_file()
            assert path.stat().st_size==record['size'] and sha(path)==record['sha256']
            all_files[path]=record['sha256']
        all_files[folder/'package.json']=sha(folder/'package.json')
        all_files[root/'files.json']=sha(root/'files.json')
        case=root/f'runs/amr-candidate-nodes{nodes}'
        assert (case/'pipeline-complete.txt').is_file()
        assert (case/'provenance-before.txt').stat().st_size and (case/'provenance-after.txt').stat().st_size
        cases[nodes]=(root,case,package)
    validation=cases[8][0]/'validation/amr-admission-candidate-v1'
    candidate_path=validation/'coherent/provenance.json'
    candidate=json.loads(candidate_path.read_text())
    assert candidate['status']=='built' and candidate['baseline_inputs_unchanged']
    assert candidate['executable_sha256']==CANDIDATE and candidate['baseline_native_executable_sha256']==BASELINE
    # Require the same build and helper bytes in both serialized allocations.
    for name in ('coherent/provenance.json','lifecycle-build/provenance.json','evolution-build/provenance.json'):
        assert sha(validation/name)==sha(cases[2][0]/'validation/amr-admission-candidate-v1'/name)
    args.output.mkdir()
    relative=lambda path:os.path.relpath(path,args.output)
    all_files[Path(__file__).resolve()]=sha(Path(__file__).resolve())
    artifacts={relative(path):digest for path,digest in sorted(all_files.items())}
    for kind in ('lifecycle','evolution','restart'):
        proof=dict(schema='native-amr-gpu-execution-v2',case=kind,candidate=True,exit_status=0,
            fixed_dt=.0002,baseline_inputs_unchanged=True,
            baseline_native_executable_sha256=BASELINE,candidate_executable_sha256=CANDIDATE,
            candidate_build_provenance=relative(candidate_path),
            candidate_compile_database=relative(validation/'coherent/build/compile_commands.json'),
            candidate_link_file=relative(validation/'coherent/build/src/CMakeFiles/athena.dir/link.txt'),
            artifacts=artifacts,executions=[],scientific_comparison_pending=True)
        if kind=='restart':
            selections=[(2,'prefix','prefix'),(8,'continuous','continuous'),(8,'resumed','resumed')]
            executable=WORK/'coherent/build/src/athena'
            executable_sha=CANDIDATE
        else:
            selections=[(2,kind,'24'),(8,kind,'96')]
            build_path=validation/f'{kind}-build/provenance.json'
            build=json.loads(build_path.read_text())
            assert build['schema']=='native-amr-gpu-build-v3' and build['coherent_candidate']
            assert build['candidate_executable_sha256']==CANDIDATE
            executable=WORK/f'{kind}-build/amr_{kind}'
            executable_sha=build['executable_sha256']
            proof['build_provenance']=relative(build_path)
        for nodes,name,label in selections:
            root,parent,package=cases[nodes]
            folder=parent/name
            operational=json.loads((folder/'operational.json').read_text())
            assert operational['operational_passed'] and operational['case']==name
            assert operational['ranks']==nodes*12 and operational['fixed_dt']==.0002
            assert operational['job_id'].split('.')[0]==package['job_id']
            assert (folder/'exit-code.txt').read_text().strip()=='0'
            assert sha(folder/'command.txt')==operational['command_sha256']
            command=shlex.split((folder/'command.txt').read_text())
            assert command[:5]==['mpiexec','-n',str(nodes*12),'-ppn','12']
            assert command[5]=='--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
            assert command[6]=='/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh'
            assert command[7]==str(executable)
            options=command[8:]
            entry=dict(label=label,directory=relative(folder),ranks=nodes*12,exit_status=0,
                       executable_sha256=executable_sha,job_id=package['job_id'])
            if name=='resumed':
                assert options==['-r',str(SIDE/'runs/amr-candidate-nodes2/prefix/rst/amr_restart.00003.rst'),'time/nlim=4']
                entry.update(input_override=False,dt_override=False)
            elif kind=='restart':
                expected=['-i',str(WORK/'package/inputs/amr_restart.athinput')]
                if name=='prefix':expected+=['time/nlim=2']
                assert options==expected and sha(parent/'amr_restart.athinput')==RESTART_INPUT
                entry['input_sha256']=RESTART_INPUT
            else:
                assert options==[str(WORK/'package/inputs/amr_helper.athinput')]+(['--diagnostics'] if kind=='lifecycle' else [])
                assert sha(parent/'amr_helper.athinput')==HELPER_INPUT
                entry['input_sha256']=HELPER_INPUT
            proof['executions'].append(entry)
        (args.output/f'{kind}-execution.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(execution_provenance_bound=True,scientific_comparison='pending',output=str(args.output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two',type=Path,required=True,help='collection folder containing package.json/run-artifacts')
    parser.add_argument('--eight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
