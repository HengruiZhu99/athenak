#!/usr/bin/env python3
"""Read-only V4 failure localization; no evolution, repair, or restart admission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
AURORA = HERE.parent.parent
FROZEN = 'd98e2a99bbf1d5546956be3d724c82715ca704ec'
V3 = 'ec0dd5afcf9d489e06c1fdf92e802a6fb8c14679'
SOURCE = '0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f'
BINARY = 'fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff'
TARGET = 'build-sycl-mpi-v11/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_dissipative'
PREFLIGHT = AURORA/'preflight-dissipative-v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require(value, message):
    if not value:
        raise ValueError(message)


v = module('v4_preflight_verifier', PREFLIGHT/'verify.py')
i = module('interpolation_analysis', AURORA/'scri-positivity-audit-v1/run_audit.py')
s = module('surface_analysis', AURORA/'stability-projected-minkowski-v1/analyze.py')
sha, write = v.sha, s.write


def verify(raw):
    plan, binding, worker = [v.read(raw/p) for p in ['plan.json','binding.json','worker-status.json']]
    require(len(plan['commands']) == len(worker['cases']) == 1, 'Unexpected case count')
    command, case = plan['commands'][0], worker['cases'][0]
    require(command['name'] == case['name'] == 'minkowski' and command['executable'] == TARGET,
            'Wrong case/executable')
    require(worker['source_manifest_sha256'] == binding['source_manifest_sha256'] == SOURCE,
            'Scientific source differs')
    require(worker['binaries'] == binding['binaries'] == {TARGET:BINARY}, 'Scientific executable differs')
    require(worker['plan_sha256'] == binding['plan_sha256'] == sha(raw/'plan.json'), 'Plan hash differs')
    require(binding['pbs_sha256'] == sha(raw/'run_debug.pbs'), 'PBS script hash differs')
    require(worker['resources'] == binding['resources'] == plan['resources'], 'Resources differ')
    build, manifest, build_proof = v.verify_build(PREFLIGHT/'compile-evidence', ROOT)
    parser, control = v.frozen_parser(ROOT, manifest)
    control.validate_plan(ROOT, plan)
    for name,digest in binding['launcher'].items():
        require(digest == manifest['source-v11/tst/hyperbolic_shift_3d/aurora/'+name], 'Launcher differs')
    admission_path = PREFLIGHT/'admission.json'
    admission = v.read(admission_path)
    require(binding['admission']['sha256'] == sha(admission_path) and admission['passed'] is True
            and admission['phase'] == 'preflight' and admission['low_resolution_n'] == 16
            and admission['source_manifest_sha256'] == SOURCE and admission['binaries'][TARGET] == BINARY,
            'Preflight admission differs')
    require(control.dissipation_config(command) in admission['scri_dissipation_configs'],
            'Runtime epsilon is not covered by admission')
    require(admission['verifier_sha256'] == sha(PREFLIGHT/'verify.py'), 'Admitted verifier differs')
    references = [(PREFLIGHT/'raw/worker-status.json',admission['worker_status_sha256']),
                  (PREFLIGHT/'verification-v2.json',admission['verification_sha256'])]
    references += [(PREFLIGHT/'raw/output'/Path(r['path']).name,r['sha256'])
                   for r in admission['supporting_reports'].values()]
    for path,digest in references:
        require(sha(path) == digest, 'Preflight supporting evidence differs')
    require(v.read(PREFLIGHT/'verification-v2.json')['passed'] is True, 'Preflight verification did not pass')
    require(build['binaries'][TARGET] == BINARY, 'Build target mismatch')
    argv = case['argv']
    require(argv[:7] == ['mpiexec','-n','12','-ppn','12','--cpu-bind=list:'+control.CPU_BIND,'gpu_tile_compact.sh']
            and argv[7] == v.REMOTE+TARGET and len(argv[8:]) == len(command['args']), 'Actual launch differs')
    for actual, expected in zip(argv[8:],command['args']):
        if expected == '@WALL_SECONDS@':
            require(0 < float(actual) <= command['timeout_seconds']-30, 'Invalid runtime ceiling')
        else:
            require(actual == expected, 'Actual launch argument differs')
    require(command['args'][5:7] == ['-','-'] and not binding['inputs'] and not case['generated_inputs'],
            'Expected fresh analytic Minkowski state')
    evidence = []
    require(set(case['artifacts']) == set(command['artifacts']), 'Artifact coverage differs')
    for remote, expected in case['artifacts'].items():
        path = raw/'output'/Path(remote).name
        actual = dict(bytes=path.stat().st_size,sha256=sha(path))
        require(actual == expected, 'Artifact differs: '+path.name)
        evidence.append(dict(remote_path=remote,local_path=str(path),**actual))
    report = v.read(raw/'output/minkowski.report.json')
    require(worker['status'] == case['status'] == case['report_status'] == report['status'] == 'failed',
            'Failure outcome differs')
    require(case['returncode'] == 2 and case['timed_out'] is False and case['storage_error'] is None,
            'Failure was infrastructure-related')
    require(report['error'] == 'nonpositive interpolated scri geometry' and report['checkpoint_admitted'] is False,
            'Wrong rejection/checkpoint admission')
    require(report['backend'] == 'SYCL' and report['mpi_size'] == 12 and report['n'] == 16 and report['dt'] == .0005
            and report['start_time'] == 0 and report['scri_dissipation_epsilon'] == 2
            and report['scri_dissipation_degree'] == 7 and report['checkpoint_header_bytes'] == 88,
            'Wrong evolution configuration')
    require([r['check'] for r in case['checks']] == command['checks'], 'Report gates differ')
    for check in case['checks']:
        observed, passed = v.check_value(check['check'], report)
        require(observed == check['observed'] and passed == check['passed'], 'Scientific check replay differs')
    affinity = worker['affinity']
    v.placement(affinity['records'])
    require(affinity['passed'] and affinity['returncode'] == 0 and not affinity['timed_out']
            and affinity['storage_error'] is None, 'Affinity probe failed')
    records = [json.loads(x.split('=',1)[1]) for x in (raw/'output/affinity.log').read_text().splitlines()
               if x.startswith('AFFINITY_JSON=')]
    require(sorted(records,key=lambda r:r['rank']) == sorted(affinity['records'],key=lambda r:r['rank']),
            'Affinity log differs')
    terminal, release, submission = [v.read(raw/p) for p in ['collected.json','allocation-released.json','submission.json']]
    require(terminal['job_id'] == release['job_id'] == worker['job_id'] == submission['job_id'], 'Job identity differs')
    v.verify_release(release)
    require(terminal['scheduler']['job_state'] == 'F' and terminal['scheduler']['Exit_status'] == 1,
            'Missing terminal failure receipt')
    require(submission['plan_sha256'] == binding['plan_sha256'] and submission['source_manifest_sha256'] == SOURCE,
            'Submission binding differs')
    require(worker['storage_at_finish']['bytes'] < 5_000_000_000, 'Campaign-home cap exceeded')
    result = dict(evidence_verified=True, scientific_status='failed', source_manifest_sha256=SOURCE,
        binary_sha256=BINARY, checks_recomputed=len(case['checks']), failed_checks=[c for c in case['checks'] if not c['passed']],
        source_git_blobs_verified=build_proof['git_blob_hashes_verified'], build_receipt_sha256=build_proof['build_receipt_sha256'],
        admission_sha256=sha(admission_path), preflight_supporting_hashes_verified=len(references),
        terminal=dict(job_id=worker['job_id'],state='F',exit_status=1,allocation_released=True),
        one_node_twelve_tiles_verified=True, artifacts=evidence,
        metadata_sha256={p:sha(raw/p) for p in ['plan.json','binding.json','worker-status.json','collected.json',
                                              'submission.json','allocation-released.json','run_debug.pbs','output/affinity.log']})
    return report, result


def replay(out, build, n, stage, saved, failed):
    provenance = dict(frozen_numerical_commit=FROZEN, source_comparison=[],
        audit_cpp_sha256=sha(i.HERE/"scri_interpolation_audit.cpp"))
    for relative in i.HEADERS:
        frozen = subprocess.check_output(["git", "show", f"{FROZEN}:{relative}"], cwd=ROOT)
        require(frozen == (ROOT/relative).read_bytes(), "audit header changed: "+relative)
        provenance["source_comparison"].append(dict(path=relative, sha256=sha(ROOT/relative),
                                                    identical_to_frozen=True,
            identical_to_v3=(frozen == subprocess.check_output(["git", "show", f"{V3}:{relative}"],cwd=ROOT))))
    flags_path = build/"tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_mpi.dir/flags.make"
    flags = dict((key, shlex.split(value)) for key, value in
                 (line.split(" = ", 1) for line in flags_path.read_text().splitlines() if " = " in line))
    provenance["flags_file"] = dict(path=str(flags_path), sha256=sha(flags_path))
    with tempfile.TemporaryDirectory(prefix="v4-scri-readonly-") as folder:
        tmp = Path(folder)
        exe = tmp/"audit"
        command = (["/usr/bin/c++"] + flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
                   + [i.HERE/"scri_interpolation_audit.cpp", "-o", exe])
        command += [build/f"kokkos/{part}/src/libkokkos{part}.a" for part in ["containers", "algorithms", "core", "simd"]]
        write(out/"audit-build.json", i.run(command, timeout=120))
        controls = []
        for name in ["positive_identity", "negative_chi", "indefinite_metric", "nonfinite_auxiliary"]:
            size = 6
            data = np.zeros((6*size*size, 31), dtype="<f8"); data[:,0] = 1
            if name == "negative_chi": data[:,0] = -1
            elif name == "indefinite_metric": data[:,1] = -2
            elif name == "nonfinite_auxiliary": data[:,30] = np.nan
            payload, result = tmp/(name+".bin"), tmp/(name+".json")
            with payload.open("wb") as f:
                f.seek(8*31*19*size**3); data.tofile(f)
            execution = i.run([exe, size, payload, result])
            result_data = json.loads(result.read_text())
            expected = {k:0 for k in result_data["rejection_counts"]}
            if name != "positive_identity":
                key = dict(negative_chi="nonpositive_chi", indefinite_metric="metric_invalid",
                           nonfinite_auxiliary="nonfinite_field")[name]
                expected[key] = result_data["unique_global_boundary_queries"]
            require(result_data["rejection_counts"] == expected, "audit control failed: "+name)
            controls.append(dict(name=name, passed=True, counts=expected, seconds=execution["seconds"]))
        write(out/"audit-controls.json", controls)
        write(out/"audit-execution.json", i.run([exe, n, failed, out/"interpolation-audit.json"]))
    data = json.loads((out/"interpolation-audit.json").read_text())
    details = []
    for query in data["rejected"]:
        value = i.interpolate(stage, query)
        require(np.allclose(value, query["interpolated_fields"], rtol=0, atol=2e-13),
                "independent interpolation replay mismatch")
        previous = i.interpolate(saved, query)
        eigen = np.linalg.eigvalsh(i.metric(previous))
        weights = np.asarray(query["weights"])
        details.append(dict(query_id=query["query_id"], boundary_xyz=query["boundary_xyz"],
            donor_block=query["donor_block"], donor_face=query["donor_face"], donor_local=query["donor_local"],
            target_ghosts_block_i_j_k=query["targets"], interpolated_check=query["check"],
            eigenvalues=np.linalg.eigvalsh(np.asarray(query["check"]["metric"])),
            minimum_donor_eigenvalue=np.linalg.eigvalsh(i.metric(stage[query["donor_surface_ids"]])).min(),
            donor_chi_range=[query["donor_field_min"][0], query["donor_field_max"][0]],
            minimum_donor_determinant=query["donor_min_determinant"],
            weights=dict(sum=weights.sum(), l1=abs(weights).sum(), minimum=weights.min(), maximum=weights.max()),
            rollback_same_query=dict(chi=previous[0], determinant=np.linalg.det(i.metric(previous)), eigenvalues=eigen)))
    write(out/"interpolation-summary.json", dict(
        scope="Immediate rejection localization, not identification of the first growing mode",
        host_replay="Frozen stencil and Geometry on CPU binary64; not a bitwise SYCL replay",
        query_count_scope="Global exact-coordinate deduplication; not sum of rank-local invalid counters",
        q_zero_for_invalid_metric="Geometry returns before constructing its inverse; q=0 is consequent, not independent",
        nodal_surface=data["nodal_surface"], unique_global_boundary_queries=data["unique_global_boundary_queries"],
        rejected_exact_coordinate_queries=data["rejected_queries"],
        distinct_failed_directions_at_12_decimals=len({tuple(round(x,12) for x in q["boundary_xyz"]) for q in details}),
        rejection_counts=data["rejection_counts"], interpolant_ranges=data["interpolants"], rejected=details))
    write(out/"audit-provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'output/hyperbolic-core-shell/aurora/stability-dissipative-minkowski-v1')
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build-hyperbolic-mpi')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    raw = args.raw_dir.resolve()
    report, proof = verify(raw)
    checkpoint, failed = [raw/'output'/name for name in ['minkowski.checkpoint','minkowski.failed-stage.bin']]
    header = v.checkpoint(checkpoint)
    n = report['n']; nv = 31*19*n**3; ns = 31*6*n*n
    require(header['magic'] == 'HYP3D_MPI_V4' and header['header_bytes'] == 88 and
            header['scri_dissipation_epsilon'] == 2 and header['scri_dissipation_degree'] == 7 and
            header['n'] == n and header['kind'] == 0 and header['admitted'] is False,
            'Checkpoint configuration differs')
    require(header['time'] == report['t'] and header['steps'] == report['steps'] and header['dt'] == report['dt'],
            'Checkpoint/report clock differs')
    require(failed.stat().st_size == 8*(nv+ns), 'Failed-stage layout differs')
    saved = np.memmap(checkpoint,dtype='<f8',mode='r',offset=88+16*nv,shape=(6*n*n,31))
    stage = np.memmap(failed,dtype='<f8',mode='r',offset=8*nv,shape=(6*n*n,31))
    require(np.isfinite(stage).all(), 'Nonfinite failed-stage surface cannot use eigenvalue audit')
    row = int(np.argmin(np.linalg.eigvalsh(s.tensors(stage))[:,0]))
    write(out/'surface-audit.json',dict(checkpoint_schema='HYP3D_MPI_V4',header_bytes=88,
        scri_dissipation_epsilon=2,scri_dissipation_degree=7,admitted=0,accepted_time=header['time'],
        steps=header['steps'],dt=header['dt'],diagnostic_time=report['diagnostic_time'],
        diagnostics_current=report['diagnostics_current'],attempted_interval=[header['time'],header['time']+header['dt']],
        failed_stage_abscissa='not recorded',selected_row='minimum failed-stage nodal metric eigenvalue',
        accepted_rollback=s.surface_audit(saved,row),failed_stage=s.surface_audit(stage,row)))
    replay(out,args.build_dir.resolve(),n,stage,saved,failed)
    helpers = [Path(__file__),Path(i.__file__),Path(s.__file__),Path(s.a.__file__),Path(v.__file__),i.HERE/'scri_interpolation_audit.cpp']
    for artifact in proof['artifacts']:
        require(sha(Path(artifact['local_path'])) == artifact['sha256'], 'Scientific input changed during audit')
        artifact['unchanged_after_read'] = True
    proof.update(checkpoint=header, helpers=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in helpers],
        scope='Read-only immediate rejection localization, not the initial growing-mode cause; no evolution, repair, restart or admission.')
    write(out/'evidence-verification.json',proof)
    print(json.dumps(dict(output=str(out),checks_recomputed=proof['checks_recomputed'],
                         artifacts_verified=len(proof['artifacts']),scientific_status='failed')))


if __name__ == '__main__':
    main()
