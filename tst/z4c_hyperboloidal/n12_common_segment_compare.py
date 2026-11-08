#!/usr/bin/env python3
"""Admit one completed N12 common-time segment using the unchanged science gates.

The previous segment must already be admitted. Complete fields are compared at
the new endpoint; all accumulated ordinary history and stage records are checked
again for coverage across the handoff. This never submits or evolves anything.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex

from aurora.common_segment_audit import audit
from aurora.longrun_admission import observed_main
from n12_checkpoint import execute, require, sha256
from n12_long_history import STEPS, REFERENCE_SHA, read_outputs, assess
from native_step_metrics import compare_step_metrics

NATIVE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'


def compare(args):
    runs = [path.resolve() for path in args.runs]
    segment = len(runs)
    require(1 <= segment <= 10 and len(set(runs)) == segment,
            'runs must be the unique completed common-time prefix')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    end = STEPS[segment]
    result = dict(schema='native-n12-common-science-admission-v1', passed=False,
                  segment=segment, end_cycle=end, through_5m_complete=False,
                  missing_future_common_steps=list(STEPS[segment+1:]),
                  comparisons={}, input_sha256={})
    try:
        proof = json.loads(args.observer_build.read_text())
        qualification = json.loads(args.observer_qualification.read_text())
        observed_main(qualification,proof,sha256(args.observer_build))
        require(proof['schema'] == 'native-step-observer-build-v1' and
                proof['status'] == 'built' and proof['build_exit_code'] == 0 and
                proof['native_executable_sha256'] == NATIVE and
                proof['runtime_inputs_unchanged'] is True and
                proof['overlay']['inverse_byte_exact'] is True and
                proof['original_main_fp_flags_preserved'] is True,
                'observer is not bound to the unchanged admitted original runtime')
        source = Path(__file__).resolve().parents[2]/'src/z4c_hyperboloidal/diagnostics.cpp'
        hashes = [digest for path,digest in proof['sources'].items()
                  if path.endswith('/src/z4c_hyperboloidal/diagnostics.cpp')]
        require(hashes == [sha256(source)], 'reviewed horizon/mask source differs')
        text = source.read_text()
        require('RequireAll(horizon_.found && horizon_.complete_surface && surface_.Valid()' in text and
                'mask(q)=surface_.Outside(x)?1:0;' in text,
                'complete-horizon/full-mask runtime guard changed')
        result['input_sha256'][str(args.observer_build.resolve())] = sha256(args.observer_build)
        result['input_sha256'][str(args.observer_qualification.resolve())] = \
            sha256(args.observer_qualification)
        predecessor = None
        if segment > 1:
            predecessor = runs[-2]/'rst'/f'native_n12_5M.{segment-1:05d}.rst'
            admission_path = args.previous_admission
            require(admission_path is not None, 'previous science admission is required')
            prior = json.loads(admission_path.read_text())
            require(prior['schema'] == result['schema'] and prior['passed'] is True and
                    prior['segment'] == segment-1 and prior['end_cycle'] == STEPS[segment-1] and
                    prior['checkpoint_sha256'] == sha256(predecessor),
                    'previous admission is missing, failed or bound to another checkpoint')
            result['input_sha256'][str(admission_path.resolve())] = sha256(admission_path)
        run = runs[-1]
        require((run/'exit-code.txt').read_text().strip() == '0',
                'native evolution did not exit successfully')
        command = shlex.split((run/'command.txt').read_text())
        require(command.count(proof['executable']) == 1 and command.count('-n') == 1 and
                command[command.index('-n')+1] == '24' and '-t' not in command,
                'unexpected executable, decomposition or software time stop')
        require(('-r' in command) == (segment > 1) and
                ('-i' in command) == (segment == 1), 'wrong fresh/restart entry path')
        if segment > 1:
            recorded = (run/'predecessor.sha256').read_text().split()
            require(len(recorded) == 2 and recorded[0] == sha256(predecessor) and
                    command.count('-r') == 1 and
                    command[command.index('-r')+1] == recorded[1],
                    'run did not restart the admitted predecessor checkpoint')
            result['input_sha256'][str(run/'predecessor.sha256')] = \
                sha256(run/'predecessor.sha256')
        result['structural'] = audit(run,args.initial_reference,segment,predecessor)
        require(result['structural']['structural_cadence_passed'] is True,
                'segment structural/cadence audit failed')
        binding_path = args.reference_root/'results.json'
        binding = json.loads(binding_path.read_text())
        require(binding['schema'] == 'native-n12-historical-reference-v1' and
                binding['passed'] is True and binding['common_steps'] == list(STEPS),
                'unbound historical reference snapshots')
        result['input_sha256'][str(binding_path.resolve())] = sha256(binding_path)
        cycles = (0,end) if segment == 1 else (end,)
        for cycle in cycles:
            checkpoint = run/'rst'/f'native_n12_5M.{0 if cycle == 0 else segment:05d}.rst'
            reference = args.reference_root/f'step-{cycle}'
            for name,digest in binding['snapshots'][str(cycle)]['sha256'].items():
                require(sha256(reference/name) == digest, 'bound historical snapshot changed')
            fields = execute(checkpoint,args.mapping,output/f'state-{cycle}',cycle,reference)
            result['comparisons'][str(cycle)] = fields
            require(fields['passed'] is True and fields['writer_mpi_ranks'] == 24 and
                    fields['mapping_sha256'] == binding['mapping_sha256'],
                    'complete field comparison or reference mapping failed')
        endpoint = run/'rst'/f'native_n12_5M.{segment:05d}.rst'
        result['checkpoint_sha256'] = sha256(endpoint)
        result['checkpoint'] = str(endpoint)
        require(sha256(args.reference_history) == REFERENCE_SHA,
                'historical constraint/horizon table changed')
        with args.reference_history.open() as stream:
            reference = {int(row['step']):{key:float(value) for key,value in row.items()}
                         for row in csv.DictReader(stream)}
        histories,horizons,files,extra = read_outputs(runs)
        history = assess(histories,horizons,reference)
        result['history'] = history
        expected = {str(step) for step in STEPS[:segment+1]}
        missing = [dict(step=step,error='missing common history or horizon')
                   for step in STEPS[segment+1:]]
        require(set(history['times']) == expected and history['failures'] == missing and not extra,
                'completed-prefix history/horizon gate or cadence failed')
        result['history_completed_prefix_passed'] = True
        # The full-history report remains false until all eleven times exist.
        result['stage_metrics'] = compare_step_metrics(runs,args.reference_steps,end,
            gate='long',ranks=24,historical_sha256=args.reference_steps_sha256)
        require(result['stage_metrics']['passed'] is True, 'per-step clock/CFL gate failed')
        files += [run/'command.txt',run/'exit-code.txt',args.reference_history,
                  args.initial_reference]
        result['input_sha256'].update({str(path.resolve()):sha256(path) for path in files})
        result.update(passed=True, through_5m_complete=segment == 10,
            chi_scope='native records decoded and checked finite; all available historical aggregate differences reported under the declared threshold-sensitive floor policy; no historical per-cell record equivalence claim',
            scope='completed N12 science prefix only; AMR, performance and scaling goal requirements remain separate')
    except Exception as error:
        result['validation_error'] = f'{type(error).__name__}: {error}'
    finally:
        result['source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
            sha256(Path(__file__).with_name(name)) for name in
            ('n12_common_segment_compare.py','n12_checkpoint.py','n12_long_history.py',
             'native_step_metrics.py')}
        result['source_sha256'][str(Path(__file__).parent/'aurora/common_segment_audit.py')] = \
            sha256(Path(__file__).parent/'aurora/common_segment_audit.py')
        result['source_sha256'][str(Path(__file__).parent/'aurora/longrun_admission.py')] = \
            sha256(Path(__file__).parent/'aurora/longrun_admission.py')
        (output/'continuation-admission.json').write_text(
            json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--reference-history',type=Path,required=True)
    parser.add_argument('--reference-steps',type=Path,required=True)
    parser.add_argument('--reference-steps-sha256',required=True)
    parser.add_argument('--initial-reference',type=Path,required=True)
    parser.add_argument('--observer-build',type=Path,required=True)
    parser.add_argument('--observer-qualification',type=Path,required=True)
    parser.add_argument('--previous-admission',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps({key:result[key] for key in ('passed','segment','end_cycle',
        'through_5m_complete','validation_error') if key in result},indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
