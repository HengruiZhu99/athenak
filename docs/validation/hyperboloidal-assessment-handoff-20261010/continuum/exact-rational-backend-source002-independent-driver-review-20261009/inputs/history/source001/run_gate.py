"""HELD one-shot CPU exact-rational82 units; root120s process-group cap required."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

P=Path(__file__).resolve().parent
ROOT=P.parents[2]


def main():
    attempt=P/'attempts/units001'
    attempt.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(P))
    from gate_common import admit,check,load,pin,require,save,sha
    started=time.monotonic()
    receipt=dict(completed=False,passed=False,returncode=1,inputs_unchanged=False,
        stage='standalone-exact-rational-CPU-units82',commands=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        both_dependency_closures_passed=False,no_native_RWM_or_production_adoption=True)
    protected=[]

    def checkpoint(): save(attempt/'receipt.json',receipt)

    def run(command,name,env,input_path=None):
        remaining=120-(time.monotonic()-started)
        require(remaining>0,'whole-unit internal120s cap exhausted')
        cap=min(60,remaining)
        output,error=attempt/(name+'.stdout'),attempt/(name+'.stderr')
        record=dict(name=name,argv=command,timeout_seconds=cap,returncode=None)
        receipt['active_command']=record;checkpoint()
        tick=time.monotonic()
        failure=None
        try:
            with output.open('xb') as out,error.open('xb') as err:
                if input_path is None:
                    done=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,
                                        check=False,timeout=cap)
                else:
                    with Path(input_path).open('rb') as source:
                        done=subprocess.run(command,cwd=ROOT,env=env,stdin=source,stdout=out,stderr=err,
                                            check=False,timeout=cap)
            record['returncode']=done.returncode
        except BaseException as exc:
            record.update(exception=repr(exc),timeout=isinstance(exc,subprocess.TimeoutExpired))
            failure=exc
        finally:
            record.update(seconds=time.monotonic()-tick,
                stdout=pin(output) if output.exists() else None,stderr=pin(error) if error.exists() else None,
                input=None if input_path is None else pin(input_path))
            receipt['commands'].append(record);receipt.pop('active_command',None);checkpoint()
        if failure is not None: raise failure
        require(record['returncode']==0 and error.stat().st_size==0,name+': actual exit/stderr failure')
        return output

    try:
        checkpoint()
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--authorization',required=True)
        parser.add_argument('--recipe',default=str(P/'recipe.json'))
        args=parser.parse_args()
        recipe,auth,protected=admit(args.recipe,args.authorization)
        save(attempt/'inputs-before.json',protected)
        receipt.update(source_index=pin(P/'source-index.json'),recipe=pin(P/'recipe.json'),
            authorization=pin(args.authorization),python=pin(sys.executable),
            environment=recipe['environment'],protected_count=len(protected))
        for name in recipe['local_sources']+['source-index.json','external-pins.json','recipe.json']:
            shutil.copyfile(P/name,attempt/name)
            require(sha(P/name)==sha(attempt/name),'copied input differs')
        shutil.copyfile(args.authorization,attempt/'authorization.json')
        env=dict(os.environ)
        run([recipe['compiler']['path'],'--version'],'compiler-version',env)
        run([recipe['python']['path'],'-I','-B','--version'],'python-version',env)
        declared={str(Path(item['path']).resolve()):item for item in protected}
        # Compile BOTH modes and close BOTH header sets before any probe or
        # Fraction/registry import. Unknown dependencies stop this exact attempt.
        for mode in ('release','debug'):
            executable,dep=attempt/('probe-'+mode),attempt/('probe-'+mode+'.d')
            command=[recipe['compiler']['path']]+recipe[mode+'_flags']+[
                '-MD','-MF',str(dep),str(attempt/'probe.cpp'),'-o',str(executable)]
            run(command,'compile-'+mode,env)
            receipt[mode+'_executable']=pin(executable)
            require(dep.is_file(),'compiler did not produce dependencies')
            dependencies=[]
            words=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
            require(bool(words),'empty dependency list')
            missing=[]
            for word in words:
                path=Path(word)
                path=(path if path.is_absolute() else ROOT/path).resolve()
                item=pin(path);dependencies.append(item)
                if path.parent==attempt:
                    require(path.name in ('probe.cpp','exact_dyadic_ratio.hpp') and sha(path)==sha(P/path.name),
                            'unexpected/changed copied local dependency')
                elif str(path) not in declared:
                    missing.append(item)
                else:
                    check(declared[str(path)])
            save(attempt/('dependencies-'+mode+'.json'),dependencies)
            receipt[mode+'_dependencies']=pin(attempt/('dependencies-'+mode+'.json'))
            if missing:
                save(attempt/('unprotected-dependencies-'+mode+'.json'),missing)
            checkpoint()
            require(not missing,'external compiler headers outside frozen closed baseline; fresh additive attempt required')
        for item in protected:check(item)
        receipt['both_dependency_closures_passed']=True;checkpoint()
        common=[recipe['python']['path'],'-I','-B',str(P/'oracle_driver.py'),
                '--recipe',str(P/'recipe.json'),'--authorization',str(Path(args.authorization).resolve())]
        run(common+['--mode','prepare'],'prepare-registry',env)
        registry_receipt=load(attempt/'registry-receipt.json')
        generated=[pin(attempt/'registry-receipt.json'),registry_receipt['registry'],registry_receipt['protocol']]
        for item in generated:check(item)
        reports={}
        for mode in ('release','debug'):
            for item in protected+generated:check(item)
            raw=run([str(attempt/('probe-'+mode))],'probe-'+mode,env,attempt/'cases.txt')
            reportpath=attempt/('oracle-'+mode+'.json')
            run(common+['--mode','check','--input',str(raw),'--report',str(reportpath)],'oracle-'+mode,env)
            report=load(reportpath)
            require(report.get('passed') is True and report.get('fixed_cases')==82
                    and report.get('exact_full_operand_comparisons')==76,
                    'all82 Fraction controls/full76 operands required')
            receipt[mode+'_oracle_report']=pin(reportpath);reports[mode]=report
        equal=(attempt/'probe-release.stdout').read_bytes()==(attempt/'probe-debug.stdout').read_bytes()
        require(equal,'Release/ASanUB native output bytes differ')
        require(reports['release']['base_report']==reports['debug']['base_report']
                and reports['release']['carry_range_report']==reports['debug']['carry_range_report'],
                'Release/ASanUB complete oracle rows differ')
        for item in protected+generated:check(item)
        receipt.update(completed=True,passed=True,returncode=0,fixed_cases=82,base_cases=74,
            carry_range_cases=8,release_debug_probe_byte_equal=True,release_debug_exact_oracle_rows_equal=True)
    except BaseException as exc:
        receipt.update(exception=repr(exc),exception_type=type(exc).__name__)
        (attempt/'exception.txt').write_text(traceback.format_exc())
    finally:
        drift=[]
        for item in protected:
            try:check(item)
            except BaseException as exc:drift.append(dict(path=item['path'],error=repr(exc)))
        receipt['inputs_unchanged']=bool(protected) and not drift
        receipt['input_drift']=drift
        if not receipt['inputs_unchanged']:receipt.update(completed=False,passed=False,returncode=1)
        save(attempt/'inputs-after.json',[pin(item['path']) for item in protected if Path(item['path']).is_file()])
        receipt.update(seconds=time.monotonic()-started,completed_utc=datetime.now(timezone.utc).isoformat())
        receipt['outputs']=[pin(path) for path in sorted(attempt.rglob('*')) if path.is_file() and path.name!='receipt.json']
        checkpoint()
    print(receipt['passed'],str(attempt/'receipt.json'))
    return receipt['returncode']


if __name__=='__main__':
    raise SystemExit(main())
