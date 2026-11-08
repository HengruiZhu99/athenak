#!/usr/bin/env python3
"""Generate an isolated original-control entry point with native old diagnostics.

The archived original source predates harmonic-mask diagnostics. Preserve all
timed code; an independent v8 diagnostic-only process evaluates the saved final
state. This prepares files only and never builds, evolves or submits a job.
"""
import argparse
import json
from pathlib import Path
import shutil

from legacy_stage_trace_gpu import require,sha

WRAPPER='abdc68c25086835224d2373832c545f634ee1f7075080274918051d2def0d604'
BUILDER='6c7b1fdf9607672f8ae1f98261402cce4a8b6785656d6f8a0938184f0c617615'
HELPER='b0f3a2146513893d54b7ea23d4478372c55d4180ba4123ef739707a8e7b8803f'

DIAGNOSTIC=r'''Diagnostics Diagnose(Evolution &e,const std::filesystem::path &input,
                     const std::filesystem::path &out) {
  (void)input;(void)out;
  // Original API: full domain, fixed r>.94 exterior, and fixed-cut sums only.
  // Harmonic exterior/FastFlow are evaluated in a separate untimed process.
  const auto before_bulk=Records(e,0),before_surface=Records(e,1);
  Buffer edge;e.RHS(dt,edge);
  const auto diag=e.kernel()->Diagnostics(e.exchange()->padded,.94,
      e.spherical_surface_residual,0,false,{.94,.16,.25});
  e.SynchronizeError(diag.invalid_cells,"invalid original benchmark diagnostics");
  Diagnostics result;
  const Real values[10]={diag.total.volume,diag.total.h2,diag.total.m2_physical,
    diag.total.theta2,diag.total.z2_physical,diag.exterior.volume,diag.exterior.h2,
    diag.exterior.m2_physical,diag.exterior.theta2,diag.exterior.z2_physical};
  const Real fixed[5]={diag.fixed[0].volume,diag.fixed[0].h2,diag.fixed[0].m2_physical,
                       diag.fixed[0].theta2,diag.fixed[0].z2_physical};
  std::copy_n(values,10,result.sums);std::copy_n(fixed,5,result.fixed);
  MPI_Allreduce(MPI_IN_PLACE,result.sums,10,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,result.fixed,5,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  Real maxima[3]={e.compatibility_max,e.clock_max,e.cfl};
  MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  result.compatibility=maxima[0];result.clock=maxima[1];result.cfl=maxima[2];
  for(Real value:result.sums)Require(std::isfinite(value)&&value>=0,"invalid original norm sum");
  for(Real value:result.fixed)Require(std::isfinite(value)&&value>=0,"invalid original fixed-cut sum");
  Require(std::isfinite(maxima[0])&&maxima[0]<=1e-7&&std::isfinite(maxima[1])&&
          std::isfinite(maxima[2]),"invalid original compatibility/clock/CFL");
  const auto after_bulk=Records(e,0),after_surface=Records(e,1);
  All(before_bulk==after_bulk&&before_surface==after_surface,"diagnostics changed chi history");
  return result;
}
'''


def generate(output):
    source=Path(__file__).parent
    original=source/'legacy_n12_benchmark.cpp'
    for name,digest in [('legacy_n12_benchmark.cpp',WRAPPER),('legacy_n12_benchmark.py',BUILDER),
                        ('legacy_stage_trace_gpu.py',HELPER)]:
        require(sha(source/name)==digest,'qualified measurement source changed: '+name)
    text=original.read_text();original_text=text;changes=[]
    def replace(old,new,label):
        nonlocal text
        require(text.count(old)==1,'ambiguous source change: '+label)
        text=text.replace(old,new);changes.append(dict(label=label,old=old,new=new))
    replace('#include "tst/hyperbolic_shift_3d/endpoint_policy/fastflow_history.hpp"\n',
            '', 'remove unavailable diagnostic-only FastFlow helper')
    begin=text.index('Diagnostics Diagnose(');end=text.index('}  // namespace legacy_n12_benchmark',begin)
    replace(text[begin:end],DIAGNOSTIC,'original full/fixed diagnostic API only')
    replace('    All(controls.cells_per_axis==12&&controls.nlim==end_cycle&&controls.dt==dt&&\n'
            '        controls.tlim==2&&controls.ko==.1&&controls.chi_floor==1e-10&&!controls.track_puncture,\n',
            '    ParameterInput policy_input;std::ifstream policy_file(input);policy_input.LoadFromStream(policy_file);\n'
            '    All(policy_input.GetInteger("mesh","cells_per_axis")==12&&controls.nlim==end_cycle&&controls.dt==dt&&\n'
            '        controls.tlim==2&&controls.ko==.1&&controls.chi_floor==1e-10&&\n'
            '        !policy_input.GetBoolean("tracker","enabled"),\n',
            'validate N12/fixed-origin input using original ParameterInput API')
    replace('const double write_start=MPI_Wtime();Snapshot(e,out);Kokkos::fence();',
            'const double write_start=MPI_Wtime();Snapshot(e,out);\n'
            '    e.WriteCheckpoint((out/"endpoint-restart.bin").string(),2,dt,true,mesh.amr_cartesian_core);\n'
            '    Kokkos::fence();','save original accepted endpoint for separate diagnostics')
    replace('legacy-n12-matched-benchmark-v1','legacy-n12-original-control-benchmark-v1','identify original control')
    replace('full_and_horizon_exterior_sums','full_and_fixed_exterior_sums','identify original fixed-cut observable')
    replace('diagnostic_and_horizon_output_seconds','diagnostic_output_seconds','separate control diagnostic cost')
    # Mechanical inversion proves no seed-state, warmup or timed Step statement
    # changed outside explicit input validation and untimed diagnostic/output sites.
    inverse=text
    for change in reversed(changes):
        if change['new']:
            require(inverse.count(change['new'])==1,'ambiguous inverse change')
            inverse=inverse.replace(change['new'],change['old'])
        else:
            anchor='#include "tst/hyperbolic_shift_3d/endpoint_policy/run_controls.hpp"\n'
            require(inverse.count(anchor)==1,'missing include inversion anchor')
            inverse=inverse.replace(anchor,anchor+change['old'])
    require(inverse==original_text,'control overlay changed unrelated source')
    output.mkdir(parents=True,exist_ok=False)
    (output/'legacy_n12_benchmark.cpp').write_text(text)
    for name in ('legacy_n12_benchmark.py','legacy_stage_trace_gpu.py','legacy_n12_benchmark_contract.md'):
        shutil.copyfile(source/name,output/name)
    shutil.copyfile(__file__,output/Path(__file__).name)
    proof=dict(schema='legacy-n12-original-control-entry-overlay-v1',passed=True,
        source_sha256={str((source/name).resolve()):sha(source/name) for name in
            ('legacy_n12_benchmark.cpp','legacy_n12_benchmark.py','legacy_stage_trace_gpu.py',
             'legacy_n12_benchmark_contract.md',Path(__file__).name)},
        changes=changes,exact_inverse_to_qualified_measurement_entry=True,
        timed_step_statements_unchanged=True,archive_numerical_sources_modified=False,
        separate_endpoint_diagnostics_required=True,gpu_execution=False,
        output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'generation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(passed=True,output=str(output.resolve()),
                         generation_sha256=sha(output/'generation.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output)
