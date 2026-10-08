"""Separate physical ADM gauge storage must preserve ordinary Cauchy evolution."""
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .overhaul_utils import ROOT, run_case

spec = importlib.util.spec_from_file_location(
    'binary_reader', ROOT / 'vis/python/bin_convert.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

SETTINGS = """
<mesh>
x1min=-2
x1max=2
x2min=-2
x2max=2
x3min=-2
x3max=2
<problem>
amp=0.0001
punc_1_rest_mass=0.1
punc_2_rest_mass=0.1
punc_1_velocity_x1=0.1
punc_2_velocity_x1=-0.1
<time>
nlim=4
tlim=1
cfl_number=0.01
<output1>
file_type=bin
variable=z4c
id=z4c
dt=0.0001
ghost_zones=true
<output2>
file_type=bin
variable=adm
id=adm
dt=0.0001
ghost_zones=true
"""


def fields(directory, kind, last=True):
    files = sorted((directory / 'bin').glob(f'*.{kind}.*.bin'))
    assert files
    data = reader.read_binary(str(files[-1] if last else files[0]))['mb_data']
    assert all(np.isfinite(values).all() for values in data.values())
    return data


def compare(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        np.testing.assert_allclose(right[name], value, atol=1e-13, rtol=1e-11,
                                   err_msg=name)


def check_gauge_copy(directory, last=True):
    z, adm = fields(directory, 'z4c', last), fields(directory, 'adm', last)
    for suffix in ('alpha', 'betax', 'betay', 'betaz'):
        # The output reader preserves the C++ variable names.
        name = 'z4c_' + suffix
        np.testing.assert_array_equal(adm['adm_' + suffix], z[name])


@pytest.mark.parametrize('problem', ['z4c_linear_wave', 'z4c_superposed_punctures'])
@pytest.mark.parametrize('blocks', [1, 2])
def test_separate_gauge_equivalence(tmp_path, problem, blocks):
    extra = SETTINGS + f'\n<problem>\npgen_name={problem}\n<mesh>\nnx1={8*blocks}\n'
    shared, separate = tmp_path / 'shared', tmp_path / 'separate'
    run_case(shared, extra, slices=False)
    run_case(separate, extra + '\n<adm>\nseparate_z4c_gauge=true\n', slices=False)
    for last in (False, True):
        compare(fields(shared, 'z4c', last), fields(separate, 'z4c', last))
        old, new = fields(shared, 'adm', last), fields(separate, 'adm', last)
        assert set(new) - set(old) == {'adm_alpha', 'adm_betax', 'adm_betay', 'adm_betaz'}
        compare(old, {k: new[k] for k in old})
        check_gauge_copy(separate, last)


def test_separate_gauge_restart(tmp_path):
    settings = SETTINGS + """
<problem>
pgen_name=z4c_superposed_punctures
<adm>
separate_z4c_gauge=true
<output3>
file_type=rst
dt=0.0001
"""
    full, split = tmp_path / 'full', tmp_path / 'split'
    run_case(full, settings, slices=False)
    run_case(split, settings + '\n<time>\nnlim=2\n', slices=False)
    checkpoint = sorted((split / 'rst').glob('*.rst'))[-1]
    executable = str(Path(os.environ['ATHENA_OVERHAUL_EXE']).resolve())
    result = subprocess.run([executable, '-r', str(checkpoint), 'time/nlim=4'],
                            cwd=split, capture_output=True, text=True, timeout=90)
    (split / 'restart.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    compare(fields(full, 'z4c'), fields(split, 'z4c'))
    compare(fields(full, 'adm'), fields(split, 'adm'))
    check_gauge_copy(split)


def test_separate_gauge_amr_outflow(tmp_path):
    settings = SETTINGS + (ROOT / 'tst/inputs/lwave_z4c_bc.athinput').read_text()
    settings += """
<mesh>
nx1=16
nx2=16
nx3=16
<mesh_refinement>
max_nmb_per_rank=128
<time>
nlim=15
<output1>
dt=1
<output2>
dt=1
"""
    shared, separate = tmp_path / 'shared', tmp_path / 'separate'
    run_case(shared, settings, slices=False)
    run_case(separate, settings + '\n<adm>\nseparate_z4c_gauge=true\n', slices=False)
    compare(fields(shared, 'z4c'), fields(separate, 'z4c'))
    old, new = fields(shared, 'adm'), fields(separate, 'adm')
    compare(old, {k: new[k] for k in old})
    # More than the eight original blocks demonstrates actual refinement.
    assert len(next(iter(new.values()))) > 8
    check_gauge_copy(separate)
