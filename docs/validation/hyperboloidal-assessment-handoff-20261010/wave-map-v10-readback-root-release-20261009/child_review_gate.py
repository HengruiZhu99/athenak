"""UNEXECUTED: mandatory stdlib review gate before exec of unchanged frozen v10."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=['primary','radial_pair','angular_pair'])
    ap.add_argument('--release-sha256', required=True)
    args = ap.parse_args()
    if not (sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize == 0):
        raise RuntimeError('child gate requires -I -B and optimize0')
    if sha(HERE/'release.json') != args.release_sha256:
        raise RuntimeError('exact frozen release required')
    release = load(HERE/'release.json')
    pins = dict(release['pins'])
    pins[str(HERE/'authorization.json')] = release['authorization_sha256']
    pins[str(HERE/'review.json')] = release['review_sha256']
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected child input '+path)
    auth = load(HERE/'authorization.json')
    root_review = load(HERE/'review.json')
    independent = load(release['independent_review_receipt'])
    if not (sha(release['independent_review_receipt']) == release['independent_review_receipt_sha256']
            and sha(release['independent_review_index']) == release['independent_review_index_sha256']
            and independent.get('passed') is True and independent.get('inputs_unchanged') is True
            and independent.get('source_review_only') is True
            and independent.get('reviewed_source_index_sha256') == auth['source_index_sha256']
            and root_review.get('passed') is True
            and root_review.get('reviewed_source_index_sha256') == auth['source_index_sha256']
            and auth['child_review_gate_required'] is True
            and auth['independent_saved_matrix_readback_authorized'] is True
            and auth['process_group_cap_seconds'] == release['process_group_cap_seconds'] == 900
            and auth['cap_is_new_not_historical'] is True
            and auth['generator_eigenvalues_authorized'] is False):
        raise RuntimeError('exact successful independent/root review required before numerical import')
    suite = Path(release['suite'])
    recipe = load(suite/'recipe.json')
    output = suite/'attempts'/('independent-'+args.case+'001')
    if output.exists():
        raise RuntimeError('fresh fixed child destination required')
    for key, value in recipe['environment'].items():
        if os.environ.get(key) != value:
            raise RuntimeError('exact child environment differs '+key)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('one-thread child environment required')
    if os.environ.get('PYTHONOPTIMIZE') != '0':
        raise RuntimeError('optimize0 child required')
    command = [recipe['python'],'-B','-s',str(suite/'verify_retained.py'),
               '--authorization',str(HERE/'authorization.json'),
               '--authorization-sha256',release['authorization_sha256'],
               '--case',args.case,'--output',str(output)]
    # Same process group; exec preserves the root cap for gate plus actual child.
    os.execve(recipe['python'], command, dict(os.environ))


if __name__ == '__main__':
    main()
