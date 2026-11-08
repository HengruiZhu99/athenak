#!/usr/bin/env python3
"""Package/build the reviewed AMR admission overlay without changing native files.

The candidate retains original main and every original object except the three
listed admission objects. It is a different executable from the frozen N12
baseline. This tool neither launches an MPI process nor submits a GPU job.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

SOURCES = ('src/z4c_hyperboloidal/z4c_hyperboloidal.hpp',
           'src/driver/driver.cpp', 'src/mesh/mesh_refinement.cpp',
           'src/z4c_hyperboloidal/z4c_hyperboloidal_lifecycle.cpp',
           'src/z4c_hyperboloidal/fixed_step_admission.hpp')
REPLACED = tuple(name for name in SOURCES if name.endswith('.cpp'))
SAFE_DT = .0002


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def exact_patch(root, overlay):
    result = ''
    for name in SOURCES:
        original = root/name
        before = original.read_text() if original.exists() else ''
        result += ''.join(difflib.unified_diff(before.splitlines(True),
            (overlay/name).read_text().splitlines(True),
            fromfile='a/'+name if original.exists() else '/dev/null', tofile='b/'+name))
    return result


def package(args):
    root, reviewed, output = args.root.resolve(), args.reviewed_overlay.resolve(), args.output.resolve()
    generation = json.loads((reviewed/'generation.json').read_text())
    proof = json.loads((reviewed/'restart-proof/results.json').read_text())
    require(generation['schema'] == 'native-amr-admission-clean-overlay-v1'
            and generation['runtime_written'] is False, 'not the clean reviewed overlay')
    require(set(generation['overlay']) == set(SOURCES), 'unexpected candidate source set')
    require(proof['passed'] and proof['dt'] == SAFE_DT
            and proof['genuine_checkpoint_deserialization']
            and proof['restart_without_input_or_dt_override']
            and proof['frozen_runtime_unchanged'], 'clean ordinary restart proof is incomplete')
    require(sha(reviewed/'provenance.json') == proof['build_provenance_sha256'],
            'CPU proof does not bind reviewed build')
    patch = reviewed/'production.patch'
    require(sha(patch) == generation['production_patch_sha256'], 'reviewed patch changed')
    require(exact_patch(root, reviewed/'overlay') == patch.read_text(),
            'patch/overlay/frozen source mismatch')
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        source = reviewed/'overlay'/name
        require(sha(source) == generation['overlay'][name], 'reviewed source changed '+name)
        target = output/'overlay'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source, target in ((patch, 'production.patch'),
                           (reviewed/'generation.json', 'generation.json'),
                           (reviewed/'provenance.json', 'cpu-build-proof.json'),
                           (reviewed/'restart-proof/results.json', 'cpu-restart-proof.json')):
        shutil.copyfile(source, output/target)
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    record = dict(schema='native-amr-admission-package-v1', candidate=True,
        fixed_dt=SAFE_DT, files=files,
        baseline_sources={name: sha(root/name) if (root/name).exists() else None for name in SOURCES},
        replaced_sources=list(REPLACED), original_main_retained=True,
        production_patch_sha256=sha(patch), cpu_restart_proof_sha256=sha(output/'cpu-restart-proof.json'),
        scope='isolated admission candidate; no frozen runtime modifications; no GPU execution',
        packager_sha256=sha(__file__))
    (output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(package=str(output), production_patch_sha256=sha(patch)), indent=2))


def check_package(folder, root=None):
    folder = Path(folder).resolve()
    record = json.loads((folder/'manifest.json').read_text())
    require(record['schema'] == 'native-amr-admission-package-v1' and record['candidate']
            and record['fixed_dt'] == SAFE_DT and record['replaced_sources'] == list(REPLACED)
            and record['original_main_retained'], 'invalid candidate package')
    for name, digest in record['files'].items():
        require(sha(folder/name) == digest, 'candidate package file changed '+name)
    require(sha(folder/'production.patch') == record['production_patch_sha256'], 'candidate patch changed')
    if root is not None:
        for name, digest in record['baseline_sources'].items():
            source = root/name
            require((sha(source) if source.exists() else None) == digest,
                    'candidate baseline source mismatch '+name)
        require(exact_patch(root, folder/'overlay') == (folder/'production.patch').read_text(),
                'candidate patch does not reproduce exact overlay')
    return record


def compile_entries(build, database=None):
    """Use actual CMake commands; Makefile fallback does not invent flags."""
    database = database or build/'compile_commands.json'
    if database.is_file():
        return json.loads(database.read_text()), {str(database): sha(database)}
    directory = build/'src/CMakeFiles/athena.dir'
    flags, recipes = directory/'flags.make', directory/'build.make'
    variables = {k: shlex.split(v) for k,v in re.findall(r'^(CXX_\w+) = (.*)$', flags.read_text(), re.M)}
    entries = []
    for line in recipes.read_text().splitlines():
        if not line.startswith('\t') or ' -c ' not in line:
            continue
        command = shlex.split(line.strip())
        require(command[:1] == ['cd'] and command[2] == '&&', 'unsupported CMake compile recipe')
        expanded = []
        for token in command[3:]:
            expanded.extend(variables[token[2:-1]] if token.startswith('$(') else [token])
        require(expanded.count('-c') == 1, 'ambiguous CMake compile recipe')
        entries.append(dict(directory=command[1], file=expanded[expanded.index('-c')+1], arguments=expanded))
    require(entries, 'no actual CMake compilation commands')
    return entries, {str(p): sha(p) for p in (flags, recipes)}


def replace_compile(entry, source, obj, overlay):
    original = entry.get('arguments') or shlex.split(entry['command'])
    cwd = Path(entry['directory']).resolve()
    original_source = (cwd/entry['file']).resolve()
    command, count, i = [], 0, 0
    while i < len(original):
        token = original[i]
        if token == '-o':
            command.extend(['-o', str(obj)]); i += 2; continue
        if token in ('-MF', '-MT', '-MQ'):
            i += 2; continue
        if token in ('-MD', '-MMD', '-MP'):
            i += 1; continue
        if not token.startswith('-') and (cwd/token).resolve() == original_source:
            command.append(str(source)); count += 1
        else:
            command.append(token)
        i += 1
    require(count == 1 and command.count('-o') == 1 and command.count('-c') == 1,
            'unrecognized native compilation command')
    command[1:1] = ['-I'+str(overlay/'src'), '-iquote', str(original_source.parent)]
    command.extend(['-MMD', '-MF', str(obj)+'.d', '-MT', str(obj)])
    return command, cwd


def check_build(record, finished=True):
    require(record['schema'] == 'native-amr-admission-build-v1', 'wrong candidate build schema')
    for group in ('sources', 'objects', 'recipe_inputs', 'dependencies'):
        for name, digest in record.get(group, {}).items():
            require(sha(name) == digest, 'changed candidate build input '+name)
    for key in ('native_executable', 'package_manifest', 'link_file', 'original_main_object'):
        require(sha(record[key]) == record[key+'_sha256'], 'changed candidate '+key)
    check_package(Path(record['package_manifest']).parent, Path(record['root']))
    if finished:
        require(record['status'] == 'built' and record['frozen_inputs_unchanged'], 'candidate build incomplete')
        require(sha(record['executable']) == record['executable_sha256'], 'candidate executable changed')
        for replacement in record['replacements'].values():
            require(sha(replacement['object']) == replacement['object_sha256'], 'candidate object changed')


def build(args):
    directory, output, package_dir = args.build.resolve(), args.output.resolve(), args.package.resolve()
    entries, recipes = compile_entries(directory, args.compile_database)
    main = [entry for entry in entries if Path(entry['file']).as_posix().endswith('/src/main.cpp')]
    require(len(main) == 1, 'need original main compilation entry')
    root = (Path(main[0]['directory'])/main[0]['file']).resolve().parents[1]
    package_record = check_package(package_dir, root)
    output.mkdir(parents=True, exist_ok=False)
    (output/'compile_commands.json').write_text(json.dumps(entries, indent=2)+'\n')
    overlay = package_dir/'overlay'
    link_file = directory/'src/CMakeFiles/athena.dir/link.txt'
    link = shlex.split(link_file.read_text()); link_cwd = link_file.parents[2]
    require(link.count('-o') == 1, 'unknown original link command')
    objects = [(link_cwd/token).resolve() for token in link if token.endswith(('.o','.a','.so','.dylib'))]
    original_main = [path for path in objects if path.as_posix().endswith('athena.dir/main.cpp.o')]
    require(len(original_main) == 1, 'original main not linked exactly once')
    replacements, commands, rules = {}, [], ['.PHONY: all']
    for name in REPLACED:
        selected = [entry for entry in entries if (Path(entry['directory'])/entry['file']).resolve() == root/name]
        require(len(selected) == 1, 'missing/ambiguous compile entry '+name)
        obj = output/(Path(name).name+'.o')
        command, cwd = replace_compile(selected[0], overlay/name, obj, overlay)
        if name.endswith('_lifecycle.cpp'):
            require('-ffp-contract=off' in command, 'admission lifecycle contraction policy missing')
        suffix = 'athena.dir/'+name.removeprefix('src/')+'.o'
        positions = [i for i, token in enumerate(link) if token.endswith(suffix)]
        require(len(positions) == 1, 'missing/ambiguous native object '+name)
        old = (link_cwd/link[positions[0]]).resolve()
        link[positions[0]] = str(obj)
        replacements[name] = dict(original_object=str(old), original_object_sha256=sha(old), object=str(obj))
        commands.append(dict(source=name, command=command, cwd=str(cwd)))
        rules += [str(obj)+':', '\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(command)]
    executable = output/'athena-admission'
    link[link.index('-o')+1] = str(executable)
    rules[1:1] = ['all: '+str(executable)]
    rules += [str(executable)+': '+' '.join(r['object'] for r in replacements.values()),
              '\tcd '+shlex.quote(str(link_cwd))+' && '+shlex.join(link)]
    makefile = output/'candidate.mk'; makefile.write_text('\n'.join(rules)+'\n')
    sources = sorted(list((root/'src').rglob('*.cpp'))+list((root/'src').rglob('*.hpp')))
    sources += [overlay/name for name in SOURCES]+[Path(__file__).resolve()]
    record = dict(schema='native-amr-admission-build-v1', status='building', candidate=True,
        root=str(root), native_executable=str(directory/'src/athena'),
        native_executable_sha256=sha(directory/'src/athena'),
        package_manifest=str(package_dir/'manifest.json'), package_manifest_sha256=sha(package_dir/'manifest.json'),
        production_patch_sha256=package_record['production_patch_sha256'], fixed_dt=SAFE_DT,
        original_main_object=str(original_main[0]), original_main_object_sha256=sha(original_main[0]),
        original_main_retained=True, replacements=replacements, compile_commands=commands,
        link_file=str(link_file), link_file_sha256=sha(link_file), link_command=link, link_cwd=str(link_cwd),
        recipe_inputs=recipes, sources={str(p):sha(p) for p in sources},
        objects={str(p):sha(p) for p in objects}, executable=str(executable))
    path = output/'provenance.json'; path.write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    with (output/'build.log').open('w') as log:
        result = subprocess.run(['make','-j64','-f',str(makefile)], stdout=log, stderr=subprocess.STDOUT)
    record.update(build_seconds=time.monotonic()-start, build_exit_code=result.returncode,
                  status='failed' if result.returncode else 'built')
    path.write_text(json.dumps(record, indent=2)+'\n')
    check_build(record, finished=False); result.check_returncode()
    dependencies = {}
    for entry in commands:
        obj = replacements[entry['source']]['object']
        replacements[entry['source']]['object_sha256'] = sha(obj)
        text = Path(obj+'.d').read_text().replace('\\\n', ' ')
        for token in shlex.split(text.split(':',1)[1]):
            p = (Path(entry['cwd'])/token).resolve(); dependencies[str(p)] = sha(p)
    record.update(dependencies=dependencies, executable_sha256=sha(executable), frozen_inputs_unchanged=True)
    path.write_text(json.dumps(record, indent=2)+'\n'); check_build(record)
    print(json.dumps({key:record[key] for key in ('status','native_executable_sha256','executable_sha256',
          'production_patch_sha256','frozen_inputs_unchanged')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('package'); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reviewed-overlay',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('build'); p.add_argument('--build',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compile-database',type=Path)
    p = sub.add_parser('verify'); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        check_build(json.loads((args.output/'provenance.json').read_text())); print('Verified isolated candidate and frozen baseline')
    else:
        {'package':package,'build':build}[args.command](args)
