#!/bin/bash -l
# Link a test-only checkpoint writer against the frozen production objects.
set -euo pipefail
module load oneapi/release/2025.3.1
root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-regularity-20261110
out=$root/pulse-writer-v1
[[ ! -e "$out" && -f "$root/scri_pulse.cpp" ]]
mkdir "$out"
cp "$root/scri_pulse.cpp" "$out/source.cpp"
sha256sum -c "$root/build-v1/executable.sha256" > "$out/production-check.txt"
module list > "$out/modules.txt" 2>&1
python3 - "$root" "$out" > "$out/build.log" 2>&1 <<'PY'
import hashlib,json,pathlib,re,shlex,subprocess,sys
root,out=map(pathlib.Path,sys.argv[1:]);build=root/'build-v1'
flags=(build/'src/CMakeFiles/athena.dir/flags.make').read_text()
options=[]
for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
 options+=shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M).group(1))
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
compiler=link[0]
version=subprocess.check_output([compiler,'--version'],universal_newlines=True)
assert '2025.3.2' in version,version
(out/'compiler.txt').write_text(version)
compile=[compiler,*options,'-c',str(out/'source.cpp'),'-o',str(out/'pulse.o')]
assert link.count('CMakeFiles/athena.dir/main.cpp.o')==1
link=[str(out/'pulse.o') if v=='CMakeFiles/athena.dir/main.cpp.o' else v for v in link]
link[link.index('-o')+1]=str(out/'scri_pulse')
(out/'commands.json').write_text(json.dumps({'compile':compile,'link':link},indent=2)+'\n')
subprocess.run(compile,check=True)
subprocess.run(link,cwd=str(build/'src'),check=True)
(out/'source.sha256').write_text(hashlib.sha256((out/'source.cpp').read_bytes()).hexdigest()+'  source.cpp\n')
PY
sha256sum "$out/scri_pulse" > "$out/executable.sha256"
sha256sum -c "$root/build-v1/executable.sha256" >> "$out/production-check.txt"
printf 'Pulse writer build complete: %s\n' "$out"
