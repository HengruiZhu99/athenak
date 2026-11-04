"""Build an analysis-only reader from production RF2 source; never rebuild campaign."""
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=ROOT/'output/refined-convergence/endpoint-diagnostic';OUT.mkdir(exist_ok=True)
target=ROOT/'build-refined-convergence/tst/hyperbolic_shift_3d'
src=ROOT/'tst/hyperbolic_shift_3d/amr_evolution.cpp'
text=src.read_text();needle='int main(int argc, char** argv) {'
assert text.count(needle)==1
source=OUT/'reader.cpp';source.write_text(text.replace(needle,'int unused_production_driver(int argc, char** argv) {')+'\n'+(HERE/'endpoint_diagnostic.inc').read_text())
flags={}
for line in (target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
obj=OUT/'reader.o';binary=OUT/'reader'
cmd=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(src.parent),'-c',str(source),'-o',str(obj)]
subprocess.run(cmd,check=True)
link=shlex.split((target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/link.txt').read_text())
link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj)
link[link.index('-o')+1]=str(binary)
subprocess.run(link,cwd=target,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build.json').write_text(json.dumps(dict(source_sha256=sha(src),body_sha256=sha(HERE/'endpoint_diagnostic.inc'),generated_sha256=sha(source),binary_sha256=sha(binary),compile=cmd,link=link,qualification='Separate local analysis binary; campaign source/checkpoints/binaries unchanged.'),indent=2)+'\n')
print(binary)
