"""Build a separate static-SMR ownership probe using the existing partition reader.

No production file/binary is edited. This checks transfer-free ownership changes,
not the uniform-mesh Repartition API or a dynamically regridding algorithm.
"""
from pathlib import Path
import json,hashlib,shlex,subprocess,struct,argparse
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
O=Path(args.output).resolve() if args.output else R/'output/refined-convergence/measured-partition-probe';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=R/'tst/hyperbolic_shift_3d'
body=(source/'amr_evolution.cpp').read_text()
old='    auto descriptor = amr.Descriptor();';assert body.count(old)==1
body=body.replace(old,'''    const char* partition_path=std::getenv("HYP3D_ANALYSIS_PARTITION");
    if(!partition_path||!partition_path[0]||policy!="smr")
      throw std::runtime_error("analysis partition probe requires explicit file and static SMR");
    auto descriptor = amr.Descriptor();''')
old='(n, 2, kind, "", &descriptor)';assert body.count(old)==2
body=body.replace(old,'(n, 2, kind, partition_path, &descriptor)')
(O/'amr_evolution.cpp').write_text(body);(O/'coupled_mpi.cpp').write_bytes((source/'coupled_mpi.cpp').read_bytes())
target=R/'build-refined-convergence/tst/hyperbolic_shift_3d';F=target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir'
flags={}
for line in (F/'flags.make').read_text().splitlines():
    if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
cmd=['/usr/bin/c++']+flags['CXX_DEFINES']+['-DHYPERBOLIC_MEASURED_PARTITION=1']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(source),'-c',str(O/'amr_evolution.cpp'),'-o',str(O/'probe.o')]
link=shlex.split((F/'link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(O/'probe.o');link[link.index('-o')+1]=str(O/'probe')
record=dict(compile=cmd,link=link,production_driver_sha256=sha(source/'amr_evolution.cpp'),analysis_driver_sha256=sha(O/'amr_evolution.cpp'),unchanged_coupled_driver_sha256=sha(O/'coupled_mpi.cpp'),qualification='Static SMR initial ownership only, via existing strict descriptor reader. Does not use fixed-uniform-mesh Repartition. No field interpolation, equation, timestep, ghost stencil or production campaign change. No performance claim before measurement.')
(O/'build.json').write_text(json.dumps(record,indent=2)+'\n')
with (O/'build.log').open('w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(link,cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True)
record['binary_sha256']=sha(O/'probe');(O/'build.json').write_text(json.dumps(record,indent=2)+'\n')
# Derive exact total from the already verified native N12 t0 framing.
p=R/'output/refined-convergence/reference-initial-resolution/n12.checkpoint'
with p.open('rb') as f:
    def zero():
        while True:
            c=f.read(1);assert c
            if c==b'\0':return
    def u64():return struct.unpack('<Q',f.read(8))[0]
    zero();f.read(u64())
    for _ in range(31):zero()
    n,kind,sub=struct.unpack('<3i',f.read(12));assert n==12 and kind==2 and sub==4
    f.seek(16+24+24+128,1);leaves=u64();f.seek(20*leaves,1);length=u64();assert length%31==0;total=length//31
descriptor_source=sha(D/'reference-rank-cost/analysis.json')
for name,cut in [('equal',total//2),('unequal',total//3),('unequal-offset',total//3+17)]:
    (O/(name+'.partition')).write_text(f'HYP3D_COST_PARTITION_V1 12 {total} 2 {descriptor_source}\n0 {cut} {total}\n')
print(json.dumps(dict(output=str(O),n=12,total=total,leaves=leaves,binary_sha256=record['binary_sha256'])))
