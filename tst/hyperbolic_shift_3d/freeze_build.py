"""Bind a just-built executable to its C++ inputs; invoke immediately after build."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('build',type=Path);a=p.parse_args();build=a.build.resolve()
files=sorted([*ROOT.joinpath('src').rglob('*.hpp'),*ROOT.joinpath('tst/hyperbolic_shift_3d').glob('*.cpp'),*ROOT.joinpath('tst/hyperbolic_shift_3d').glob('*.hpp')])
files.extend([ROOT/'CMakeLists.txt',ROOT/'tst/hyperbolic_shift_3d/CMakeLists.txt'])
archive=build/'coupled-source.tar.gz'
with tarfile.open(archive,'w:gz') as out:
 for f in files:out.add(f,arcname=str(f.relative_to(ROOT)))
metadata=dict(binary_sha256=digest(build/'tst/hyperbolic_shift_3d/hyperbolic_shift_3d_coupled'),
 archive_sha256=digest(archive),files={str(f.relative_to(ROOT)):digest(f) for f in files})
(build/'coupled-source.json').write_text(json.dumps(metadata,indent=2)+'\n')
