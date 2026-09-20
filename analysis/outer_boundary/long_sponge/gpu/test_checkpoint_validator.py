from pathlib import Path
import argparse,importlib.util,json,struct,tempfile
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('validator',here/'check_minkowski_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
parser=argparse.ArgumentParser(description='Reject a finite but indefinite ghost metric in a real zero checkpoint.')
parser.add_argument('--fixture',type=Path,required=True,help='One-rank evolved M0 zero run with cycle3 final checkpoint')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();fixture=args.fixture
valid=m.validate(fixture,1,3,True);assert valid['passed']
source=next(p for p in (fixture/'rst/rank_00000000').glob('*.rst') if m.checkpoint(p)['cycle']==3)
raw=bytearray(source.read_bytes());end=raw.index(b'<par_end>\n')+len(b'<par_end>\n');total,level=struct.unpack_from('<ii',raw,end);ng,nx,ny,nz=struct.unpack_from('<19i',raw,end+8+72+76)[:4];cells=(nx+2*ng)*(ny+2*ng)*(nz+2*ng);payload=end+276+20*total;stride=struct.unpack_from('<Q',raw,payload-8)[0];offset=payload+stride-25*cells*8
# A positive determinant is insufficient: diag(-1,-1,+1) at one ghost corner.
struct.pack_into('<d',raw,offset+cells*8,-2.);struct.pack_into('<d',raw,offset+4*cells*8,-2.)
with tempfile.TemporaryDirectory(prefix='checkpoint-test-',dir=here) as tmp:
 run=Path(tmp);d=run/'rst/rank_00000000';d.mkdir(parents=True);(d/source.name).write_bytes(raw);bad=m.validate(run,1,3,False,False)
 assert not bad['passed'] and bad['all_payload_finite'] and bad['invalid_metric_cells_including_ghosts']==1
 assert bad['invalid_samples'][0]['values']['detg']==1 and bad['invalid_samples'][0]['values']['gxx']==-1
result={'valid_real_fixture_passed':True,'finite_positive_determinant_indefinite_ghost_rejected':True,'invalid_sample':bad['invalid_samples'][0]};args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
