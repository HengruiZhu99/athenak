"""Parse frozen decimal data only; no geometry/kernel imports or calls."""
from pathlib import Path
from decimal import Decimal,getcontext
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PAYLOAD=ROOT/'build-layer-research/continuum/nonlinear-minkowski-wave-map-oracle-20261009/attempt002/oracle-110.json'
def main():
    getcontext().prec=256
    assert sha(PAYLOAD)=='411594e363e92e0d1998150cc19b4d627549cda6ec625841282f430461b84f38'
    v=json.loads(PAYLOAD.read_text());assert v['digits']==110 and len(v['cases'])==48
    sym=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    fields=[('chi',),*[("metric",i,j)for i,j in sym],('P',),*[("A",i,j)for i,j in sym],*[("Lambda",i)for i in range(3)],('Theta',),('alpha',),*[("beta",i)for i in range(3)]]
    config=set(range(7))|set(range(18,22));rows=[];expected=[]
    def take(source,path):
        for key in path:source=source[key]
        return source
    def table(jet):
        assert isinstance(jet['order'],int)
        result={tuple(e['multiindex']):e['value']for e in jet['ordinary']}
        assert len(result)==len(jet['ordinary'])
        assert all(len(m)==4 and sum(m)<=jet['order']for m in result)
        return result
    def extract(jet,second):
        assert jet['order']>=2 if second else jet['order']>=1
        t=table(jet);order=[(0,0,0,0)]+[tuple(int(k==i+1)for k in range(4))for i in range(3)]
        if second:order +=[tuple(int(k==i+1)+int(k==j+1)for k in range(4))for i in range(3)for j in range(3)]
        return [float(t[m])for m in order]
    for index,case in enumerate(v['cases']):
        assert case['passed']
        point=[float(x)for x in case['point']];assert len(point)==4
        row=point[:];rates=[]
        for col,path in enumerate(fields):
            jet=take(case['fields'],path);row+=extract(jet,col in config)
            rate=take(case['exact_time_derivatives'],path)
            assert Decimal(str(rate))==Decimal(table(jet)[(1,0,0,0)]),(index,path)
            rates.append(float(rate))
        row+=extract(case['fields']['omega'],True);assert len(row)==204
        assert all(math.isfinite(x)for x in row+rates)
        o=float(table(case['fields']['omega'])[(0,0,0,0)])
        connection=[float(x)for a in case['scaled_reference_connection']for b in a for x in b];assert len(connection)==36
        # The source export is unscaled Fbar. Multiply in Decimal before the
        # single binary64 conversion, preserving the declared comparison scale.
        od=Decimal(table(case['fields']['omega'])[(0,0,0,0)])
        source=[float(od*Decimal(x))for x in case['source_Fbar']]
        expected.append({'index':index,'point_name':case['point_name'],'point':point,'epsilon':float(case['epsilon']),
          'rhs22':rates,'scaled_reference_connection36':connection,'scaled_source4':source,'omega':o})
        rows.append(' '.join(format(x,'.17g')for x in row)+'\n')
    data=''.join(rows).encode();out=HERE/'prepared-input001';out.mkdir(exist_ok=False)
    (out/'input.txt').write_bytes(data);(out/'expected.json').write_text(json.dumps(expected,indent=2,allow_nan=False)+'\n')
    receipt={'passed':True,'scope':'decimal parsing/schema/time-coefficient readback only; no scientific kernel query',
      'payload_sha256':sha(PAYLOAD),'source_sha256':sha(__file__),'input_sha256':sha(out/'input.txt'),
      'expected_sha256':sha(out/'expected.json'),'cases':48,'columns_per_input_row':204,
      'raw22_field_paths':[list(p)for p in fields],'configuration_columns':sorted(config),
      'conversion':'ordinary decimal derivatives directly to binary64 once; scaled source product in Decimal before binary64'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
