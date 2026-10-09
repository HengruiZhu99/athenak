"""Independent high precision radial formulas versus saved owner C++ Cartesian jets.

No compilation or source API call. New arithmetic uses the previously independently
checked radial oracle, with closed Cartesian chain rules independent of owner Taylor.
"""
from pathlib import Path
import hashlib,json,sys,time
import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ORACLE=ROOT/'build-layer-research/continuum/einstein-reference-jet-oracle-20261009'
sys.path.insert(0,str(ORACLE))
from radial_oracle import exact,component,side_at
from cartesian_composition import scalar_cart,vector_cart,tensor_cart

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ORDER=[(i,j,k)for i in range(4)for j in range(4-i)for k in range(4-i-j)]

def values(r,side):
    c=lambda key:component(r,side,key,False)+component(r,side,key,True)
    o,a,chi,P,gr,ar,at=[c(k)for k in ['omega','alpha','chi','P','g_radial','A_radial','A_tangent']]
    bar=gr/chi;phys=bar/o**2;tan=1/o**2
    return dict(omega=o,alpha=a,chi=chi,P=P,physical_lapse=a/o,
                beta=c('beta'),g_radial=gr,g_tangent=chi,
                bar_radial=bar,bar_tangent=mp.mpf(1),
                physical_radial=phys,physical_tangent=tan,
                A_radial=ar,A_tangent=at,
                K_radial=ar/(o*chi)+phys*P/3,
                K_tangent=at/(o*chi)+tan*P/3)

def main():
    plan=json.loads((HERE/'plan.json').read_text())
    for path,h in plan['pins'].items():assert sha(ROOT/path)==h,path
    assert sha(__file__)==plan['source_sha256']
    out=HERE/'attempt001';out.mkdir(exist_ok=False);start=time.monotonic()
    saved=[json.loads(line)for line in (ROOT/plan['native_export']).read_text().splitlines()]
    assert len(saved)==8
    results=[]
    for digits in [100,130]:
        rows=[]
        with mp.workdps(digits):
            for rowidx,row in enumerate(saved):
                xyz=[exact(x)for x in row['point']];r=mp.sqrt(sum(x*x for x in xyz));side=side_at(r)
                jets={key:[mp.diff(lambda q:values(q,side)[key],r,n)for n in range(4)]for key in values(r,side)}
                for name,actual in row['fields'].items():
                    assert len(actual)==20
                    for m,native in zip(ORDER,actual):
                        if name in ['omega','alpha','chi','P','physical_lapse']:
                            expected=scalar_cart(jets[name],xyz,m)
                        elif name.startswith('beta'):
                            expected=vector_cart(jets['beta'],xyz,m,int(name[4:]))
                        else:
                            family=name[:-2];i,j=map(int,name[-2:])
                            expected=tensor_cart(jets[family+'_radial'],jets[family+'_tangent'],xyz,m,i,j)
                        err=abs(exact(native)-expected)/max(1,abs(expected))
                        rows.append({'point':rowidx,'field':name,'multiindex':m,'native':native,
                                     'expected':mp.nstr(expected,digits),'native_scaled':float(err)})
        results.append(rows)
        (out/('comparisons-%d.json'%digits)).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    with mp.workdps(150):
        precision=max(abs(mp.mpf(lo['expected'])-mp.mpf(hi['expected']))/max(1,abs(mp.mpf(hi['expected'])))for lo,hi in zip(*results))
    maximum=max(row['native_scaled']for rows in results for row in rows)
    worst=max(results[1],key=lambda row:row['native_scaled'])
    receipt={'scope':'saved C++ reference Cartesian exports only; no source/API call, coordinate-lift test, Cdot or evolution',
             'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'plan.json'),
             'rows_per_precision':len(results[0]),'precisions':[100,130],
             'native_scaled_max':maximum,'precision_scaled_max':float(precision),'worst':worst,
             'native_threshold':2e-10,'precision_threshold':1e-80,'seconds':time.monotonic()-start,
             'passed':maximum<=2e-10 and precision<=mp.mpf('1e-80')}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2));assert receipt['passed']

if __name__=='__main__':main()
