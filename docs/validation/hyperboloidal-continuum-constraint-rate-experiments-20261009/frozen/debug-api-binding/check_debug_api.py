#!/usr/bin/env python3
"""Three Release/ASan API calls; source binding only, no scientific batch."""
from pathlib import Path
import hashlib,json,math,subprocess,time
P=Path(__file__).resolve().parents[2]/'boundary/total-j-finite-rb-control-20261009'
O=Path(__file__).resolve().parent/'debug-api-binding'
assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
release=P/'radial-bridge-release';debug=P/'build-attempts/debug-003/radial-bridge-debug'
assert sha(release)=='2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15'
assert sha(debug)=='75193a3b023eb0004f285fabb1a532efd24881db4cb660b3fa96e084a6e95507'
assert sha(P/'constraint_rate_api.hpp')=='d60ea8266abebf5263eb78b81997b60af472dc7b1fe102bd6f2c21baca1a6017'
auth={'continuum_constraint_rate_api_admitted':True,'executable_sha256':sha(debug),
      'api_sha256':sha(P/'constraint_rate_api.hpp'),
      'scope':'three API calls, each with origin/core/transition representatives, per boundary authorization; no duplicate full scientific batch'}
(O/'authorization.json').write_text(json.dumps(auth,indent=2)+'\n')
n=[1/math.sqrt(14),2/math.sqrt(14),3/math.sqrt(14)]
points=[[0.,0.,0.],[.025*v for v in n],[.5*v for v in n]]
rows={'--manufactured-rate-batch':[[2,1,10,0,6,*points[0]],[2,1,18,0,2,*points[1]],[1,1,2,0,5,*points[2]]],
      '--constraint-rate-batch':[],'--subsidiary-batch':[]}
for mode,count in [('--constraint-rate-batch',22),('--subsidiary-batch',8)]:
    for x in points:
        row=list(x)
        for f in range(count):
            row += [.003*math.sin(f+1)]+[.01*(f+1)*(i+1) for i in range(3)]+[
                .005*math.sin((f+1)*(i+j+1)) for i in range(3) for j in range(3)]
        rows[mode].append(row)
results=[];before={str(p):sha(p) for p in [Path(__file__),release,debug,P/'constraint_rate_api.hpp']}
for mode,data in rows.items():
    text=''.join(' '.join(format(v,'.17g') for v in row)+'\n' for row in data)
    stem=mode.removeprefix('--');(O/(stem+'.input')).write_text(text);output=[];calls=[]
    for name,exe in [('release',release),('debug',debug)]:
        command=[str(exe),mode];start=time.monotonic();r=subprocess.run(command,input=text,text=True,capture_output=True)
        (O/(stem+'.'+name+'.stdout')).write_text(r.stdout);(O/(stem+'.'+name+'.stderr')).write_text(r.stderr)
        values=[[float(v) for v in line.split()] for line in r.stdout.splitlines()]
        calls.append({'command':command,'seconds':time.monotonic()-start,'returncode':r.returncode,
            'stderr_bytes':len(r.stderr.encode()),'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest()})
        assert r.returncode==0 and not r.stderr and len(values)==3
        assert all(len(v)==(30 if mode=='--manufactured-rate-batch' else 8) and all(map(math.isfinite,v)) for v in values)
        output.append(values)
    absolute=max(abs(a-b) for x,y in zip(*output) for a,b in zip(x,y))
    scale=max(1.,*(math.sqrt(math.fsum(v*v for v in x)) for matrix in output for x in matrix))
    results.append({'mode':mode,'rows':3,'calls':calls,'absolute_peak_difference':absolute,
                    'scaled_peak_difference':absolute/scale,'passed':absolute/scale<=5e-11})
after={str(p):sha(p) for p in [Path(__file__),release,debug,P/'constraint_rate_api.hpp']}
report={'passed_three_mode_release_asan_binding':all(r['passed'] for r in results) and before==after,
        'scope':auth['scope'],'results':results,'source_before':before,'source_after':after,'sources_unchanged':before==after}
(O/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2));assert report['passed_three_mode_release_asan_binding']
