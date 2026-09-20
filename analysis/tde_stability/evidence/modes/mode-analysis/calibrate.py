from mode_operator import *
op=Operator('amplitude_calibration')
v=np.fromfile(ROOT/'late_seed_500M.bin').reshape(SHAPE)
rs={}
for e in [1e-2,3e-3,1e-3,3e-4,1e-4,3e-5]:
    rs[e]=op.response(v,steps=40,epsilon=e,label=f'eps{e:g}')
    rs[e].tofile(ROOT/f'response40_eps{e:g}.bin')
    if len(rs)>1:
        prev=list(rs)[-2]
        print(e,discrepancy(rs[e],rs[prev]),flush=True)
r={str(e):discrepancy(x,rs[1e-3]) for e,x in rs.items()}
(ROOT/'amplitude-calibration.json').write_text(json.dumps(r,indent=2)+'\n')
