"""Prepare a selected screen survivor's prescribed gates; never launch them."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from make_inputs import parse,write
from make_hybrid_controls import candidate,CANDIDATES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',choices=CANDIDATES,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    evidence=a.evidence/'della-cuda';sources={};cases=[]
    def load(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return parse(path.read_text())
    def save(name,p,gate):
        write(a.output/(name+'.athinput'),p)
        cases.append(dict(input=name+'.athinput',gate=gate,qualification='not run'))
    def setup(base,order=6,mask=(.125,.5),half=False):
        p=deepcopy(base);candidate(p,a.candidate,*mask,ceiling=.00625 if half else .0125)
        p['pc_gh']['spatial_order']=str(order)
        p['time']['cfl_number']='.1' if half else '.2'
        for k,v in p.items():
            if k.startswith('output') and v.get('file_type')=='rst':v['dt']='1'
        return p
    masks={'small':(.0625,.25),'standard':(.125,.5),'large':(.25,1)} if a.candidate in ['R4','R16','P1'] else {'standard':(.125,.5)}
    base=evidence/'rate16-qualification'
    for tag,mask in masks.items():
        for order in [6,2]:
            for r in [16,20,24]:
                p=setup(load(base/f'pcgh-l16-o6-rk3-uniform-r{r}-R8/used_input.athinput'),order,mask)
                p['time']['tlim']='12'
                save(f'uniform-r{r}-o{order}-{tag}',p,'uniform resolution/mask')
                p=setup(load((base/f'pcgh-l16-o6-rk3-smr-r{r}-R64/used_input.athinput') if r!=16 else
                    evidence/'single-discretization/pcgh-l16-o6-rk3-smr-r16-R64/used_input.athinput'),order,mask)
                p['time']['tlim']='20'
                save(f'large-smr-r{r}-o{order}-{tag}',p,'large-domain resolution/mask')
            for n in [64,128,256]:
                p=setup(load(base/f'pcgh-l16-o6-rk3-core{n}-R8/used_input.athinput'),order,mask)
                p['time']['tlim']='6'
                save(f'core{n}-o{order}-{tag}',p,'fine-core resolution/mask')
    # Halve both CFL and the research ceiling, at unchanged mask and mesh.
    for n in [128,256]:
        if n==128:
            original=base/'pcgh-l16-o6-rk3-uniform-r16-R8/used_input.athinput';end='12';name='uniform128'
        else:
            original=base/'pcgh-l16-o6-rk3-core256-R8/used_input.athinput';end='6';name='core256'
        for order in [6,2]:
            p=setup(load(original),order,half=True);p['time']['tlim']=end
            save(f'{name}-o{order}-half-step',p,'timestep refinement')
    for order in [6,2]:
        p=setup(load(base/'pcgh-l16-o6-rk3-core256-R8/used_input.athinput'),order)
        p['time']['tlim']='6'
        for d in [1,2,3]:
            p['refined_region8'][f'x{d}min']='-0.062499'
            p['refined_region8'][f'x{d}max']='0.062499'
        save(f'core256-o{order}-boundary-outward',p,'innermost boundary sensitivity; actual tree must be recorded')
    for order in [6,2]:
        for n in [32,48,64]:
            original=evidence/f'order6-amr-ko03/fast-shift/p-l16-n{n}-smr1/used_input.athinput'
            for family in ['p','Q','L','B']:
                for location,x in [('inside',.75),('crossing',0)]:
                    p=setup(load(original),order)
                    p['problem']['pulse_family']=family
                    p['pc_gh']['reduction_center_x']=str(x)
                    p['time']['tlim']='.25';p['output1']['dt']='.125'
                    save(f'interface-{family}-n{n}-o{order}-{location}',p,'taper/interface pulse')
    (a.output/'manifest.json').write_text(json.dumps(dict(candidate=a.candidate,
        authorization='Prepared conditional gates; execute only after screen review.',
        source_sha256=sources,cases=cases),indent=2)+'\n')


if __name__=='__main__':main()
