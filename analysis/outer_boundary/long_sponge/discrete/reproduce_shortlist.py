"""Reproduce the final requested eight-case shortlist in a separate folder."""
from pathlib import Path
from screen import defaults,run

out=Path('reproduced-shortlist');out.mkdir(exist_ok=True)
for eta,ld in [(2.,.1),(.02,.01)]:
    for q in [0.,1/64,1/32,1/8]:
        path=out/f'profile-r0001-k0-eta{eta}-q{q}.json'
        if path.exists():
            print(f'Preserving existing {path}',flush=True);continue
        p=defaults(n=64,h=64,width=1536,rate=.001,kappa=0,angle=q)
        p.update(sponge_end_cells=4.,eta=eta,lapse_damping=ld)
        run(p,path)
