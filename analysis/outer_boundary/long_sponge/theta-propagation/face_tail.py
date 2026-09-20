#!/usr/bin/env python3
"""Inspect fixed active face-center Theta values from previously validated bins.

Reads only included manifests; confirms cached raw hashes for every accessed
file. This does not reconstruct characteristic traces, which also need metric
and evolved Gamma fields unavailable in Theta-only dumps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from aggregate import parse

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('analysis',type=Path)
a=p.parse_args()
manifest=json.loads((a.analysis/'theta_primary/manifest.json').read_text())
run=Path(manifest['source_run'])
rows=[]
for item in manifest['included'][::max(1,len(manifest['included'])//16)]+manifest['included'][-1:]:
    vals=[]
    for rank in range(8):
        path=run/'bin'/('rank_%08d'%rank)/item['filename']
        header,phash,(loc,theta),digest=parse(path)
        assert digest==item['rank_sha256'][rank]
        # Each octant supplies one near-axis center on each of its three faces.
        center=[0 if loc[d] else 31 for d in range(3)]
        for axis in range(3):
            ix=list(center)
            ix[axis]=31 if loc[axis] else 0
            value=float(theta[ix[2],ix[1],ix[0]])
            vals.append(value)
    rows.append({'time':item['time'],'cycle':item['cycle'],
                 'face_center24_min':min(vals),'face_center24_max':max(vals),
                 'face_center24_mean':float(np.mean(vals))})
initial=rows[0]['face_center24_mean']
for row in rows:
    row['mean_over_initial']=row['face_center24_mean']/initial
out={'scope':'24 first-active face-near-axis cells: one coordinate ±2016, other two ±32. Theta only, not incoming characteristic state.',
     'source_manifest_sha256':hashlib.sha256((a.analysis/'theta_primary/manifest.json').read_bytes()).hexdigest(),
     'initial_analytic_Theta_sigma384':float(1e-6*np.exp(-(2016**2+2*32**2)/(2*384**2))),
     'hypothetical_sigma256_same_amplitude':float(1e-6*np.exp(-(2016**2+2*32**2)/(2*256**2))),
     'rows':rows}
(a.analysis/'face-tail.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
