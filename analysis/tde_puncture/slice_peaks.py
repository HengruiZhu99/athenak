#!/usr/bin/env python3
"""Peak locations from actual output cells; never upsample an SMR slice to a full grid.

Locations are maxima ON THE OUTPUT SLICE, not global 3-D maxima. Two-dimensional
cuts can miss the global peak. The radial and temporal labels are preserved.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
import bin_convert


def peaks(path):
    d=bin_convert.read_binary(str(path))
    variable='dens' if 'dens' in d['var_names'] else 'z4c_Theta'
    values=d['mb_data'][variable]
    volume=all(all(n>1 for n in v.shape) for v in values)
    result=[]
    for region in ['all','r<1','r>=1']:
        best=None
        for m,(geom,index,logical,v) in enumerate(zip(d['mb_geometry'],d['mb_index'],d['mb_logical'],values)):
            sizes=[d['nx1_mb'],d['nx2_mb'],d['nx3_mb']]
            axes=[geom[2*a]+(index[2*a]+np.arange(v.shape[2-a])+.5)*(geom[2*a+1]-geom[2*a])/sizes[a]
                  for a in range(3)]
            z,y,x=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij');rad=np.sqrt(x*x+y*y+z*z)
            mask=np.ones(v.shape,dtype=bool) if region=='all' else (rad<1 if region=='r<1' else rad>=1)
            if not mask.any():continue
            data=np.where(mask,abs(v),-np.inf);idx=np.unravel_index(np.argmax(data),data.shape)
            row={'region':region,'value':float(v[idx]),'absolute_value':float(data[idx]),
                 'nonzero_peak':bool(data[idx]>0),
                 'xyz_M':[float(x[idx]),float(y[idx]),float(z[idx])],
                 'coordinate_radius_M':float(rad[idx]),'file_block_index':m,
                 'logical_level':int(logical[3]),
                 'block_bounds_M':[float(q) for q in geom],
                 'cell_spacing_M':[float((geom[2*a+1]-geom[2*a])/sizes[a]) for a in range(3)]}
            if best is None or row['absolute_value']>best['absolute_value']:best=row
        if best is not None:result.append(best)
    nonfinite=[name for name in d['var_names']
               if any(not np.isfinite(v).all() for v in d['mb_data'][name])]
    return {'file':str(path),'time_M':float(d['time']),'cycle':int(d['cycle']),
            'output_precision_bits':values[0].dtype.itemsize*8,
            'variable':variable,'scope':('full active-volume maxima at output time; not first-injection diagnosis'
                if volume else 'slice maxima only; not global 3D or first-injection diagnosis'),
            'all_output_variables_finite':not nonfinite,'nonfinite_variables':nonfinite,
            'all_residual_fields_exact_zero':(all(np.count_nonzero(v)==0 for name in d['var_names']
                for v in d['mb_data'][name]) if variable=='z4c_Theta' else None),
            'finite':bool(all(np.isfinite(v).all() for v in values)), 'peaks':result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps([peaks(f) for f in a.files],indent=2)+'\n')
