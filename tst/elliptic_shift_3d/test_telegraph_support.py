#!/usr/bin/env python3
"""Selected .24-wide AF support against the existing independent 1D code."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from telegraph import window
data=np.array(json.loads(subprocess.check_output([sys.argv[1]],text=True))['selected_telegraph_support'])
w,dw=window(data[:,0],width=.24,inner_profile='af-regular')
error=float(np.max(np.abs(data[:,1:]-np.column_stack([w,dw]))))
assert error<3e-13,error
old,_=window(data[:,0],width=.08,inner_profile='af-regular')
assert np.max(abs(w-old))>.5
print(json.dumps(dict(cases=len(data),selected_width=.24,outer=.8,inner_profile='af-regular',value_and_derivative_error=error,old_narrow_width_difference=float(np.max(abs(w-old))))))
