#!/usr/bin/env python3
"""Cumulative stage prefix extraction cannot import a later failure peak."""
import json
from trajectory import prefix_stage_extrema,merge_report_extrema
rows=[]
for time,peak,report in [(0,0,0),(1,2,2.5),(2,100,200)]:
 row=dict(t=time,all_RK_stages={'max_H_rms':peak,'max_accepted_H_rms':peak,'min_alpha':1/(time+1)},H_rms=report)
 rows.append(row)
prefix=prefix_stage_extrema(rows,1)
assert prefix['max_H_rms']==2 and prefix['max_accepted_H_rms']==2.5 and prefix['min_alpha']==.5,prefix
for endpoint in [.5,3]:
 try:prefix_stage_extrema(rows,endpoint)
 except ValueError:pass
 else:raise AssertionError('Prefix without an exact final report accepted')
try:prefix_stage_extrema([dict(t=0,H_rms=0),dict(t=1,H_rms=1)],1)
except ValueError:pass
else:raise AssertionError('Sparse sampled diagnostics promoted to all-step accuracy')
manifest={'max_H_rms':900.,'max_accepted_H_rms':10.,'max_scri_pole':.1,'min_alpha':.2}
merge_report_extrema(manifest,[dict(H_rms=20.,scri_pole=.3,min_alpha=.1)])
assert manifest=={'max_H_rms':900.,'max_accepted_H_rms':20.,'max_scri_pole':.3,'min_alpha':.1}
print(json.dumps(dict(later_stage_peaks_excluded=True,last_accepted_state_included=True,missing_endpoint_and_sparse_instrumentation_rejected=True)))
