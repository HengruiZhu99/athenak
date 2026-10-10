"""Observations only. Exact Context arithmetic and endpoint key/order stay unchanged."""
import json
from pathlib import Path
import time


class ObservedEndpointCache(dict):
    """The ordinary insertion-ordered dict protocol plus integer counters."""
    def __init__(self):
        super().__init__()
        self.hits=0
        self.misses=0
        self.evictions=0

    def __contains__(self,key):
        present=super().__contains__(key)
        if present:self.hits+=1
        else:self.misses+=1
        return present

    def __delitem__(self,key):
        super().__delitem__(key)
        self.evictions+=1

    def counters(self):
        return {'hits':self.hits,'misses':self.misses,'evictions':self.evictions,
                'entries':len(self),'semantics':'same exact Fraction/precision keys and FIFO insertion order'}


class ProgressObserver:
    def __init__(self,out,roots,started,cache):
        self.out=Path(out)
        self.roots=roots
        self.started=started
        self.cache=cache
        self.root_nodes=[0 for _ in roots]
        self.root_leaves=[0 for _ in roots]
        self.root_max_depth=[0 for _ in roots]
        self.method_evaluations={k:0 for k in ('regular','separated','upper_tail','lower_tail')}
        self.last=None

    @staticmethod
    def box_record(box):
        rlo,rhi,tlo,thi=box
        return {'R':[str(rlo),str(rhi)],'T':[str(tlo),str(thi)],
                'R_width':str(rhi-rlo),'T_width':str(thi-tlo)}

    def active(self,root_id,path,box,depth):
        self.root_nodes[root_id]+=1
        self.root_max_depth[root_id]=max(self.root_max_depth[root_id],depth)
        self.last={'root':root_id,'sigma':str(self.roots[root_id][0]),
                   'path':path,'depth':depth,'box':self.box_record(box),
                   'method':None,'lower':None}

    def lower(self,method,low):
        self.method_evaluations[method]+=1
        self.last['method']=method
        self.last['lower']=str(low)

    def leaf(self,root_id):
        self.root_leaves[root_id]+=1

    def progress(self,nodes,leaves,stack,counts,minimum,status):
        pending=[]
        per_root=[0 for _ in self.roots]
        for root_id,path,box,depth in reversed(stack):
            per_root[root_id]+=1
            pending.append({'root':root_id,'sigma':str(self.roots[root_id][0]),
                            'path':path,'depth':depth,'box':self.box_record(box)})
        row={'nodes':nodes,'leaves':leaves,'pending':len(stack),
             'elapsed_seconds':time.monotonic()-self.started,'status':status,
             'most_recent_node':self.last,'active_node_role':'most recently visited; pending stack is future work',
             'method_evaluations':dict(self.method_evaluations),'positive_leaf_methods':dict(counts),
             'minimum_observed_positive_leaf_lower':None if minimum is None else str(minimum),
             'root_counters':[{'root':i,'sigma':str(sigma),'initial_box':self.box_record(box),
                 'nodes':self.root_nodes[i],'leaves':self.root_leaves[i],
                 'maximum_visited_depth':self.root_max_depth[i],'pending_nodes':per_root[i]}
                 for i,(sigma,box) in enumerate(self.roots)],
             'remaining_pending_boxes_in_DFS_order':pending,'cache':self.cache.counters(),
             'coverage_fraction_claimed':False,'global_slicing_accepted':False,
             'completion_requires_full_producer_footer_and_independent_replay':True}
        raw=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        (self.out/'progress.json').write_text(raw)
        print(raw.rstrip('\n'),flush=True)
