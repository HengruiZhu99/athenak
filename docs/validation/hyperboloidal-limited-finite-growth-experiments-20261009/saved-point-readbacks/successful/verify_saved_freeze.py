#!/usr/bin/env python3
"""Hash/JSON/saved-case arithmetic verification only; no scientific rerun."""
from pathlib import Path
import argparse,hashlib,json,math
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def norm(real,imag):return math.hypot(*real,*imag)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    index=read(a.root/'index.json');checked=0;skipped=[]
    for item in index['files']:
        f=a.root/item['path']
        assert not (f.suffix in ('.npz','.npy','.jsonl') and item['role']!='large_payload')
        if not f.exists():
            if a.metadata_only and item['role']=='large_payload':skipped.append(item['path']);continue
            raise RuntimeError('missing indexed artifact '+item['path'])
        assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];checked+=1
        if f.suffix=='.json':read(f)
    assert index['original_SciPy_expm_attempt_remains_failed'] is True
    assert index['both_ordinary_FD_attempts_remain_failed'] is True
    assert index['general_nongauge_continuum_comparator_unresolved'] is True
    arithmetic=[]
    for label,name in [('failed-and-degree','failed-growth-readback001'),('failed-and-degree','N12-projected-readback001'),
                       ('failed-and-degree','N16-projected-readback001')]+[('successful',f'N{n}-readback001') for n in (8,12,16)]:
        folder=a.root/label/name;receipt=read(folder/'receipt.json');summary=read(folder/'summary.json');casefile=folder/'cases.jsonl'
        assert receipt['error'] is None and receipt['sources_unchanged'] is True
        assert receipt['both_ordinary_FD_attempts_remain_failed'] is True
        assert receipt['general_nongauge_continuum_comparator_unresolved'] is True
        if not casefile.exists():
            assert a.metadata_only;arithmetic.append({'run':name,'saved_case_arithmetic_skipped':'large cases.jsonl absent by compact policy'});continue
        rows=[json.loads(v,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s))) for v in casefile.read_text().splitlines()]
        assert len(rows)==receipt['case_count'];maximum=0.;groups={}
        if name.startswith('N') and label=='failed-and-degree':
            names=[v['name'] for v in summary['witnesses']]
            for row in rows:
                column=names.index(row['witness'])
                for key in ('initial','bulk','SAT','total'):groups.setdefault((key,column),[]).append((row[key+'_physical8'],[0.]*8))
        else:
            for row in rows:groups.setdefault((row['column_group'],row['column']),[]).append((row['physical8_real'],row['physical8_imag']))
        for (key,column),vectors in groups.items():
            assert len(vectors)==21
            samples=[norm(r,i) for r,i in vectors]
            computed={'sample_l2_peak':max(samples),'sample_l2_RMS':math.sqrt(math.fsum(v*v for v in samples)/21),
              'H_peak':max(math.hypot(r[0],i[0]) for r,i in vectors),
              'M_coordinate_l2_peak':max(norm(r[1:4],i[1:4]) for r,i in vectors),
              'Z_coordinate_l2_peak':max(norm(r[4:7],i[4:7]) for r,i in vectors),
              'Theta_physical_peak':max(math.hypot(r[7],i[7]) for r,i in vectors)}
            for field,value in computed.items():
                expected=summary['stats'][key][column][field];scaled=abs(value-expected)/max(1.,abs(value),abs(expected));maximum=max(maximum,scaled)
                assert scaled<=5e-11
        arithmetic.append({'run':name,'cases':len(rows),'max_scaled_summary_reconstruction':maximum})
    print(json.dumps({'index_sha256':sha(a.root/'index.json'),'checked_files':checked,'omitted_large_payload_files':skipped,
      'metadata_only':a.metadata_only,'saved_case_arithmetic':arithmetic,'passed':True},indent=2,allow_nan=False))
if __name__=='__main__':main()
