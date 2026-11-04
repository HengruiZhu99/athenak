#!/usr/bin/env python3
"""Read the distinct AMR1 identity and compare ownership-independent payloads."""
import argparse,hashlib,json,math,pathlib,struct
import numpy as np

def read(path):
    raw=pathlib.Path(path).read_bytes();pos=raw.index(b'\0')+1;science=raw[:pos-1].decode()
    if not science.startswith('HYP3D_AMR1;'):raise ValueError('not AMR1')
    names=[]
    for _ in range(31):
        end=raw.index(b'\0',pos);names.append(raw[pos:end].decode());pos=end+1
    fmt='<iiiQQddd3d8d8d';meta=struct.unpack_from(fmt,raw,pos);pos+=struct.calcsize(fmt)
    count,=struct.unpack_from('<Q',raw,pos);pos+=8
    leaves=np.frombuffer(raw,dtype='<i4',count=4*count,offset=pos).reshape(-1,4).copy();pos+=16*count
    ages=np.frombuffer(raw,dtype='<i4',count=count,offset=pos).copy();pos+=4*count
    arrays=[]
    for _ in range(8):
        size,=struct.unpack_from('<Q',raw,pos);pos+=8
        arrays.append(np.frombuffer(raw,dtype='<f8',count=size,offset=pos).copy());pos+=8*size
    tail,=struct.unpack_from('<Q',raw,pos);pos+=8
    if 'checksum-fnv1a64' in science:pos+=8
    if tail!=0x414d5231434f4d50 or pos!=len(raw):raise ValueError('incomplete/trailing checkpoint')
    return dict(science=science,names=names,meta=meta,leaves=leaves,ages=ages,arrays=arrays,sha256=hashlib.sha256(raw).hexdigest())
def compare(a,b):
    assert a['science']==b['science'] and np.array_equal(a['leaves'],b['leaves'])
    diffs=[]
    for x,y in zip(a['arrays'],b['arrays']):
        assert x.shape==y.shape
        diffs.append(dict(bitwise=x.tobytes()==y.tobytes(),max_absolute=float(np.max(np.abs(x-y))) if x.size else 0))
    per_field=np.max(np.abs(a['arrays'][0]-b['arrays'][0]).reshape(-1,31),axis=0)
    return dict(a_sha256=a['sha256'],b_sha256=b['sha256'],arrays=diffs,field_errors=dict(zip(a['names'],map(float,per_field))),ages_equal=np.array_equal(a['ages'],b['ages']),metadata_equal=a['meta']==b['meta'],metadata_max_absolute=float(np.max(np.abs(np.asarray(a['meta'])-np.asarray(b['meta'])))))
def core_grid(s):
    n=s['meta'][0];levels=s['leaves'][:,3];assert np.all(levels==levels[0]),'uniform core required'
    subdivisions=2**int(levels[0]);out=np.empty((subdivisions*n,)*3+(31,))
    blocks=s['arrays'][0][:len(levels)*n**3*31].reshape(len(levels),n,n,n,31)
    for m,(x,y,z,level) in enumerate(s['leaves']):out[z*n:(z+1)*n,y*n:(y+1)*n,x*n:(x+1)*n]=blocks[m]
    return out

def uniform_compare(a,b):
    fine=core_grid(b);nf=fine.shape[0];n=a['meta'][0];errors=[]
    blocks=a['arrays'][0][:len(a['leaves'])*n**3*31].reshape(-1,n,n,n,31)
    for m,(lx,ly,lz,level) in enumerate(a['leaves']):
        matrices=[]
        for loc in [lx,ly,lz]:
            w=np.zeros((n,nf))
            for i in range(n):
                t=(loc+(i+.5)/n)/(2**int(level))*nf-.5
                start=max(0,min(math.floor(t)-2,nf-6))
                for j in range(6):
                    weight=1.
                    for k in range(6):
                        if k!=j:weight*=(t-start-k)/(j-k)
                    w[i,start+j]=weight
            matrices.append(w)
        sampled=np.einsum('ai,kjif->kjaf',matrices[0],fine,optimize=True)
        sampled=np.einsum('bj,kjaf->kbaf',matrices[1],sampled,optimize=True)
        sampled=np.einsum('ck,kbaf->cbaf',matrices[2],sampled,optimize=True)
        errors.append((blocks[m]-sampled).reshape(-1,31))
    delta=np.concatenate(errors)
    return dict(amr_leaves=len(a['leaves']),fine_cells=nf,max_field_error=dict(zip(a['names'],map(float,np.max(abs(delta),axis=0)))),rms_field_error=dict(zip(a['names'],map(float,np.sqrt(np.mean(delta**2,axis=0))))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('--uniform',action='store_true');p.add_argument('--output',required=True);args=p.parse_args()
    a,b=read(args.a),read(args.b);result=uniform_compare(a,b) if args.uniform else compare(a,b)
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
