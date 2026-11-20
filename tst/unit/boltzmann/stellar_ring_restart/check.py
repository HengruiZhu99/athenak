"""Compare the active native checkpoint fields at the fourth cycle on eight ranks."""
from pathlib import Path
import re
import struct
import numpy as np

def fields(folder):
    log=(folder/'stdout').read_text()
    if 'Terminating on cycle limit' not in log or not re.search(r'^time=.* cycle=4$',log,re.M):
        raise ValueError('short test did not reach cycle four')
    blocks=int(re.search(r'Current number of MeshBlocks = (\d+)',log)[1])
    file=sorted((folder/'rst').glob('*.rst'))[-1]
    phase=1152*32**2
    stride=phase+25*33**2
    offset=file.stat().st_size-blocks*stride*8
    with file.open('rb') as stream:
        stream.seek(offset-8)
        assert struct.unpack('=Q',stream.read(8))[0]==stride*8
    data=np.memmap(file,dtype='=f8',offset=offset,mode='r',shape=(blocks,stride))
    return (data[:,:phase].reshape(blocks,1152,32,32)[:,:,4:-4,4:-4],
            data[:,phase:].reshape(blocks,25,33,33)[:,:,4:-4,4:-4])

a=fields(Path('direct'))
b=fields(Path('resumed'))
for label, ref, continued in zip(('photon bins','geometry vertices'),a,b):
    assert ref.shape==continued.shape
    err=max(np.max(np.abs(x-y)) for x,y in zip(ref,continued))
    for x,y in zip(ref,continued):
        np.testing.assert_allclose(x,y,rtol=2e-12,atol=3e-15)
    print(label,ref.size,'max absolute difference',err,flush=True)
