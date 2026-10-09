"""Independent saved-field audit; no original reader, kernel or executable calls."""
from pathlib import Path
import hashlib
import json
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ABI = ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/abi.json'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def read(path, n):
    blob = path.read_bytes()
    end = blob.index(b'<par_end>\n')+len(b'<par_end>\n')
    cursor = end
    def get(fmt):
        nonlocal cursor
        result = struct.unpack_from('<'+fmt, blob, cursor)
        cursor += struct.calcsize('<'+fmt)
        return result
    assert get('2i') == (1, 0)
    region = get('9d')
    mesh, block = get('19i'), get('19i')
    t, dt = get('2d')
    cycle, = get('i')
    assert get('4i') == (0, 0, 0, 0)
    cost, = get('f')
    last_output, = get('d')
    size, = get('Q')
    assert cursor-end == 288
    assert mesh[:10] == block[:10] and mesh[10:] == (0,)*9
    assert block[:4] == (3, n, n, n)
    assert block[4:10] == (3, n+2, 3, n+2, 3, n+2)
    assert region == (-1.1, -1.1, -1.1, 1.1, 1.1, 1.1, 2.2/n, 2.2/n, 2.2/n)
    assert np.isfinite([t, dt, cost, last_output]).all() and t >= 0 and dt > 0 and cycle >= 0
    assert size == 25*(n+6)**3*8 and len(blob) == cursor+size
    data = np.frombuffer(blob, dtype='<f8', offset=cursor).reshape(25, n+6, n+6, n+6)
    return t, dt, cycle, data


def main():
    np.seterr(all='raise')
    name = sys.argv[1]
    release = json.loads((HERE/'release.json').read_text())
    spec, = [x for x in release['cases'] if x['name'] == name]
    launch_path = HERE/'batch001'/name/'launch-receipt.json'
    launch = json.loads(launch_path.read_text())
    assert launch['passed_native_process_and_provenance'] and launch['returncode'] == 0
    assert launch['sources_before_after_equal']
    n = int(name.split('-N')[1].split('-')[0])
    reference = '-reference-' in name
    directory = Path(launch['output_directory'])
    paths = sorted(directory.rglob('*.rst'))
    history_path, = directory.glob('*.z4c.user.hst')
    pins = {str(p): sha(p) for p in paths+[history_path, launch_path, ABI, Path(__file__).resolve(), HERE/'release.json']}
    abi = json.loads(ABI.read_text())
    assert {k:abi[k] for k in ['Real','RegionSize','RegionIndcs','LogicalLocation','IOWrapperSizeT','nz4c','LayoutRight','little_endian']} == {
        'Real':8,'RegionSize':72,'RegionIndcs':76,'LogicalLocation':16,'IOWrapperSizeT':8,'nz4c':25,'LayoutRight':1,'little_endian':1}
    for rel, digest in abi['source_sha256'].items():
        assert sha(ROOT/rel) == digest
        pins[str(ROOT/rel)] = digest
    output = HERE/'field-audits'/(name+'-001')
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'source-and-input-pins-before.json', pins)
    history = np.atleast_2d(np.loadtxt(history_path))
    assert history.shape[1] == 15 and np.isfinite(history).all()
    assert (history[:,1] > 0).all() and (np.diff(history[:,0]) > 0).all()
    h = 2.2/n
    c = -1.1+(0.5-3)*h+np.arange(n+6)*h
    zz, yy, xx = np.meshgrid(c, c, c, indexing='ij')
    mask = xx*xx+yy*yy+zz*zz < 1
    rows = []
    last = -1.
    for p in paths:
        t, dt, cycle, q = read(p, n)
        assert t > last
        last = t
        active = q[:,mask]
        assert np.isfinite(active).all() and (active[[0,18]] > 0).all()
        a,b,c0,d,e,f = (active[k] for k in [1,2,3,4,5,6])
        cof = [d*f-e*e, c0*e-b*f, b*e-c0*d, a*f-c0*c0, b*c0-a*e, a*d-b*b]
        det = a*cof[0]+b*cof[1]+c0*cof[2]
        assert (a > 0).all() and (cof[5] > 0).all() and (det > 0).all()
        tr = (cof[0]*active[8]+2*cof[1]*active[9]+2*cof[2]*active[10]+
              cof[3]*active[11]+2*cof[4]*active[12]+cof[5]*active[13])/det
        det_error = float(np.max(np.abs(det-1)))
        trace_error = float(np.max(np.abs(tr)))
        tol = 1e-11 if reference else 1e-10
        assert det_error <= tol and trace_error <= tol
        match = np.flatnonzero(np.abs(history[:,0]-t) <= 1e-12)
        assert len(match) == 1
        hr = history[match[0]]
        if reference:
            assert np.max(np.abs(hr[2:6])) <= 1e-9
        assert abs(float(hr[6])-det_error) <= 5e-15
        assert abs(float(hr[7])-trace_error) <= 5e-15
        rows.append({'path':str(p),'time':t,'cycle':cycle,'restart_dt':dt,
            'active_cells':int(mask.sum()),'alpha_min':float(active[18].min()),
            'chi_min':float(active[0].min()),'det_max':det_error,'trace_max':trace_error,
            'history_H_Mcon_Zcon_Theta':hr[2:6].tolist()})
    assert rows[0]['time'] == 0 and len(rows) >= 2
    assert rows[-1]['time'] == (0.05 if reference else 0.02)
    for p,digest in pins.items():
        assert sha(p) == digest
    dump(output/'rows.json', rows)
    dump(output/'receipt.json', {'passed':True,'case':name,'arrays':len(rows),
        'source_sha256':sha(Path(__file__)),'pins_before_after_equal':True,
        'pins':pins,'rows_sha256':sha(output/'rows.json'),
        'scope':'Independent direct struct parser and elementwise SPD/minor/normal/history audit of saved binary64 fields. No original reader import, scientific kernel, native executable, propagation or PDE eigensolve. Owner aggregate reports were known before this source; source pinned before direct binary-field reads.'})
    print(json.dumps({'passed':True,'case':name,'arrays':len(rows),'receipt_sha256':sha(output/'receipt.json')}))


if __name__ == '__main__':
    main()
