#!/usr/bin/env python3
"""Read supported double-precision MHD+Z4c checkpoints without changing them.

Parser extracted from the archived checkpoint reader; no execution/submission
helper is included. Used only by checkpoint_face_trace.py's narrower fixture.
"""
import argparse
from array import array
import json
import hashlib
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


def checkpoint(path):
    with path.open('rb') as stream:
        prefix = stream.read(262144)
        marker = b'<par_end>\n'
        stop = prefix.find(marker)
        assert stop >= 0, 'Missing parameter header'
        end = stop + len(marker)
        params, block = {}, None
        for line in prefix[:stop].decode().splitlines():
            line = line.split('#', 1)[0].strip()
            if line.startswith('<'):
                block = line[1:-1]
                params[block] = {}
            elif '=' in line:
                key, value = line.split('=', 1)
                params[block][key.strip()] = value.strip()
        assert {'mhd', 'z4c'} <= params.keys()
        assert not {'hydro', 'radiation', 'turbulence'} & params.keys()
        assert not any(k.startswith('co_') and k.endswith('_type') for k in params['z4c'])
        assert not any(k.startswith('dump_horizon_') and v == 'true'
                       for k, v in params['z4c'].items())
        stream.seek(end)
        total, level = struct.unpack('<ii', stream.read(8))
        stream.read(72)  # RegionSize.
        root_indices = struct.unpack('<19i', stream.read(76))
        assert root_indices[10:] == (0,)*9, 'Uninitialized root coarse indices'
        indices = struct.unpack('<19i', stream.read(76))
        time, dt, cycle = struct.unpack('<ddi', stream.read(20))
        ng, nx, ny, nz = indices[:4]
        assert min(nx, ny, nz) > 1 and ng > 0
        stream.seek(20 * total + 16, 1)  # Locations/costs and Z4c output times.
        stride, = struct.unpack('<Q', stream.read(8))
        n1, n2, n3 = nx + 2*ng, ny + 2*ng, nz + 2*ng
        cells = n1*n2*n3
        nmhd = 5 + int(params['mhd'].get('nscalars', '0'))
        faces = (n1+1)*n2*n3 + n1*(n2+1)*n3 + n1*n2*(n3+1)
        offset = 8*(nmhd*cells + faces)
        assert stride == offset + 8*25*cells, 'Unsupported checkpoint payload'
        payload_start = stream.tell()
        stream.seek(0)
        header_hash = hashlib.sha256(stream.read(payload_start)).hexdigest()
        payload = stream.read()
        assert payload and len(payload) % stride == 0
        assert sys.byteorder == 'little', 'This reader expects a little-endian host'
        state = []
        for start in range(0, len(payload), stride):
            values = array('d')
            values.frombytes(payload[start+offset:start+stride])
            assert all(math.isfinite(x) for x in values)
            state.append(values)
        active = [((n*n3+k)*n2+j)*n1+i for n in range(25)
                  for k in range(ng, ng+nz) for j in range(ng, ng+ny)
                  for i in range(ng, ng+nx)]
        return {'time': time, 'dt': dt, 'cycle': cycle, 'total': total,
                'state': state, 'active': active, 'cells': cells, 'level': level,
                'header_hash': header_hash}
