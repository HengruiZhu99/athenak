"""Validate the layout and epoch of a published SMR photon-ring restart."""
import struct


def validate_restart(path, expected_time=None, expected_cycle=None):
    # Native double precision, signed prograde sky, four ghosts, 25 Z4c fields.
    stride = 8*(24*48*44**2 + 25*45**2)
    offset = path.stat().st_size-76*stride
    if not 0 < offset < 4*1024*1024:
        raise RuntimeError(f'incomplete checkpoint {path}')
    with path.open('rb') as file:
        header = file.read(256*1024)
        begin = header.index(b'<par_end>\n') + len(b'<par_end>\n')
        nblocks, = struct.unpack_from('=i',header,begin)
        # Two leading integers, RegionSize (9 doubles), two RegionIndcs (19 integers each).
        t, dt, cycle = struct.unpack_from('=ddi',header,begin+8+9*8+2*19*4)
        if nblocks != 76 or (expected_cycle is not None and cycle != expected_cycle) or (
                expected_time is not None and abs(t-expected_time)>1.e-8):
            raise ValueError(f'checkpoint epoch mismatch {path}: t={t}, cycle={cycle}')
        file.seek(offset-8)
        if struct.unpack('=Q',file.read(8))[0] != stride:
            raise RuntimeError(f'bad checkpoint field stride {path}')
