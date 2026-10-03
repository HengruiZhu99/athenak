"""Harmonic-cache allocation screening and actual per-horizon log witnesses."""
import re


def harmonic_table_bytes(lmax,ntheta,mode='dense'):
    """One double-precision copy, including factorized mode's 15 tiny tables."""
    if lmax<2 or ntheta<lmax+1:raise ValueError('harmonic order/quadrature outside supported control')
    if mode=='dense':return 8*2*ntheta**2*(12*(lmax+1)**2+3*(lmax+1))
    if mode=='factorized':return 8*(3*ntheta*(lmax+1)**2+4*ntheta*(lmax+1)+15)
    raise ValueError('unknown harmonic storage mode')


def storage_evidence(stdout,mode,lmax,ntheta,surface_count=1):
    rows=[dict(horizon=int(h),mode=m,lmax=int(l),ntheta=int(n),host_bytes=int(hb),
               device_bytes=int(db),unique_bytes=int(ub)) for h,m,l,n,hb,db,ub in re.findall(
        r'FastFlow harmonic_storage horizon=(\d+) mode=(dense|factorized) lmax=(\d+) ntheta=(\d+) '
        r'host_bytes=(\d+) device_bytes=(\d+) unique_bytes=(\d+)',stdout)]
    expected=harmonic_table_bytes(lmax,ntheta,mode)
    # These drivers declare one Serial consumer. A two-copy GPU allocation
    # needs its own budget and execution protocol; it cannot pass this screen.
    passed=bool(len(rows)==surface_count and sorted(r['horizon'] for r in rows)==list(range(surface_count))
        and all(r['mode']==mode and r['lmax']==lmax and r['ntheta']==ntheta
            and all(r[key]==expected for key in ('host_bytes','device_bytes','unique_bytes')) for r in rows))
    return dict(requested_mode=mode,expected_bytes_per_horizon=expected,records=rows,
                serial_allocation_verified=passed,passed=passed,measured_process_peak=False)
