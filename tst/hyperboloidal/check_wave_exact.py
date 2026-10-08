"""Independent high-precision audit of the conformal dipole wave test.

Requires mpmath. Differentiate the physical outgoing/ingoing solution, rather
than copying the C++ closed derivative expressions. This verifies both PDE forms.
"""
import mpmath as mp

mp.mp.dps = 45


def pulse(s):
    return mp.mpf('.01') * mp.exp(-((s - mp.mpf('.6')) / mp.mpf('.2'))**2)


def phi(t, r):
    outgoing = t + (1 - r) / (1 + r)
    ingoing = t + (1 + r) / (1 - r)
    omega = (1 - r*r) / 2
    return (-(mp.diff(pulse, outgoing) + mp.diff(pulse, ingoing)) / r
            - omega * (pulse(outgoing) - pulse(ingoing)) / (r*r))


def pi(t, r):
    return (mp.diff(lambda t: phi(t, r), t)
            + r * mp.diff(lambda r: phi(t, r), r)) / ((1 + r*r) / 2)


def audit():
    maximum = mp.mpf(0)
    for t in map(mp.mpf, ('0', '.3', '.8')):
        for r in map(mp.mpf, ('.05', '.3', '.8', '.99')):
            alpha = (1 + r*r) / 2
            value = phi(t, r)
            dr = mp.diff(lambda r: phi(t, r), r)
            drr = mp.diff(lambda r: phi(t, r), r, 2)
            dt = mp.diff(lambda t: phi(t, r), t)
            dtr = mp.diff(lambda r: mp.diff(lambda t: phi(t, r), t), r)
            # Angular eigenvalue l(l+1)=2 for this dipole.
            laplacian = drr + 2*dr/r - 2*value/(r*r)
            rhs = (-r * mp.diff(lambda r: pi(t, r), r) + alpha*laplacian
                   + r*dr - 3*pi(t, r) - (1/(alpha*alpha) - 1)*value)
            maximum = max(maximum, abs(mp.diff(lambda t: pi(t, r), t) - rhs))
            coordinate_rhs = (-2*r*dtr + alpha*alpha*laplacian - r*r*drr
                              + (alpha - 4 + r*r/alpha)*r*dr
                              + (-3 + r*r/alpha)*dt - (1/alpha - alpha)*value)
            maximum = max(maximum, abs(mp.diff(lambda t: phi(t, r), t, 2)
                                       - coordinate_rhs))
    assert maximum < mp.mpf('1e-35'), maximum
    print('PASS normal/coordinate conformal dipole PDE residual:', mp.nstr(maximum, 10))


if __name__ == '__main__':
    audit()
