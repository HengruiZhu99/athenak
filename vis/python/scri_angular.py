"""Cartesian angular derivatives and spin-weighted harmonics about the x axis."""
import math
import numpy as np
from numpy.polynomial import legendre as leg


def scalar_basis(x, lmax):
    """Real orthonormal Y_lm, tangential Cartesian gradients and Hessians.

    Index l*l+l+m: positive m is sqrt(2) Re Y_lm, negative m is
    sqrt(2) Im Y_l|m|. Derivatives extend the function constantly in radius.
    """
    radius = np.linalg.norm(x, axis=-1)
    n = x/radius[:, None]; mu = n[:, 0]
    transverse2 = 1-mu*mu
    if np.min(transverse2) < 1e-14:
        raise ValueError('basis derivative chart excludes the two x-axis poles')
    phi = np.arctan2(x[:, 2], x[:, 1])
    grad_mu = (np.array([1., 0, 0])-mu[:, None]*n)/radius[:, None]
    hess_mu = (-np.eye(3)[0][None, :, None]*n[:, None, :]
               - n[:, :, None]*np.eye(3)[0][None, None, :]
               - mu[:, None, None]*np.eye(3)+3*mu[:, None, None]*n[:, :, None]*n[:, None, :])/radius[:, None, None]**2
    yz2 = x[:, 1]**2+x[:, 2]**2
    grad_phi = np.column_stack((np.zeros(len(x)), -x[:, 2]/yz2, x[:, 1]/yz2))
    hess_phi = np.zeros((len(x), 3, 3))
    hess_phi[:, 1, 1] = 2*x[:, 1]*x[:, 2]/yz2**2
    hess_phi[:, 2, 2] = -hess_phi[:, 1, 1]
    hess_phi[:, 1, 2] = hess_phi[:, 2, 1] = (x[:, 2]**2-x[:, 1]**2)/yz2**2
    size = (lmax+1)**2
    values = np.empty((len(x), size)); gradient = np.empty((len(x), size, 3)); hessian = np.empty((len(x), size, 3, 3))
    for ell in range(lmax+1):
        polynomial = np.zeros(ell+1); polynomial[-1] = 1
        for m in range(ell+1):
            derivative = leg.legder(polynomial, m)
            q, dq, ddq = [leg.legval(mu, leg.legder(derivative, order)) for order in range(3)]
            p = m/2
            b = transverse2**p
            db = -2*p*mu*transverse2**(p-1)
            ddb = -2*p*transverse2**(p-1)+4*p*(p-1)*mu*mu*transverse2**(p-2)
            norm = (-1)**m*math.sqrt((2*ell+1)/(4*math.pi)*math.factorial(ell-m)/math.factorial(ell+m))
            f, df, ddf = norm*b*q, norm*(db*q+b*dq), norm*(ddb*q+2*db*dq+b*ddq)
            phase = np.exp(1j*m*phi)
            y = f*phase
            dy = phase[:, None]*(df[:, None]*grad_mu+1j*m*f[:, None]*grad_phi)
            ddy = phase[:, None, None]*(ddf[:, None, None]*grad_mu[:, :, None]*grad_mu[:, None, :]
                    +df[:, None, None]*hess_mu+1j*m*df[:, None, None]*(grad_mu[:, :, None]*grad_phi[:, None, :]+grad_phi[:, :, None]*grad_mu[:, None, :])
                    +1j*m*f[:, None, None]*hess_phi-m*m*f[:, None, None]*grad_phi[:, :, None]*grad_phi[:, None, :])
            for sign in ([0] if m == 0 else [1, -1]):
                idx = ell*ell+ell+sign*m
                part = np.real if sign >= 0 else np.imag
                scale = 1 if m == 0 else np.sqrt(2)
                values[:, idx] = scale*part(y)
                gradient[:, idx] = scale*part(dy)
                hessian[:, idx] = scale*part(ddy)
    return values, gradient, hessian


def spin_basis(x, lmax, spin=-2):
    """Complex spin harmonics; ordered (ell,m), ell>=abs(spin), -ell<=m<=ell.

    Uses the Wigner-d polynomial with the Condon--Shortley scalar convention.
    Polarization basis is (e_theta, e_phi) about Cartesian x.
    """
    x = x/np.linalg.norm(x, axis=-1)[:, None]
    mu = np.clip(x[:, 0], -1, 1); phi = np.arctan2(x[:, 2], x[:, 1])
    c, s = np.sqrt((1+mu)/2), np.sqrt((1-mu)/2)
    labels = [(ell, m) for ell in range(abs(spin), lmax+1) for m in range(-ell, ell+1)]
    values = np.empty((len(x), len(labels)), dtype=complex)
    for column, (ell, m) in enumerate(labels):
        mp = -spin; d = np.zeros(len(x))
        numerator = math.sqrt(math.factorial(ell+m)*math.factorial(ell-m)*math.factorial(ell+mp)*math.factorial(ell-mp))
        for k in range(max(0, m-mp), min(ell+m, ell-mp)+1):
            denominator = math.factorial(ell+m-k)*math.factorial(k)*math.factorial(mp-m+k)*math.factorial(ell-mp-k)
            d += (-1)**k*numerator/denominator*c**(2*ell+m-mp-2*k)*s**(mp-m+2*k)
        values[:, column] = (-1)**spin*np.sqrt((2*ell+1)/(4*np.pi))*d*np.exp(1j*m*phi)
    return values, labels


def weighted_projection(values, weights):
    """Interpolation coefficients on the native quadrature, not reported modes."""
    weighted = values.conj().T*weights[None, :]
    gram = weighted@values
    if np.linalg.cond(gram) > 1e8:
        raise ValueError('angular interpolation is ill conditioned')
    return np.linalg.solve(gram, weighted)
