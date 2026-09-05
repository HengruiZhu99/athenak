"""Independent configuration Jacobian for the nonlinear reduction subsidiary system.

For e_i^A=(r_i,q_iab,ell_i/2,b_i^a), the undamped source is
M = (partial_i beta^j) tensor I_11 + I_3 tensor C.
The full 33-component bound includes the three algebraic trace directions,
so its Euclidean energy bound is conservative on the physical 30-dimensional
reduction subspace. Coefficients are evaluated on a specified background.
"""
import numpy as np

PAIRS = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]


def tensor(values):
    result = np.zeros((3,3), dtype=np.asarray(values).dtype)
    for value, (a,b) in zip(values, PAIRS):
        result[a,b] = result[b,a] = value
    return result


def configuration(u):
    return np.r_[u[0:7], u[0]*u[18], u[19:22]]


def sources(x, u, eta=2., z0=.1, z1=.5):
    w, alpha = x[0], x[7]
    g = tensor(x[1:7]); inv = np.linalg.inv(g)
    K, A, Z, p, L = u[7], tensor(u[8:14]), u[14:17], u[22:25], u[43:46]
    Q = np.array([tensor(u[25+6*i:31+6*i]) for i in range(3)])
    B = u[46:55].reshape(3,3)
    z = alpha*w*w
    if z.real <= z0:
        switch = 0.
    elif z.real >= z1:
        switch = 1.
    else:
        t = (z-z0)/(z1-z0)
        switch = t*t*(3-2*t)
    gamma_down = np.einsum('jk,jak->a', inv, Q)-.5*np.einsum('jk,ajk->a', inv, Q)
    Fg = -2*alpha*A+B@g+g@B.T-2*g*np.trace(B)/3
    Fbeta = inv@gamma_down-Z-eta*x[8:11]
    Fbeta += switch*inv@(alpha*alpha*w*p-.5*alpha*w*w*L)
    return np.r_[w*(alpha*K-np.trace(B))/3,
                 [Fg[a,b] for a,b in PAIRS], -2*alpha*K, Fbeta]


def coefficient(u, eta=2., z0=.1, z1=.5):
    x = configuration(u)
    C = np.empty((11,11))
    for j in range(11):
        perturbed = x.astype(complex)
        perturbed[j] += 1e-25j
        C[:,j] = sources(perturbed, u, eta, z0, z1).imag/1e-25
    gradients = np.column_stack([u[22:25], u[25:43].reshape(3,6),
                                  u[43:46]/2, u[46:55].reshape(3,3)])
    C[:,8:11] += gradients.T
    inv = np.linalg.inv(tensor(u[1:7]))
    Aup = inv@tensor(u[8:14])@inv
    contraction = np.array([(1 if a==b else 2)*Aup[a,b] for a,b in PAIRS])
    C[1:7,1:7] -= 2*x[7]/3*np.outer(u[1:7], contraction)
    return C


def estimates(u, derivative_beta, rate=1., eta=2., z0=.1, z1=.5):
    C = coefficient(u, eta, z0, z1)
    D = derivative_beta
    mu = (np.linalg.eigvalsh((C+C.T)/2).max()
          +np.linalg.eigvalsh((D+D.T)/2).max()-.5*np.trace(D))
    abscissa = np.linalg.eigvals(C).real.max()+np.linalg.eigvals(D).real.max()-rate
    return dict(energy_rate_margin=float(rate-mu),
                frozen_spectral_abscissa=float(abscissa))


if __name__ == '__main__':
    # Independent explicit initial-slice matrix from the nonlinear derivation.
    for w in [.25,1.]:
        u = np.zeros(55);u[0]=w;u[[1,4,6]]=1;u[18]=1
        p = np.array([1/8,1/16,-1/32]);u[22:25]=p;u[43:46]=2*p
        expected = np.zeros((11,11));expected[8:11,8:11]=-2*np.eye(3)
        expected[0,8:11]=expected[7,8:11]=p
        switch = 0 if w==.25 else 1
        expected[8:11,0]=-switch*w*w*p
        expected[8:11,7]=switch*w*w*p
        assert np.max(abs(coefficient(u)-expected)) < 1e-15
    print('PASS independent wormhole-slice source matrix, switch zero and one')
