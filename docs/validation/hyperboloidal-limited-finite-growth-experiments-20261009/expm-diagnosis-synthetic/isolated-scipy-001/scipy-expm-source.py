def expm(A):
    """Compute the matrix exponential of an array.

    Parameters
    ----------
    A : ndarray
        Input with last two dimensions are square ``(..., n, n)``.

    Returns
    -------
    eA : ndarray
        The resulting matrix exponential with the same shape of ``A``

    Notes
    -----
    Implements the algorithm given in [1], which is essentially a Pade
    approximation with a variable order that is decided based on the array
    data.

    For input with size ``n``, the memory usage is in the worst case in the
    order of ``8*(n**2)``. If the input data is not of single and double
    precision of real and complex dtypes, it is copied to a new array.

    For cases ``n >= 400``, the exact 1-norm computation cost, breaks even with
    1-norm estimation and from that point on the estimation scheme given in
    [2] is used to decide on the approximation order.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham, (2009), "A New Scaling
           and Squaring Algorithm for the Matrix Exponential", SIAM J. Matrix
           Anal. Appl. 31(3):970-989, :doi:`10.1137/09074721X`

    .. [2] Nicholas J. Higham and Francoise Tisseur (2000), "A Block Algorithm
           for Matrix 1-Norm Estimation, with an Application to 1-Norm
           Pseudospectra." SIAM J. Matrix Anal. Appl. 21(4):1185-1201,
           :doi:`10.1137/S0895479899356080`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    Matrix version of the formula exp(0) = 1:

    >>> expm(np.zeros((3, 2, 2)))
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]]])

    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    a = np.asarray(A)
    if a.size == 1 and a.ndim < 2:
        return np.array([[np.exp(a.item())]])

    if a.ndim < 2:
        raise LinAlgError('The input array must be at least two-dimensional')
    if a.shape[-1] != a.shape[-2]:
        raise LinAlgError('Last 2 dimensions of the array must be square')
    n = a.shape[-1]
    # Empty array
    if min(*a.shape) == 0:
        return np.empty_like(a)

    # Scalar case
    if a.shape[-2:] == (1, 1):
        return np.exp(a)

    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(np.float64)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)

    # An explicit formula for 2x2 case exists (formula (2.2) in [1]). However, without
    # Kahan's method, numerical instabilities can occur (See gh-19584). Hence removed
    # here until we have a more stable implementation.

    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # working memory to hold intermediate arrays
    Am = np.empty((5, n, n), dtype=a.dtype)

    # Main loop to go through the slices of an ndarray and passing to expm
    for ind in product(*[range(x) for x in a.shape[:-2]]):
        aw = a[ind]

        lu = bandwidth(aw)
        if not any(lu):  # a is diagonal?
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue

        # Generic/triangular case; copy the slice into scratch and send.
        # Am will be mutated by pick_pade_structure
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)

        if s != 0:  # scaling needed
            Am[:4] *= [[[2**(-s)]], [[4**(-s)]], [[16**(-s)]], [[64**(-s)]]]

        pade_UV_calc(Am, n, m)
        eAw = Am[0]

        if s != 0:  # squaring needed

            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # This branch implements Code Fragment 2.1 of [1]

                diag_aw = np.diag(aw)
                # einsum returns a writable view
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # super/sub diagonal
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)

                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw

                    # diagonal
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd

            else:  # generic
                for _ in range(s):
                    eAw = eAw @ eAw

        # Zero out the entries from np.empty in case of triangular input
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw

    return eA
