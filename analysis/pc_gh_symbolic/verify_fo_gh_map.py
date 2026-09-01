#!/usr/bin/env python3
"""Exact round-trip audit of the PC-GH/standard FO-GH variable map."""

import sympy as sp


DIM = 3
R = sp.Rational


def assert_matrix_zero(name, value):
    for i in range(value.rows):
        for j in range(value.cols):
            residual = sp.simplify(sp.cancel(value[i, j]))
            if residual != 0:
                raise AssertionError(f"{name} ({i},{j}) failed: {residual}")


def assert_scalar_zero(name, value):
    residual = sp.simplify(sp.cancel(value))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def make_trace_free(seed, metric, inverse):
    return seed - metric * sp.trace(inverse * seed) / 3


def pc_to_fo(pc, gauge):
    alpha = pc["alpha"]
    A = alpha**2
    chi = pc["chi"]
    gt = pc["gt"]
    gtu = gt.inv()
    beta = pc["beta"]
    gamma = gt / chi

    dgamma = [pc["Q"][k] / chi - gt * pc["X"][k] / chi**2
              for k in range(DIM)]
    phi = []
    for k in range(DIM):
        block = sp.zeros(4)
        block[1:4, 1:4] = dgamma[k]
        block[1:4, 0] = dgamma[k] * beta + gamma * pc["B"].row(k).T
        block[0, 1:4] = block[1:4, 0].T
        block[0, 0] = (
            -pc["Y"][k]
            + (beta.T * dgamma[k] * beta)[0]
            + 2 * (beta.T * gamma * pc["B"].row(k).T)[0]
        )
        phi.append(block)

    vbeta = (
        gauge["hvec"] + A * chi * pc["Lambda"]
        + gtu * (A * pc["X"] - chi * pc["Y"]) / 2
    )
    kij = (pc["at"] + gt * pc["K"] / 3) / chi
    d0gamma = -2 * alpha * kij + gamma * pc["B"].T + pc["B"] * gamma
    d0A = 2 * A * (alpha * pc["pi"] - gauge["hperp"])
    d0g = sp.zeros(4)
    d0g[1:4, 1:4] = d0gamma
    d0g[1:4, 0] = d0gamma * beta + gamma * vbeta
    d0g[0, 1:4] = d0g[1:4, 0].T
    d0g[0, 0] = (
        -d0A + (beta.T * d0gamma * beta)[0]
        + 2 * (beta.T * gamma * vbeta)[0]
    )

    g = sp.zeros(4)
    g[1:4, 1:4] = gamma
    g[1:4, 0] = gamma * beta
    g[0, 1:4] = g[1:4, 0].T
    g[0, 0] = -A + (beta.T * gamma * beta)[0]
    return {"g": g, "phi": phi, "Pi": -d0g / alpha}


def fo_to_pc(fo, gauge, alpha):
    g = fo["g"]
    gamma = g[1:4, 1:4]
    gammau = gamma.inv()
    beta = gammau * g[1:4, 0]
    A = (beta.T * gamma * beta)[0] - g[0, 0]
    assert_scalar_zero("positive-root lapse consistency", A - alpha**2)
    chi = sp.real_root(gamma.det(), 3) ** -1
    gt = chi * gamma
    gtu = gt.inv()

    dgamma = [fo["phi"][k][1:4, 1:4] for k in range(DIM)]
    B = sp.zeros(DIM)
    X = sp.zeros(DIM, 1)
    Y = sp.zeros(DIM, 1)
    Q = []
    for k in range(DIM):
        B[k, :] = (
            gammau * (fo["phi"][k][1:4, 0] - dgamma[k] * beta)
        ).T
        X[k] = -chi * sp.trace(gammau * dgamma[k]) / 3
        Y[k] = (
            -fo["phi"][k][0, 0]
            + (beta.T * dgamma[k] * beta)[0]
            + 2 * (beta.T * gamma * B.row(k).T)[0]
        )
        Q.append(chi * dgamma[k] + X[k] * gamma)

    d0g = -alpha * fo["Pi"]
    d0gamma = d0g[1:4, 1:4]
    d0beta = gammau * (d0g[1:4, 0] - d0gamma * beta)
    d0A = (
        -d0g[0, 0] + (beta.T * d0gamma * beta)[0]
        + 2 * (beta.T * gamma * d0beta)[0]
    )
    kij = -(d0gamma - gamma * B.T - B * gamma) / (2 * alpha)
    K = sp.trace(gammau * kij)
    at = chi * kij - gt * K / 3
    pi = (d0A / (2 * A) + gauge["hperp"]) / alpha
    Lambda = (
        d0beta - gauge["hvec"] - gtu * (A * X - chi * Y) / 2
    ) / (A * chi)
    return {
        "alpha": alpha, "chi": chi, "gt": gt, "beta": beta,
        "X": X, "Y": Y, "Q": Q, "B": B, "K": K, "at": at,
        "pi": pi, "Lambda": Lambda,
    }


def main():
    lower = sp.Matrix([
        [R(2), 0, 0],
        [R(1, 5), R(3), 0],
        [R(-1, 7), R(2, 9), R(1, 6)],
    ])
    gt = lower * lower.T
    assert_scalar_zero("det(gtilde)", gt.det() - 1)
    gtu = gt.inv()

    q_seeds = [
        sp.Matrix([[2, 1, -1], [1, -3, 2], [-1, 2, 1]]),
        sp.Matrix([[1, -2, 3], [-2, 4, 1], [3, 1, -2]]),
        sp.Matrix([[-1, 3, 2], [3, 2, -4], [2, -4, 5]]),
    ]
    Q = [make_trace_free(seed, gt, gtu) for seed in q_seeds]
    at = make_trace_free(
        sp.Matrix([[R(1, 3), R(-2, 5), R(1, 7)],
                   [R(-2, 5), R(3, 8), R(2, 9)],
                   [R(1, 7), R(2, 9), R(-1, 4)]]),
        gt,
        gtu,
    )
    pc = {
        "alpha": R(3, 2),
        "chi": R(2, 3),
        "gt": gt,
        "beta": sp.Matrix([R(1, 7), R(-2, 11), R(3, 13)]),
        "X": sp.Matrix([R(2, 5), R(-1, 3), R(4, 9)]),
        "Y": sp.Matrix([R(-3, 7), R(5, 8), R(1, 6)]),
        "Q": Q,
        "B": sp.Matrix([[R(1, 3), R(2, 7), R(-1, 5)],
                         [R(-2, 9), R(3, 11), R(4, 13)],
                         [R(5, 17), R(-1, 4), R(2, 15)]]),
        "K": R(-5, 12),
        "at": at,
        "pi": R(7, 10),
        "Lambda": sp.Matrix([R(1, 8), R(-3, 14), R(2, 9)]),
    }
    gauge = {
        "hperp": R(-2, 7),
        "hvec": sp.Matrix([R(1, 9), R(2, 11), R(-4, 15)]),
    }

    recovered = fo_to_pc(pc_to_fo(pc, gauge), gauge, pc["alpha"])
    for name in ("gt", "beta", "X", "Y", "B", "at", "Lambda"):
        assert_matrix_zero(f"round trip {name}", recovered[name] - pc[name])
    for k in range(DIM):
        assert_matrix_zero(f"round trip Q[{k}]", recovered["Q"][k] - pc["Q"][k])
        assert_scalar_zero(
            f"Q[{k}] trace constraint", sp.trace(gtu * recovered["Q"][k])
        )
    for name in ("chi", "K", "pi"):
        assert_scalar_zero(f"round trip {name}", recovered[name] - pc[name])
    assert_scalar_zero("A round trip", recovered["alpha"]**2 - pc["alpha"]**2)
    assert_scalar_zero("Atilde trace constraint", sp.trace(gtu * recovered["at"]))

    print("PASS: exact PC-GH -> FO-GH -> PC-GH round trip")
    print("PASS: conformal metric, Q, and Atilde algebraic constraints")


if __name__ == "__main__":
    main()
