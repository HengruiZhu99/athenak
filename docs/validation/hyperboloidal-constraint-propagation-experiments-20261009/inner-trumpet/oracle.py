"""100-digit independent local slice limits and derivative scaling."""
import json
from pathlib import Path
import mpmath as mp
mp.mp.dps = 100

def state(r, M, R0, p):
    delta = mp.mpf(".05")*M*(r/M)**p
    R = R0+delta
    rp = p*delta/r
    C4 = R0**3*(2*M-R0)
    A = R**4-C4/4
    B = delta*(R0**2*(4*R0-6*M)+delta*(6*R0**2-6*M*R0
              +delta*(4*R0-2*M+delta)))
    alpha = -2*B/(C4+mp.sqrt(C4**2+4*A*B))
    S = mp.sqrt(alpha**2-1+2*M/R)
    beta = alpha*S/rp
    gamma_r, gamma_t = rp**2/alpha**2, (R/r)**2
    chi = (gamma_r*gamma_t**2)**(-mp.mpf(1)/3)
    gr, gt = chi*gamma_r, chi*gamma_t
    return [alpha, beta, chi, gr, gt, R, S]

rows=[]
for mass, ratio, exponent in [(".5", "1.4", ".75"), (".5", "1.4", "1"),
        (".5", "1.4", "1.5"), (".5", "1.4", str(mp.mpf(2)/3)),
        (".5", "1.35", "1.25"), ("1", "1.4", "1")]:
    M, R0, p = mp.mpf(mass), mp.mpf(mass)*mp.mpf(ratio), mp.mpf(exponent)
    S0 = mp.sqrt(2*M/R0-1)
    K0 = (3*M-2*R0)/(R0**2*S0)
    aR0 = 2*K0/S0
    v = 2*K0/p
    zeta = p/(aR0*R0)
    lambda0 = 2*(zeta**(mp.mpf(2)/3)-zeta**(-mp.mpf(4)/3))
    errors=[]
    for scale in ["1e-10", "1e-20", "1e-40"]:
        r=M*mp.mpf(scale)
        alpha,beta,chi,gr,gt,R,S=state(r,M,R0,p)
        derivative=lambda i: mp.diff(lambda x: state(x,M,R0,p)[i],r)
        bp=derivative(1)
        ap=derivative(0)
        Rp=derivative(5)
        K=derivative(6)/Rp+2*S/R
        invgrprime=mp.diff(lambda x: 1/state(x,M,R0,p)[3],r)
        Lam=-invgrprime+2*(1/gt-1/gr)/r
        driver=mp.mpf(3)/8*alpha**2*chi*Lam
        bd=beta*bp+driver
        balanced=bd-v*beta
        lapse=beta*ap-alpha*(alpha+2)*K
        assert abs(lapse/alpha*M)<mp.mpf("1e-55")
        assert abs(gr*gt**2-1)<mp.mpf("1e-95")
        errors.append({"r_over_M":scale,"beta_slope_error":float(abs(beta/r-v)*M),
            "connection_limit_error":float(abs(r*Lam-lambda0)),
            "undamped_limit_error":float(abs(bd/r-v*v)*M*M),
            "damped_limit_error":float(abs(balanced/r)*M*M),
            "driver_over_r_M2":float(driver/r*M*M)})
    assert max(errors[-1][k] for k in ["beta_slope_error", "connection_limit_error",
               "undamped_limit_error", "damped_limit_error"]) < 1e-25
    rows.append({"M":mass,"R0_over_M":ratio,"p":exponent,
        "K0_M":float(K0*M),"v_M":float(v*M),"p_iso":float(aR0*R0),
        "zeta":float(zeta),"limits":errors})
# External pure-gauge/flat-normalization result is a numerical comparison only.
# Paper 0905.0450 Eq (28) uses C_paper^2=C4/4 and opposite K sign.
Cpaper2=mp.mpf(8)/243*(13*mp.sqrt(13)-35)
C4=4*Cpaper2
root=mp.findroot(lambda R:R**3*(2-R)-C4,(mp.mpf("1.3"),mp.mpf("1.45")))
out={"passed":True,"digits":100,"local_cases":rows,
     "critical_integral_zero_lapse_root_recomputed_over_M":float(root),
     "critical_Cpaper2_over_M4":float(Cpaper2),
     "critical_C4_over_M4":float(C4),"paper_text_radius_over_M":1.3955,
     "paper_text_radius_consistent_with_Eqs28_to31":False,
     "either_external_radius_is_layer_target":False}
Path(__file__).with_name("oracle-result.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
