# Flat spatial height with cutoff powers: rejected native family

All work remained ignored scratch; no tracked source changes or black-hole
combination were made. The original flat-height failed variant is retained in
`report.md`, `flat_height.hpp` and `tangent-compare/`.

For this family `Omega=(1-w^p)+w^p Omega_out`, then
`b=sqrt((-r Omega')*(2Omega-rOmega'))`, `alpha=L=Omega-rOmega'`,
`chi=1`, `gtilde=I`, `Lambda=0`. Powers2,3,4 preserve monotonicity,
spacelike slices and exact Cauchy/CMC branches. Their inner boost behaves as
`sqrt(2p r0(1-Omega_out(r0))/width)*exp(p*g/2)/s`, so all derivatives remain
flat at r0 and the exponential decay softens relative to power1.

The logarithmic boost formula uses

```
F_p = p (1-Omega_out)g'/(1+e)^(p+1)
       + (-Omega_out')/(1+e)^p
z = r exp(p*g) F_p
log b = [log r + p*g + log F_p + log(2Omega+z)]/2.
```

The original flat-height consumed curvature/P/A formulas apply with the new
Omega jets. Compactification gradient/Hessian, Nref and BoxOmega/Omega are
recomputed; they must not be inherited from the power1 compactification.
`power_symbolic.py` checks the third cutoff derivatives and exact z factor.
The independent H/M/mass/Box proof applies to every monotone Omega.

Power2 default H/M/geometric/gauge fixedpoint residuals are
7.06e-14/4.71e-14/9.10e-13/7.11e-15. Power2 Release and ASAN/UBSan audits
pass. Powers3/4 Release audits pass. Each includes1,498 axis/oblique points,
all consumed Cartesian jets, exact endpoints and small-Omega parity, plus800
signed full20-field outer tensor/gauge regular/pole probes. The100-digit boost
oracles pass. Exact core/outer behavior and outer principal/pole matrices stay
unchanged. Existing tiny-Omega roundoff amplification is separated from the
regular fixedpoint gate. The numerical-range caveat for arbitrary tiny scales
remains as documented for the first prototype.

Dense broad a=.5,r0=.2,r1=.8 peaks:

| power | max alpha | max abs(P) | max abs(Axx) | max gauge speed |
|---|---:|---:|---:|---:|
| 1 | 2 | 9.55368 | 2.88793 | 4.30577 |
| 2 | 2.33373 | 12.35698 | 4.93676 | 5.10675 |
| 3 | 2.66448 | 14.70280 | 6.57203 | 5.74984 |
| 4 | 2.94532 | 16.75581 | 7.95103 | 6.29565 |

Native actual Cartesian Hdot RMS, same broad geometry, degree2,
symmetric ghost plans, unchanged0.1 lapse/0.02 shift/width.5 angular pulse:

| reference | N24 | N36 | N48 |
|---|---:|---:|---:|
| current production | .797146 | .370212 | .216247 |
| flat power1 | 5.8861 | 5.3953 | 5.0517 |
| flat power2 | 2.21766 | 1.29418 | .822434 |
| flat power3 | 1.36522 | .660376 | .345565 |
| flat power4 | .957066 | .432597 | .253792 |

Higher powers improve the rejected power1 height, but none beats the current
reference RMS. Power4 lowers N48 peak Hdot to1.61116 from production3.58918,
while increasing RMS17.4%, Mdot to.296528 from.242945 and Zdot to.012669
from.004225. It spreads the source more broadly and gives no overall
constraint-source improvement. Native initial H remains <=2.00e-14;
continuum checks pass at suitable resolutions (maximum1.03e-5).
Power2 dt consistency passes over1e-5→1e-7. Powers3/4 did not trigger
degree3 or N64/72 extensions because neither passed the production RMS gate.
This family is stopped without production adoption or an evolution claim.

Reproduction/provenance:

- `flat_power.hpp`, frozen SHA256
  `7d3c22bb5e23627a5da83d542fd4100d109fad43612ee3f995c223bd50df1135`.
- `run_power_audit.py`, `power_receipt.json`, `power_provenance.json`.
- `run_power3_audit.py`, `power3_receipt.json`.
- `run_power4_audit.py`, `power4_receipt.json`.
- Native power2: `power2-tangent/results.json`.
- Native powers3/4: `power-family-tangent/results.json`.

All receipts retain commands, executable hashes, durations, and negative
outcomes. Native source commit differs after a documentation-only parent
commit; critical source hashes are recorded independently in native receipts.
