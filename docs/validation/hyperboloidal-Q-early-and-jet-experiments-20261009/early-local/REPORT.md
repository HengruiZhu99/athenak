# Earlier feedback weights: local attribution audit

Status: source, reference, principal and unchanged outer-pole gates pass for the fixed finite-Omega configurations below. The sampled frozen primitive spectra retain positive roots. Neither mode is accepted as a stable evolution; no native or global evolution was run by this audit.

This is a separate continuation of the frozen conformal-Q/null-feedback candidate. The stored physical P and its geometric evolution remain unchanged. The alpha-only blend is exact physical-P lapse at W=0 and exact original conformal-Q lapse at W=1. Inputs explicitly use physical_trace_lapse=false, preferred_source=true and scri_lapse_damping=1/a. The null-feedback coefficient is sigma=5. Only its prescribed weight changes: form4 uses V=W_gauge for the supplied gauge interval (.45,.85), and form5 uses V=SmoothCutoff(.65,.85). The frozen late-onset form2 uses V=SmoothCutoff(.85,.95).

## Exact helper/API and source convention

Include early_feedback.hpp together with its inputs/ dependencies; call earlynf::Gauge(p,u,g,{5,0}) for the one proposed W control, or {5,1} for the separately examined smooth(.65,.85) mode. Assemble with qnf::Assemble so pole.beta is included exactly once. The helper calls frozen qnf::Gauge(p,u,g,{.85,.95,0,true}), then adds

    Delta pole.beta^i = V sigma Omega_i [alpha^2 (Nraw-Nraw_ref)] / |dOmega|_Euclidean^2.

The weighted null numerator is factored without division by live alpha. Frozen qnf's guard still requires g.r1<=.85. Both new weights are one by .85, and the entire gauge, outer source identity and leading full20 pole are identical to the frozen late-onset sigma5 candidate for r>=.95.

This deliberately extends feedback into 0<W<1, where the alpha-only blend already selects an algebraic spatial-source extension rather than the old preferred conformal Box identity. The full preferred Box identity is claimed only in W=1. The independent generic 4D deltaBox identity, Delta Box(Omega)=V sigma (Nraw-Nraw_ref)/Omega, is checked at all sampled finite-Omega radii. No new full transition preferred-Box identity is asserted.

Admissibility is limited to the tested geometry (.05,.95), S=1, a>=.5 and gauge r1<=.85. With outer=(1-r^2)/(2a), Omega=1-w_geo+w_geo outer, its derivative w_geo' (outer-1)-w_geo r/a is strictly negative on both earlier supports. The denominator therefore stays positive there. The code rejects nonpositive norm and does not floor it. Fixed sigma5 is finite; this is not a parameter-general NaN/inf validation or geometry admission.

## Reproducible gates

receipt.json records 376 unchanged inputs: 365 tracked src/CMake inputs and 11 scratch inputs. Seven compile/run/principal commands plus one separately pinned postprocessing command all exit zero with empty stderr. The postprocessor source was added after the initial source snapshot and has its own SHA256 field; it is not counted among the 376. Launch HEAD is ad68e9701060cec99651368e36377f482653bc32; production implementation is unchanged 27c19d20696ea6dd4704032c51dfd026218f64f2. No frozen old collector was rerun.

There are 504 actual compiled oblique/SPD principal cases: maximum symbol error 3.552714e-15, left-eigenfield error 8.881784e-16 and normalized basis condition 11.529153; harmonic endpoint is complete. Source/reference gates use 192 sampled rows, 32 outer gauge comparisons and 160 leading gauge-pole columns. Maximum reference residual is 7.549466e-15, independent generic 4D feedback Box error 2.664535e-15, and identical outer/leading-gauge-pole errors are exactly zero. These are the same leading pole matrices as the previously reconstructed full20 sigma5 gate, not a new nonlinear exact-scri closure proof.

The 560 new actual full20 dual Fourier matrices cover the same 280 parameter points as the frozen controls: a=.5,.75,1,2; r=.45,.65,.8,.85,.9,.95,.98; k=0,4,16,64,256; radial and oblique (.36,-.48,.8) directions. All use kappa_input=10. Delta from the late helper occurs only in the real radial beta row and value columns alpha,chi,beta_radial,g_radial_radial; error outside those entries is exactly zero. Outer matrices and W=0 baseline matrices are exactly identical. The full old 1120-matrix payload remains frozen separately.

## Frequency convention and retained negatives

k is the unscaled Cartesian coordinate phase: delta u=v exp(+i k n.x), with constant primitive20 amplitude, d=ikn and dd=-k^2 nn. It is not an Omega-normalized amplitude or physical orthonormal wave number. Reference jets are included; frozen amplitude coefficients are not spatially differentiated. These matrices are neither a coefficient-aware subsidiary generator nor the global boundary/KO discrete operator.

For a=.5, maximum Re(lambda) over sampled radii/directions at each k is:

| k | C0 physical-P/spatialnorm | late V(.85,.95) | early V=W | early V(.65,.85) |
|---:|---:|---:|---:|---:|
| 0 | 1.518016 | 20.399344 | 1.518016 | 1.518016 |
| 4 | 1.531952 | 22.467616 | 3.343106 | 3.169341 |
| 16 | 1.269652 | 21.298014 | 8.745175 | 8.745175 |
| 64 | -0.259155 | 20.168941 | 20.168941 | 20.168941 |
| 256 | -0.253218 | 15.290290 | 15.137307 | 15.137307 |

Both earlier modes retain their full sampled worst at r=.98,k64,radial, Re(lambda)=20.168941, Im(lambda)=-28.804166. The earlier W k4 maximum is at r=.8, radial, with Im(lambda)=1.325920; the smooth65 maximum is likewise at .8 with Im(lambda)=1.269901. The k16 maximum is r=.98, radial, with Im(lambda)=-3.776856. All-a worst earlier maxima are 20.168941,15.061212,12.445687,6.921416 respectively. check-report.json preserves all by-k/by-radius maxima and positive-root counts, rather than excluding these roots.

N16/span2.2 Cartesian Nyquist is 22.847947 and N24/span2.1 is 35.903916. Thus sampled k0,4,16 are resolved in both grids; the retained positive k16 mode is already resolved. This sampled subset is not an exhaustive below-Nyquist sweep, and no k32/direct-Nyquist probe was added. The positive k64 warning also remains relevant to refinement/continuum behavior, although it is beyond these coarse coordinate Nyquists.

The onset-region resolved growth is substantially smaller than the late control. One cheap actual global V=W control is therefore justified as an attribution experiment, with its initial source/Jv/stage/short-canonical gates first. This is not a request for long native evolution, t6 or a continuum stability claim. Earlier physical-P preferred/sigma controls used a different lapse and restricted feedback to the harmonic collar; this deliberately distinct Q/alpha-blend transition continuation does not repeat that source definition.
