"""HELD analytic identities with absolute errors and fixed summand scales."""
import mpmath as mp
from taylor3 import Jet, determinant, inverse, indices, factorial
from gaussian3 import wave
from geometry import (metric_from_adm,metric_from_embedding,connection,curvature,
                      embedding_connection,adm_constraints,extract_fields)
from oracle import graph_comparisons
from reference3 import reference_adm


def number(value):
    if not mp.isfinite(value):
        raise ArithmeticError("nonfinite analytic diagnostic operand")
    return mp.nstr(value,mp.mp.dps,strip_zeros=False)


def zero_row(name,terms,admitted,branch):
    terms = list(terms)
    residual, total = mp.fsum(terms),mp.fsum(abs(v) for v in terms)
    return dict(name=name,branch=branch,admission_gate=admitted,
                terms=[number(v) for v in terms],absolute=number(abs(residual)),
                signed=number(residual),term_sum=number(total),
                scaled=number(abs(residual)/max(1,total)))


def jet_rows(name,jet,admitted,branch):
    return [zero_row(name+"/"+"".join(map(str,m)),[jet.c[m]*factorial(m)],admitted,branch)
            for m in indices(jet.order)]


def comparison(name,left,right,admitted,branch):
    terms = [left,-right]
    row = zero_row(name,terms,admitted,branch)
    row["component_scale"] = number(max(1,abs(left),abs(right)))
    row["scaled"] = number(abs(left-right)/max(1,abs(left),abs(right)))
    return row


def scaled_and_unscaled_row(name,scaled_terms,omega,admitted,branch):
    """One gate, with its unscaled residual/operands retained as diagnostics."""
    row = zero_row(name,scaled_terms,admitted,branch)
    unscaled = zero_row(name+'/unscaled',[v/omega for v in scaled_terms],False,branch)
    row['unscaled_diagnostic'] = unscaled
    return row


def all_checks(data,graph_radius):
    ref, compact = data["ref"],data["compact"]
    omega, embedding = ref["omega"],data["embedding"]
    gamma, alpha, beta = compact["gamma"],compact["alpha"],compact["beta"]
    bar = metric_from_adm(alpha,beta,gamma)
    cb = curvature(bar,(0,1,2,3))
    bi,Gb,Rb = cb["inverse"],cb["Gamma"],cb["Ricci"]
    branch = data["primary"]
    checks = []
    # Normal identities are checked at every available coefficient, including
    # their time coefficients. They are not enforced by a projection step.
    checks += jet_rows("det_gtilde_minus_one",determinant(compact["gt"])-1,True,branch)
    gti = inverse(compact["gt"])
    traceA = sum(gti[i][j]*compact["A"][i][j] for i in range(3) for j in range(3))
    checks += jet_rows("trace_A",traceA,True,branch)
    checks += jet_rows("implicit_inverse",data["inverse_residual"],True,branch)
    # Complete bounded conformal vacuum identities, not Riemann[bar g]=0.
    Hess = [[omega.derivative(a,b)-mp.fsum(Gb[c][a][b].v*omega.derivative(c) for c in range(4))
             for b in range(4)] for a in range(4)]
    Box = mp.fsum(bi[a][b].v*Hess[a][b] for a in range(4) for b in range(4))
    N = mp.fsum(bi[a][b].v*omega.derivative(a)*omega.derivative(b) for a in range(4) for b in range(4))
    for a in range(4):
        for b in range(a,4):
            checks.append(zero_row("conformal_vacuum_%d%d"%(a,b),
                [omega.v**2*Rb[a][b],2*omega.v*Hess[a][b],
                 omega.v*bar[a][b].v*Box,-3*bar[a][b].v*N],True,branch))
    checks.append(zero_row("conformal_vacuum_trace",
        [omega.v**2*cb["scalar"],6*omega.v*Box,-12*N],True,branch))
    wedges = [(a,b) for a in range(4) for b in range(a+1,4)]
    for l,(a,b) in enumerate(wedges):
        for c,d in wedges[l:]:
            terms = [bar[a][e].v*cb["Riemann"][e][b][c][d] for e in range(4)]
            terms += [-bar[a][c].v*Rb[d][b]/2,bar[a][d].v*Rb[c][b]/2,
                      bar[b][c].v*Rb[d][a]/2,-bar[b][d].v*Rb[c][a]/2,
                      cb["scalar"]*bar[a][c].v*bar[d][b].v/6,
                      -cb["scalar"]*bar[a][d].v*bar[c][b].v/6]
            checks.append(zero_row("Weyl_%d%d%d%d"%(a,b,c,d),terms,True,branch))
    # Physical constraints: H, Mxyz, Zxyz, Theta, without Omega rescaling.
    for label,terms in zip(["H","Mx","My","Mz","Zx","Zy","Zz","Theta"],adm_constraints(compact,omega)):
        checks.append(zero_row("physical_"+label,terms,True,branch))
    # The field Lambda is the contracted metric connection by construction;
    # retain that definition explicitly rather than imply independent Z data.
    # Independent raw graph and physical curvature are admitted only here.
    finite = ref["rvalue"] <= graph_radius
    graph = graph_comparisons(data)
    pairs = [("alpha",graph["alpha"])]+[("beta%d"%i,p) for i,p in enumerate(graph["beta"])]
    pairs += [("gamma%d"%i,p) for i,p in enumerate(graph["gamma"])]
    for label,(left,right) in pairs:
        for m in indices(min(left.order,right.order)):
            checks.append(comparison("graph_"+label+"/"+"".join(map(str,m)),
                left.c[m]*factorial(m),right.c[m]*factorial(m),finite,"raw_graph"))
    for k,(left,right) in enumerate(graph["physical_K"]):
        checks.append(comparison("graph_K%d"%k,left,right,finite,"raw_graph"))
    physical = metric_from_embedding(embedding)
    cp = curvature(physical,(0,1,2,3))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    checks.append(zero_row("physical_Riemann_%d%d%d%d"%(a,b,c,d),
                        cp["terms"][(a,b,c,d)],finite,"raw_physical_embedding"))
    for b in range(4):
        for d in range(b,4):
            pieces = [piece for a in range(4) for piece in cp['terms'][(a,b,a,d)]]
            checks.append(zero_row('physical_Ricci_%d%d'%(b,d),pieces,finite,'raw_physical_embedding'))
    # Reference connections from independent embedding and ADM four-metric.
    Ghat = embedding_connection(ref["embedding"])
    reference_metric = metric_from_adm(*reference_adm(ref))
    _,Ghat_metric = connection(reference_metric,(0,1,2,3))
    Gphysical = embedding_connection(embedding)
    scaled_Ghat = [[[omega.v*Ghat[a][b][c].v for c in range(4)] for b in range(4)] for a in range(4)]
    scaled_F = []
    for a in range(4):
        source_terms = [bi[b][c].v*scaled_Ghat[a][b][c] for b in range(4) for c in range(4)]
        source_terms += [-2*bi[a][i].v*omega.derivative(i) for i in range(4)]
        scaled_F.append(mp.fsum(source_terms))
        # Actual expected source contract is exported separately below.
        wave_terms = [omega.v*bi[b][c].v*(Gphysical[a][b][c].v-Ghat[a][b][c].v)
                      for b in range(4) for c in range(4)]
        checks.append(scaled_and_unscaled_row("physical_wave_map_%d"%a,wave_terms,omega.v,True,branch))
        # This is equivalent to the nonzero expected source, not a zero-source
        # assertion. All contractions include time and mixed indices.
        contracted_bar = [omega.v*bi[b][c].v*Gb[a][b][c].v for b in range(4) for c in range(4)]
        checks.append(scaled_and_unscaled_row("conformal_source_%d"%a,contracted_bar+[-scaled_F[-1]],omega.v,True,branch))
        for b in range(4):
            for c in range(b,4):
                checks.append(comparison("reference_Gamma_%d%d%d"%(a,b,c),
                    Ghat[a][b][c].v,Ghat_metric[a][b][c].v,finite,"reference_embedding_vs_metric"))
                terms = [omega.v*Gphysical[a][b][c].v,-omega.v*Gb[a][b][c].v,
                         int(a==b)*omega.derivative(c),int(a==c)*omega.derivative(b),
                         -bar[b][c].v*mp.fsum(bi[a][d].v*omega.derivative(d) for d in range(4))]
                checks.append(zero_row("connection_conformal_%d%d%d"%(a,b,c),terms,True,branch))
    # Scalar F solves the inertial flat wave equation. Derivatives here are in
    # inertial coordinates and are not native-chart derivatives miscontracted.
    inertial = [Jet.variable(embedding[A].v,A,2) for A in range(4)]
    F = wave(inertial[0],inertial[1:],data["sigma"])
    checks.append(zero_row("inertial_scalar_wave",[-F.derivative(0,0)]+[F.derivative(i,i) for i in (1,2,3)],True,"inertial"))
    if data["epsilon"] == 0:
        for k,rate in enumerate(compact["rates"]):
            checks.append(zero_row("reference_rate_%d"%k,[rate],True,branch))
    return checks,dict(physical_metric=physical,bar_metric=bar,
        scaled_reference_connection=scaled_Ghat,scaled_source=scaled_F,
        source=[v/omega.v for v in scaled_F],Box=Box,N=N)


def spatial_export(jet):
    return dict(order=jet.order,ordinary=[dict(multiindex=list(m[1:]),value=number(jet.c[m]*factorial(m)))
                 for m in indices(jet.order) if m[0]==0])


def complete_export(jet):
    return dict(order=jet.order,ordinary=[dict(multiindex=list(m),value=number(jet.c[m]*factorial(m)))
                                          for m in indices(jet.order)])


def export_record(data,identity_rows,auxiliary):
    ref,compact = data["ref"],data["compact"]
    return dict(refused=False,branch=data["primary"],event=[number(v) for v in ref["event"]],
                J=number(data["J"].v),D=number(data["D"].v),
                Omega=spatial_export(ref["omega"].truncate(2)),
                Omega_native3=complete_export(ref["omega"]),
                embedding3=[complete_export(v) for v in data["embedding"]],
                reference_embedding3=[complete_export(v) for v in ref["embedding"]],
                physical_metric2=[[complete_export(v) for v in row] for row in auxiliary["physical_metric"]],
                conformal_metric2=[[complete_export(v) for v in row] for row in auxiliary["bar_metric"]],
                fields=[spatial_export(v) for v in compact["fields"]],
                exact_time_rates=[number(v) for v in compact["rates"]],
                scaled_reference_connection=[[[number(v) for v in row] for row in plane] for plane in auxiliary["scaled_reference_connection"]],
                scaled_source=[number(v) for v in auxiliary["scaled_source"]],
                unscaled_source=[number(v) for v in auxiliary["source"]],
                checks=identity_rows,
                missing_seconds=["P","Aij","Lambda_xyz","Theta"],
                gauge_binding="physical-reference RWM Einstein sector only; not compound inner BM",
                native_query_admitted=False)
