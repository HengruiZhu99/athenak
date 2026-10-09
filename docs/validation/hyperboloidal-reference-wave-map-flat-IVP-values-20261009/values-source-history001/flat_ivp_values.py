"""HELD: independent exact-flat native-pulse integral, scalar values only.

Imports stdlib and mpmath only. It never imports native/reference source,
queries a kernel, integrates a PDE in time, reconstructs an inverse map,
computes numerical map derivatives, or claims native acceptance.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import mpmath as mp


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def number(value):
    if not mp.isfinite(value):
        raise ArithmeticError("nonfinite multiprecision value")
    return mp.nstr(value, mp.mp.dps)


def dot(a, b):
    return mp.fsum(x*y for x, y in zip(a, b))


def norm(a):
    return mp.sqrt(dot(a, a))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def scaled(a, b):
    return max(abs(x-y)/max(1, abs(x), abs(y)) for x, y in zip(a, b))


def frame(axis):
    """A fixed event-axis frame; no derivatives of this chart are claimed."""
    axis = [x/norm(axis) for x in axis]
    j = min(range(3), key=lambda i: abs(axis[i]))
    trial = [mp.mpf(int(i == j)) for i in range(3)]
    e1 = cross(axis, trial)
    e1 = [x/norm(e1) for x in e1]
    return axis, e1, cross(axis, e1)


def direction(basis, mu, az):
    if not (-1 <= mu <= 1):
        raise ArithmeticError("coarea cosine outside exact admissible interval")
    axis, e1, e2 = basis
    transverse = mp.sqrt((1-mu)*(1+mu))
    return [mu*axis[i] + transverse*(mp.cos(az)*e1[i]+mp.sin(az)*e2[i]) for i in range(3)]


@functools.lru_cache(None)
def gauss(order, dps):
    if mp.mp.dps != dps:
        raise RuntimeError("quadrature cache precision mismatch")
    nodes, weights = mp.gauss_quadrature(order, "legendre")
    return tuple(nodes), tuple(weights)


def integrate(function, lower, upper, order):
    if upper < lower:
        raise ArithmeticError("reversed integration interval")
    if upper == lower:
        return mp.mpf(0)
    nodes, weights = gauss(order, mp.mp.dps)
    mid, half = (lower+upper)/2, (upper-lower)/2
    return half*mp.fsum(w*function(mid+half*x) for x, w in zip(nodes, weights))


class Layer:
    """Independent S1/a.5/.05-.95 formulas, no native imports or tables."""
    def __init__(self, recipe, height_order):
        self.a = mp.mpf(recipe["a"])
        self.r0 = mp.mpf(recipe["geometry_r0"])
        self.r1 = mp.mpf(recipe["geometry_r1"])
        self.panels = tuple(mp.mpf(x) for x in recipe["height_panels"])
        self.order = height_order
        self.root_tolerance = mp.mpf(recipe["root_tolerance"])
        self.root_iterations = recipe["maximum_root_iterations"]
        self.maximum_root_absolute_residual = mp.mpf(0)
        if self.a != mp.mpf(".5") or self.r0 != mp.mpf(".05") or self.r1 != mp.mpf(".95"):
            raise RuntimeError("this independent prototype is deliberately fixed to the reviewed geometry")
        if self.panels[0] != self.r0 or self.panels[-1] != self.r1:
            raise RuntimeError("height panel endpoint mismatch")
        self.prefix = [ -self.r0 ]
        for lo, hi in zip(self.panels[:-1], self.panels[1:]):
            self.prefix.append(self.prefix[-1] + integrate(self.defect_derivative, lo, hi, self.order))
        q1 = self.r1/self.reference(self.r1)[0]
        self.outer_constant = self.prefix[-1] - self.a**2/(mp.sqrt(q1*q1+self.a*self.a)+q1)

    @functools.lru_cache(None)
    def reference(self, r):
        if not (0 <= r < 1):
            raise ArithmeticError("reference radius must satisfy 0<=r<1")
        if r <= self.r0:
            return mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(1)
        outer = (1-r)*(1+r)/(2*self.a)
        outer_d = -r/self.a
        if r >= self.r1:
            omega, omega_d, b = outer, outer_d, r/self.a
        else:
            width = self.r1-self.r0
            s, t = (r-self.r0)/width, (self.r1-r)/width
            logit = -1/s+1/t
            # Logistic and its complement are both evaluated from the smaller
            # exponential. No subtraction of 1-w and no endpoint clipping.
            e = mp.exp(-abs(logit))
            small = e/(1+e)
            w, one_minus_w = (small, 1/(1+e)) if logit <= 0 else (1/(1+e), small)
            wp = e/(1+e)**2 * (1/s**2+1/t**2)/width
            omega = one_minus_w+w*outer
            omega_d = wp*(outer-1)+w*outer_d
            b = r*w/self.a
        h = mp.sqrt(omega*omega+b*b)
        L = omega-r*omega_d
        if not (omega > 0 and h > 0 and L > 0):
            raise ArithmeticError("invalid independent reference")
        return omega, omega_d, b, h, L

    def defect_derivative(self, r):
        omega, omega_d, b, h, L = self.reference(r)
        return -L/(h*(h+b))

    @functools.lru_cache(None)
    def defect(self, r):
        if r <= self.r0:
            return -r
        if r >= self.r1:
            omega = self.reference(r)[0]
            q = r/omega
            return self.outer_constant+self.a**2/(mp.sqrt(q*q+self.a*self.a)+q)
        panel = next(i for i in range(len(self.panels)-1) if r <= self.panels[i+1])
        return self.prefix[panel]+integrate(self.defect_derivative, self.panels[panel], r, self.order)

    def radius(self, r):
        return r/self.reference(r)[0]

    def height(self, r):
        return self.radius(r)+self.defect(r)

    def bisect(self, function, increasing=True):
        # r=1 is an analytic bracket endpoint; never evaluate Omega there.
        lo, hi = mp.mpf(0), mp.mpf(1)
        for count in range(self.root_iterations):
            mid = (lo+hi)/2
            value = function(mid)
            if not mp.isfinite(value):
                raise ArithmeticError("nonfinite root value")
            if (value > 0) == increasing:
                hi = mid
            else:
                lo = mid
            if hi-lo < self.root_tolerance:
                r = (lo+hi)/2
                residual = abs(function(r))
                self.maximum_root_absolute_residual = max(self.maximum_root_absolute_residual, residual)
                if residual > mp.mpf("1e-40"):
                    raise ArithmeticError("fixed root absolute residual gate")
                return r
        raise ArithmeticError("fixed root iteration limit")

    def source_bounds(self, event_r, tau):
        Re = self.radius(event_r)
        uret = self.defect(event_r)+tau
        if uret < 0:
            if not (self.outer_constant < uret < 0):
                raise ArithmeticError("event outside graph future domain")
            lower = self.bisect(lambda r: self.defect(r)-uret, increasing=False)
        else:
            lower = mp.mpf(0) if uret == 0 else self.bisect(lambda r: 2*self.radius(r)+self.defect(r)-uret)
        upper = self.bisect(lambda r: 2*self.radius(r)+self.defect(r)-(2*Re+uret))
        if not (0 <= lower < upper < 1):
            raise ArithmeticError("invalid source intersection endpoints")
        return Re, uret, lower, upper

    def r_of_q(self, q):
        if q < 0:
            raise ArithmeticError("negative physical radius")
        if q <= self.r0:
            return q
        q1 = self.radius(self.r1)
        if q >= q1:
            return q/(mp.sqrt(q*q+self.a*self.a)+self.a)
        return self.bisect(lambda r: self.radius(r)-q)


def pulse_pi(layer, r, nu, amplitudes):
    """Complete fixed-inertial Pi=s/Omega^2, factored near scri."""
    omega, omega_d, b, h, L = layer.reference(r)
    x, y, z = [r*v for v in nu]
    lapse_shape = 1+mp.mpf(".2")*x+mp.mpf(".3")*y*z
    shift_shape = [1+mp.mpf(".3")*y*z, mp.mpf(".2")*x, mp.mpf(".1")*x*y]
    amp_a, amp_b = [mp.mpf(v) for v in amplitudes]
    exponential = mp.exp(-r*r/mp.mpf(".35")**2)
    f = ((1-r)*(1+r))**4*exponential
    f_over_omega3 = (2*layer.a)**4*omega*exponential if r >= layer.r1 else f/omega**3
    alpha = h+amp_a*f*lapse_shape
    if not alpha > 0:
        raise ArithmeticError("invalid initial live lapse")
    bn = dot(shift_shape, nu)
    bt = [shift_shape[i]-bn*nu[i] for i in range(3)]
    pi0 = -f_over_omega3*(h*amp_a*lapse_shape+(b*L/h)*amp_b*bn)/alpha
    pin = b*amp_a*lapse_shape+L*amp_b*bn
    pis = [-f_over_omega3*(pin*nu[i]+omega*amp_b*bt[i])/alpha for i in range(3)]
    return [pi0]+pis


def azimuth_average(function, count):
    rows = [function(2*mp.pi*k/count) for k in range(count)]
    return [mp.fsum(row[i] for row in rows)/count for i in range(4)]


def center_value(layer, tau, nmu, naz, amplitudes):
    T = tau  # H(0)=0 exactly.
    rs = layer.bisect(lambda r: 2*layer.radius(r)+layer.defect(r)-T)
    omega, omega_d, b, h, L = layer.reference(rs)
    q = layer.radius(rs)
    nodes, weights = gauss(nmu, mp.mp.dps)
    basis = frame([mp.mpf(0), mp.mpf(0), mp.mpf(1)])
    rows = []
    for mu, weight in zip(nodes, weights):
        average = azimuth_average(lambda az: pulse_pi(layer, rs, direction(basis, mu, az), amplitudes), naz)
        rows.append([weight*v/2 for v in average])
    averaged_pi = [mp.fsum(row[i] for row in rows) for i in range(4)]
    factor = q*omega**3/(h+b)
    return [factor*v for v in averaged_pi], {"center_source_radius": number(rs)}


def coarea(layer, event, nrad, naz):
    xyz = [mp.mpf(v) for v in event["xyz"]]
    re, tau = norm(xyz), mp.mpf(event["tau_reference"])
    if not tau > 0:
        raise ArithmeticError("values events require positive reference time")
    if re == 0:
        result, meta = center_value(layer, tau, nrad, naz, event["amplitudes"])
        return result, result, meta
    basis = frame(xyz)
    omega_e = layer.reference(re)[0]
    Re, uret, lower, upper = layer.source_bounds(re, tau)
    breaks = sorted(set([lower, upper]+[x for x in layer.panels if lower < x < upper]))
    nodes, weights = gauss(nrad, mp.mp.dps)
    sums = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid, half = (lo+hi)/2, (hi-lo)/2
        for node, weight in zip(nodes, weights):
            r = mid+half*node
            omega, omega_d, b, h, L = layer.reference(r)
            q, w = layer.radius(r), uret-layer.defect(r)
            mu = 1-w/q+w/Re-w*w/(2*Re*q)
            averaged_pi = azimuth_average(lambda az: pulse_pi(layer, r, direction(basis, mu, az), event["amplitudes"]), naz)
            # azimuth integral/(4pi) = azimuth average/2.
            factor = half*weight*r*L/(2*re*h)
            sums.append([factor*v for v in averaged_pi])
    phi = [mp.fsum(row[i] for row in sums) for i in range(4)]
    return [omega_e*v for v in phi], phi, {"source_r_lower": number(lower), "source_r_upper": number(upper), "segments": len(breaks)-1}


def ray(layer, event, nmu, naz):
    """Independent constant-frame past-null-ray parameterization."""
    xyz = [mp.mpf(v) for v in event["xyz"]]
    re, tau = norm(xyz), mp.mpf(event["tau_reference"])
    omega_e, ode, be, he, Le = layer.reference(re)
    Re, T = layer.radius(re), layer.height(re)+tau
    axis = [v/re for v in xyz] if re else [mp.mpf(0),mp.mpf(0),mp.mpf(1)]
    X = [Re*v for v in axis]
    basis = frame(axis)
    gamma, boost = (he/omega_e, be/omega_e) if event.get("ray_boost", False) else (mp.mpf(1),mp.mpf(0))
    center_r = layer.bisect(lambda r: 2*layer.radius(r)+layer.defect(r)-T) if re == 0 else None
    core_only = Re+T <= layer.r0
    outer_only = False
    if re:
        _,_,source_lower,source_upper = layer.source_bounds(re,tau)
        outer_only = source_lower >= layer.r1
    nodes, weights = gauss(nmu, mp.mp.dps)
    output = []
    for mu, weight in zip(nodes, weights):
        k0 = gamma+boost*mu
        for j in range(naz):
            az = 2*mp.pi*j/naz
            transverse = mp.sqrt((1-mu)*(1+mu))
            spatial = [transverse*(mp.cos(az)*basis[1][i]+mp.sin(az)*basis[2][i])+(gamma*mu+boost)*axis[i] for i in range(3)]
            if center_r is not None:
                ell=layer.radius(center_r)
            elif core_only:
                ell=T/k0
            elif outer_only:
                T0=T-layer.outer_constant
                denominator=2*(T0*k0-dot(X,spatial))
                if not denominator > 0:
                    raise ArithmeticError("invalid analytic outer ray root")
                ell=(T0*T0-Re*Re-layer.a*layer.a)/denominator
            else:
                lo, hi = mp.mpf(0), T/k0
                for iteration in range(layer.root_iterations):
                    ell = (lo+hi)/2
                    y = [X[i]-ell*spatial[i] for i in range(3)]
                    q = norm(y)
                    r = layer.r_of_q(q)
                    residual = T-ell*k0-layer.height(r)
                    if residual > 0:
                        lo = ell
                    else:
                        hi = ell
                    if hi-lo < layer.root_tolerance:
                        break
                else:
                    raise ArithmeticError("fixed ray root iteration limit")
                ell = (lo+hi)/2
            y = [X[i]-ell*spatial[i] for i in range(3)]
            q = norm(y)
            r = center_r if center_r is not None else layer.r_of_q(q)
            if abs(T-ell*k0-layer.height(r)) > mp.mpf("1e-40"):
                raise ArithmeticError("fixed ray root absolute residual gate")
            nu = [v/q for v in y] if q else [mp.mpf(0)]*3
            omega, omega_d, b, h, L = layer.reference(r)
            pi = pulse_pi(layer, r, nu, event["amplitudes"])
            # h*K = h*k0-b*nu.kspace, written as nonnegative terms.
            # k0-nu.kspace = |kspace-k0*nu|^2/(2*k0).
            difference = [spatial[i]-k0*nu[i] for i in range(3)]
            hK = omega*omega*k0/(h+b)+b*dot(difference,difference)/(2*k0)
            if not hK > 0:
                raise ArithmeticError("nonpositive exact ray denominator")
            factor = weight*ell*omega**3/(2*naz*hK)
            output.append([factor*v for v in pi])
    value = [mp.fsum(row[i] for row in output) for i in range(4)]
    return value, [v/omega_e for v in value]


def flat_constant(order, naz, Re, T, velocity):
    lower, upper = abs(Re-T), Re+T
    if Re == 0:
        # Independent flat ray sphere integral, not the assigned target value.
        nodes, weights = gauss(order, mp.mp.dps)
        return T*mp.fsum(w*velocity/naz for w in weights for j in range(naz))/2
    return integrate(lambda q: q*velocity/(2*Re), lower, upper, order)


def pure_cmc(order, naz, degree):
    a = mp.mpf(".5")
    X = [mp.mpf(".3"),mp.mpf("-.2"),mp.mpf(".1")]
    Re = norm(X)
    T = mp.sqrt(Re*Re+a*a)+mp.mpf(".2")
    uret = T-Re
    upper = ((T+Re)**2-a*a)/(2*(T+Re))
    lower = (uret*uret-a*a)/(2*uret) if uret >= a else (a*a-uret*uret)/(2*uret)
    basis = frame(X)
    def polynomial(y):
        if degree == 0:
            return mp.mpf(1)
        if degree == 1:
            return y[0]+mp.mpf(".3")*y[1]-mp.mpf(".2")*y[2]
        if degree == 2:
            return y[0]*y[1]+mp.mpf(".3")*(y[1]**2-y[2]**2)
        raise RuntimeError("unadmitted exact harmonic degree")
    def integrand(q):
        H = mp.sqrt(q*q+a*a)
        D = T-H
        mu = (Re*Re+q*q-D*D)/(2*Re*q)
        values = []
        for j in range(naz):
            nu = direction(basis, mu, 2*mp.pi*j/naz)
            values.append(2*(degree+1)*polynomial([q*v for v in nu])/a)
        return q*(a/H)*mp.fsum(values)/(2*Re*naz)
    numerical = integrate(integrand, lower, upper, order)
    z = T*T-Re*Re
    exact = polynomial(X)*(1-a**(2*degree+2)/z**(degree+1))
    return numerical, exact


def initial_audit(layer, xyz, amplitudes):
    r = norm(xyz)
    nu = [v/r for v in xyz] if r else [mp.mpf(0)]*3
    omega, omega_d, b, h, L = layer.reference(r)
    pi = pulse_pi(layer, r, nu, amplitudes)
    s = [omega**2*v for v in pi]
    f = ((1-r)*(1+r))**4*mp.exp(-r*r/mp.mpf(".35")**2)
    x,y,z = xyz
    da = mp.mpf(amplitudes[0])*f*(1+mp.mpf(".2")*x+mp.mpf(".3")*y*z)
    db = [mp.mpf(amplitudes[1])*f*v for v in [1+mp.mpf(".3")*y*z,mp.mpf(".2")*x,mp.mpf(".1")*x*y]]
    alpha = h+da
    betahat = [-b*h/L*v for v in nu]
    beta = [betahat[i]+db[i] for i in range(3)]
    n = [h/omega]+[b/omega*v for v in nu]
    E = []
    for i in range(3):
        E.append([b*L/(h*omega**2)*nu[i]]+[(mp.mpf(int(i==j))-nu[i]*nu[j])/omega+L/omega**2*nu[i]*nu[j] for j in range(3)])
    direct = [omega*((1/alpha-1/h)*int(A==0)-mp.fsum((beta[i]/alpha-betahat[i]/h)*E[i][A] for i in range(3))) for A in range(4)]
    ncov = [-n[0]]+n[1:]
    J = mp.matrix([[int(A==B)-s[A]*ncov[B] for B in range(4)] for A in range(4)])
    det = mp.det(J)
    return {"s": [number(v) for v in s],"pi": [number(v) for v in pi],"alpha":number(alpha),"normal_data_scaled_error":number(scaled(s,direct)),"detJ":number(det),"detJ_expected":number(h/alpha),"determinant_scaled_error":number(scaled([det],[h/alpha]))}


def package_pins(root):
    return {str(f):sha(f) for f in sorted(Path(root).rglob("*.py"))}


def run(recipe, output):
    all_rows, all_initial, all_controls, heights, rays = [],[],[],[],[]
    for dps in recipe["precisions"]:
        mp.mp.dps = dps
        for level in recipe["levels"]:
            started = time.monotonic()
            layer = Layer(recipe, level["height_order"])
            for probe in recipe["height_probe_radii"]:
                r = mp.mpf(probe)
                heights.append({"dps":dps,"level":level["name"],"r":probe,"defect":number(layer.defect(r)),"outer_constant":number(layer.outer_constant)})
            for event in recipe["events"]:
                begin = time.monotonic()
                value, phi, metadata = coarea(layer,event,level["radial_order"],level["azimuth_order"])
                row={"dps":dps,"level":level["name"],"name":event["name"],"u":list(map(number,value)),"phi":list(map(number,phi)),"seconds":time.monotonic()-begin,"metadata":metadata,"maximum_root_absolute_residual":number(layer.maximum_root_absolute_residual),"interpretation":"reference evaluation event; no native target inverse"}
                all_rows.append(row)
                write_json(output/"partial-values.json",all_rows)
            for Re,T,velocity in recipe["flat_constant_controls"]:
                v=flat_constant(level["radial_order"],level["azimuth_order"],mp.mpf(Re),mp.mpf(T),mp.mpf(velocity))
                exact=mp.mpf(velocity)*mp.mpf(T)
                all_controls.append({"kind":"flat_constant","dps":dps,"level":level["name"],"Re":Re,"T":T,"value":number(v),"exact":number(exact),"scaled_error":number(scaled([v],[exact]))})
            for degree in recipe["pure_cmc_degrees"]:
                v,exact=pure_cmc(level["radial_order"],level["azimuth_order"],degree)
                all_controls.append({"kind":"pure_CMC_zero_Dirichlet","degree":degree,"dps":dps,"level":level["name"],"value":number(v),"exact":number(exact),"scaled_error":number(scaled([v],[exact]))})
            write_json(output/"partial-controls.json",all_controls)
            write_json(output/"partial-height.json",heights)
            print("level-complete",dps,level["name"],time.monotonic()-started,flush=True)
        final_layer=Layer(recipe,recipe["levels"][-1]["height_order"])
        for point in recipe["initial_points"]:
            xyz=[mp.mpf(v) for v in point]
            for amplitudes in recipe["initial_amplitude_controls"]:
                all_initial.append({"dps":dps,"xyz":point,"amplitudes":amplitudes,**initial_audit(final_layer,xyz,amplitudes)})
        for event in recipe["ray_events"]:
            cv,cp,cm=coarea(final_layer,event,128,128)
            for level in recipe["ray_levels"]:
                begin=time.monotonic()
                value,phi=ray(final_layer,event,level[0],level[1])
                rays.append({"dps":dps,"name":event["name"],"orders":level,"u":list(map(number,value)),"phi":list(map(number,phi)),"coarea_u":list(map(number,cv)),"seconds":time.monotonic()-begin})
                write_json(output/"partial-rays.json",rays)
        write_json(output/"partial-initial.json",all_initial)
    # All comparison logic uses saved numerical values only; no eigensolve,
    # finite-difference map derivative, inversion or time advance is present.
    mp.mp.dps=max(recipe["precisions"])
    checks=[]
    def check(name,error,tolerance):
        checks.append({"name":name,"error":number(error),"tolerance":tolerance,"passed":bool(error<=mp.mpf(tolerance))})
    lookup={(r["dps"],r["level"],r["name"]):[mp.mpf(v) for v in r["u"]] for r in all_rows}
    for event in recipe["events"]:
        name=event["name"]
        for dps in recipe["precisions"]:
            final=lookup[(dps,"full128",name)]
            for other in ["full64","rad128az64","rad64az128"]:
                check("coarea_convergence/%s/%d/%s"%(name,dps,other),scaled(final,lookup[(dps,other,name)]),recipe["tolerances"]["value_convergence"])
        check("precision/"+name,scaled(lookup[(recipe["precisions"][0],"full128",name)],lookup[(recipe["precisions"][1],"full128",name)]),recipe["tolerances"]["precision"])
        if event["amplitudes"]==["0","0"]:
            check("zero_pulse/"+name,max(abs(v) for v in lookup[(recipe["precisions"][1],"full128",name)]),"0")
    for row in all_initial:
        for key in ["normal_data_scaled_error","determinant_scaled_error"]:
            check("initial/%s/%s/%s"%(row["dps"],row["xyz"],key),mp.mpf(row[key]),recipe["tolerances"]["initial_data"])
    for row in all_controls:
        if row["level"]=="full128":
            check("control/%s/%s/%s"%(row["kind"],row.get("degree",row.get("Re")),row["dps"]),mp.mpf(row["scaled_error"]),recipe["tolerances"]["value_convergence"])
    for r in recipe["height_probe_radii"]:
        for dps in recipe["precisions"]:
            vals={row["level"]:mp.mpf(row["defect"]) for row in heights if row["dps"]==dps and row["r"]==r}
            check("height/%d/%s"%(dps,r),scaled([vals["full128"]],[vals["full64"]]),recipe["tolerances"]["height_convergence"])
    for event in recipe["ray_events"]:
        name=event["name"]
        for dps in recipe["precisions"]:
            rows=[row for row in rays if row["name"]==name and row["dps"]==dps]
            final=[mp.mpf(v) for v in rows[-1]["u"]]
            previous=[mp.mpf(v) for v in rows[-2]["u"]]
            check("ray_convergence/%s/%d"%(name,dps),scaled(final,previous),recipe["tolerances"]["ray_comparison"])
            value=[mp.mpf(v) for v in rows[-1]["coarea_u"]]
            check("ray_coarea/%s/%d"%(name,dps),scaled(final,value),recipe["tolerances"]["ray_comparison"])
    for filename,data in [("values.json",all_rows),("initial-data.json",all_initial),("controls.json",all_controls),("height.json",heights),("rays.json",rays),("checks.json",checks)]:
        write_json(output/filename,data)
    return {"passed_scalar_values_only":all(row["passed"] for row in checks),"failed_checks":[r for r in checks if not r["passed"]],"checks":len(checks),"coarea_rows":len(all_rows),"initial_rows":len(all_initial),"control_rows":len(all_controls),"ray_rows":len(rays),"scope":"Converged finite-event scalar values and initial data only. No numerical map derivatives/inversion, native target coverage, global caustic exclusion, PDE evolution or native acceptance."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    receipt={"kind":"Independent exact-flat native-pulse scalar-value integral attempt","accepted_native":False,"scientific_scope":"values_only"}
    protected_before={}
    try:
        recipe_path=Path(args.recipe).resolve()
        auth_path=Path(args.authorization).resolve()
        recipe=json.loads(recipe_path.read_text())
        auth=json.loads(auth_path.read_text())
        required={str(Path(__file__).resolve()):sha(__file__),str(recipe_path):sha(recipe_path),str(HERE/"VALUES-PLAN.md"):sha(HERE/"VALUES-PLAN.md")}
        if auth.get("scalar_values_execution_admitted") is not True or auth.get("source_pins") != required:
            raise PermissionError("missing exact root scalar-values release")
        if Path(auth["fresh_output_path"]).resolve()!=out:
            raise PermissionError("output path not admitted")
        for key,value in recipe["dependency_pins"].items():
            if sha(key)!=value:
                raise RuntimeError("dependency pin mismatch: "+key)
        if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
            raise RuntimeError("bytecode creation must be disabled")
        if sys.version_info[:2]!=(3,9) or mp.__version__!="1.3.0":
            raise RuntimeError("unreviewed Python/mpmath runtime")
        runtime=Path(sys.executable).resolve()
        if str(runtime)!=recipe["python_runtime_path"] or sha(runtime)!=recipe["python_runtime_sha256"]:
            raise RuntimeError("Python executable identity mismatch")
        mp_root=Path(mp.__file__).resolve().parent
        mp_pins=package_pins(mp_root)
        if mp_pins!=recipe["mpmath_python_pins"]:
            raise RuntimeError("mpmath package source inventory mismatch")
        protected_before={**required,**recipe["dependency_pins"],**mp_pins,str(runtime):sha(runtime),str(auth_path):sha(auth_path)}
        receipt.update({"source_before":protected_before,"authorization":str(auth_path),"output":str(out),"runtime":str(runtime),"python":platform.python_version(),"mpmath":mp.__version__,"command":sys.argv,"environment":{key:os.environ.get(key) for key in ["PYTHONDONTWRITEBYTECODE","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","PYTHONPATH"]}})
        write_json(out/"before.json",receipt)
        result=run(recipe,out)
        receipt.update(result)
        if not result["passed_scalar_values_only"]:
            raise ArithmeticError("fixed scalar-values gate failed; see checks.json")
    except BaseException as exc:
        receipt.update({"passed_scalar_values_only":False,"exception_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        def after_hash(key):
            try:
                return sha(key)
            except BaseException as exc:
                return "READ_ERROR:"+type(exc).__name__+":"+str(exc)
        receipt["source_after"]={key:after_hash(key) for key in protected_before}
        receipt["sources_unchanged"]=receipt["source_after"]==protected_before
        if not receipt["sources_unchanged"]:
            receipt["passed_scalar_values_only"]=False
            receipt["provenance_error"]="protected source/runtime drift"
        receipt["seconds"]=time.monotonic()-begin
        receipt["output_pins"]={str(f):sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!="receipt.json"}
        write_json(out/"receipt.json",receipt)
        if not receipt["sources_unchanged"]:
            raise RuntimeError("protected source/runtime drift; failed receipt retained")


if __name__=="__main__":
    main()
