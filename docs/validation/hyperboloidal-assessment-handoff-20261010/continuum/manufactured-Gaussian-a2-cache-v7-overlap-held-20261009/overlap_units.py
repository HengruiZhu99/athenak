"""Fixed exact 2-sigma remainder/dispatch controls; no import-time evaluation.

Dispatch controls replace only the two formula callables with zero-valued
recorders, restore them in finally, and never assert positivity. Separate point
overlap controls use all four unchanged formula implementations without stubs.
"""
from fractions import Fraction as Q
from math import factorial


def run_overlap_units(recipe):
    from interval import Context, pow2
    import producer_bounds as producer
    import replay_bounds as replay
    rows = []

    def require(condition, name):
        rows.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise RuntimeError("overlap unit failed: " + name)

    K = recipe["series_order"]
    if K != 32 or recipe["series_interface_sigma_multiple"] != 2:
        raise RuntimeError("only the fixed reviewed K32 / 2-sigma units are defined")
    den = (2*K+3)*(2*K+5)*(2*K+7)
    scaled = [2*producer.moment(2*K+7)*Q(2)**(2*K+2)/(den*factorial(2*K+2)),
              2*producer.moment(2*K+8)*Q(2)**(2*K+2)/(den*factorial(2*K+2)),
              producer.moment(2*K+7)*Q(2)**(2*K)/(den*factorial(2*K+1))]
    pencil = [Q(2)**103*factorial(35)/(67*69*71*factorial(66)),
               Q(2)**31*68*70*72/factorial(36),
               Q(2)**100*factorial(35)/(67*69*71*factorial(65))]
    for j, bound in enumerate(scaled):
        require(bound == pencil[j], "2sigma-remainder-pencil-identity:" + str(j))
        require(bound < pow2([-70, -71, -68][j]), "2sigma-remainder-dyadic-bound:" + str(j))
    sigmas = [Q(x) for x in recipe["sigma"]]
    if sigmas != [Q(7, 20), Q(1, 2)]:
        raise RuntimeError("unexpected profile registry")
    for sigma in sigmas:
        rhi = 2*sigma
        dimensional = [2*replay.derivative_bound(2*K+7,sigma)*rhi**(2*K+2)/(den*factorial(2*K+2)),
                       2*replay.derivative_bound(2*K+8,sigma)*rhi**(2*K+2)/(den*factorial(2*K+2)),
                       replay.derivative_bound(2*K+7,sigma)*rhi**(2*K)/(den*factorial(2*K+1))]
        for j, bound in enumerate(dimensional):
            require(bound == scaled[j]/sigma**(j+1),
                    "replay-dimensional-remainder-identity:" + str((sigma,j)))
        roots = [(Q(1,50), 2*sigma), (2*sigma, Q(4))]
        require(roots[0][0] == Q(1,50) and roots[1][1] == Q(4), "domain-endpoints:" + str(sigma))
        require(roots[0][1] == roots[1][0] == 2*sigma, "domain-no-gap-shared-endpoint:" + str(sigma))
        require(roots[0][0] < roots[0][1] < roots[1][1], "domain-positive-widths:" + str(sigma))

    for module, entry_name in [(producer, "coefficients"), (replay, "lower_bound")]:
        context = Context(recipe["bits"] if module is producer else recipe["replay_bits"])
        original_regular, original_separated = module.regular, module.separated
        calls = []

        def regular_record(*args):
            calls.append("regular")
            return [context.point(0) for _ in range(3)]

        def separated_record(*args):
            calls.append("separated")
            return [context.point(0) for _ in range(3)]

        try:
            module.regular, module.separated = regular_record, separated_record
            for sigma in sigmas:
                cases = [(sigma/2,sigma,"regular"), (3*sigma/4,5*sigma/4,"regular"),
                         (sigma,2*sigma,"regular"), (2*sigma,2*sigma,"regular"),
                         (2*sigma,3*sigma,"separated"), (3*sigma,4*sigma,"separated"),
                         (3*sigma/2,5*sigma/2,"reject")]
                for lo,hi,expected in cases:
                    calls.clear()
                    rejected = False
                    try:
                        getattr(module,entry_name)(context,(lo,hi,Q(0),Q(0)),sigma,K)
                    except ValueError:
                        rejected = True
                    require((rejected and calls == []) if expected == "reject" else
                            (not rejected and calls == [expected]),
                            "actual-dispatch:" + str((module.__name__,sigma,lo,hi,expected)))
        finally:
            module.regular, module.separated = original_regular, original_separated
        if module.regular is not original_regular or module.separated is not original_separated:
            raise RuntimeError("dispatch controls did not restore original formulas")

    pctx, rctx = Context(recipe["bits"]), Context(recipe["replay_bits"])
    operands = []
    for sigma in sigmas:
        radius = 2*sigma
        for time in [Q(0),sigma]:
            families = [producer.regular(pctx,pctx.point(radius),pctx.point(time),radius,sigma,K),
                        producer.separated(pctx,pctx.point(radius),pctx.point(time),sigma),
                        replay.regular(rctx,rctx.point(radius),rctx.point(time),radius,sigma,K),
                        replay.separated(rctx,rctx.point(radius),rctx.point(time),sigma)]
            for j in range(3):
                intervals = [family[j] for family in families]
                require(max(x.lo for x in intervals) <= min(x.hi for x in intervals),
                        "four-formula-point-overlap:" + str((sigma,time,j)))
                operands.append({"sigma":str(sigma),"R":str(radius),"T":str(time),"field":j,
                                 "intervals":[[str(x.lo),str(x.hi)] for x in intervals]})
    if len(rows) != recipe["expected_overlap_unit_count"]:
        raise RuntimeError("overlap-unit registry count mismatch")
    return {"passed":True,"case_count":len(rows),"cases":rows,"point_operands":operands,
            "domain_certificate_evaluated":False,"domain_positivity_claimed":False,
            "dispatch_controls_use_restored_zero_recorders":True,
            "point_overlap_controls_use_actual_unchanged_formulas":True}
