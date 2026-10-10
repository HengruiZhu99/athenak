"""Fixed exact cache controls; callable only after the owning unit admission."""
def run_cache_units():
    from fractions import Fraction as Q
    from interval import Context
    from interval_uncached import Context as UncachedContext
    rows = []
    def require(ok, name, **metadata):
        rows.append(dict(name=name, passed=bool(ok), **metadata))
        if not ok:
            raise RuntimeError("cache unit failed: " + name)
    def same(actual, expected, ctx):
        return actual.ctx is ctx and (actual.lo, actual.hi) == (expected.lo, expected.hi)
    def evaluate(ctx, q):
        return ctx.exp_neg(ctx.point(q))
    arguments = [Q(0), Q(1,32), Q(1,16)-Q(1,2**20), Q(1,16),
                 Q(1,16)+Q(1,2**20), Q(1,8), Q(3,2), Q(32), Q(600)]
    regular_contexts = {}
    for bits in (256,384):
        ctx, baseline = Context(bits), UncachedContext(bits)
        regular_contexts[bits] = ctx
        for index,q in enumerate(arguments):
            before = dict(ctx._exp_endpoint_audit)
            actual, repeated = evaluate(ctx,q), evaluate(ctx,q)
            expected = evaluate(baseline,q)
            delta = {k:ctx._exp_endpoint_audit[k]-before[k] for k in before}
            require(same(actual,expected,ctx) and repeated.lo==actual.lo and repeated.hi==actual.hi
                    and repeated.ctx is ctx and delta=={"hits":3,"misses":1,"evictions":0},
                    "cold-warm-equality:"+str(bits)+":"+str(index), argument=str(q), audit_delta=delta)
        before = dict(ctx._exp_endpoint_audit)
        actual=evaluate(ctx,Q(2,32));expected=evaluate(baseline,Q(1,16))
        require(same(actual,expected,ctx) and ctx._exp_endpoint_audit["misses"]==before["misses"]
                and ctx._exp_endpoint_audit["hits"]==before["hits"]+2,
                "canonical-Fraction-key:"+str(bits))
        small, baseline = Context(bits,2), UncachedContext(bits)
        okay=True
        for q in (Q(0),Q(1,32),Q(1,16),Q(0)):
            okay = okay and same(evaluate(small,q),evaluate(baseline,q),small)
        require(okay and small._exp_endpoint_audit=={"hits":4,"misses":4,"evictions":2}
                and list(small._exp_endpoint_cache)==[(bits,Q(1,16)),(bits,Q(0))],
                "FIFO-eviction-recomputation:"+str(bits),audit=small._exp_endpoint_audit)
        disabled, baseline=Context(bits,0),UncachedContext(bits)
        first,second=evaluate(disabled,Q(1,16)),evaluate(disabled,Q(1,16))
        expected=evaluate(baseline,Q(1,16))
        require(same(first,expected,disabled) and same(second,expected,disabled)
                and disabled._exp_endpoint_audit=={"hits":0,"misses":4,"evictions":0}
                and not disabled._exp_endpoint_cache,"disabled-cache:"+str(bits))
        left,right=Context(bits),Context(bits)
        xl,xr=evaluate(left,Q(1,16)),evaluate(right,Q(1,16))
        require(xl.ctx is left and xr.ctx is right and xl is not xr
                and (xl.lo,xl.hi)==(xr.lo,xr.hi)
                and left._exp_endpoint_cache is not right._exp_endpoint_cache
                and left._exp_endpoint_audit==right._exp_endpoint_audit=={"hits":1,"misses":1,"evictions":0},
                "separate-context-ownership:"+str(bits))
        bad=Context(bits);negative_rejected=False
        try:evaluate(bad,Q(-1))
        except ValueError:negative_rejected=True
        require(negative_rejected and not bad._exp_endpoint_cache
                and bad._exp_endpoint_audit=={"hits":0,"misses":0,"evictions":0},
                "negative-rejection-not-cached:"+str(bits))
        bad=Context(bits);range_rejected=False
        try:evaluate(bad,Q(2**64,16)+1)
        except ValueError:range_rejected=True
        require(range_rejected and not bad._exp_endpoint_cache
                and bad._exp_endpoint_audit=={"hits":0,"misses":1,"evictions":0},
                "range-rejection-not-cached:"+str(bits))
    x,y=regular_contexts[256],regular_contexts[384]
    before_y=dict(y._exp_endpoint_audit);evaluate(x,Q(1,16))
    require(x._exp_endpoint_cache is not y._exp_endpoint_cache
            and y._exp_endpoint_audit==before_y
            and all(k[0]==256 for k in x._exp_endpoint_cache)
            and all(k[0]==384 for k in y._exp_endpoint_cache),"cross-precision-keys-and-state")
    for cap in (False,-1,Q(1,2),4097):
        rejected=False
        try:Context(256,cap)
        except ValueError:rejected=True
        require(rejected,"invalid-cache-cap:"+str(cap))
    if len(rows)!=35:
        raise RuntimeError("fixed cache-unit registry count mismatch")
    return {"passed":True,"case_count":len(rows),"cases":rows,
            "domain_boxes_evaluated":0,"cache_cap_default":4096,
            "baseline_uncached_body_exact_v2":True,"measured_speedup_claim":False}
