"""SOURCE-ONLY monotone compact-radius root; unchanged derivative algebra.

This module has not been imported/executed in source preparation. Parent
must admit the fresh comparison pilot before any scientific execution.
All signs below are evaluated with the SAME fixed height quadrature as
old code, not validated interval-arithmetic enclosure claims.
"""
import mpmath as mp
import derivative_core as core


class AcceleratedNativeGraph(core.NativeGraph):
    def __init__(self, recipe, height_order, amplitudes):
        super().__init__(recipe, height_order, amplitudes)
        self.compact_context = {}
        self.last_root = None
        self.max_newton = recipe["compact_root_newton_iterations"]
        self.absolute_gate = mp.mpf(recipe["tolerances"]["root"])

    def _endpoint_bracket(self, fun, increasing=True):
        """Retain both endpoint brackets; never call reference at r=1."""
        lo, hi = mp.mpf(0), mp.mpf(1)
        for count in range(self.layer.root_iterations):
            mid=(lo+hi)/2
            value=fun(mid)
            if not mp.isfinite(value):
                raise ArithmeticError("nonfinite compact endpoint equation")
            if (value>0)==increasing:
                hi=mid
            else:
                lo=mid
            if hi-lo<self.layer.root_tolerance:
                if not hi<1:
                    raise ArithmeticError("compact endpoint failed to leave scri")
                fl,fh=fun(lo),fun(hi)
                if not (fl<=0<=fh if increasing else fl>=0>=fh):
                    raise ArithmeticError("compact endpoint sign bracket failed")
                if abs(fun((lo+hi)/2))>self.absolute_gate:
                    raise ArithmeticError("compact endpoint absolute residual failed")
                return lo,hi,count+1
        raise ArithmeticError("fixed compact endpoint iteration limit")

    def _event_context(self,T,X):
        key=(T,tuple(X))
        if key in self.compact_context:
            return self.compact_context[key]
        if len(self.compact_context)>=8:
            raise RuntimeError("declared compact root-context cache exceeded")
        Re=core.vc.norm(X)
        re=self.layer.r_of_q(Re)
        tau=T-self.layer.height(re)
        if not tau>0:
            raise ArithmeticError("future native graph event required")
        uret=self.layer.defect(re)+tau
        if Re==0:
            lo,hi,it=self._endpoint_bracket(lambda r:2*self.layer.radius(r)+self.layer.defect(r)-T)
            record={"lo":lo,"hi":hi,"outer_only":lo>=self.layer.r1,"endpoint_iterations":it}
        else:
            if uret<0:
                if not self.layer.outer_constant<uret<0:
                    raise ArithmeticError("compact event outside graph future domain")
                ll,lh,li=self._endpoint_bracket(lambda r:self.layer.defect(r)-uret,increasing=False)
            elif uret==0:
                ll,lh,li=mp.mpf(0),mp.mpf(0),0
            else:
                ll,lh,li=self._endpoint_bracket(lambda r:2*self.layer.radius(r)+self.layer.defect(r)-uret)
            ul,uh,ui=self._endpoint_bracket(lambda r:2*self.layer.radius(r)+self.layer.defect(r)-(2*Re+uret))
            # Outward endpoints matter: midpoint endpoints may exclude exactly
            # parallel rays. Lower lo and upper hi retain their root brackets.
            if not 0<=ll<uh<1:
                raise ArithmeticError("invalid outward compact source bracket")
            record={"lo":ll,"hi":uh,"outer_only":ll>=self.layer.r1,"endpoint_iterations":li+ui}
        self.compact_context[key]=record
        return record

    def _at_radius(self,r,T,X,k):
        omega,od,b,h,L=self.layer.reference(r)
        q=r/omega
        ell=(T-self.layer.height(r))/k[0]
        y=[X[i]-ell*k[i+1] for i in range(3)]
        qy=core.vc.norm(y)
        F=q-qy
        qr=L/(omega*omega)
        # Exact Cartesian core branch: H_r=0, F_r=1 even when y=0.
        if b==0:
            derivative=qr
        elif qy==0:
            # F has a norm cusp here; only bisection is used at this point.
            derivative=None
        else:
            derivative=qr*(1-(b/h)*core.vc.dot(y,k[1:])/(qy*k[0]))
            if not derivative>0:
                raise ArithmeticError("nonpositive compact monotone derivative")
        return {"F":F,"ell":ell,"derivative":derivative,"height_r":qr*b/h}

    def _certificate(self,left,right,T,X,k,method,newton,fallback,context):
        if not right-left<self.layer.root_tolerance:
            return None
        A,B=self._at_radius(left,T,X,k),self._at_radius(right,T,X,k)
        if not A["F"]<=0<=B["F"]:
            return None
        if not (right-left<self.layer.root_tolerance and abs(A["ell"]-B["ell"])<self.layer.root_tolerance):
            return None
        ell=(A["ell"]+B["ell"])/2
        y=[X[i]-ell*k[i+1] for i in range(3)]
        # Original residual uses the original physical-radius inverse and
        # fixed height code. A small compact F is not substituted for it.
        original=abs(T-ell*k[0]-super().height(y))
        if original>self.absolute_gate:
            raise ArithmeticError("original absolute ray residual failed after compact root")
        self.last_root={"method":method,"accepted":True,"left":left,"right":right,
                        "radius_width":right-left,"lambda_left":A["ell"],"lambda_right":B["ell"],
                        "lambda_width":abs(A["ell"]-B["ell"]),"F_left":A["F"],"F_right":B["F"],
                        "original_g_residual":original,"newton_iterations":newton,
                        "fallback_iterations":fallback,"endpoint_iterations":context["endpoint_iterations"],
                        "radius_tolerance":self.layer.root_tolerance,"lambda_tolerance":self.layer.root_tolerance}
        return ell

    def _exact(self,ell,method,T,X,k,initial=False):
        y=[X[i]-ell*k[i+1] for i in range(3)]
        residual=abs(T-ell*k[0]-super().height(y))
        if residual>self.absolute_gate:
            raise ArithmeticError("original absolute residual failed in analytic compact branch")
        self.last_root={"method":method,"accepted":True,"lambda":ell,
                        "radius_width":mp.mpf(0),"lambda_width":mp.mpf(0),
                        "original_g_residual":residual,"newton_iterations":0,"fallback_iterations":0,
                        "endpoint_iterations":0,"initial":initial}
        return ell

    def root(self,T,X,k,initial=False):
        if not k[0]>0:
            raise ArithmeticError("nonfuture fixed null ray")
        if initial:
            return self._exact(mp.mpf(0),"initial_exact",T,X,k,initial=True)
        # Test the exact Cartesian H=0 solution BEFORE dividing by source q.
        coreell=T/k[0]
        corey=[X[i]-coreell*k[i+1] for i in range(3)]
        if core.vc.norm(corey)<=self.layer.r0:
            return self._exact(coreell,"Cartesian_core_exact",T,X,k)
        context=self._event_context(T,X)
        if context["outer_only"]:
            Re=core.vc.norm(X)
            tc=T-self.layer.outer_constant
            den=2*(tc*k[0]-core.vc.dot(X,k[1:]))
            if not den>0:
                raise ArithmeticError("nonpositive analytic outer denominator")
            return self._exact((tc*tc-Re*Re-self.a*self.a)/den,"outer_exact",T,X,k)
        lo,hi=context["lo"],context["hi"]
        A,B=self._at_radius(lo,T,X,k),self._at_radius(hi,T,X,k)
        if not A["F"]<=0<=B["F"] or not min(A["ell"],B["ell"])>=0:
            raise ArithmeticError("outward compact bracket fails fixed-ray signs or future lambda")
        x=(lo+hi)/2
        for count in range(self.max_newton):
            P=self._at_radius(x,T,X,k)
            if P["F"]<=0:
                lo=x
            else:
                hi=x
            if P["derivative"] is not None:
                candidate=x-P["F"]/P["derivative"]
            else:
                candidate=(lo+hi)/2
            if not lo<=candidate<=hi:
                candidate=(lo+hi)/2
            C=self._at_radius(candidate,T,X,k)
            # A fixed small probe, with both r and transformed-lambda widths
            # checked explicitly. Never accept an unbracketed Newton point.
            delta=self.layer.root_tolerance/(8*max(1,abs(C["height_r"]/k[0])))
            left,right=max(lo,candidate-delta),min(hi,candidate+delta)
            if left<right:
                ell=self._certificate(left,right,T,X,k,"compact_safeguarded_Newton",count+1,0,context)
                if ell is not None:
                    return ell
            x=candidate
        # Fixed fallback has the original maximum 512 and unchanged width /
        # absolute residual gates. No adaptive quadrature or tolerance change.
        for count in range(self.layer.root_iterations):
            x=(lo+hi)/2
            P=self._at_radius(x,T,X,k)
            if P["F"]<=0:
                lo=x
            else:
                hi=x
            ell=self._certificate(lo,hi,T,X,k,"compact_fixed_bisection_fallback",self.max_newton,count+1,context)
            if ell is not None:
                return ell
        raise ArithmeticError("fixed compact fallback iteration limit")
