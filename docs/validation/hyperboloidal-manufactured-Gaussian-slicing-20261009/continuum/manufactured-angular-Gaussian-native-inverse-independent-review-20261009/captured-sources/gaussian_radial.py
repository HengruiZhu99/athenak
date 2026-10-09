"""Exact extracted Gaussian radial function definitions; no imports/evaluation."""

def functions(mp):
    def derivatives(t,sigma,n):
        x=t/sigma;e=mp.exp(-x*x/2);h0=mp.mpf(1);h1=x
        values=[sigma**4*e]
        if n:values.append(-sigma**3*h1*e)
        for k in range(1,n):
            h0,h1=h1,x*h1-k*h0
            values.append((-1)**(k+1)*sigma**(3-k)*h1*e)
        return values
    def radial(t,r,sigma,terms):
        if r==0:
            f=derivatives(t,sigma,6)
            return -2*f[5]/15,-2*f[6]/15,mp.mpf(0),'origin'
        if r<=sigma/8:
            f=derivatives(t,sigma,2*terms+6)
            c=mp.mpf(0);ct=mp.mpf(0);cr=mp.mpf(0)
            for j in range(terms):
                factor=-8*(j+2)*(j+1)/mp.factorial(2*j+5)
                c+=factor*f[2*j+5]*r**(2*j)
                ct+=factor*f[2*j+6]*r**(2*j)
                if j:cr+=factor*f[2*j+5]*(2*j)*r**(2*j-1)
            return c,ct,cr,'origin_series'
        u=derivatives(t-r,sigma,3);v=derivatives(t+r,sigma,3)
        c=(u[2]-v[2])/r**3+3*(u[1]+v[1])/r**4+3*(u[0]-v[0])/r**5
        ct=(u[3]-v[3])/r**3+3*(u[2]+v[2])/r**4+3*(u[1]-v[1])/r**5
        cr=-(u[3]+v[3])/r**3-6*(u[2]-v[2])/r**4-15*(u[1]+v[1])/r**5-15*(u[0]-v[0])/r**6
        return c,ct,cr,'advanced_retarded'
    return derivatives, radial
