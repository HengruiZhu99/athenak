This additive analytic observation leaves the approved plan and its sampled orientation/ADM gates unchanged.

For F(s)=exp(-((s-u0)/sigma)^2),

    phi=-sigma*integral[-1,1] F'(T+sR) ds.

Consequently |partial_z phi|<=sigma*sup|F''|*integral[-1,1]|s|ds. The center follows by continuity; away from it partial_z R=z/R has magnitude at most one. Writing q=(s-u0)/sigma,

    F''=(4q^2-2)*exp(-q^2)/sigma^2.

The extrema of the absolute value occur at q=0, q^2=3/2, its zeros, or infinity. The corresponding nonzero values are 2/sigma^2 and 4exp(-3/2)/sigma^2. Since exp(3/2)>1+3/2>2, the latter is less than 2/sigma^2. Thus sup|F''|=2/sigma^2 and |partial_z phi|<=2/sigma.

For the fixed epsilon<=.1 and sigma=.35, D=1+epsilon*phi_z>=1-2epsilon/sigma>=3/7. At fixed T,X,Y the scalar map z->z+epsilon phi is strictly increasing and onto (phi is bounded). It therefore has a unique global inverse in z. The full map is triangular, preserving T,X,Y, so this establishes map invertibility for this analytic Gaussian family. This is not a claim about a general wave map or a numerical inversion result.

This bound does not establish that the coordinate t=constant surfaces remain spacelike after the inverse map. Positive physical spatial metric and lapse still have to pass the separately declared sampled ADM gates. It supplies no evolution, boundary, scri, puncture or black-hole stability claim. The first active-coordinate perturbation is -epsilon*a^A*phi, retaining the inverse-map minus sign.
