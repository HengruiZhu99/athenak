# Fixed-frame covector components in the sparse identities

This note supplements, without modifying, immutable-Q-Einstein-cubic-null-curvature-timejet-20261009/index.json (SHA0302a9da72f1a36f5fd6bf7dc9311fccc08c97ac1eace3282423a2ee38517f13).

M_n,M_y,M_z and Z_n,Z_y,Z_z in the sparse identities denote the native covector components projected onto the **fixed local rational Cartesian frame at the base point**. Angular Taylor coefficients differentiate those fixed-frame components. They do not silently rotate the frame to the normal/tangents at a neighboring angular point. Nraw is a scalar. All chart/frame derivatives used in the induced two-metric curvature are separately retained by the actual construction. The momentum and spatial Z components are not multiplied by Omega or normalized by their physical metric norms.

The parameter-specific identities saved in check-report.json have the following common expression, verified exactly for the reconstructed matrices at the four tested a values .5,.75,1,2:

    L+deltaR[q]/a^2 = H000/6-2H100/a+H200/a^2+(H020+H002)/3
      -2M_n000/a+2M_n100/a^2-(M_y010+M_z001)/a
      -12Z_n000/a^2+4Z_n100/a^3
      +4Theta000/a-12Theta100/a^2+4Theta200/a^3
      +N000+2N100/a.

Subscripts are Taylor powers [Omega,y,z], so H020 is half the corresponding repeated angular derivative. This compact expression records the four checked parameter-specific identities; no symbolic-in-a or uniform-a nonlinear/PDE proof is added.
