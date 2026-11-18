# Use dyadic time levels on the numerical coupling graph

Fix the base interval for the run and the time levels for the lifetime of each mesh topology. Cartesian blocks inherit their spatial level relative to the Cartesian root. Spherical regions receive one exact level each. Automatic assignments use the initial characteristic and relaxation bounds, then grade neighboring regions. A change of more than one level across any actual donor edge is invalid.

The bidirectional overlap correction evaluates a single discrepancy using every contributing field. Giving all participants a common time level preserves that operator and allows independent radial subcycling farther out. Only participating Cartesian blocks can be promoted. Scri and its closure support also share a level. The startup report gives the measured membership and promotion counts for the constructed mesh.
