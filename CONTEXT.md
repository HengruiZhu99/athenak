# Hyperboloidal evolution

This glossary establishes the language used to design time refinement for hyperboloidal evolution, including evolution at future null infinity.

## Language

**Base time interval**:
The coarsest time interval, denoted by ΔT, against which temporal refinement levels are defined.
_Avoid_: Global timestep, when referring to an interval that need not be the timestep used throughout the domain.

**Temporal refinement level**:
A nonnegative integer n identifying a timestep Δtₙ = ΔT / 2ⁿ within the time-refinement hierarchy. Increasing the temporal refinement level by one halves the timestep.
_Avoid_: Refinement level without specifying whether it is temporal or spatial.

**Common endpoint** is a time at which every block has accepted its solution over the same base interval. Refinement decisions, topology transfers, checkpoints and comparisons between temporal resolutions use common endpoints. A checkpoint at a regrid endpoint contains the accepted post-regrid mesh and the block refinement ages.

**Numerical coupling edge** connects a recipient block to a block supplying values to its spatial operator. Interpolation and overlap-agreement stencils can couple blocks beyond immediate geometric neighbors.

**Overlap-agreement group** consists of the blocks contributing to or receiving the bidirectional discrepancy correction. Each member has the same temporal refinement level.

**Spherical region** is a physical radial interval of the atlas, including all its radial subdivisions and angular patches. Each spherical region has one temporal refinement level.

**Block ownership** assigns a stable mesh block identity to an execution rank and a local storage slot. Ownership may change at a common endpoint.

## Example dialogue

**Researcher:** “This region is at temporal refinement level two.”

**Developer:** “Its timestep is one quarter of the base time interval; that says nothing about the number of Runge–Kutta stages per step.”

**Researcher:** “Correct: temporal refinement counts steps, not stages.”
