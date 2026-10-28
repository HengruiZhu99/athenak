Final100M coefficient-four Gamma-driver assessment.

All spatial and temporal trajectories were already completed; this assessment launches no new evolutions. The source/reference equations remain unchanged.

Entry points: assessment.json (conclusions and caveats), profiles/summary.json (374 exact spatial profiles and334 exact temporal profiles), history/assessment.json (1368 shared diagnostic samples), plots/terminal.json (endpoint quantities, peak drift, radius), verification.json (source and regression checks). Each profile comparison includes all ten fields. The norm is an unscaled mixed RMS, not a gauge-invariant physical distance.

Figures: plots/exterior-constraints-100M.{png,pdf}, plots/horizon-100M.{png,pdf}, plots/convergence-100M.{png,pdf}, inner-100M/inner-profiles.png. All data are retained without smoothing; logarithmic plots omit only nonpositive values.

Reproduce with assess_completed_campaign.py --cases [the five final paths] --output [fresh directory], assess_history_refinement.py with the three final spatial paths, and plot_completed_campaign.py --cases [case JSON] --evidence [evidence directory]. Exact commands and case paths are in commands.json. Final endpoint comparisons also use the independent existing assess_refinement.py. Raw frozen checkpoints remain under output; their hashes and full continuation ancestry are recorded.

The positive exterior result does not remove local near-puncture failures or mass-error crossing exceptions. A single endpoint or successful compiler run is not the basis for the convergence assessment.
