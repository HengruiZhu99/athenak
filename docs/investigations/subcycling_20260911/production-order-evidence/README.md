Production spatial-order qualification

CPU fixture tests with spatial_order=4 and extrap_order=2, using --production-orders on test_live_subcycling.py and test_live_amr.py (also --mixed). Live vs frozen differences are below 1.2e-15; split restarts agree exactly for ratios 1 and 2. Real refine/coarsen and mixed events pass. These are smooth gauge-pulse fixtures, not strong-field Brill reproduction, production gauge stiffness qualification, or GPU performance evidence. Executable hashes are recorded in each result.

Fixed-time temporal refinement at dt=.004,.002,.001 over duration=.016 passes: successive RMS difference ratios 16.12742 synchronous and 16.03511 subcycled. This establishes fourth-order behavior for the production spatial-order smooth fixture, not Brill strong-field accuracy. GPU qualification script is prepared for pinned befad6a4; not yet executed because its build remains active.
