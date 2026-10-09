# Saved C0 t2 endpoint inspection

This reads existing observations only. H/M/Z and Euclidean component amplification come from the saved canonical Taylor histories. Configuration-H1/momentum-L2 component amplification comes from the existing native field-derivative analysis of Arnoldi histories, whose states agree with canonical at all 81 times/two seeds within 1.76e-13 relative. That component norm is not a physical energy or proved symmetrizer.

| Gauge/seed | H peak(value,time); t2 | M peak(value,time); t2 | Z peak(value,time); t2 | Component-unit amplification peak; t2 |
| --- | --- | --- | --- | --- |
| Production gauge |2.32295,1.875;1.72047|3.56432,1.825;1.21774|.949591,1.8;.272567|47.0983,2;47.0983|
| Spatial norm gauge |2.19266,2;2.19266|2.56829,1.825;1.22040|.911376,1.8;.369710|35.7290,2;35.7290|
| Production shell |.730122,0;.00347946|.317302,.025;.00387373|.112328,0;.000969100|1,0;.0748867|
| Spatial norm shell |.730122,0;.00250440|.316514,.025;.00228468|.112328,0;.000996051|1,0;.0169943|

Production gauge H/M/Z decrease over both the final .025 step and final .1 interval. Its component-unit norm nevertheless rises to its sampled maximum at t2. Spatial-norm gauge H and component-unit norm also rise to sampled maxima at t2. Its M falls; Z rises on the last step but falls net over the last .1 interval. Both gauge Euclidean norms have earlier sampled peaks at t1.775, followed by endpoint rebounds: production106.991→106.518 and spatialnorm98.5575→91.9671.

Both shell seeds have much larger early H/M/Z and component peaks. Production shell component norm/H fall at the endpoint; M/Z rise on the last step but fall net over the last .1 interval. Spatial-norm shell H/M/Z and component norm rise over both final intervals but remain far below their early peaks. These mixed and oscillatory histories do not support a uniform monotone-growth description or a post-peak all-time decay claim. Peaks are sampled at .025 spacing, not optimized continuous-time extrema. A separately requested fresh t6 exploratory extension is outside this receipt.

`results.json` retains exact peak/time/endpoint values, final-step and final-.1 slopes, source hashes and the canonical-versus-Arnoldi agreement pin. `inspect_saved_histories.py` can reproduce this read-only extraction. No original source, state, matrix, receipt or frozen archive is modified.
