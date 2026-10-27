# Exploratory N513 terminal duration evidence

The run reached coordinate t500 normally. All10,001 saved states were validated; final relative apparent-horizon mass drift is0.109%. This is coarse duration evidence, not the principal N1025 accuracy certificate, whose run is still active.

The12MB raw `summary.json` is preserved locally but excluded from Git. `summary.json.gz` contains its exact bytes (not rounded or thinned), matching the SHA-256 in terminal-assessment.json. The plot helper reads either form. The original profile archive remains under the run output path; source-manifest.json identifies every validated snapshot.

From the repository root, redraw the plots with:

```sh
python3 docs/elliptic-shift-1d/paper/make_repair_duration_figure.py --data docs/elliptic-shift-1d/evidence/telegraph-attenuated-n513-terminal-v1 --output /tmp/coarse-duration-status
```

No evolution or restart is performed. The collector sorted numeric report indices (including10000/10001) and excluded the AF coefficient before computing positive-radius exponentials. Collection passed with warnings promoted to errors.

The copied terminal assessment was recorded while the principal fine run was still active. Its historical scope text is preserved. The now-completed N1025 assessment is in ../attenuated-duration/.
