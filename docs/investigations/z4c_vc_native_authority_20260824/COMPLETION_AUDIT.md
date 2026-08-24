# Controlling-prompt completion audit

This audit maps the staged native-authority prompt to committed evidence. A
later stage is intentionally absent when an earlier gate requires stopping.

| Requirement | Disposition | Authoritative evidence |
|---|---|---|
| Source authority and clean numerical baseline | Passed | `AUTHORITY.md`; reviewed commit `6dd20656...`, reviewed tree `551b16fa...` |
| Fresh build and regression baseline | Passed | focused A100 20/20; host 140/141 concurrently with the sole timeout passing alone in 177.96 s; `evidence/host-regression-summary.txt` |
| Fresh native N256 record to `t=2.5 M` | Passed | job 57524377; authority SHA-256 `fd08e6b...`; run evidence and scheduler provenance in `evidence/perlmutter/runs/n256_native_record_t2p5_v2_dethist/` |
| Historical comparison only through event 3 | Passed | `NATIVE_AUTHORITY_COMPARISON.md`; exact first four tree checksums, then documented repaired divergence |
| N256 deterministic record/replay | Passed under AthenaK's tested numerical-payload contract | exact hierarchy, event times, histories, diagnostics, all 33 binary payloads and six restart payloads; `analysis/record_replay_verification.json` |
| N128/N256/N512 early common tree | Passed | exact replay ledgers and `EARLY_CONVERGENCE.md` |
| Early constraint and state gate | Passed | monotone C/H/M/Z, positive effective order, positive chi and SPD pivots, finite states, zero sampled shared-node spreads; `analysis/history-v1/` and `analysis/fields-v1/` |
| AMR-event and causal gate | Passed through the early authority | event-3 C/H/M/Z all decrease at every resolution; earlier absolute event injection decreases with resolution; protected-radius margins remain positive; `amr_event_constraint_jumps.csv` and `causal_protection.csv` |
| Native-AMR characterization | Completed | `NATIVE_AMR_HEALTH.md`, full 160-state authority, and `analysis/native_summary.json` |
| Tau approximately 4 | Passed | jobs 57525084, 57525355, 57525422, and 57525474; `TAU4.md` |
| Tau approximately 7 | Failed | N256 develops fixed-hierarchy constraint growth and then a refinement cascade; job 57525753 cancelled at the fail gate; `TAU7.md` |
| Tau approximately 10.5 and full interval | Correctly not attempted | prohibited by the failed tau-7 gate; `TAU10P5.md` |
| Figure 3 reproduction | Not claimed | only an unshifted partial overlay through the qualified tau-4 window; `FIGURE3.md` |

## Record/replay hash qualification

Whole binary and restart containers cannot be byte-identical while preserving
honest authenticated parameter provenance: their text headers record
`amr_history_mode=record` versus `replay` and different basenames. AthenaK's
production integration test therefore defines numerical identity as bytes
after the `<par_end>` marker. Under that repository contract, every binary and
restart payload is byte-identical. Raw and payload hashes are both retained;
the distinction is disclosed rather than normalized away.

## Final disposition

The exact allowed verdict is `NATIVE_AMR_UNSTABLE`. The goal is complete at
that failure disposition: subsequent milestones are forbidden, not missing
work. No result beyond central proper time approximately four is qualified as
convergent, and no source-level cause is claimed.
