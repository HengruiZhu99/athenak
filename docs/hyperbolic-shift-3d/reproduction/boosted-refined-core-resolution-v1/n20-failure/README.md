# N20 BC2 terminal numerical failure

Job9037163 stopped at last accepted6.2082M/10347steps, native1933.148s
(32.219min), PBS walltime36:44. Native return2, workerfailed, PBSexit1; no
timeout/storage/report error, four-node affinity passed. Last recorded
diagnostic6.15M. Rejected RK-stage abscissa is not separately recorded.
Frozen source/binary/plan/input and all report configuration checks passed
except the expected failed status and unadmitted checkpoint. All six worker
artifacts are locally hash/length verified. No endpoint admission or restart.
`preserve_n20_failure.py` reproduces preservation checks and localization.

The complete failed capture is finite. Exactly one volume cell has a
nonpositive metric eigenvalue: cell328016 at(-.01575,.00225,.00225), r=.0160682,
distance.00905513 from the last accepted tracker. Metric eigenvalues
[-.00278346,12.61146,14.46264], determinant-.507689. Chi.000444621 and conformal
lapse.0534579 remain positive there; D components are approximately
[172133,22098,39691]. Surface metric minimum eigenvalue.837567 remains positive.
This identifies the immediate near-puncture geometry failure, not its root
cause. The tracker remains inside the zero-inner-shift/gauge plateau: final
position(-.0242092,-.0000665139,-.00000178612), cube margin.155791.
The puncture did not reach the stitching transition beginning at r=.2.

Null surface/ghost maxima1.55e-15/1.67e-15, pole/clock5.27666/12.0409.
Chi floor events80987stage/27153final, no lapse floor or surface chi floor
events. Maximum CFL.273561. Large floor counts remain an accuracy qualification.
At6.15M own-candidate exterior H=.0385219,M=.0596346; this is not a fixed-domain
endpoint comparison. Forty of42 complete saved candidates pass the user's
post-only expansion RMS1e-2 criterion, including the last candidate RMS.00589889.
Twenty-six were accepted online at1e-3 with recorded masses; missing masses
remain gaps, and full native finder success is not inferred from RMS alone.

N16 BC2 survived10M whereas this N20 failed. Thus uniform refinement has not
monotonically cured the instability; no converged boosted evolution is
established. Space/time and MPI rank count differ; no isolated spatial cause
or parallel scaling conclusion follows. Fresh N24 was independently authorized
and submitted as9037378 after failure preservation/localization and lock release.

N24 needed281092425bytes beyond the10GB capacity including reserves. Only the
failed N20 remote checkpoint was retired after verified local preservation
and fresh remote rehash; its481985232-byte local original is retained with
SHA256c4656aaae6b20b16adcde79ff111b25236a861543f744d83f14757db596ba49e.
The failed-stage capture and remaining remote evidence are retained. See the
retirement receipt in the parent directory and the local-verification record.
