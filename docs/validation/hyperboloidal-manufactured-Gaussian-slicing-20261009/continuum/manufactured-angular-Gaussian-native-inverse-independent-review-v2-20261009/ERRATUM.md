The first saved-only reviewer checked all 54,432 rows then failed its own exact
printed-maxima equality assumption. Each row is printed at its 80/110-digit
level; the final stored maximum is printed at 110 digits. Exact textual-decimal
equality is therefore stronger than the producer serialization promises.

V2 preserves that failure and compares these maxima using the sum of one
decimal significant-digit ulp per saved level and final aggregate. This is a
serialization consistency check, not a scientific tolerance change or an
interval/rounding theorem for mpmath evaluation. Every row still meets the
original 1e-50 root residual, 1e-55 width and 1e-30 identity/convergence gates.
No wave evaluation, inverse solution or scientific import is performed.
