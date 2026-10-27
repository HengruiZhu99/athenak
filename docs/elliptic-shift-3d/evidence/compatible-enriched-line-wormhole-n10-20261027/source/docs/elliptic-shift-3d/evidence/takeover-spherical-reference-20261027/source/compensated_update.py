"""Selectable Kahan accumulation of accepted RK increments, no field projection."""
def accumulate(state,correction,increment):
    adjusted=increment-correction
    updated=state+adjusted
    correction=(updated-state)-adjusted
    return updated,correction
