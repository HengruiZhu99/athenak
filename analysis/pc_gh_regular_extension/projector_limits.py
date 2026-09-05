"""Check that spectral projectors have no poles at nonzero speed crossings."""
import sympy as s
from hyperbolicity import symmetric_legacy

a, chi = s.symbols('a chi', positive=True)
p = symmetric_legacy(a, chi)
p2 = p*p
speeds2 = [s.Integer(1), 2*a*chi, a*a*chi, (4-a*a*chi)/3]
denominators = set()
for family, speed in enumerate([s.Integer(0)]+speeds2):
    projector = s.eye(50)
    for other in [s.Integer(0)]+speeds2:
        if other == speed:
            continue
        projector = ((p2-other*s.eye(50))*projector).applyfunc(s.cancel)/(speed-other)
    projector = projector.applyfunc(s.cancel)
    assert all(s.cancel(value) == 0 for value in (p2-speed*s.eye(50))*projector)
    factors = set()
    for value in projector:
        if value:
            for factor, power in s.factor_list(s.denom(value))[1]:
                factors.add(str(factor))
    print(f'Family {family}, speed squared {speed}, denominator factors: {sorted(factors)}')
    denominators.update(factors)
print('All denominator factors:', sorted(denominators))

assert denominators <= {'a', 'chi', 'a - 2'}
print('PASS: no spectral-projector poles at the three permitted speed crossings')
