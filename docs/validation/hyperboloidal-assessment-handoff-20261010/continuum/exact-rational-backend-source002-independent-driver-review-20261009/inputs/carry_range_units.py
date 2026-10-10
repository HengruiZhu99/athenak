"""Eight additional exact carry/range controls; base74 remains byte-identical."""

EXPECTED_CASES = 8


def registry():
    one, half = "3ff0000000000000", "3fe0000000000000"
    minimum, maximum = "0000000000000001", "7fefffffffffffff"
    cases = []

    def pair(name, atoms, operations, numerator, denominator, category):
        cases.append(dict(id=name+"_positive",atoms=atoms,operations=operations,
                          numerator=numerator,denominator=denominator,category=category))
        negative = operations+[("N",numerator,denominator)]
        cases.append(dict(id=name+"_negative",atoms=atoms,operations=negative,
                          numerator=len(atoms)+len(negative)-1,denominator=denominator,category=category))

    pair("subnormal_to_normal_tie",["0010000000000000",minimum,half,one],
         [("*",1,2),("-",0,4)],5,3,"finite")
    pair("normal_binade_carry",["3fefffffffffffff","3c90000000000000",one],
         [("+",0,1)],3,2,"finite")
    pair("strong_range_above_max",[maximum,minimum,one],[("+",0,1)],3,2,"exact_overflow")
    pair("strong_range_below_max",[maximum,minimum,one],[("-",0,1)],3,2,"finite")
    if len(cases)!=EXPECTED_CASES or len({case["id"] for case in cases})!=EXPECTED_CASES:
        raise RuntimeError("fixed8 carry/range registry drift")
    return cases


def check_output(text, base):
    cases,lines=registry(),text.splitlines()
    if len(lines)!=EXPECTED_CASES:
        raise RuntimeError("carry/range output count differs")
    checks=[]
    for case,line in zip(cases,lines):
        want=base.expected(case)
        fields=line.split("\t")
        if fields[:2]!=[want["id"],want["category"]] or len(fields)!=9 or fields[2]!=want["bits"]:
            raise RuntimeError("carry/range ID/category/bits mismatch: "+case["id"])
        numerator=(int(fields[3]),int(fields[4]),fields[5])
        denominator=(int(fields[6]),int(fields[7]),fields[8])
        if numerator!=want["numerator"] or denominator!=want["denominator"]:
            raise RuntimeError("carry/range full operand mismatch: "+case["id"])
        checks.append(dict(id=case["id"],category=want["category"],passed=True))
    return dict(passed=True,checks=len(checks),expected_checks=EXPECTED_CASES,
                exact_full_operand_comparisons=len(checks),rows=checks)
