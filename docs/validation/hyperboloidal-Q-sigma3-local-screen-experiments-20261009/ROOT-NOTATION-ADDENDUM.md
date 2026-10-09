# Additive row/column clarification

The original byte-preserved `DERIVATION.md` contains the sentence “Only β value columns change” after the σ3-minus-σ5 feedback formula. The changed output **rows** are the three β rows. Their input value columns include α, χ, the symmetric metric entries and β, through δN. No derivative columns are added by changing σ.

The original formula, actual C++ source and independent checker already implement the row statement correctly. This clarification changes no source, payload, tolerance or numerical result. The original prose remains preserved; the reviewed public draft uses the correct row wording.
