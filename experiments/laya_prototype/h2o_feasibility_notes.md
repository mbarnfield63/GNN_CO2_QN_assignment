# H2O QN-Scheme Desk-Check (Step 6)

Code-free desk check, per PLAN.md: is the real blocker to H2O generalization (a) graph-construction
effort (a row classifier would sidestep this), or (b) defining the right label/feature
representation for an asymmetric top (blocks either approach equally)? Grounded in one web search
plus established spectroscopy background; not an exhaustive literature review -- treat the specific
numeric claims as indicative, not authoritative, if this becomes load-bearing for a real decision.

## CO2 vs. H2O structural differences

**Rotational structure.** CO2 is linear (D∞h at equilibrium): rotational structure is described by a
single quantum number J (plus parity and, for the bending mode, vibrational angular momentum l2).
H2O is a near-prolate asymmetric top (C2v): rotational structure needs the pair of pseudo-quantum
numbers **(Ka, Kc)** alongside J -- there is no single-number rotational ladder analogous to CO2's J
progression. This is a genuine representation difference, independent of GNN vs. row-classifier: the
`combinatorial_class_id` scheme (built from `(m1, m2, m3, r)`, i.e. CO2's AFGL polyad notation) has
no direct H2O analogue and would need to be redesigned around `(v1, v2, v3, Ka, Kc, ...)` from
scratch.

**Vibrational structure.** CO2's `polyad = 2*t1 + t2 + 3*t3` grouping exists because of a strong
2:1 Fermi resonance between the symmetric stretch and bend, which is why different notational
conventions (Herzberg vs. AFGL) disagree on how to label individual states within a polyad and why
this project's "combinatorial class ID" reconciliation was needed in the first place.

The initial assumption going into this desk-check was that H2O's vibrational modes are separated
enough that this kind of polyad ambiguity mostly doesn't arise -- **this is only partially true.**
A web search on water's high-energy vibrational structure turned up an explicit reference to a
"polyad structure" in water's vibrational energy levels, and confirms that water's anharmonic,
"floppy" vibrational motion produces near-resonant, overlapping vibrational bands whose states
interact in ways not easily predicted by perturbation theory -- closely analogous to CO2's polyad
problem, addressed instead via variational (not perturbative) calculation in the water line-list
literature (Polyansky, Tennyson, Zobov, Kyuberis, Yurchenko, Lodi -- MNRAS 2018 water line list, and
earlier work). So H2O *does* have a real near-degenerate-state disambiguation problem, not just a
different rotational representation.

## Answering the plan's question: (a) or (b)?

Both, but they don't weigh equally, and they interact with the GNN-vs-classifier choice
differently:

- **(a) Graph-construction effort -- partially confirmed, and a row classifier would sidestep it.**
  H2O's polyad-like vibrational resonance structure means an analogous graph (edges linking states
  that Fermi/Coriolis-mix, playing the role CO2's inter-isotope polyad chains play) is a real,
  nontrivial engineering task -- comparable in kind to what this project already solved for CO2. A
  row-based classifier genuinely sidesteps having to construct this graph, same as it does here.
- **(b) Label/representation design -- the harder, more novel piece, and architecture-independent.**
  Redefining the combinatorial class scheme around `(v1, v2, v3, Ka, Kc)` instead of `(m1, m2, m3,
  r)` is unavoidable regardless of GNN vs. classifier. Unlike CO2, where AFGL/Herzberg were both
  established conventions this project reconciled, an equivalent "combinatorial class ID for H2O"
  convention doesn't have the same off-the-shelf precedent to build on -- this is closer to a new
  research problem than a re-application of the existing pipeline's ideas.

## Bottom line for the Laya motivation

The original framing ("does graph-vs-tabular even matter for H2O, or is generalization blocked
regardless") turns out to have a real answer on both sides. A row classifier *would* meaningfully
reduce one genuine piece of H2O-porting effort (no polyad-analogue graph to build), which is a
legitimate point in its favor beyond just the empirical accuracy result in `results.md`. But the
label/representation redesign is probably the larger and more novel piece of work either way, and
neither this prototype nor the existing GNN pipeline gets any of that work "for free." H2O
generalization stays a multi-month effort regardless of which classifier architecture underlies it;
the classifier choice affects the *shape* of that effort more than its overall size.

Sources:
- [Vibrational energy levels of the water molecule -- the polyad structure (ResearchGate figure)](https://www.researchgate.net/figure/brational-energy-levels-of-the-water-molecule-The-polyad-structure-is-indicated-by_fig2_12181466)
- [The assignment of quantum numbers in the theoretical spectra of H2O isotopologues (ResearchGate)](https://www.researchgate.net/publication/257842633_The_assignment_of_quantum_numbers_in_the_theoretical_spectra_of_the_H2O-O-16_H2O-O-17_and_H2O-O-18_molecules_calculated_by_variational_methods_in_the_region_0-26000_cm-1)
- [A high-accuracy computed water line list, MNRAS (Polyansky et al.)](https://academic.oup.com/mnras/article/368/3/1087/1022611)
