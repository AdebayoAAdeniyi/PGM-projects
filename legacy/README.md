# Original coursework scripts

These are the scripts as I first wrote them for CMPUT 651, kept unchanged so
the provenance of the work is visible. They contain the machine-specific
paths and the helper imports they ran with at the time, so they will not run
as-is on another machine.

The version to read and run is the package in
[`pgm_portfolio/`](../pgm_portfolio/), which implements the same models with
portable I/O, tests and a command-line runner.

| Script | What it does |
|---|---|
| `Coding_Assignment_1_CMPUT651_D1.py` | Supervised HMM counts with add-k smoothing; joint log probability of a sentence under two candidate tag sequences |
| `Coding_Assignment_1_CMPUT651_D2.py` | Smoothing sweep over the development split, selection by mean joint log probability, random-tag comparison |
| `Coding_Assignment_2_CMPUT651_D1.py` | Max-product (Viterbi) decoding with back-pointers |
| `Coding_Assignment_2_CMPUT651_D2.py` | Viterbi accuracy sweep, word and sentence accuracy, most-frequent-tag baseline, accuracy plot |
| `PGM-projects-main/` | The same work as published to my original GitHub repository |

What changed in the rewrite, and why, is in
[`docs/IMPLEMENTATION_NOTES.md`](../docs/IMPLEMENTATION_NOTES.md). The two
substantive differences are that the rewrite keeps `log 0 = -inf` instead of
`log(p + 1e-10)`, and that it caches the log-parameters once per model
rather than recomputing the emission matrix for every sentence.
