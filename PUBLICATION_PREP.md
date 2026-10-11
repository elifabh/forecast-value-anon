# Publication preparation — draft branch

This branch preserves the original `main` branch. It does not signify journal or NeurIPS acceptance and has not been assigned a DOI.

## Working manuscript
Proposed title: *Operational Value of Day-Ahead Solar Forecasts for Storage-Based Voltage Management: Evidence from Two Distribution Feeders*.

The previous workshop paper is a rejected submission, and the revised manuscript should be marked as an **unreviewed preprint** unless a journal subsequently accepts it.

## What the existing evidence supports
- Full 2,429-step violation rates (persistence vs NWP): 11.28% vs 9.80% on IEEE 33-bus; 7.49% vs 6.34% on the 109-bus component.
- Paired daily-block contrasts over 50 *complete* days: +1.54 percentage points [0.29, 3.12] and +1.21 [0.46, 2.17].
- No detectable profit difference under the stated bootstrap analysis; this does not demonstrate equivalence.
- Required curtailment is a **model-based proxy** under uniform/proportional reduction, not measured operator curtailment.
- Error-based and network-value-based forecaster rankings are **not shown to differ** by the available experiment.

## Important code clarification
In `safe_layer.py` and `gg2/safe_layer_ob.py`, **positive storage power means charging**. For overvoltage, the correction attempts to move *toward the maximum feasible charging power*, checks AC voltage, and uses a scalar regula-falsi root search (or a finite scan fallback). It does **not** simply “shrink dispatch.” Uncorrectable violations remain possible when power or energy headroom is exhausted.

## Author review and reproducibility checklist
- [ ] Confirm the author-name parsing, spelling, institutional affiliation, coauthor permissions (if relevant), and funding disclosure.
- [ ] Confirm **all manuscript numbers** against the committed logs, separating the 2,429-step and 2,400-step windows.
- [ ] Specify the exact forecast-error metric, target units, train/validation/test dates, and leakage safeguards from the actual code.
- [ ] Explain battery prices and profit calculation, including treatment of PV export revenue and feed-in tariffs.
- [ ] Audit forecast loaders (including API-key requirements), raw-data licenses, dataset versions, and conditions for reproducibility.
- [ ] Identify a stable commit, test executable installation instructions, and freeze dependency versions before tagging a release.
- [ ] Add an explicit **software license** only after author approval; the workshop paper's CC BY notice does not by itself license this code.
- [ ] Author-verify the revised paper's methods, findings, references, AI-use declaration, and preprint wording.
- [ ] Check the next journal's preprint and prior-submission rules before public deposit.
- [ ] Merge reviewed metadata/documentation onto `main`. Then connect the **public** repository to Zenodo and enable it before creating a GitHub release, so Zenodo can archive it with a *software* DOI.
- [ ] Separately deposit the author-approved PDF on Zenodo as **Publication → Preprint** if desired; a preprint DOI and a software DOI identify different artifacts.
- [ ] After publishing, allow time for Google Scholar to index the preprint (indexing is not guaranteed).

Original public code: https://github.com/elifabh/forecast-value-anon
Original discussion: https://openreview.net/forum?id=T1ePi5dwtu

**Status:** Draft planning material only. No release, Zenodo deposit, DOI registration, or Google Scholar indexing has been initiated by this branch.
