# CPAPOutcomes

[![DOI](https://img.shields.io/badge/DOI-10.5664%2Fjcsm.9750-blue)](https://doi.org/10.5664/jcsm.9750)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Citation File Format](https://img.shields.io/badge/citation-CITATION.cff-green)](CITATION.cff)

Legacy Python analysis code supporting **"Electronic health record-derived outcomes in obstructive sleep apnea managed with positive airway pressure tracking systems."** The project analyzes restricted EHR-derived workbooks for newly diagnosed obstructive sleep apnea patients treated with CPAP, comparing cardiometabolic and utilization outcomes before and after CPAP initiation and by PAP-tracking goal status.

## Article And Repository

| Item | Link or identifier |
| --- | --- |
| Final article | [Journal of Clinical Sleep Medicine](https://jcsm.aasm.org/doi/10.5664/jcsm.9750) |
| DOI | [10.5664/jcsm.9750](https://doi.org/10.5664/jcsm.9750) |
| PubMed | [PMID 34725036](https://pubmed.ncbi.nlm.nih.gov/34725036/) |
| NLM full-text record | [PMCID PMC8883092](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883092/) |
| Code repository | <https://github.com/reblocke/CPAPOutcomes> |
| Paper-aligned commit | `f64bc309860a34a8e6153da438086896c218486f` |

The PMC article is linked for open access reading and machine discovery. Full manuscript text is not mirrored in this repository.

## Authors

Primary article authors: Brian W. Locke, Sarah E. Neill, Heather E. Howe, Michael C. Crotty, Jaewhan Kim, and Krishna M. Sundar. Repository maintainer: Brian W. Locke (`@reblocke`; ORCID `0000-0002-3588-5238`).

Article affiliations include the University of Utah Division of Pulmonary, Critical Care, and Sleep Medicine; Owensboro Health Medical Group; University of Utah Health Enterprise Data Warehouse; and the University of Utah Department of Physical Therapy and Athletic Training. The repository does not include private funding or disclosure forms. The final article is the source of record for disclosures; it reports Krishna M. Sundar's Hypnoscure and Merck advisory relationships, with no conflicts reported by the other authors.

## Data Access

The source workbooks are EHR-derived clinical data and are **not public**. Do not commit raw workbooks, split workbooks, patient identifiers, row-level exports, logs containing row-level values, or PHI. To rerun the code, supply compatible local de-identified/restricted workbooks under `data/private/` or pass explicit paths with the command-line options below.

Expected private inputs:

| Workbook | Default local path | Purpose |
| --- | --- | --- |
| Combined cohort workbook | `data/private/Full n977 (minus UARS) w dAHI use and outcomes.xlsm` | Primary workbook with all expected sheets |
| At-goal subset workbook | `data/private/Full (minus UARS) meeting goals.xlsx` | Participants meeting machine AHI and nightly use thresholds |
| Not-at-goal subset workbook | `data/private/Full (minus UARS) not meeting goals.xlsx` | Participants not meeting one or both thresholds |

See [data_dictionary.md](data_dictionary.md) and [data_dictionary.csv](data_dictionary.csv) for the expected sheets, columns, derived variables, and review flags.

## Repository Layout

| Path | Role |
| --- | --- |
| `CPAPOutcomeStats.py` | Legacy Python analysis script for workbook loading, pre/post comparisons, subgroup comparisons, and the aggregate distribution figure |
| `CITATION.cff` | Structured citation metadata for the repository and preferred article citation |
| `llms.txt` | Machine-readable repository summary and agent guidance |
| `AGENTS.md` | Repository-specific instructions for future coding agents |
| `data_dictionary.md` / `data_dictionary.csv` | Human-readable and machine-usable data dictionary |
| `requirements.txt` | Python runtime dependencies |
| `output.txt` | Historical aggregate text output from the paper-aligned workflow |
| `combined difference histos.png` | Historical aggregate figure from the paper-aligned workflow |

## Quick Start

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the legacy analysis from the repository root:

```bash
python CPAPOutcomeStats.py \
  --combined-input "data/private/Full n977 (minus UARS) w dAHI use and outcomes.xlsm" \
  --at-goal-input "data/private/Full (minus UARS) meeting goals.xlsx" \
  --not-at-goal-input "data/private/Full (minus UARS) not meeting goals.xlsx" \
  --output-dir outputs/python
```

Generated text and figures are written under `outputs/python/` by default:

| Output | Meaning |
| --- | --- |
| `outputs/python/output.txt` | Console-style aggregate summary of severity distributions, baseline/post comparisons, and subgroup tests |
| `outputs/python/combined difference histos.png` | Distribution figure for pre/post differences across systolic BP, diastolic BP, SpO2, BMI, creatinine, and A1c |
| `outputs/python/PAT_IDs of goal or not.xlsx` | Optional split-helper output with restricted patient IDs if `stratify_by_goals()` is run |
| `outputs/python/Full (minus UARS) meeting goals.xlsx` | Optional restricted at-goal split workbook if `stratify_by_goals()` is run |
| `outputs/python/Full (minus UARS) not meeting goals.xlsx` | Optional restricted not-at-goal split workbook if `stratify_by_goals()` is run |

## Dependencies

| Dependency | Use |
| --- | --- |
| `pandas` and `openpyxl` | Excel workbook loading and writing |
| `numpy` and `scipy` | Numeric summaries and statistical tests |
| `scikit-learn` | Legacy imported utilities retained for compatibility |
| `matplotlib`, `seaborn`, and `matplotlib-venn` | Historical plotting dependencies |

Full execution requires the restricted local workbooks. `python CPAPOutcomeStats.py --help` can be run without data.

## Paper-Code Mapping

The script calculates pre/post CPAP differences for systolic BP, diastolic BP, SpO2, BMI, creatinine, and A1c, compares patients meeting PAP-tracking goals with those not meeting goals, and summarizes OSA severity and age differences among patients with complete vs missing systolic BP data. The tracked `output.txt` and `combined difference histos.png` are historical aggregate artifacts from the legacy Python workflow and are not a substitute for rerunning the analysis on the governed source data.

## Citation

If using this repository or reproducing its results, cite the final article and the repository commit or release used.

> Locke BW, Neill SE, Howe HE, Crotty MC, Kim J, Sundar KM. Electronic health record-derived outcomes in obstructive sleep apnea managed with positive airway pressure tracking systems. *J Clin Sleep Med.* 2022;18(3):885-894. doi:10.5664/jcsm.9750

```bibtex
@article{Locke2022_CPAP_EHR_Outcomes,
  title   = {Electronic health record-derived outcomes in obstructive sleep apnea managed with positive airway pressure tracking systems},
  author  = {Locke, Brian W. and Neill, Sarah E. and Howe, Heather E. and Crotty, Michael C. and Kim, Jaewhan and Sundar, Krishna M.},
  journal = {Journal of Clinical Sleep Medicine},
  year    = {2022},
  volume  = {18},
  number  = {3},
  pages   = {885--894},
  doi     = {10.5664/jcsm.9750},
  pmcid   = {PMC8883092},
  pmid    = {34725036}
}
```

Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

## License

Repository code and documentation are released under the MIT License; see [LICENSE](LICENSE). Restricted clinical data, private workbooks, third-party materials, and publisher-formatted article files are excluded.

## Contact

For public repository issues, use GitHub issues or pull requests. For questions involving restricted clinical data or governed analysis workbooks, contact the maintainer through established institutional channels rather than posting private data in this repository.
