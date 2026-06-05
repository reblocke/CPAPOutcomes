# AGENTS

## Project Purpose

This public repository contains legacy Python analysis code for the Journal of Clinical Sleep Medicine article "Electronic health record-derived outcomes in obstructive sleep apnea managed with positive airway pressure tracking systems" (DOI `10.5664/jcsm.9750`, PMID `34725036`, PMCID `PMC8883092`).

## Public And Data-Safety Rules

- Treat the repository as public.
- Do not commit raw EHR workbooks, derived row-level exports, patient identifiers, PHI, credentials, local paths, or private drafts.
- Do not copy publisher-formatted article files or full manuscript text into Markdown. Link the DOI, PubMed, and PMC records instead.
- Keep generated patient-derived outputs under ignored local directories such as `outputs/` unless intentionally preserving aggregate historical artifacts.
- The tracked `output.txt` and `combined difference histos.png` are aggregate historical outputs. Do not replace them with outputs from restricted data unless the change is explicitly reviewed.

## How To Orient Quickly

1. Read `README.md` for scope, article identifiers, run command, data restrictions, citation, and license.
2. Read `llms.txt` for a compact machine-readable summary and agent cautions.
3. Use `data_dictionary.md` and `data_dictionary.csv` for expected workbook sheets, variables, and derived outputs.
4. Inspect `CPAPOutcomeStats.py` before running; full execution requires local restricted workbooks.

## Workflow

From the repository root:

```bash
python -m pip install -r requirements.txt
python CPAPOutcomeStats.py --help
python CPAPOutcomeStats.py \
  --combined-input "data/private/Full n977 (minus UARS) w dAHI use and outcomes.xlsm" \
  --at-goal-input "data/private/Full (minus UARS) meeting goals.xlsx" \
  --not-at-goal-input "data/private/Full (minus UARS) not meeting goals.xlsx" \
  --output-dir outputs/python
```

Full execution should fail clearly if the restricted local workbooks are absent.

## Verification Before Publishing Changes

- Run `python -m py_compile CPAPOutcomeStats.py`.
- Run `python CPAPOutcomeStats.py --help`.
- Validate `CITATION.cff` after citation edits.
- Parse `data_dictionary.csv` after dictionary edits.
- Run `git diff --check`.
- Search for stale local paths, generic placeholder text, and restricted-data filenames before pushing.

## Documentation Standards

- Keep `README.md`, `llms.txt`, `AGENTS.md`, `CITATION.cff`, and the data dictionary internally consistent.
- Preserve the article DOI, PMID, PMCID, and paper-aligned commit unless a source-backed correction is made.
- Flag uncertain variable definitions as `needs_review` rather than inventing definitions.
