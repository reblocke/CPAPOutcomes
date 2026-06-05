# CPAPOutcomes Data Dictionary

This dictionary documents the restricted workbook structure expected by `CPAPOutcomeStats.py`. It is inferred from the public analysis script and should be reviewed against the governed source workbooks before any formal data release or rerun report.

## Data Boundary

The workbooks are EHR-derived clinical data and are not public. `PAT_ID`, source workbooks, split workbooks, and any row-level derived exports should remain in ignored local folders such as `data/private/` and `outputs/python/`.

## Expected Source Workbooks

| Local workbook | Purpose | Public status |
| --- | --- | --- |
| `Full n977 (minus UARS) w dAHI use and outcomes.xlsm` | Combined analysis workbook with all expected sheets | Restricted |
| `Full (minus UARS) meeting goals.xlsx` | Participant subset meeting CPAP tracking goals | Restricted |
| `Full (minus UARS) not meeting goals.xlsx` | Participant subset not meeting one or both tracking goals | Restricted |

## Expected Sheets

| Sheet | Unit of observation | Notes |
| --- | --- | --- |
| `Population` | Participant | Contains identifiers, OSA severity inputs, CPAP tracking variables, and age used for missing-data comparisons |
| `Systolic` | Participant-metric row | Systolic blood pressure pre/post values |
| `Diastolic` | Participant-metric row | Diastolic blood pressure pre/post values |
| `SP02` | Participant-metric row | Oxygen saturation sheet name used by the script; confirm whether the workbook label intentionally uses zero in `SP02` |
| `BMI` | Participant-metric row | Body mass index pre/post values |
| `CREAT` | Participant-metric row | Serum creatinine pre/post values |
| `HBA1C` | Participant-metric row | Glycated hemoglobin pre/post values |

## Key Source Variables

| Variable | Sheet(s) | Definition | Review status |
| --- | --- | --- | --- |
| `PAT_ID` | All expected sheets | Restricted participant identifier used to align rows across sheets and to split at-goal vs not-at-goal workbooks | needs_review |
| `MACHINE_AHI_AFTER` | `Population` | Residual AHI measured by the PAP adherence tracking device after treatment initiation | needs_review |
| `AHI_AFTER_AVERAGE_HRS` | `Population` | Average nightly PAP use in hours after treatment initiation | needs_review |
| `DIAGNOSTIC_AHI_BEFORE` | `Population` | Diagnostic apnea-hypopnea index before CPAP treatment | needs_review |
| `CPAP_DME_REQUEST_PAT_AGE` | `Population` | Patient age at CPAP durable medical equipment request; used in complete vs incomplete systolic BP comparisons | needs_review |
| `Before Index (Mean)` | Metric sheets | Pre-CPAP baseline mean for the sheet-specific metric | needs_review |
| `6 Months After (Mean)` | Metric sheets | Six-month post-index mean; present in workbook but excluded from the default 9-15 month analysis window | needs_review |
| `12 Months After (Mean)` | Metric sheets | Twelve-month post-index mean used as the primary post-CPAP value | needs_review |
| `18 Months After (Mean)` | Metric sheets | Eighteen-month post-index mean; present in workbook but excluded from the default 9-15 month analysis window | needs_review |

## Derived Variables And Rules

| Derived item | Source | Rule |
| --- | --- | --- |
| CPAP tracking goal status | `MACHINE_AHI_AFTER`, `AHI_AFTER_AVERAGE_HRS` | At goal when machine AHI is less than 5 and average use is greater than 4 hours; not at goal otherwise |
| OSA severity | `DIAGNOSTIC_AHI_BEFORE` | `mild` when less than 15, `moderate` when less than 30, `severe` when 30 or greater; missing AHI returns missing severity |
| Baseline value | Metric sheets | `Before Index (Mean)` when both baseline and 12-month post values are available under the default complete-data setting |
| Post value | Metric sheets | `12 Months After (Mean)` when baseline is present under the default complete-data setting |
| Pre/post difference | Metric sheets | `12 Months After (Mean) - Before Index (Mean)`; 6- and 18-month alternatives are intentionally not used by default |
| `complete_data` | Metric sheets | Indicator set to 1 when both default baseline and post values are present; 0 otherwise |

## Generated Outputs

| Output | Default path | Data sensitivity |
| --- | --- | --- |
| `output.txt` | `outputs/python/output.txt` | Aggregate statistical summaries; review before sharing |
| `combined difference histos.png` | `outputs/python/combined difference histos.png` | Aggregate figure; review before sharing |
| `PAT_IDs of goal or not.xlsx` | `outputs/python/PAT_IDs of goal or not.xlsx` | Restricted row-level patient ID output |
| `Full (minus UARS) meeting goals.xlsx` | `outputs/python/Full (minus UARS) meeting goals.xlsx` | Restricted row-level split workbook |
| `Full (minus UARS) not meeting goals.xlsx` | `outputs/python/Full (minus UARS) not meeting goals.xlsx` | Restricted row-level split workbook |

## Unresolved Review Flags

- Confirm exact workbook column names, data types, missing-value conventions, and units against the governed workbooks.
- Confirm whether `SP02` is the intended sheet label or a historical typo for `SPO2`.
- Confirm whether the tracked historical aggregate outputs match the final paper-aligned commit before using them as release artifacts.
