# Collected Data and Readiness

## Dataset obtained

- **Dataset:** Predictive Maintenance Engine Data, uploaded to Hugging Face by user `jskswamy`.
- **Source:** https://huggingface.co/datasets/jskswamy/predictive-maintenance-data. File used: `engine_data.csv` (1,282,265 bytes).
- **Access and permissions:** Public download, no login. The dataset card states the MIT license. The card gives no original source, author or citation, so the license of the underlying data cannot be confirmed.
- **Nature of the data:** Described on the card as "engine sensor readings from commercial diesel vehicles". The card does not say whether the data is real or synthetic, how it was collected, or how the labels were assigned. **Provenance is unverified.** The same 19,535 rows with the same class counts also appear in other third-party uploads (for example `dhani10/engine-condition-dataset`), and none of them names an original source.

## Sample size, unit of observation, key variables

- **Sample size:** 19,535 records, 7 columns.
- **Unit of observation:** One row of engine sensor readings at one moment. The data does not say whether the rows come from one engine or many, and there is no time or engine ID column, so no history per engine.
- **Key variables:** Engine RPM, lubricating oil pressure (bar), fuel pressure (bar), coolant pressure (bar), lubricating oil temperature (C), coolant temperature (C).
- **Target:** `Engine Condition`, where 0 = Normal and 1 = Needs Maintenance. Counts: 7,218 Normal (36.9%) and 12,317 Needs Maintenance (63.1%). The majority of rows are labeled "needs maintenance", so this is a fairly balanced classification task, not rare-failure detection.
- **Not used:** The repository also holds pre-split files with 16 extra derived features. Their method is not documented, so they are not used. We make our own split from `engine_data.csv`.

## Cleaned data ready for analysis

Checks completed on the raw file:
- Missing cells: 0. Duplicate rows: 0. Every row has 7 values.
- Value ranges: coolant temperature ranges from 61.7 to 195.5 C (mean 78.4). Only 2 rows exceed 100 C, which is implausible for a liquid-cooled engine; these are the clearest suspect values. No pressure column contains an exact zero (minimum values are small positive numbers).
- Outliers (1.5 x IQR rule per sensor): RPM 464 (2.4%), lub oil pressure 66 (0.3%), fuel pressure 1,135 (5.8%), coolant pressure 785 (4.0%), lub oil temperature 2,617 (13.4%), coolant temperature 2 (0.01%). In total 4,636 rows (23.7%) have at least one IQR outlier; with a 3 standard deviation rule it is 1,198 rows (6.1%). The share of "Needs Maintenance" among outlier rows (57.9%) is close to the overall share (63.1%). Lub oil temperature has a narrow, right-skewed distribution (71 to 90 C), so its high count reflects distribution shape rather than bad readings.

Status of cleaning:
- **Records now: 19,535 (nothing removed).** Decision: keep all rows, because there is no evidence the outliers are errors. Only the 2 rows with coolant temperature above 100 C are suspicious. As a sensitivity check, the model is re-run with outlier rows removed (see `notebooks/engine_condition_modeling.ipynb`, section 6); report the resulting row count and result here once run.
- **TODO:** stratified train/test split and a baseline model. No model has been run on this dataset yet.
- No collection or labeling work remains, but the label definition is unknown and cannot be checked from the data.

## How the data support the business decision

The decision is which engines to check or service first. Given the six sensor readings, a model can estimate the probability that an engine is in the "Needs Maintenance" state and rank readings from highest to lowest, so a maintenance team can choose how many to inspect. The sensors (oil, fuel and coolant pressure, temperatures, RPM) are ones a fleet or workshop already monitors, which makes the use case easy to explain.

## Limits

- Provenance, real or synthetic status and label definition are unknown, so results show that the method works on this data, not how it will perform on a real fleet.
- No time or engine ID, so no trend over time and no remaining-useful-life estimate.
- Binary label only, so the model cannot say why an engine needs maintenance.
- No cost data. Savings estimates use stated assumptions with a sensitivity analysis.

---
**To confirm before submitting (delete this section):**
- Run `notebooks/engine_condition_modeling.ipynb` and fill in the sensitivity-test row count and results above, plus the baseline model result.
- Be ready to answer "where does this data come from?" honestly: third-party upload, origin unverified.
- If a source is found later (original author, license), add it under "Source".
