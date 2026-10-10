# Processed data schema

`data.csv` is the full training matrix. `test.csv` is the fixed 336-window test matrix and has one additional final column named `type`.

The first 340 columns are parsed by the supplied code as follows:

| Zero-based columns | Shape | Meaning |
|---|---:|---|
| 0:5 | 5 | worker static features |
| 5:215 | 30 × 7 | 30-minute historical sequence |
| 215:335 | 15 × 8 | 15-minute future sequence |
| 335:339 | 4 | time-window metadata used by the model implementation |
| 339 | 1 | final mPSI target |

Within the historical and future arrays, the supplied parser uses the control channels and physiological channels defined in `src/prepare_selected_control_sequences.py` and `src/pdcan_scenario_core.py`. The five future control channels are humidity, air temperature, wind speed, radiant-environment proxy, and oxygen-consumption proxy. The three physiological targets are HR, skin temperature, and core temperature.

The final `type` column in `test.csv` identifies the fixed test categories used by the manuscript. Rows are retained with their original order because the scenario scripts use zero-based row IDs for auditability.

These are processed files reconstructed from FACT/PROSPIE. The source dataset record is: Havenith, G., Davey, S., Downie, V., Griggs, K., and Richmond, V. (2024), *Dataset for “Prediction of Core Body Temperature from Multiple Variables” and “The physiological strain index does not reliably identify individuals at risk of reaching a thermal tolerance limit” PROSPIE project*, Loughborough University Research Repository, https://doi.org/10.17028/rd.lboro.26076577.v1. Consult and cite that record for source-data provenance and reuse conditions.

## SHA-256 checksums

```text
data.csv  1aeec39f82cde7e078816e7303e332c14a269f34c8789b72f8630b50e2061029
test.csv  6a58b661d10805c2cc41dc6ce36b8f273a53a934d60ae0285c7fd0ccf2448872
```

