# PDCAN: physiology-driven heat-strain intervention simulation

This repository accompanies our manuscript, *A physiology-driven decision support model for simulating heat strain interventions in construction*. We provide the processed data, source code, model artifacts, result tables, and figure files needed to inspect the analysis and reproduce the main computational workflows.

The study is organised around a simple question: given a worker’s current physiological state, recent history, and a specified future control sequence, can the model project how heart rate, skin temperature, core temperature, and heat-strain risk may evolve over the next 15 minutes? The repository follows the same order as the manuscript: model definition and forecasting, model comparison and component ablation, and bounded control-sequence scenario simulation.

## What is included

```text
data/
  data.csv                         processed training matrix
  test.csv                         fixed 336-window test matrix; final column is `type`
src/
  pdcan_scenario_core.py           PDCAN architecture, scaling, loss, and inference
  prepare_selected_control_sequences.py
                                   selection and donor matching for Section 5.6
  run_selected_control_scenarios.py
                                   fixed-epoch fitting, scenario inference, and validation
  make_selected_scenario_figures.py
                                   publication-format exports for Figs. 10 and 11
notebooks/
  model_comparison.ipynb           PDCAN, LSTM, GRU, and Transformer comparison
  pdcan_ablation.ipynb             PDCAN component-ablation analysis
models/pdcan_scenario/
  final_model_state_dict.pt        checkpoint paired with the scenario results
  scalers_train_full.json          training-only scaling parameters
  hyperparameters.json             PDCAN hyperparameters
  fixed_epochs.json                fixed training epoch count
  training_provenance.json         seed, data hash, software, and device provenance
results/
  model_comparison/                comparison tables
  ablation/                        component-ablation tables
  scenario/                        scenario inputs, predictions, and diagnostics
  figures/                         model-comparison and factual-prediction figures
figures/manuscript/                figures appearing in the manuscript, Figs. 1–11
FIGURES.md                          figure-by-figure guide
```

The old exploratory `alternative_control_scenarios_outputs` directory is not included. It used an earlier all-window protocol and is separate from the selected transition-window experiment reported in Section 5.6.

## Main figures

The figure files are collected in [`FIGURES.md`](FIGURES.md). A few representative figures are shown here to make the analysis easier to follow.

### Forecasting setup

<img src="figures/manuscript/Fig01_prediction_timeline.png" alt="Forecasting timeline" width="720">

The model receives a 30-minute historical window and a predefined 15-minute future control sequence. The control sequence begins at the decision time and is used to project physiological trajectories.

### PDCAN architecture

<img src="figures/manuscript/Fig02_pdcan_architecture.png" alt="PDCAN architecture" width="900">

Static worker characteristics, historical physiological and environmental states, and future control conditions are encoded jointly. The decoder produces the three physiological trajectories and the final mPSI estimate.

### Forecasting performance

<img src="figures/manuscript/Fig05_hr_mae_by_horizon.png" alt="Heart-rate MAE by forecasting horizon" width="760">

The step-wise evaluation compares PDCAN with the three baseline sequence models over the 15-minute horizon.

<img src="figures/manuscript/Fig08_mpsi_predicted_vs_observed.png" alt="Predicted versus observed mPSI" width="760">

The mPSI comparison evaluates the model’s final risk estimate against the observed test-set value.

### Bounded intervention scenarios

<img src="figures/manuscript/Fig10_scenario_trajectories.png" alt="Scenario physiological trajectories" width="900">

For six representative workload-transition blocks, the same current state is paired with four future control sequences. The alternative sequences are model projections; the observed workload-reduction sequence is the factual reference.

<img src="figures/manuscript/Fig11_scenario_final_mpsi.png" alt="Scenario final mPSI" width="760">

This paired comparison shows how the projected final mPSI changes across the four control sequences. The results are descriptive and support scenario comparison rather than causal-effect estimation.

## Reproduce the Section 5.6 scenario experiment

The scripts can be run from the repository root with Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python src/run_selected_control_scenarios.py --stage prepare
python src/run_selected_control_scenarios.py --stage train
python src/run_selected_control_scenarios.py --stage scenarios
python src/make_selected_scenario_figures.py
```

`--stage all` combines preparation, fixed-epoch fitting, and scenario inference. The screening protocol identifies 30 Type 3 workload-transition windows in the fixed test set. Six representative windows (sample IDs 255, 271, 287, 303, 319, and 335; zero-based) form the primary descriptive cohort, while 24 overlapping windows are retained for supplementary sensitivity analysis.

The four future control sequences are continued work, rest, rest plus environmental improvement, and the observed workload-reduction sequence. Only the last sequence has paired physiological observations. The other sequences are model-generated projections and should not be interpreted as causal intervention effects.

## Reproduce the comparison and ablation analyses

The two notebooks preserve the fixed-epoch comparison and PDCAN component-ablation workflows used to generate the manuscript tables. The corresponding CSV and Excel result files are already provided under `results/model_comparison/` and `results/ablation/`. The notebooks use the processed matrices in `data/`.

## Data provenance and availability

The processed matrices were reconstructed from the public FACT dataset (PROSPIE). The original dataset record is:

> Havenith, G., Davey, S., Downie, V., Griggs, K., and Richmond, V. (2024). *Dataset for “Prediction of Core Body Temperature from Multiple Variables” and “The physiological strain index does not reliably identify individuals at risk of reaching a thermal tolerance limit” PROSPIE project*. Loughborough University Research Repository. https://doi.org/10.17028/rd.lboro.26076577.v1

The original dataset should be obtained from and cited at its source record. Before public release, the redistribution terms for the processed matrices should be checked. If the source terms do not permit redistribution, the data files should be removed from the public repository and replaced by a documented reconstruction procedure.

## Suggested Data Availability statement

> The processed data analysed in this study were reconstructed from the publicly available FACT dataset (PROSPIE). The processed training and fixed-test datasets, selected scenario inputs, PDCAN model checkpoint and scaling parameters, analysis scripts, notebooks, result tables, and figure source files are available at https://github.com/Tianchenxcl/pdcan-heat-strain-intervention-simulation. The original FACT/PROSPIE dataset should be obtained from and cited at its original public source.


