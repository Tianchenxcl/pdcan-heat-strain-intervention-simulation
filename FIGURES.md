# Figures from the manuscript

This page places the manuscript figures in one sequence so that readers can move from the forecasting setup to the model architecture, predictive evaluation, component analysis, and intervention scenarios.

## Fig. 1. Schematic diagram of the model prediction timeline

<img src="figures/manuscript/Fig01_prediction_timeline.png" alt="Model prediction timeline" width="800">

The blue segment is the 30-minute historical input window. At the decision time, a predefined future control sequence is supplied to the model, which projects physiological states over the following 15 minutes.

## Fig. 2. Structure of PDCAN

<img src="figures/manuscript/Fig02_pdcan_architecture.png" alt="PDCAN architecture" width="1000">

PDCAN combines static worker features, historical physiological and environmental states, and future control conditions. Its outputs are HR, skin-temperature, and core-temperature trajectories, together with the final mPSI estimate.

## Fig. 3. PDCAN encoder

<img src="figures/manuscript/Fig03_encoder_detail.png" alt="PDCAN encoder" width="1000">

The encoder forms a static embedding, extracts local temporal patterns with a TCN and GRU, and represents the future control sequence before fusing the three information streams.

## Fig. 4. Control-aware recurrent physiological decoder

<img src="figures/manuscript/Fig04_decoder_detail.png" alt="Control-aware recurrent decoder" width="1000">

The decoder rolls forward across the 15-minute horizon using the supplied control sequence and produces the multivariable physiological trajectories used by the risk head.

## Fig. 5. Heart-rate MAE across prediction steps

<img src="figures/manuscript/Fig05_hr_mae_by_horizon.png" alt="Heart-rate MAE" width="850">

This figure compares step-wise heart-rate errors for PDCAN, LSTM, Transformer, and GRU.

## Fig. 6. Skin-temperature MAE across prediction steps

<img src="figures/manuscript/Fig06_tskin_mae_by_horizon.png" alt="Skin-temperature MAE" width="850">

This figure compares the skin-temperature forecasting errors across the 15-minute horizon.

## Fig. 7. Core-temperature MAE across prediction steps

<img src="figures/manuscript/Fig07_tcore_mae_by_horizon.png" alt="Core-temperature MAE" width="850">

This figure compares the core-temperature forecasting errors across the 15-minute horizon.

## Fig. 8. Predicted versus observed mPSI

<img src="figures/manuscript/Fig08_mpsi_predicted_vs_observed.png" alt="Predicted versus observed mPSI" width="850">

The scatter plot shows the agreement between predicted and observed final mPSI values in the fixed test set.

## Fig. 9. Distribution of mPSI residuals

<img src="figures/manuscript/Fig09_mpsi_residuals.png" alt="mPSI residual distribution" width="850">

The residual distribution provides a complementary view of the final mPSI prediction error.

## Fig. 10. Individual and median physiological trajectories under control-sequence scenarios

<img src="figures/manuscript/Fig10_scenario_trajectories.png" alt="Scenario physiological trajectories" width="1000">

The figure compares predicted HR, skin temperature, and core temperature under continued work, rest, rest plus environmental improvement, and observed workload reduction for the six representative transition blocks.

## Fig. 11. Predicted final mPSI across control-sequence scenarios

<img src="figures/manuscript/Fig11_scenario_final_mpsi.png" alt="Scenario final mPSI" width="850">

The paired lines show the projected change in final mPSI for each representative transition block. The alternative scenarios are model projections; the observed workload-reduction sequence is the factual reference.

