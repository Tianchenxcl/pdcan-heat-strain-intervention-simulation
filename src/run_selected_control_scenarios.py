"""Run the selected transition-window control-intervention experiment.

Recommended sequence (from this folder):
  python run_selected_control_scenarios.py --stage train
  python run_selected_control_scenarios.py --stage scenarios

``--stage all`` performs both. The script uses the supplied PDCAN architecture
and training data, then replaces only future control inputs. Alternative
physiological outcomes are projections; the factual workload-reduction input is
the only scenario with observed physiological labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import pdcan_scenario_core as core
from prepare_selected_control_sequences import BASE_DIR, OUTPUT_DIR, SCENARIOS, prepare

CONTROL_NAMES = ["humidity", "air_temperature", "wind_speed", "radiant_environment", "oxygen_consumption"]
PHYS_NAMES = ["HR", "Tskin", "Tcore"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "pdcan_scenario"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_model(train_raw, scalers):
    """Train the fixed PDCAN once, without using scenario outputs or test labels."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hp = core.BEST_HYPERPARAMS["PDCAN"]
    epochs = core.FIXED_EPOCHS["PDCAN"]
    core.seed_everything(core.SEED + 101)
    model = core.build_model("PDCAN", hp, scalers).to(core.DEVICE)
    loader = core.make_loader(core.transform_raw(train_raw, scalers), hp["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    amp = torch.cuda.amp.GradScaler(enabled=(core.USE_AMP and core.DEVICE.type == "cuda"))
    history = []
    for epoch in range(epochs):
        loss, tf = core.train_one_epoch(model, loader, optimizer, hp, epoch, epochs, amp)
        scheduler.step(loss)
        history.append({"epoch": epoch + 1, "train_loss": loss, "teacher_forcing_ratio": tf,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"PDCAN | epoch {epoch+1:03d}/{epochs:03d} | loss {loss:.5f} | tf {tf:.3f}")
    checkpoint = MODEL_DIR / "final_model_state_dict.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    core.save_scalers(scalers, str(MODEL_DIR / "scalers_train_full.json"))
    (MODEL_DIR / "hyperparameters.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")
    (MODEL_DIR / "fixed_epochs.json").write_text(json.dumps({"PDCAN": epochs}, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(MODEL_DIR / "training_history_fixed_epoch.csv", index=False, encoding="utf-8-sig")
    provenance = {"source": "New fixed-epoch fit for this scenario experiment; not the unavailable original manuscript checkpoint",
        "seed": core.SEED + 101, "epochs": epochs, "train_rows": len(train_raw.y_seq),
        "train_sha256": sha256(BASE_DIR / "data.csv"), "checkpoint_sha256": sha256(checkpoint),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "device": str(core.DEVICE),
        "test_labels_used_for_training_or_selection": False,
        "use_future_context": True, "use_autoregressive_feedback": False, "use_mpsi_residual": True}
    (MODEL_DIR / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Saved:", checkpoint)


def saved_scalers():
    """Use the scalers paired with the checkpoint; never refit them on test inputs."""
    state = json.loads((MODEL_DIR / "scalers_train_full.json").read_text(encoding="utf-8"))
    objects = []
    for name in ["static", "hist", "ctrl", "y_phys", "y_mpsi"]:
        obj = core.NanStandardizer()
        obj.mean_ = np.asarray(state[name + "_mean"], dtype=np.float32)
        obj.scale_ = np.asarray(state[name + "_scale"], dtype=np.float32)
        if np.any(obj.scale_ <= 0):
            raise ValueError("Invalid saved scaling parameters.")
        objects.append(obj)
    core.BEST_HYPERPARAMS["PDCAN"] = json.loads((MODEL_DIR / "hyperparameters.json").read_text(encoding="utf-8"))
    return core.Scalers(*objects)


def slice_data(data, ids, n_total):
    return {k: (v[ids] if len(v) == n_total else v) for k, v in data.items()}


def predictions_long(predictions, selected):
    rows = []
    for scenario, result in predictions.items():
        phys, mpsi = result
        for n, sample_id in enumerate(selected.sample_id.astype(int)):
            for minute in range(15):
                rows.append({
                    "sample_id": int(sample_id), "source_row_number": int(sample_id + 2),
                    "cohort": str(selected.iloc[n].cohort), "scenario": scenario,
                    "minute": minute + 1, "predicted_HR": float(phys[n, minute, 0]),
                    "predicted_Tskin": float(phys[n, minute, 1]), "predicted_Tcore": float(phys[n, minute, 2]),
                    "predicted_mPSI_final": float(mpsi[n]) if minute == 14 else np.nan,
                })
    return pd.DataFrame(rows)


def final_outcomes(predictions, selected):
    rows = []
    for scenario, (phys, mpsi) in predictions.items():
        for n, sample_id in enumerate(selected.sample_id.astype(int)):
            rows.append({"sample_id": int(sample_id), "source_row_number": int(sample_id + 2),
                "cohort": str(selected.iloc[n].cohort), "intervention_start_minute": int(selected.iloc[n].intervention_start_minute),
                "scenario": scenario, "final_HR": float(phys[n, -1, 0]), "final_Tskin": float(phys[n, -1, 1]),
                "final_Tcore": float(phys[n, -1, 2]), "final_mPSI": float(mpsi[n])})
    out = pd.DataFrame(rows)
    base = out[out.scenario == "continued_work"].set_index("sample_id")
    factual = out[out.scenario == "workload_reduction"].set_index("sample_id")
    for ref_name, ref in [("continued_work", base), ("workload_reduction_model_reference", factual)]:
        for col in ["final_HR", "final_Tskin", "final_Tcore", "final_mPSI"]:
            out[f"delta_{col}_vs_{ref_name}"] = out.apply(lambda r: r[col] - ref.loc[r.sample_id, col], axis=1)
    return out


def summarize(final):
    rows = []
    for cohort, gcohort in final.groupby("cohort", sort=True):
        for scenario, g in gcohort.groupby("scenario", sort=False):
            row = {"cohort": cohort, "scenario": scenario, "n": len(g)}
            for col in ["final_HR", "final_Tskin", "final_Tcore", "final_mPSI",
                        "delta_final_HR_vs_continued_work", "delta_final_Tcore_vs_continued_work",
                        "delta_final_mPSI_vs_continued_work"]:
                row[f"{col}_median"] = float(g[col].median())
                row[f"{col}_q25"] = float(g[col].quantile(.25))
                row[f"{col}_q75"] = float(g[col].quantile(.75))
            rows.append(row)
    return pd.DataFrame(rows)


def directional(final):
    rows = []
    base = final[final.scenario == "continued_work"].set_index("sample_id")
    for cohort, gc in final.groupby("cohort", sort=True):
        for scenario in ["rest", "rest_environment", "workload_reduction"]:
            g = gc[gc.scenario == scenario]
            d = g.set_index("sample_id").join(base[["final_HR", "final_Tcore", "final_mPSI"]], rsuffix="_continued")
            rows.append({"cohort": cohort, "scenario": scenario, "n": len(d),
                "fraction_lower_HR": float((d.final_HR < d.final_HR_continued).mean()),
                "fraction_lower_Tcore": float((d.final_Tcore < d.final_Tcore_continued).mean()),
                "fraction_lower_mPSI": float((d.final_mPSI < d.final_mPSI_continued).mean())})
    return pd.DataFrame(rows)


def factual_observed(test_raw, selected, factual_pred):
    rows = []
    for n, i in enumerate(selected.sample_id.astype(int)):
        rows.append({"sample_id": i, "cohort": str(selected.iloc[n].cohort),
            "observed_final_HR": float(test_raw.y_seq[i, -1, 0]), "observed_final_Tskin": float(test_raw.y_seq[i, -1, 1]),
            "observed_final_Tcore": float(test_raw.y_seq[i, -1, 2]), "observed_final_mPSI": float(test_raw.y_mpsi[i, 0]),
            "predicted_final_HR": float(factual_pred[0][n, -1, 0]), "predicted_final_Tskin": float(factual_pred[0][n, -1, 1]),
            "predicted_final_Tcore": float(factual_pred[0][n, -1, 2]), "predicted_final_mPSI": float(factual_pred[1][n])})
    return pd.DataFrame(rows)


def plot_outputs(final, predictions, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    primary_ids = set(final.loc[final.cohort == "primary", "sample_id"])
    for ax, var, idx in zip(axes, ["HR", "Tskin", "Tcore"], range(3)):
        for scenario, (phys, _) in predictions.items():
            scenario_ids = final.loc[final.scenario == scenario, "sample_id"].to_numpy()
            arr = phys[np.isin(scenario_ids, list(primary_ids)), :, idx]
            x = np.arange(1, 16)
            ax.plot(x, np.median(arr, axis=0), label=scenario)
            ax.fill_between(x, np.quantile(arr, .25, axis=0), np.quantile(arr, .75, axis=0), alpha=.12)
        ax.set_title(f"Primary-window predicted {var}")
        ax.set_xlabel("Future minute")
        ax.set_ylabel(var)
        ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7)
    fig.savefig(out / "primary_predicted_trajectories.png", dpi=220)
    plt.close(fig)

    p = final[final.cohort == "primary"]
    ref = p[p.scenario == "continued_work"].set_index("sample_id")
    data, labels = [], []
    for scenario in ["rest", "rest_environment", "workload_reduction"]:
        g = p[p.scenario == scenario].set_index("sample_id")
        data.append((g.final_mPSI - ref.final_mPSI).to_numpy())
        labels.append(scenario)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks([1, 2, 3], labels)
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Predicted final mPSI change vs continued work")
    ax.set_title(f"Primary transition windows (n={len(primary_ids)})")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "primary_delta_mPSI_boxplot.png", dpi=220)
    plt.close(fig)


def run_scenarios(selected, controls, train_raw, test_raw, scalers):
    checkpoint = MODEL_DIR / "final_model_state_dict.pt"
    model = core.load_checkpoint_model(scalers, checkpoint)
    ids = selected.sample_id.astype(int).to_numpy()
    test_data = slice_data(core.transform_raw(test_raw, scalers), ids, len(test_raw.y_seq))
    # Abort on misalignment rather than attaching predictions to incorrect CSV rows.
    np.testing.assert_array_equal(test_raw.source_row[ids, 0], ids + 2)
    np.testing.assert_array_equal(controls[:, 3], test_raw.u_ctrl[ids])
    predictions = {}
    for j, scenario in enumerate(SCENARIOS):
        print("Predicting", scenario)
        predictions[scenario] = core.predict_one_control_scenario(model, test_data, controls[:, j], scalers)
        if not all(np.isfinite(p).all() for p in predictions[scenario]):
            raise ValueError(f"Nonfinite prediction in {scenario}.")

    # One factual evaluation over all 336 windows. No alternative has truth labels.
    factual_all = core.predict_one_control_scenario(model, core.transform_raw(test_raw, scalers), test_raw.u_ctrl, scalers)
    np.testing.assert_allclose(factual_all[0][ids], predictions["workload_reduction"][0], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(factual_all[1][ids], predictions["workload_reduction"][1], rtol=1e-5, atol=1e-4)
    metric_rows = []
    for group, group_ids in [("all336", np.arange(len(test_raw.y_seq))),
                             ("primary", selected.loc[selected.cohort == "primary", "sample_id"].to_numpy(int)),
                             ("supplementary", selected.loc[selected.cohort == "supplementary", "sample_id"].to_numpy(int))]:
        for v, name in enumerate(PHYS_NAMES + ["mPSI"]):
            actual = test_raw.y_seq[group_ids, :, v] if v < 3 else test_raw.y_mpsi[group_ids, 0]
            predicted = factual_all[0][group_ids, :, v] if v < 3 else factual_all[1][group_ids]
            mae, rmse, r2, me = core.mae_rmse_r2_me(actual, predicted)
            metric_rows.append({"group": group, "n_windows": len(group_ids), "variable": name,
                "evaluation": "all_15_steps" if v < 3 else "final_only", "MAE": mae, "RMSE": rmse, "R2": r2, "ME": me})
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "factual_prediction_metrics.csv", index=False, encoding="utf-8-sig")

    # A full-future encoder can change projections before a delayed onset even
    # with an identical control prefix. Report this behavior, do not suppress it.
    prefix_rows = []
    for n, row in selected.iterrows():
        k = int(row.intervention_start_minute) - 1
        if k == 0:
            continue
        for scenario in SCENARIOS[1:]:
            delta = predictions[scenario][0][n, :k] - predictions["continued_work"][0][n, :k]
            prefix_rows.append({"sample_id": int(row.sample_id), "scenario": scenario,
                "n_preintervention_steps": k, **{f"max_abs_preonset_delta_{name}": float(abs(delta[:, j]).max()) for j, name in enumerate(PHYS_NAMES)}})
    pd.DataFrame(prefix_rows).to_csv(OUTPUT_DIR / "delayed_onset_prefix_diagnostics.csv", index=False, encoding="utf-8-sig")
    pred_long = predictions_long(predictions, selected)
    final = final_outcomes(predictions, selected)
    summary = summarize(final)
    direction = directional(final)
    observed = factual_observed(test_raw, selected, predictions["workload_reduction"])
    out = OUTPUT_DIR
    pred_long.to_csv(out / "scenario_predictions_long.csv", index=False, encoding="utf-8-sig")
    final.to_csv(out / "scenario_final_outcomes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(out / "scenario_directional_consistency.csv", index=False, encoding="utf-8-sig")
    observed.to_csv(out / "factual_observed_outcomes.csv", index=False, encoding="utf-8-sig")
    plot_outputs(final, predictions, out)
    metadata = json.loads((out / "selection_metadata.json").read_text(encoding="utf-8"))
    metadata.update({"model": "PDCAN", "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
        "prediction_outputs": ["scenario_predictions_long.csv", "scenario_final_outcomes.csv", "scenario_summary.csv"],
        "outcome_rule": "Only workload_reduction is compared with observed physiological labels; alternatives are model projections.",
        "statistical_rule": "Descriptive only: primary n=6; supplementary windows overlap; participant/session identifiers are unavailable.",
        "mpsi_rule": "Native final risk output includes the formula and learned residual; it is related to predicted HR/Tcore and is not an independent validation outcome.",
        "future_context_limit": "PDCAN uses the entire future control plan in its encoder, so delayed-plan forecasts can differ before intervention onset; see prefix diagnostics.",
        "model_scalers_sha256": sha256(MODEL_DIR / "scalers_train_full.json"),
        "model_provenance": (MODEL_DIR / "training_provenance.json").relative_to(PROJECT_ROOT).as_posix(),
        "validation": {"finite_outputs": True, "factual_control_identity": True, "factual_prediction_identity": True},})
    (out / "scenario_experiment_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Scenario outputs written to", out)
    print(summary[["cohort", "scenario", "n", "final_mPSI_median", "delta_final_mPSI_vs_continued_work_median"]].to_string(index=False))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "scenarios", "all"], default="all")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
        help="Checkpoint directory containing final_model_state_dict.pt, scalers_train_full.json and hyperparameters.json")
    args = parser.parse_args()
    globals()["MODEL_DIR"] = args.model_dir.resolve()
    selected, controls = prepare()
    if args.stage == "prepare":
        return
    train_raw = core.load_csv(str(BASE_DIR / "data.csv"), name="train_full")
    test_raw = core.load_csv(str(BASE_DIR / "test.csv"), name="test")
    scalers = core.fit_scalers(train_raw)
    if args.stage in {"train", "all"}:
        train_model(train_raw, scalers)
    if args.stage in {"scenarios", "all"}:
        run_scenarios(selected, controls, train_raw, test_raw, saved_scalers())


if __name__ == "__main__":
    main()
