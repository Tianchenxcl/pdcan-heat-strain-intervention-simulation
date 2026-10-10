"""Build auditable scenario inputs without training or using future outcomes.

Run this file directly for data preparation, or import prepare() from the runner.
Sample IDs are zero-based original CSV data-row indices; Excel row = ID + 2.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "scenario"
CONTROL_NAMES = ["humidity", "air_temperature", "wind_speed", "radiant_environment", "oxygen_consumption"]
SCENARIOS = ["continued_work", "rest", "rest_environment", "workload_reduction"]
REST_LEVEL = 3.5
EPS = 0.1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arrays(path, require_type=False):
    df = pd.read_csv(path)
    if df.shape[1] < (341 if require_type else 340):
        raise ValueError(f"Unexpected column count in {path}: {df.shape[1]}")
    a = df.iloc[:, :340].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Missing/nonfinite data: stop rather than silently change row IDs.")
    labels = df.iloc[:, 340].to_numpy(int) if require_type else None
    return df, a, labels, a[:, :5], a[:, 5:215].reshape(-1, 30, 7), a[:, 215:335].reshape(-1, 15, 8)


def anchor(static, hist):
    # Only static and past/current inputs: no future physiology or future controls.
    return np.concatenate([static, hist[:, -5:, :5].mean(1), hist[:, -5:, 5:7].mean(1), hist[:, -1, :5]], axis=1)


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def prepare(output_dir=OUTPUT_DIR, train_path=BASE_DIR / "data.csv", test_path=BASE_DIR / "test.csv"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, arr, labels, static, hist, future = read_arrays(test_path, require_type=True)
    _, _, _, train_static, train_hist, train_future = read_arrays(train_path)
    ctrl = future[:, :, :5]
    current = hist[:, -1, 4]
    w = ctrl[:, :, 4]
    conditions = {
        "is_type3": labels == 3,
        "currently_working": current > REST_LEVEL + EPS,
        "future_remains_work": (w > REST_LEVEL + EPS).all(1),
        "future_nonincreasing": (np.diff(w, axis=1) <= EPS).all(1),
        "future_not_above_current": (w <= current[:, None] + EPS).all(1),
        "final_workload_reduced": w[:, -1] < current - EPS,
    }
    eligible = np.logical_and.reduce(list(conditions.values()))
    stable = np.ptp(hist[:, -5:, 4], axis=1) <= EPS
    immediate = w[:, 0] < current - EPS
    primary = eligible & stable & immediate
    ids = np.flatnonzero(eligible)
    if len(ids) == 0:
        raise ValueError("No eligible workload-reduction windows.")

    # Donor matching is scaled with training data only; model scalers are separate.
    train_a = anchor(train_static, train_hist)
    scale = train_a.std(0)
    scale[scale < 1e-6] = 1.0
    az = (anchor(static, hist) - train_a.mean(0)) / scale
    full_rest = np.isclose(w, REST_LEVEL, atol=1e-4, rtol=0).all(1)
    rest_pool = np.flatnonzero((labels == 1) & full_rest)
    env_pool = np.flatnonzero((labels == 2) & full_rest)

    def nearest(i, candidates):
        if not len(candidates):
            raise ValueError(f"No eligible rest/environment donor for sample {i}.")
        distances = np.linalg.norm(az[candidates] - az[i], axis=1)
        # np.argmin resolves ties in original row order, deterministically.
        j = int(np.argmin(distances))
        return int(candidates[j]), float(distances[j])

    controls = np.empty((len(ids), 4, 15, 5), np.float32)
    selected, matches = [], []
    for n, i in enumerate(ids):
        onset = int(np.flatnonzero(w[i] < current[i] - EPS)[0])
        cohort = "primary" if primary[i] else "supplementary"
        # The prefix must truly be continued work, not a changing-intensity ramp.
        np.testing.assert_allclose(w[i, :onset], current[i], atol=EPS, rtol=0)
        rest_id, rest_distance = nearest(i, rest_pool)
        duration = 15 - onset
        # Preserve a real joint environment sequence; do not mix its 4 channels.
        # Cooler air and non-higher radiation are required, not presumed by type.
        env_ok = (
            (ctrl[env_pool, :duration, 1] < ctrl[i, onset:, 1] - EPS).all(1)
            & (ctrl[env_pool, :duration, 3] <= ctrl[i, onset:, 3] + EPS).all(1)
        )
        env_id, env_distance = nearest(i, env_pool[env_ok])
        continued = ctrl[i].copy()
        continued[:, 4] = current[i]
        rest = continued.copy()
        rest[onset:, 4] = ctrl[rest_id, :duration, 4]
        restenv = rest.copy()
        restenv[onset:, :4] = ctrl[env_id, :duration, :4]
        factual = ctrl[i].copy()
        controls[n] = np.stack([continued, rest, restenv, factual])
        # Mechanical invariants: same history, same prefix, componentwise changes.
        np.testing.assert_array_equal(controls[n, 3], ctrl[i])
        np.testing.assert_array_equal(rest[:, :4], continued[:, :4])
        np.testing.assert_array_equal(restenv[:, 4], rest[:, 4])
        for s in range(4):
            np.testing.assert_allclose(controls[n, s, :onset], ctrl[i, :onset], atol=EPS, rtol=0)
        selected.append({
            "sample_id": int(i), "sample_number_one_based": int(i + 1), "source_row_number": int(i + 2),
            "type": int(labels[i]), "cohort": cohort, "intervention_start_minute": onset + 1,
            "current_oxygen_consumption": float(current[i]), "final_reduced_oxygen_consumption": float(w[i, -1]),
            "reduction_fraction": float(1 - w[i, -1] / current[i]),
            "past5_workload_range": float(np.ptp(hist[i, -5:, 4])),
            "rest_donor_sample_id": rest_id, "environment_donor_sample_id": env_id,
            "target_air_temperature_mean": float(ctrl[i, onset:, 1].mean()),
            "rest_environment_air_temperature_mean": float(restenv[onset:, 1].mean()),
        })
        for scenario, donor, distance, pool_n in [("rest", rest_id, rest_distance, len(rest_pool)), ("rest_environment", env_id, env_distance, int(env_ok.sum()))]:
            matches.append({"sample_id": int(i), "cohort": cohort, "scenario": scenario,
                "donor_sample_id": donor, "donor_source_row_number": donor + 2,
                "donor_type": int(labels[donor]), "matching_distance": distance,
                "eligible_donors": pool_n, "donor_sequence_start_minute": 1,
                "substituted_target_start_minute": onset + 1})

    selected = pd.DataFrame(selected)
    audit = pd.DataFrame({"sample_id": np.arange(len(labels)), "source_row_number": np.arange(len(labels)) + 2,
        "type": labels, **conditions, "stable_past5": stable, "immediate_drop": immediate,
        "eligible": eligible, "primary": primary})
    long, observations = [], []
    for n, i in enumerate(ids):
        for minute in range(15):
            # Actual future columns: Tcore=5, Tskin=6, HR=7. mPSI is final only.
            observations.append({"sample_id": int(i), "cohort": selected.iloc[n]["cohort"], "minute": minute + 1,
                "observed_HR": float(future[i, minute, 7]), "observed_Tskin": float(future[i, minute, 6]),
                "observed_Tcore": float(future[i, minute, 5]),
                "observed_mPSI_final_only": float(arr[i, 339]) if minute == 14 else None})
            for j, scenario in enumerate(SCENARIOS):
                long.append({"sample_id": int(i), "source_row_number": int(i + 2), "cohort": selected.iloc[n]["cohort"],
                    "scenario": scenario, "minute": minute + 1,
                    **{name: float(controls[n, j, minute, c]) for c, name in enumerate(CONTROL_NAMES)}})
    csv(selected, out / "selected_samples.csv")
    csv(selected[selected.cohort == "primary"], out / "primary_samples.csv")
    csv(audit, out / "selection_audit_all336.csv")
    csv(pd.DataFrame(matches), out / "scenario_donor_matches.csv")
    csv(pd.DataFrame(long), out / "scenario_controls_long.csv")
    csv(pd.DataFrame(long).query("cohort == 'primary'"), out / "primary_scenario_controls.csv")
    csv(pd.DataFrame(observations), out / "selected_original_observations.csv")
    original = df.iloc[ids].copy()
    original.insert(0, "original_sample_id", ids)
    csv(original, out / "selected_original_rows.csv")
    # No future physiological labels in this inference-input bundle.
    meta = np.concatenate([arr[ids, 335:339], static[ids, 1:2]], axis=1)
    np.savez_compressed(out / "scenario_inputs.npz", sample_ids=ids, controls=controls,
        scenario_names=np.asarray(SCENARIOS), x_static=static[ids], x_hist=hist[ids],
        init_phys=hist[ids, -1][:, [6, 5]], meta=meta)

    support = []
    training_ctrl = train_future[:, :, :5].reshape(-1, 5)
    for j, scenario in enumerate(SCENARIOS):
        for c, name in enumerate(CONTROL_NAMES):
            z = controls[:, j, :, c]
            lo, hi = float(training_ctrl[:, c].min()), float(training_ctrl[:, c].max())
            support.append({"scenario": scenario, "control": name, "scenario_min": float(z.min()),
                "scenario_max": float(z.max()), "train_min": lo, "train_max": hi,
                "fraction_outside_train_marginal_range": float(((z < lo - 1e-5) | (z > hi + 1e-5)).mean())})
    csv(pd.DataFrame(support), out / "control_support_diagnostics.csv")
    metadata = {
        "protocol_version": "selected-componentwise-v1", "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
        "n_test": len(labels), "n_selected": len(ids), "n_primary": int(primary.sum()),
        "n_supplementary": int((eligible & ~primary).sum()), "primary_ids": np.flatnonzero(primary).tolist(),
        "scenario_order": SCENARIOS, "control_order": CONTROL_NAMES, "rest_level": REST_LEVEL,
        "rest_pool_n": len(rest_pool), "rest_environment_pool_n": len(env_pool),
        "screening": "Type 3, current and future workload above rest, non-increasing future workload, no future increase over current, final drop > 0.1.",
        "primary": "Eligible + past 5-minute workload range <= 0.1 + immediate drop.",
        "matching": "Euclidean distance in static, recent and current historical input features; standardized with training inputs only; no future physiology or predictions.",
        "donor_use": "Type-1 oxygen sequence only for rest; Type-2 complete joint environmental sequence only for rest_environment; no donor outcomes are used.",
        "reference": "continued_work is a synthetic model reference; workload_reduction preserves factual inputs and alone has observed outcome labels.",
        "interpretation": "Model scenario projections, not causal effects or counterfactual accuracy validation.",
        "dependence": "Overlapping windows exist. Participant/session IDs are unavailable in these CSVs; no independence, confidence interval or significance claim is made.",
        "support_limit": "Marginal range checks do not demonstrate joint or history-conditioned support.",
        "prospective_limit": "Type and future measured workload are used to select this retrospective demonstration; this is not a prospective treatment-selection algorithm.",
    }
    (out / "selection_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(ids)}/{len(labels)} windows: {primary.sum()} primary, {(eligible & ~primary).sum()} supplementary.")
    print("Primary sample IDs:", np.flatnonzero(primary).tolist())
    return selected, controls


if __name__ == "__main__":
    prepare()
