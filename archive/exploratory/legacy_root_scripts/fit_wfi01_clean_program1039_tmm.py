"""Fit one WFI tile with all clean Program 1039 data.

This is a diagnostic fit, not a production rerun.  It keeps every Program
1039 observation before and after the physical reset except rows whose
production correction table says that a temperature-dependent gain term would
be applied.  ACM/count corrections are retained; the temperature terms are
excluded from the input.  The resulting TMM model is evaluated on the omitted
early rows to estimate a per-LED gain correction.

Any use of this correction on Program 1063 would be an extrapolation: Program
1063 is not included in this fit or in the validation rows.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


ROOT = Path(__file__).resolve().parent
FULL_STATISTICS = ROOT / (
    "20261011/reduction_stage_comparison/inputs/wfi01/scaled_full/"
    "wfi01_tile_log_statistics_512_tile_000.csv"
)
CORRECTIONS = ROOT / (
    "20261011/corrections/"
    "wfi01_count_space_unified_temperature_corrections.csv"
)
WARM_START = ROOT / (
    "20261011/corrected_statistics_program1063_4h/production/wfi01/"
    "fits/tile_000/posterior_samples.npz"
)
FULL_WARM_STATISTICS = ROOT / (
    "20261011/reduction_stage_comparison/inputs/wfi01/scaled_full_4h/"
    "wfi01_tile_log_statistics_512_tile_000.csv"
)
OUTPUT_DIR = ROOT / (
    "20261011/reduction_stage_comparison/"
    "clean_program1039_tmm_wfi01_tile000"
)

RESET_DAY = 12.907397928240743
PROGRAM = 1039
EXTRAPOLATION_PROGRAM = 1063
ZERO_TOLERANCE = 1.0e-14
TEMPERATURE_MNEMONIC = "WFI_MCE_ADC_WTSM00"
TEMPERATURE_KEY = TEMPERATURE_MNEMONIC.lower()


def detector_name(statistics_path):
    return production.detector_name_from_statistics_path(statistics_path)


def tile_statistics_name(detector, description, tile_id):
    return (
        f"{detector.lower()}_tile_log_statistics_512_{description}_"
        f"tile_{tile_id:03d}.csv"
    )


def tile_label(detector, tile_id):
    return f"{detector} tile {tile_id:03d}"


def load_corrections(path, tile_id):
    with Path(path).open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if int(row["tile_id"]) == tile_id
        ]
    corrections = {int(row["group"]): row for row in rows}
    if not corrections:
        raise ValueError(f"No corrections found for tile {tile_id}: {path}")
    return corrections


def correction_value(row, field):
    return float(row.get(field) or 0.0)


def apply_log_correction_to_moments(row, log_correction):
    """Apply a count-space multiplicative correction to saved log moments."""
    count = float(row["n_pair_valid"])
    sums = float(row["sum_log_change"])
    row["sum_log_change"] = repr(sums + count * log_correction)
    row["sum_squared_log_change"] = repr(
        float(row["sum_squared_log_change"])
        + 2.0 * log_correction * sums
        + count * log_correction * log_correction
    )


def write_program_statistics(
    source_path,
    corrections,
    output_path,
    *,
    program,
    exclude_temperature_rows,
    tile_id=0,
):
    """Write one program's stats with ACM corrections but no temp gain term."""
    with Path(source_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    selected = []
    excluded = []
    for source_row in source_rows:
        if int(source_row["tile_id"]) != tile_id:
            continue
        if int(source_row.get("program") or -1) != program:
            continue
        group = int(source_row["group"])
        correction = corrections[group]
        pre_temperature = correction_value(correction, "log_gain_correction")
        post_temperature = correction_value(
            correction, "temperature_gain_log_correction"
        )
        if exclude_temperature_rows and (
            abs(pre_temperature) > ZERO_TOLERANCE
            or abs(post_temperature) > ZERO_TOLERANCE
        ):
            excluded.append(source_row)
            continue
        row = dict(source_row)
        total = correction_value(correction, "count_log_correction")
        acm_only = total - pre_temperature - post_temperature
        apply_log_correction_to_moments(row, acm_only)
        selected.append(row)
    if not selected:
        raise ValueError(f"No rows selected for Program {program}")
    selected.sort(key=lambda row: int(row["group"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected, excluded


def strip_sample_leading_axes(value, target_shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(target_shape):
        raise ValueError(f"Warm-start shape {value.shape} does not match {target_shape}")
    return value.copy()


def warm_start_values(initial_values, data, warm_start, warm_statistics):
    """Transfer fixed optical values and interpolate the old rate trajectory."""
    values = {name: np.asarray(value, dtype=float).copy() for name, value in initial_values.items()}
    with np.load(warm_start, allow_pickle=False) as source:
        for name in (
            "ar_tile_raw",
            "initial_thickness_log_nm",
            "reset_initial_thickness_log_nm",
            "post_reset_tile_log_offset",
            "absolute_hgcdte_n_log_scale",
            "absolute_hgcdte_k_log_scale",
        ):
            if name in source:
                values[name] = strip_sample_leading_axes(source[name], values[name].shape)
        if "incidence_angle_deg" in source and "incidence_angle_deg" in values:
            values["incidence_angle_deg"] = float(np.asarray(source["incidence_angle_deg"]).reshape(-1)[0])
        old_rates = np.asarray(source["node_instantaneous_rate_nm_day"], dtype=float)
        old_rates = old_rates.reshape(-1)
    old_data = production.load_tile_statistics(warm_statistics)
    if old_rates.size != old_data["node_days"].size:
        raise ValueError("Warm-start rate nodes do not match warm-start statistics")
    values["np_log_rate_nodes"] = np.log(
        np.interp(
            data["node_days"],
            old_data["node_days"],
            old_rates,
        )
    )[None, None, :]
    return values


def run_actual_tmm_fit(
    statistics_path,
    warm_start,
    warm_statistics,
    output_dir,
    reset_day,
    detector,
    tile_id,
):
    """Run the production TMM MAP machinery on the clean diagnostic input."""
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = production.load_tile_statistics(statistics_path)
    data["reset_day"] = float(reset_day)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    initial = production.initial_values_for_data(data, constant_rate=False)
    initial = warm_start_values(initial, data, warm_start, warm_statistics)
    rate_reference = np.asarray(initial["np_log_rate_nodes"], dtype=float)
    model = production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=0.1,
        rate_prior_log_mean=rate_reference,
        constant_rate=False,
    )
    map_values, diagnostics = production.run_least_squares_map_optimization(
        model,
        initial,
        jax,
    )
    evaluated = production._evaluate_model(model, map_values)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    observed += np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(data["days"])[None, :] >= reset_day
    )
    fit_diagnostics = {
        "observed": observed,
        "prediction": np.asarray(evaluated["_map_least_squares_prediction"]),
    }
    model_metadata = {
        "fit_incidence_angle": 1,
        "incidence_pupil_half_angle_deg": 0.0,
        "trajectory_regularization": "maxent",
        "trajectory_smoothness_weight": 0.0,
        "trajectory_entropy_weight": 0.1,
        "trajectory_entropy_reference": (
            f"existing_{detector.lower()}_tile{tile_id:03d}_map_"
            "interpolated_to_clean_nodes"
        ),
        "temperature_gain_correction_excluded": 1,
        "fit_program": PROGRAM,
        "physical_reset_day": float(reset_day),
    }
    production.write_outputs(
        output_dir,
        data,
        evaluated,
        statistics_path,
        diagnostics,
        fit_diagnostics,
        model_metadata,
    )
    return data, context, map_values, diagnostics, evaluated


def _parse_utc(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def load_temperature_series(path, mnemonic, bucket_seconds=300):
    buckets = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("mnemonic", "")).strip() != mnemonic:
                continue
            try:
                value = float(row["EUValue"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            second = _parse_utc(row["ObsTime"]).timestamp()
            bucket = int(second) // bucket_seconds
            total, count = buckets.get(bucket, (0.0, 0))
            buckets[bucket] = (total + value, count + 1)
    if not buckets:
        raise ValueError(f"No telemetry samples found for {mnemonic}")
    ordered = sorted(buckets.items())
    seconds = np.asarray([(key + 0.5) * bucket_seconds for key, _ in ordered])
    values = np.asarray([total / count for _, (total, count) in ordered])
    return seconds, values


def evaluate_clean_fit(
    source_path,
    corrections,
    fit_data,
    evaluated,
    map_values,
    reset_day,
    output_dir,
    mpa_temperature_telemetry,
    detector,
    tile_id,
):
    """Evaluate the fitted TMM on all Program 1039 rows, including exclusions."""
    all_path = output_dir / tile_statistics_name(
        detector, "all_program1039_acm_only", tile_id
    )
    all_rows, _ = write_program_statistics(
        source_path,
        corrections,
        all_path,
        program=PROGRAM,
        exclude_temperature_rows=False,
        tile_id=tile_id,
    )
    eval_data = production.load_tile_statistics(all_path)
    clean_group_to_index = {
        int(group): index for index, group in enumerate(fit_data["groups"])
    }
    eval_data["first_by_led"] = np.asarray(
        [
            int(np.flatnonzero(eval_data["groups"] == fit_data["groups"][ref])[0])
            for ref in fit_data["first_by_led"]
        ],
        dtype=int,
    )
    eval_context = production.load_response_context(eval_data["sca_names"])
    predictor = production.build_prediction_function(eval_data, eval_context)
    fit_ice = np.asarray(evaluated["ice"])[0]
    fit_rates = np.asarray(evaluated["rates"])[0]
    pre_reset = fit_data["days"] < reset_day
    median_rate = float(np.median(fit_rates[pre_reset]))
    fit_ice_by_group = {
        int(group): float(fit_ice[index])
        for index, group in enumerate(fit_data["groups"])
    }
    fit_ref_by_led = {
        led: int(fit_data["groups"][fit_data["first_by_led"][led - 1]])
        for led in range(1, 7)
    }
    fit_ref_day_by_led = {
        led: float(
            fit_data["days"][fit_data["first_by_led"][led - 1]]
        )
        for led in range(1, 7)
    }
    fit_ref_ice_by_led = {
        led: fit_ice_by_group[fit_ref_by_led[led]] for led in range(1, 7)
    }
    eval_ice = np.empty(len(eval_data["groups"]), dtype=float)
    for index, group in enumerate(eval_data["groups"]):
        group = int(group)
        if group in fit_ice_by_group:
            eval_ice[index] = fit_ice_by_group[group]
            continue
        led = int(eval_data["led"][index])
        eval_ice[index] = fit_ref_ice_by_led[led] + median_rate * (
            float(eval_data["days"][index]) - fit_ref_day_by_led[led]
        )
    ar_parameters = np.asarray(evaluated["ar_parameters"]).reshape(-1, 9)[0]
    hgcdte_parameters = np.asarray(evaluated["hgcdte"]).reshape(-1, 2)[0]
    incidence_angle = float(np.asarray(map_values["incidence_angle_deg"]).reshape(-1)[0])
    prediction = np.asarray(
        predictor(eval_ice[None, :], ar_parameters, hgcdte_parameters, incidence_angle)
    )[0].copy()
    post_reset_offset = float(
        np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1)[0]
    )
    prediction += post_reset_offset * (eval_data["days"] >= reset_day)
    observed = eval_data["sums"][0] / eval_data["counts"][0]

    telemetry_specs = {
        TEMPERATURE_MNEMONIC: Path(mpa_temperature_telemetry),
    }
    exposure_seconds = np.asarray(
        [_parse_utc(row["exposure_start_time"]).timestamp() for row in all_rows]
    )
    first_exposure = float(np.min(exposure_seconds))
    temperature_columns = {}
    for mnemonic, telemetry_path in telemetry_specs.items():
        seconds, values = load_temperature_series(telemetry_path, mnemonic)
        relative = (seconds - first_exposure) / 86400.0
        stable = (relative >= 3.0) & (relative <= 11.3)
        operating = float(np.median(values[stable]))
        temperature_columns[mnemonic] = (
            np.interp(exposure_seconds, seconds, values),
            operating,
        )

    fields = [
        "tile_id", "group", "led", "program", "repeat", "days",
        "exposure_start_time", "observed_log_change", "tmm_log_change",
        "inferred_log_gain_correction", "existing_log_gain_correction",
        "is_temperature_corrected_row", "is_fit_row",
    ]
    for mnemonic in telemetry_specs:
        key = mnemonic.lower()
        fields.extend([f"temperature_{key}", f"delta_temperature_{key}"])
    comparison_rows = []
    for index, source_row in enumerate(all_rows):
        group = int(source_row["group"])
        correction = corrections[group]
        gain = correction_value(correction, "log_gain_correction")
        post_gain = correction_value(correction, "temperature_gain_log_correction")
        row = {
            "tile_id": tile_id,
            "group": group,
            "led": int(eval_data["led"][index]),
            "program": PROGRAM,
            "repeat": int(eval_data["repeat"][index]),
            "days": float(eval_data["days"][index]),
            "exposure_start_time": source_row["exposure_start_time"],
            "observed_log_change": float(observed[index]),
            "tmm_log_change": float(prediction[index]),
            "inferred_log_gain_correction": float(prediction[index] - observed[index]),
            "existing_log_gain_correction": gain,
            "is_temperature_corrected_row": int(
                abs(gain) > ZERO_TOLERANCE or abs(post_gain) > ZERO_TOLERANCE
            ),
            "is_fit_row": int(group in clean_group_to_index),
        }
        for mnemonic, (temperature, operating) in temperature_columns.items():
            key = mnemonic.lower()
            row[f"temperature_{key}"] = float(temperature[index])
            row[f"delta_temperature_{key}"] = float(temperature[index] - operating)
        comparison_rows.append(row)
    comparison_path = output_dir / "program1039_tmm_gain_comparison.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_rows = []
    for led in range(1, 7):
        selected = [
            row for row in comparison_rows
            if row["led"] == led and row["is_temperature_corrected_row"]
        ]
        inferred = np.asarray([row["inferred_log_gain_correction"] for row in selected])
        existing = np.asarray([row["existing_log_gain_correction"] for row in selected])
        summary_rows.append(
            {
                "led": led,
                "excluded_rows": len(selected),
                "inferred_mean_log_gain": float(np.mean(inferred)),
                "existing_mean_log_gain": float(np.mean(existing)),
                "inferred_minus_existing_mean_log": float(np.mean(inferred - existing)),
                "inferred_minus_existing_rmse_log": float(
                    np.sqrt(np.mean((inferred - existing) ** 2))
                ),
            }
        )
    with (output_dir / "program1039_tmm_gain_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_gain_vs_temperature(output_dir, comparison_rows, detector, tile_id)
    return comparison_rows, summary_rows, median_rate


def evaluate_program1063_extrapolation(
    source_path,
    corrections,
    fit_data,
    evaluated,
    map_values,
    reset_day,
    output_dir,
    mpa_temperature_telemetry,
    detector,
    tile_id,
):
    """Evaluate the Program 1039 fit on Program 1063 without refitting it."""
    statistics_path = output_dir / tile_statistics_name(
        detector, "program1063_acm_only", tile_id
    )
    rows, _ = write_program_statistics(
        source_path,
        corrections,
        statistics_path,
        program=EXTRAPOLATION_PROGRAM,
        exclude_temperature_rows=False,
        tile_id=tile_id,
    )
    eval_data = production.load_tile_statistics(statistics_path)
    eval_data["reset_day"] = float(reset_day)
    eval_data["node_days"] = np.asarray(fit_data["node_days"], dtype=float)

    # The 1063 sufficient statistics already use the P1039 baseline.  Include
    # the six clean P1039 rows that were the fitted predictor's references so
    # its internal response reference is not silently replaced by the first
    # row of the 1063-only subset.  These rows retain their P1039-relative
    # observed values; using the absolute group-zero rows here would force the
    # model reference to zero and erase the correction being estimated.
    reference_statistics_path = output_dir / (
        tile_statistics_name(
            detector, "program1039_reference_program1063_acm_only", tile_id
        )
    )
    program1039_rows, _ = write_program_statistics(
        source_path,
        corrections,
        output_dir / tile_statistics_name(
            detector, "program1039_acm_only", tile_id
        ),
        program=PROGRAM,
        exclude_temperature_rows=False,
        tile_id=tile_id,
    )
    program1039_by_group = {
        int(row["group"]): row for row in program1039_rows
    }
    reference_groups = [
        int(group) for group in fit_data["groups"][fit_data["first_by_led"]]
    ]
    reference_rows = [program1039_by_group[group] for group in reference_groups]
    combined_rows = reference_rows + rows
    with reference_statistics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    predictor_data = production.load_tile_statistics(reference_statistics_path)
    predictor_data["reset_day"] = float(reset_day)
    predictor_data["node_days"] = np.asarray(fit_data["node_days"], dtype=float)
    eval_context = production.load_response_context(predictor_data["sca_names"])
    predictor = production.build_prediction_function(predictor_data, eval_context)

    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    extrapolated_ice, _, _, _ = production._trajectory_state(
        predictor_data,
        map_values["np_log_rate_nodes"],
        map_values["initial_thickness_log_nm"],
        map_values["reset_initial_thickness_log_nm"],
        jax,
        jnp,
    )
    extrapolated_ice = np.asarray(extrapolated_ice)[0]
    ar_parameters = np.asarray(evaluated["ar_parameters"]).reshape(-1, 9)[0]
    hgcdte_parameters = np.asarray(evaluated["hgcdte"]).reshape(-1, 2)[0]
    incidence_angle = float(
        np.asarray(map_values["incidence_angle_deg"]).reshape(-1)[0]
    )
    prediction_all = np.asarray(
        predictor(
            extrapolated_ice[None, :],
            ar_parameters,
            hgcdte_parameters,
            incidence_angle,
        )
    )[0]
    prediction_indices = np.flatnonzero(
        predictor_data["program"] == EXTRAPOLATION_PROGRAM
    )
    if not np.array_equal(
        predictor_data["groups"][prediction_indices], eval_data["groups"]
    ):
        raise ValueError("Combined predictor rows do not match Program 1063 rows")
    prediction = prediction_all[prediction_indices].copy()
    post_reset_offset = float(
        np.asarray(map_values["post_reset_tile_log_offset"]).reshape(-1)[0]
    )
    observed = eval_data["sums"][0] / eval_data["counts"][0]
    comparison_observed = observed + post_reset_offset * (
        eval_data["days"] >= reset_day
    )

    program1039_rows = []
    with Path(source_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["tile_id"]) == tile_id
                and int(row.get("program") or -1) == PROGRAM
            ):
                program1039_rows.append(row)
    reference_exposure = min(
        _parse_utc(row["exposure_start_time"]) for row in program1039_rows
    ).timestamp()
    exposure_seconds = np.asarray(
        [_parse_utc(row["exposure_start_time"]).timestamp() for row in rows]
    )
    seconds, values = load_temperature_series(
        Path(mpa_temperature_telemetry), TEMPERATURE_MNEMONIC
    )
    relative = (seconds - reference_exposure) / 86400.0
    stable = (relative >= 3.0) & (relative <= 11.3)
    operating = float(np.median(values[stable]))
    temperature = np.interp(exposure_seconds, seconds, values)

    fit_max_day = float(np.max(fit_data["days"]))
    fields = [
        "tile_id", "group", "led", "program", "repeat", "days",
        "exposure_start_time", "observed_log_change", "tmm_log_change",
        "inferred_log_gain_correction", "existing_log_gain_correction",
        "existing_temperature_gain_correction", "existing_total_gain_correction",
        "is_temperature_corrected_row", "is_beyond_program1039_fit_time",
    ]
    key = TEMPERATURE_KEY
    fields.extend([f"temperature_{key}", f"delta_temperature_{key}"])
    comparison_rows = []
    for index, source_row in enumerate(rows):
        group = int(source_row["group"])
        correction = corrections[group]
        pre_gain = correction_value(correction, "log_gain_correction")
        temperature_gain = correction_value(
            correction, "temperature_gain_log_correction"
        )
        total_gain = pre_gain + temperature_gain
        comparison_rows.append(
            {
                "tile_id": tile_id,
                "group": group,
                "led": int(eval_data["led"][index]),
                "program": EXTRAPOLATION_PROGRAM,
                "repeat": int(eval_data["repeat"][index]),
                "days": float(eval_data["days"][index]),
                "exposure_start_time": source_row["exposure_start_time"],
                "observed_log_change": float(observed[index]),
                "tmm_log_change": float(prediction[index]),
                "inferred_log_gain_correction": float(
                    prediction[index] - comparison_observed[index]
                ),
                "existing_log_gain_correction": pre_gain,
                "existing_temperature_gain_correction": temperature_gain,
                "existing_total_gain_correction": total_gain,
                "is_temperature_corrected_row": int(
                    abs(total_gain) > ZERO_TOLERANCE
                ),
                "is_beyond_program1039_fit_time": int(
                    float(eval_data["days"][index]) > fit_max_day
                ),
                f"temperature_{key}": float(temperature[index]),
                f"delta_temperature_{key}": float(temperature[index] - operating),
            }
        )

    comparison_path = output_dir / "program1063_tmm_extrapolation_comparison.csv"
    with comparison_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_rows = []
    for led in range(1, 7):
        selected = [
            row for row in comparison_rows
            if row["led"] == led and row["is_temperature_corrected_row"]
        ]
        inferred = np.asarray(
            [row["inferred_log_gain_correction"] for row in selected],
            dtype=float,
        )
        existing = np.asarray(
            [row["existing_total_gain_correction"] for row in selected],
            dtype=float,
        )
        difference = inferred - existing
        summary_rows.append(
            {
                "led": led,
                "temperature_corrected_rows": len(selected),
                "beyond_program1039_fit_time_rows": sum(
                    row["is_beyond_program1039_fit_time"] for row in selected
                ),
                "inferred_mean_log_gain": float(np.mean(inferred)),
                "existing_mean_log_gain": float(np.mean(existing)),
                "inferred_minus_existing_mean_log": float(np.mean(difference)),
                "inferred_minus_existing_rmse_log": float(
                    np.sqrt(np.mean(difference**2))
                ),
            }
        )
    summary_path = output_dir / "program1063_tmm_extrapolation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_program1063_extrapolation(output_dir, comparison_rows, detector, tile_id)

    selected = [row for row in comparison_rows if row["is_temperature_corrected_row"]]
    inferred = np.asarray(
        [row["inferred_log_gain_correction"] for row in selected], dtype=float
    )
    existing = np.asarray(
        [row["existing_total_gain_correction"] for row in selected], dtype=float
    )
    difference = inferred - existing
    assessment = {
        "status": "extrapolation_holdout_comparison_not_validation",
        "comparison_rows": len(comparison_rows),
        "rows_with_existing_temperature_correction": len(selected),
        "rows_beyond_program1039_fit_time": sum(
            row["is_beyond_program1039_fit_time"] for row in comparison_rows
        ),
        "program1039_fit_day_max": fit_max_day,
        "program1063_day_min": float(np.min(eval_data["days"])),
        "program1063_day_max": float(np.max(eval_data["days"])),
        "inferred_vs_existing_bias_mlog": float(1000.0 * difference.mean()),
        "inferred_vs_existing_rmse_mlog": float(
            1000.0 * np.sqrt(np.mean(difference**2))
        ),
        "inferred_vs_existing_correlation": float(
            np.corrcoef(inferred, existing)[0, 1]
        ),
    }
    return comparison_rows, summary_rows, assessment


def plot_gain_vs_temperature(output_dir, rows, detector, tile_id):
    for mnemonic in (TEMPERATURE_MNEMONIC,):
        key = mnemonic.lower()
        figure, axes = plt.subplots(3, 2, figsize=(13, 13), sharey=True, constrained_layout=True)
        for axis, led in zip(axes.flat, range(1, 7)):
            selected = [
                row for row in rows
                if row["led"] == led and row["is_temperature_corrected_row"]
            ]
            selected.sort(key=lambda row: row[f"delta_temperature_{key}"])
            x = np.asarray([row[f"delta_temperature_{key}"] for row in selected])
            inferred = 1000.0 * np.asarray(
                [row["inferred_log_gain_correction"] for row in selected]
            )
            existing = 1000.0 * np.asarray(
                [row["existing_log_gain_correction"] for row in selected]
            )
            axis.plot(x, inferred, "o-", color="tab:red", label="clean Program 1039 TMM")
            axis.plot(x, existing, "s--", color="0.2", label="production")
            axis.set_title(f"LED {led}")
            axis.grid(alpha=0.25)
            axis.axhline(0.0, color="0.45", linewidth=0.8)
        for axis in axes[-1]:
            axis.set_xlabel(r"$\Delta T = T - T_{op}$ (K)")
        for axis in axes[:, 0]:
            axis.set_ylabel("gain correction (millilog)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle(
            f"{tile_label(detector, tile_id)}: all clean Program 1039 TMM extrapolation vs {mnemonic}\n"
            "Rows with a production temperature-gain term are excluded from the fit"
        )
        figure.savefig(output_dir / f"gain_vs_delta_t_{key}.png", dpi=180)
        plt.close(figure)


def plot_program1063_extrapolation(output_dir, rows, detector, tile_id):
    """Plot 1063 inferred and existing corrections against MPA temperature."""
    figure, axes = plt.subplots(
        3, 2, figsize=(13, 13), sharey=True, constrained_layout=True
    )
    for axis, led in zip(axes.flat, range(1, 7)):
        selected = [row for row in rows if row["led"] == led]
        temperature_field = f"temperature_{TEMPERATURE_KEY}"
        selected.sort(key=lambda row: float(row[temperature_field]))
        x = np.asarray([float(row[temperature_field]) for row in selected])
        inferred = 1000.0 * np.asarray(
            [row["inferred_log_gain_correction"] for row in selected]
        )
        axis.scatter(x, inferred, s=12, color="tab:red", label="Program 1039 TMM")
        corrected = [
            row for row in selected if row["is_temperature_corrected_row"]
        ]
        if corrected:
            corrected_x = np.asarray(
                [float(row[temperature_field]) for row in corrected]
            )
            existing = 1000.0 * np.asarray(
                [row["existing_total_gain_correction"] for row in corrected]
            )
            axis.scatter(
                corrected_x,
                existing,
                s=18,
                marker="s",
                color="0.2",
                label="existing Program 1063 correction",
            )
        beyond = [
            row for row in selected if row["is_beyond_program1039_fit_time"]
        ]
        if beyond:
            beyond_x = np.asarray(
                [float(row[temperature_field]) for row in beyond]
            )
            beyond_inferred = 1000.0 * np.asarray(
                [row["inferred_log_gain_correction"] for row in beyond]
            )
            axis.scatter(
                beyond_x,
                beyond_inferred,
                s=30,
                facecolors="none",
                edgecolors="tab:orange",
                label="> Program 1039 time range",
            )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        axis.axhline(0.0, color="0.45", linewidth=0.8)
    for axis in axes[-1]:
        axis.set_xlabel("MPA interface temperature (K)")
    for axis in axes[:, 0]:
        axis.set_ylabel("gain correction (millilog)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{tile_label(detector, tile_id)}: Program 1063 correction comparison vs MPA temperature\n"
        "Temperature: WFI_MCE_ADC_WTSM00 (CPR strap at MPA interface); orange markers are beyond the 1039 time range"
    )
    figure.savefig(
        output_dir
        / "gain_vs_temperature_wfi_mce_adc_wtsm00_program1063_extrapolation.png",
        dpi=180,
    )
    plt.close(figure)


def plot_count_space_diagnostics(
    output_dir, program1039_rows, program1063_rows, fit_data, detector, tile_id
):
    """Plot count-space observations, TMM predictions, and correction residuals."""
    fit_rows = [row for row in program1039_rows if row["is_fit_row"]]
    holdout_rows = [
        row for row in program1039_rows
        if row["is_temperature_corrected_row"] and not row["is_fit_row"]
    ]
    fit_max_day = float(np.max(fit_data["days"]))
    figure, axes = plt.subplots(
        6,
        2,
        figsize=(17, 25),
        sharex=True,
        constrained_layout=True,
    )

    def count_space(log_change):
        return 100.0 * np.expm1(float(log_change))

    def plot_dataset(axis, residual_axis, rows, observed_color, model_color, label):
        if not rows:
            return
        ordered = sorted(rows, key=lambda row: float(row["days"]))
        days = np.asarray([float(row["days"]) for row in ordered])
        observed = np.asarray(
            [count_space(row["observed_log_change"]) for row in ordered]
        )
        model = np.asarray(
            [count_space(row["tmm_log_change"]) for row in ordered]
        )
        residual = np.asarray(
            [
                100.0
                * np.expm1(
                    float(row["tmm_log_change"])
                    - float(row["observed_log_change"])
                )
                for row in ordered
            ]
        )
        axis.scatter(days, observed, s=9, color=observed_color, label=f"{label} data")
        if model_color is None:
            return
        axis.plot(days, model, color=model_color, linewidth=1.0, label=f"{label} TMM")
        residual_axis.scatter(days, residual, s=9, color=model_color, label=f"{label} residual")

    for axis_row, led in enumerate(range(1, 7)):
        model_axis, residual_axis = axes[axis_row]
        fit_led = [row for row in fit_rows if row["led"] == led]
        holdout_led = [row for row in holdout_rows if row["led"] == led]
        program1063_led = [row for row in program1063_rows if row["led"] == led]
        plot_dataset(
            model_axis,
            residual_axis,
            fit_led,
            "tab:blue",
            "tab:blue",
            "1039 fit",
        )
        plot_dataset(
            model_axis,
            residual_axis,
            holdout_led,
            "tab:orange",
            "tab:red",
            "1039 holdout",
        )
        plot_dataset(
            model_axis,
            residual_axis,
            program1063_led,
            "tab:green",
            None,
            "1063 corrected data (native)",
        )
        model_axis.set_title(f"LED {led}: count-space relative change")
        residual_axis.set_title(f"LED {led}: TMM residual (Program 1039 only)")
        model_axis.grid(alpha=0.25)
        residual_axis.grid(alpha=0.25)
        residual_axis.axhline(0.0, color="0.45", linewidth=0.8)
        model_axis.axvline(fit_max_day, color="tab:orange", linestyle=":", linewidth=0.9)
        residual_axis.axvline(fit_max_day, color="tab:orange", linestyle=":", linewidth=0.9)

    for model_axis, residual_axis in axes[-1:]:
        model_axis.set_xlabel("mission day")
        residual_axis.set_xlabel("mission day")
    for model_axis, residual_axis in axes:
        model_axis.set_ylabel("relative count change (%)")
        residual_axis.set_ylabel("relative residual (%)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        f"{tile_label(detector, tile_id)}: corrected count-space data and Program 1039 TMM fit\n"
        "Data: count/ACM corrections applied; log and temperature-dependent gain terms removed; "
        "Program 1063 is plotted as native corrected data only; "
        "orange vertical line = last Program 1039 fit day",
        fontsize=14,
    )
    figure.savefig(
        output_dir / "tmm_count_space_fit_extrapolation_residuals.png",
        dpi=180,
    )
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=FULL_STATISTICS)
    parser.add_argument("--corrections", type=Path, default=CORRECTIONS)
    parser.add_argument("--warm-start", type=Path, default=WARM_START)
    parser.add_argument("--warm-statistics", type=Path, default=FULL_WARM_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mpa-temperature-telemetry",
        type=Path,
        default=OUTPUT_DIR / "telemetry/wfi_mce_adc_wtsm00.csv",
    )
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = detector_name(args.statistics)
    tile_id = args.tile_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_data = production.load_tile_statistics(args.statistics)
    source_tile_id = int(source_data["tile_ids"][0])
    if source_tile_id != tile_id:
        raise ValueError(
            f"--tile-id {tile_id} does not match statistics tile {source_tile_id}"
        )
    corrections = load_corrections(args.corrections, tile_id)
    clean_path = args.output_dir / tile_statistics_name(
        detector, "clean_program1039_acm_only", tile_id
    )
    clean_rows, excluded_rows = write_program_statistics(
        args.statistics,
        corrections,
        clean_path,
        program=PROGRAM,
        exclude_temperature_rows=True,
        tile_id=tile_id,
    )
    print(
        f"Program 1039 rows: selected={len(clean_rows)}, "
        f"excluded temperature-dependent={len(excluded_rows)}",
        flush=True,
    )
    fit_data, context, map_values, diagnostics, evaluated = run_actual_tmm_fit(
        clean_path,
        args.warm_start,
        args.warm_statistics,
        args.output_dir / "tmm_fit",
        args.reset_day,
        detector,
        tile_id,
    )
    comparison_rows, summary_rows, median_rate = evaluate_clean_fit(
        args.statistics,
        corrections,
        fit_data,
        evaluated,
        map_values,
        args.reset_day,
        args.output_dir,
        args.mpa_temperature_telemetry,
        detector,
        tile_id,
    )
    (
        program1063_rows,
        program1063_summary,
        program1063_assessment,
    ) = evaluate_program1063_extrapolation(
        args.statistics,
        corrections,
        fit_data,
        evaluated,
        map_values,
        args.reset_day,
        args.output_dir,
        args.mpa_temperature_telemetry,
        detector,
        tile_id,
    )
    plot_count_space_diagnostics(
        args.output_dir,
        comparison_rows,
        program1063_rows,
        fit_data,
        detector,
        tile_id,
    )
    metadata = {
        "detector": detector,
        "tile_id": tile_id,
        "source_statistics": str(args.statistics),
        "clean_statistics": str(clean_path),
        "corrections": str(args.corrections),
        "warm_start": str(args.warm_start),
        "program": PROGRAM,
        "selected_rows": len(clean_rows),
        "excluded_temperature_rows": len(excluded_rows),
        "pre_reset_selected_rows": int(np.count_nonzero(np.asarray([float(row["days"]) for row in clean_rows]) < args.reset_day)),
        "post_reset_selected_rows": int(np.count_nonzero(np.asarray([float(row["days"]) for row in clean_rows]) >= args.reset_day)),
        "physical_reset_day": args.reset_day,
        "median_fitted_pre_reset_rate_nm_day": median_rate,
        "map_objective": diagnostics["objective"],
        "map_converged": diagnostics["converged"],
        "map_gradient_converged": diagnostics["gradient_converged"],
        "map_message": diagnostics["message"],
        "temperature_terms_excluded": [
            "log_gain_correction",
            "temperature_gain_log_correction",
        ],
        "temperature_mnemonic": TEMPERATURE_MNEMONIC,
        "temperature_telemetry": str(args.mpa_temperature_telemetry),
        "temperature_units": "K",
        "temperature_definition": "roman-mast-tools tlm_groups.yaml temp_cpr: CPR strap at MPA interface",
        "program_1063_comparison_csv": str(
            args.output_dir / "program1063_tmm_extrapolation_comparison.csv"
        ),
        "program_1063_summary_csv": str(
            args.output_dir / "program1063_tmm_extrapolation_summary.csv"
        ),
        "program_1063_plot": str(
            args.output_dir
            / "gain_vs_temperature_wfi_mce_adc_wtsm00_program1063_extrapolation.png"
        ),
        "count_space_diagnostic_plot": str(
            args.output_dir / "tmm_count_space_fit_extrapolation_residuals.png"
        ),
        "comparison_data_correction": (
            "count_log_correction - log_gain_correction - "
            "temperature_gain_log_correction"
        ),
        "program_1063_comparison": program1063_assessment,
    }
    excluded_comparison = [
        row for row in comparison_rows if row["is_temperature_corrected_row"]
    ]
    inferred = np.asarray(
        [row["inferred_log_gain_correction"] for row in excluded_comparison],
        dtype=float,
    )
    existing = np.asarray(
        [row["existing_log_gain_correction"] for row in excluded_comparison],
        dtype=float,
    )
    residual_after_existing = inferred - existing
    metadata["correction_assessment"] = {
        "status": "useful_first_order_correction_not_independently_validated",
        "comparison_rows": len(excluded_comparison),
        "inferred_vs_existing_correlation": float(
            np.corrcoef(inferred, existing)[0, 1]
        ),
        "inferred_vs_existing_bias_mlog": float(
            1000.0 * residual_after_existing.mean()
        ),
        "inferred_vs_existing_rmse_mlog": float(
            1000.0 * np.sqrt(np.mean(residual_after_existing**2))
        ),
        "uncorrected_rmse_mlog": float(1000.0 * np.sqrt(np.mean(inferred**2))),
        "existing_corrected_rmse_mlog": float(
            1000.0 * np.sqrt(np.mean(residual_after_existing**2))
        ),
    }
    metadata["program_1063_application"] = "extrapolation"
    metadata["program_1063_validation"] = (
        "not independent validation; this is a no-refit extrapolation comparison"
    )
    with (args.output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(
        f"TMM fit objective={diagnostics['objective']:.10g}, "
        f"converged={diagnostics['converged']}, "
        f"strict_gradient={diagnostics['gradient_converged']}",
        flush=True,
    )
    for row in summary_rows:
        print(
            f"LED {row['led']}: inferred-existing mean="
            f"{1000.0 * row['inferred_minus_existing_mean_log']:+.4f} mlog, "
            f"RMSE={1000.0 * row['inferred_minus_existing_rmse_log']:.4f} mlog",
            flush=True,
        )
    print(
        f"Program 1063 extrapolation rows={len(program1063_rows)}, "
        f"temperature-corrected={sum(row['is_temperature_corrected_row'] for row in program1063_rows)}, "
        f"beyond_1039_time={program1063_assessment['rows_beyond_program1039_fit_time']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
