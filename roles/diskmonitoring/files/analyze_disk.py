#!/usr/bin/env python3
 
import argparse
import glob
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
 
import pandas as pd
 
 
LOGGER = logging.getLogger("disk_monitor")
 
REQUIRED_COLUMNS = {
    "timestamp",
    "server",
    "drive",
    "total_gb",
    "free_gb",
    "used_percent",
}
 
 
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
 
 
def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--baseline-points", type=int, default=3)
    parser.add_argument("--sudden-jump", type=float, default=10.0)
    parser.add_argument("--spike-multiplier", type=float, default=1.5)
 
    return parser.parse_args()
 
 
def load_csv_files(input_dir):
    files = sorted(
        glob.glob(
            os.path.join(input_dir, "*.csv")
        )
    )
 
    if not files:
        LOGGER.warning(
            "No CSV files found in %s",
            input_dir,
        )
        return pd.DataFrame()
 
    frames = []
 
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
 
        except (
            OSError,
            ValueError,
            pd.errors.ParserError,
        ):
            LOGGER.exception(
                "Unable to read %s",
                file_path,
            )
            continue
 
        missing_columns = REQUIRED_COLUMNS.difference(
            df.columns
        )
 
        if missing_columns:
            LOGGER.warning(
                "Skipping %s; missing columns: %s",
                file_path,
                sorted(missing_columns),
            )
            continue
 
        frames.append(df)
 
    if not frames:
        return pd.DataFrame()
 
    return pd.concat(
        frames,
        ignore_index=True,
    )
 
 
def clean_data(df):
    if df.empty:
        return df
 
    df = df.copy()
 
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )
 
    for column in (
        "total_gb",
        "free_gb",
        "used_percent",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )
 
    df["server"] = (
        df["server"]
        .astype("string")
        .str.strip()
    )
 
    df["drive"] = (
        df["drive"]
        .astype("string")
        .str.strip()
    )
 
    df = df.dropna(
        subset=[
            "timestamp",
            "server",
            "drive",
            "used_percent",
        ]
    )
 
    df = df[
        df["server"].ne("")
        & df["drive"].ne("")
        & df["used_percent"].between(0, 100)
    ]
 
    df = df.sort_values(
        [
            "server",
            "drive",
            "timestamp",
        ]
    )
 
    return df.drop_duplicates(
        subset=[
            "server",
            "drive",
            "timestamp",
        ],
        keep="last",
    )
 
 
def calculate_anomalies(
    group,
    baseline_points,
    sudden_jump,
    spike_multiplier,
):
    group = (
        group
        .sort_values("timestamp")
        .copy()
    )
 
    group["growth"] = (
        group["used_percent"].diff()
    )
 
    group["baseline_growth"] = (
        group["growth"]
        .shift(1)
        .rolling(
            window=baseline_points,
            min_periods=baseline_points,
        )
        .mean()
    )
 
    group["spike"] = (
        group["growth"].gt(0)
        & group["baseline_growth"].notna()
        & group["growth"].gt(
            group["baseline_growth"]
            * spike_multiplier
        )
    )
 
    group["sudden_jump"] = (
        group["growth"].gt(
            sudden_jump
        )
    )
 
    group["anomaly"] = (
        group["spike"]
        | group["sudden_jump"]
    )
 
    return group
 
 
def generate_report(
    df,
    threshold,
    history_days,
    baseline_points,
    sudden_jump,
    spike_multiplier,
):
    if df.empty:
        return []
 
    latest_timestamp = df["timestamp"].max()
 
    window_start = (
        latest_timestamp
        - timedelta(days=history_days)
    )
 
    history = df[
        df["timestamp"] >= window_start
    ].copy()
 
    latest = (
        history
        .sort_values("timestamp")
        .groupby(
            [
                "server",
                "drive",
            ],
            as_index=False,
        )
        .tail(1)
    )
 
    latest = latest[
        latest["used_percent"] >= threshold
    ]
 
    if latest.empty:
        return []
 
    eligible = set(
        zip(
            latest["server"],
            latest["drive"],
        )
    )
 
    history = history[
        history.apply(
            lambda row: (
                row["server"],
                row["drive"],
            ) in eligible,
            axis=1,
        )
    ]
 
    results = []
 
    for key, group in history.groupby(
        [
            "server",
            "drive",
        ]
    ):
        analyzed = calculate_anomalies(
            group=group,
            baseline_points=baseline_points,
            sudden_jump=sudden_jump,
            spike_multiplier=spike_multiplier,
        )
 
        latest_row = analyzed.iloc[-1]
 
        anomalies = analyzed[
            analyzed["anomaly"]
        ]
 
        if anomalies.empty:
            first_anomaly_date = None
        else:
            first_anomaly_date = (
                anomalies.iloc[0]["timestamp"]
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
 
        positive_growth = analyzed.loc[
            analyzed["growth"] > 0,
            "growth",
        ]
 
        if positive_growth.empty:
            max_spike = 0.0
        else:
            max_spike = float(
                positive_growth.max()
            )
 
        historic_disk_usage = [
            round(float(value), 2)
            for value in group[
                "used_percent"
            ].tolist()
        ]
 
        results.append(
            {
                "server": str(key[0]),
                "drive": str(key[1]),
                "used_percent": round(
                    float(
                        latest_row[
                            "used_percent"
                        ]
                    ),
                    2,
                ),
                "anomaly_detected_30d": bool(
                    not anomalies.empty
                ),
                "anomaly_count": int(
                    len(anomalies)
                ),
                "first_anomaly_date":
                    first_anomaly_date,
                "max_spike": round(
                    max_spike,
                    2,
                ),
                "historic_disk_usage":
                    historic_disk_usage,
            }
        )
 
    results.sort(
        key=lambda item: (
            not item["anomaly_detected_30d"],
            item["server"],
            item["drive"],
        )
    )
 
    return results
 
 
def write_report(
    result,
    output,
):
    output_path = Path(output)
 
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
 
    temporary_path = output_path.with_suffix(
        ".tmp"
    )
 
    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
 
    os.replace(
        temporary_path,
        output_path,
    )
 
 
def validate_arguments(args):
    if args.history_days < 1:
        raise ValueError(
            "history-days must be greater than zero"
        )
 
    if args.baseline_points < 1:
        raise ValueError(
            "baseline-points must be greater than zero"
        )
 
    if not 0 <= args.threshold <= 100:
        raise ValueError(
            "threshold must be between 0 and 100"
        )
 
    if args.sudden_jump <= 0:
        raise ValueError(
            "sudden-jump must be greater than zero"
        )
 
    if args.spike_multiplier <= 0:
        raise ValueError(
            "spike-multiplier must be greater than zero"
        )
 
 
def main():
    configure_logging()
 
    args = parse_args()
 
    validate_arguments(args)
 
    df = load_csv_files(
        args.input_dir
    )
 
    df = clean_data(df)
 
    result = generate_report(
        df=df,
        threshold=args.threshold,
        history_days=args.history_days,
        baseline_points=args.baseline_points,
        sudden_jump=args.sudden_jump,
        spike_multiplier=args.spike_multiplier,
    )
 
    write_report(
        result=result,
        output=args.output,
    )
 
    LOGGER.info(
        "Analysis completed. Records: %d",
        len(result),
    )
 
 
if __name__ == "__main__":
    main()
