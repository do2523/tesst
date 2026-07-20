"""
Predict each Snowflake table's next-day cost with a simple linear regression model.

Expected CSV columns:
    date
    table_name
    table_cost
    query_count
    bytes_scanned
    cache_percent
    remote_io
    synchronization

Example:
    python forecast_table_cost.py --input table_daily_metrics.csv

Outputs:
    model_evaluation.csv
    next_day_predictions.csv
    actual_vs_predicted.png
    feature_coefficients.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "date",
    "table_name",
    "table_cost",
    "query_count",
    "bytes_scanned",
    "cache_percent",
    "remote_io",
    "synchronization",
]

NUMERIC_FEATURES = [
    "table_cost",
    "query_count",
    "bytes_scanned",
    "cache_percent",
    "remote_io",
    "synchronization",
    "cost_change_pct",
    "cost_3day_avg",
]

CATEGORICAL_FEATURES = ["table_name", "day_of_week"]


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nExpected columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    if df.empty:
        raise ValueError("The input CSV is empty.")


def build_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [c for c in REQUIRED_COLUMNS if c not in {"date", "table_name"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "table_name", "table_cost"])
    df = df.sort_values(["table_name", "date"]).reset_index(drop=True)

    # Features available at the end of the current day.
    df["day_of_week"] = df["date"].dt.day_name()
    df["cost_change_pct"] = (
        df.groupby("table_name")["table_cost"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )
    df["cost_3day_avg"] = (
        df.groupby("table_name")["table_cost"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    # Label: the same table's cost on its next recorded day.
    df["tomorrow_cost"] = df.groupby("table_name")["table_cost"].shift(-1)
    df["target_date"] = df.groupby("table_name")["date"].shift(-1)

    return df


def create_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def chronological_split(training_rows: pd.DataFrame):
    unique_target_dates = sorted(training_rows["target_date"].dropna().unique())

    if len(unique_target_dates) < 2:
        raise ValueError(
            "Not enough dated observations to evaluate the model. "
            "Provide at least 3 daily rows per table when possible."
        )

    test_date = unique_target_dates[-1]
    train_df = training_rows[training_rows["target_date"] < test_date].copy()
    test_df = training_rows[training_rows["target_date"] == test_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Could not create a chronological train/test split.")

    return train_df, test_df, pd.Timestamp(test_date)


def save_coefficients(pipeline: Pipeline, output_dir: Path) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "absolute_coefficient": np.abs(model.coef_),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    coefficients.to_csv(output_dir / "feature_coefficients.csv", index=False)


def save_plot(actual, predicted, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.75)

    minimum = float(min(actual.min(), predicted.min()))
    maximum = float(max(actual.max(), predicted.max()))
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")

    plt.xlabel("Actual next-day cost")
    plt.ylabel("Predicted next-day cost")
    plt.title("Snowflake Table Cost: Actual vs. Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast Snowflake table cost for the next day."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the daily table metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="forecast_output",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_input(df)
    prepared = build_training_data(df)

    training_rows = prepared.dropna(subset=["tomorrow_cost", "target_date"]).copy()
    train_df, test_df, test_date = chronological_split(training_rows)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipeline = create_pipeline()
    pipeline.fit(train_df[features], train_df["tomorrow_cost"])

    test_predictions = pipeline.predict(test_df[features])
    test_predictions = np.maximum(test_predictions, 0)

    mae = mean_absolute_error(test_df["tomorrow_cost"], test_predictions)
    rmse = mean_squared_error(
        test_df["tomorrow_cost"], test_predictions
    ) ** 0.5

    r2 = np.nan
    if len(test_df) >= 2 and test_df["tomorrow_cost"].nunique() > 1:
        r2 = r2_score(test_df["tomorrow_cost"], test_predictions)

    evaluation = pd.DataFrame(
        [
            {
                "test_target_date": test_date.date(),
                "training_rows": len(train_df),
                "test_rows": len(test_df),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        ]
    )
    evaluation.to_csv(output_dir / "model_evaluation.csv", index=False)

    comparison = test_df[
        ["table_name", "date", "target_date", "tomorrow_cost"]
    ].copy()
    comparison["predicted_tomorrow_cost"] = test_predictions
    comparison["absolute_error"] = (
        comparison["tomorrow_cost"] - comparison["predicted_tomorrow_cost"]
    ).abs()
    comparison.to_csv(output_dir / "test_predictions.csv", index=False)

    save_plot(
        test_df["tomorrow_cost"],
        pd.Series(test_predictions, index=test_df.index),
        output_dir,
    )
    save_coefficients(pipeline, output_dir)

    # Retrain on every row whose next-day target is known.
    pipeline.fit(training_rows[features], training_rows["tomorrow_cost"])

    # Predict one day beyond the latest date for every table.
    latest_rows = (
        prepared.sort_values(["table_name", "date"])
        .groupby("table_name", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["prediction_date"] = latest_rows["date"] + pd.Timedelta(days=1)
    next_day_predictions = pipeline.predict(latest_rows[features])
    next_day_predictions = np.maximum(next_day_predictions, 0)

    results = latest_rows[
        ["table_name", "date", "prediction_date", "table_cost"]
    ].copy()
    results = results.rename(columns={"table_cost": "latest_known_cost"})
    results["predicted_cost"] = next_day_predictions
    results["predicted_change"] = (
        results["predicted_cost"] - results["latest_known_cost"]
    )
    results["predicted_change_pct"] = np.where(
        results["latest_known_cost"] != 0,
        results["predicted_change"] / results["latest_known_cost"] * 100,
        np.nan,
    )
    results = results.sort_values("predicted_cost", ascending=False)
    results.to_csv(output_dir / "next_day_predictions.csv", index=False)

    print("\nModel evaluation")
    print(evaluation.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    print("\nTop next-day predictions")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
