"""
Snowflake Table Cost Intelligence MCP Server.

Exposes three tools over the Model Context Protocol:

    get_table_cost            -- look up recorded daily cost for a table
    get_most_expensive_tables -- rank tables by recent cost
    forecast_table_cost       -- predict each table's next-day cost with a
                                 linear-regression model (reuses the project's
                                 forecast pipeline)

The server reads the same daily-metrics CSV the CLI uses. Point it at a
different file with the SNOWFLAKE_METRICS_CSV environment variable.

Run directly for stdio transport (how Claude Desktop / Claude Code launch it):

    python mcp_server.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from forecast_table_cost import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_training_data,
    create_pipeline,
    validate_input,
)

DEFAULT_CSV = "sample_table_daily_metrics.csv"

mcp = FastMCP("Snowflake Table Cost Intelligence")


def _csv_path() -> Path:
    """Resolve the metrics CSV, preferring the env override."""
    configured = os.environ.get("SNOWFLAKE_METRICS_CSV", DEFAULT_CSV)
    path = Path(configured)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _load_metrics() -> pd.DataFrame:
    """Load and validate the daily metrics CSV."""
    path = _csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {path}. "
            "Set SNOWFLAKE_METRICS_CSV to point at your data."
        )
    df = pd.read_csv(path)
    validate_input(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["table_cost"] = pd.to_numeric(df["table_cost"], errors="coerce")
    df = df.dropna(subset=["date", "table_name", "table_cost"])
    return df


@mcp.tool()
def get_table_cost(table_name: str, date: str | None = None) -> dict:
    """Get the recorded daily cost for a specific Snowflake table.

    Args:
        table_name: Fully-qualified table name, e.g. "ANALYTICS.PUBLIC.ORDERS".
        date: Optional YYYY-MM-DD date. When omitted, returns the most
            recent recorded day plus a short recent-history trail.

    Returns:
        The cost for the requested day (or latest), the number of days on
        record, and a recent-history list of {date, table_cost}.
    """
    df = _load_metrics()
    table_df = df[df["table_name"] == table_name].sort_values("date")

    if table_df.empty:
        available = sorted(df["table_name"].unique().tolist())
        return {
            "error": f"No records found for table '{table_name}'.",
            "available_tables": available,
        }

    if date is not None:
        target = pd.to_datetime(date, errors="coerce")
        if pd.isna(target):
            return {"error": f"Could not parse date '{date}'. Use YYYY-MM-DD."}
        match = table_df[table_df["date"] == target]
        if match.empty:
            return {
                "error": f"No record for {table_name} on {date}.",
                "date_range": {
                    "earliest": table_df["date"].min().date().isoformat(),
                    "latest": table_df["date"].max().date().isoformat(),
                },
            }
        row = match.iloc[-1]
        return {
            "table_name": table_name,
            "date": row["date"].date().isoformat(),
            "table_cost": round(float(row["table_cost"]), 4),
        }

    latest = table_df.iloc[-1]
    history = [
        {"date": r["date"].date().isoformat(), "table_cost": round(float(r["table_cost"]), 4)}
        for _, r in table_df.tail(7).iterrows()
    ]
    return {
        "table_name": table_name,
        "latest_date": latest["date"].date().isoformat(),
        "latest_cost": round(float(latest["table_cost"]), 4),
        "days_on_record": int(len(table_df)),
        "recent_history": history,
    }


@mcp.tool()
def get_most_expensive_tables(limit: int = 5, date: str | None = None) -> dict:
    """Rank Snowflake tables by cost, most expensive first.

    Args:
        limit: Maximum number of tables to return (default 5).
        date: Optional YYYY-MM-DD date. When omitted, ranks each table by
            its most recent recorded daily cost.

    Returns:
        A ranked list of {table_name, date, table_cost}.
    """
    df = _load_metrics()
    if limit < 1:
        return {"error": "limit must be at least 1."}

    if date is not None:
        target = pd.to_datetime(date, errors="coerce")
        if pd.isna(target):
            return {"error": f"Could not parse date '{date}'. Use YYYY-MM-DD."}
        day_df = df[df["date"] == target]
        if day_df.empty:
            return {"error": f"No records on {date}."}
        ranked = day_df.sort_values("table_cost", ascending=False).head(limit)
        basis = f"cost on {date}"
    else:
        latest = df.sort_values("date").groupby("table_name", as_index=False).tail(1)
        ranked = latest.sort_values("table_cost", ascending=False).head(limit)
        basis = "most recent recorded cost per table"

    tables = [
        {
            "rank": i + 1,
            "table_name": row["table_name"],
            "date": row["date"].date().isoformat(),
            "table_cost": round(float(row["table_cost"]), 4),
        }
        for i, (_, row) in enumerate(ranked.iterrows())
    ]
    return {"basis": basis, "count": len(tables), "tables": tables}


@mcp.tool()
def forecast_table_cost(table_name: str | None = None) -> dict:
    """Predict next-day cost for Snowflake tables using linear regression.

    Trains the project's forecasting pipeline on all available history and
    predicts one day beyond each table's latest recorded date.

    Args:
        table_name: Optional fully-qualified table name to forecast just one
            table. When omitted, forecasts every table.

    Returns:
        Per-table predictions with latest known cost, predicted cost, and the
        absolute/percentage change.
    """
    df = _load_metrics()
    prepared = build_training_data(df)
    training_rows = prepared.dropna(subset=["tomorrow_cost", "target_date"]).copy()

    if training_rows.empty:
        return {
            "error": "Not enough dated history to train a forecast. "
            "Provide at least 2 daily rows per table."
        }

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    pipeline = create_pipeline()
    pipeline.fit(training_rows[features], training_rows["tomorrow_cost"])

    latest_rows = (
        prepared.sort_values(["table_name", "date"])
        .groupby("table_name", as_index=False)
        .tail(1)
        .copy()
    )

    if table_name is not None:
        latest_rows = latest_rows[latest_rows["table_name"] == table_name]
        if latest_rows.empty:
            available = sorted(df["table_name"].unique().tolist())
            return {
                "error": f"No records found for table '{table_name}'.",
                "available_tables": available,
            }

    latest_rows["prediction_date"] = latest_rows["date"] + pd.Timedelta(days=1)
    preds = np.maximum(pipeline.predict(latest_rows[features]), 0)

    results = []
    for (_, row), pred in zip(latest_rows.iterrows(), preds):
        latest_cost = float(row["table_cost"])
        predicted = float(pred)
        change = predicted - latest_cost
        change_pct = (change / latest_cost * 100) if latest_cost else None
        results.append(
            {
                "table_name": row["table_name"],
                "latest_date": row["date"].date().isoformat(),
                "prediction_date": row["prediction_date"].date().isoformat(),
                "latest_known_cost": round(latest_cost, 4),
                "predicted_cost": round(predicted, 4),
                "predicted_change": round(change, 4),
                "predicted_change_pct": (
                    round(change_pct, 2) if change_pct is not None else None
                ),
            }
        )

    results.sort(key=lambda r: r["predicted_cost"], reverse=True)
    return {
        "training_rows": int(len(training_rows)),
        "forecasts": results,
    }


if __name__ == "__main__":
    mcp.run()
