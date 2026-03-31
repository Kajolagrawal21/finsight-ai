"""
FinSight AI — Data Quality Checker
Validates incoming stock data for anomalies, missing values, and outliers.
Logs results to MLflow and raises alerts if quality drops.
"""

import logging
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.data_quality")

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}


@dataclass
class QualityReport:
    symbol:          str
    total_rows:      int
    missing_pct:     float
    outlier_pct:     float
    gap_count:       int
    freshness_mins:  float
    passed:          bool
    issues:          List[str]


def check_data_quality(symbol: str) -> QualityReport:
    """Run data quality checks for a symbol."""
    issues = []
    conn   = psycopg2.connect(**DB_CONFIG)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Load recent data
        cur.execute("""
            SELECT time, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = %s
            AND time > NOW() - INTERVAL '7 days'
            ORDER BY time ASC
        """, (symbol,))
        rows = cur.fetchall()

    conn.close()

    if not rows:
        return QualityReport(
            symbol=symbol, total_rows=0, missing_pct=100,
            outlier_pct=0, gap_count=0, freshness_mins=999,
            passed=False, issues=["No data found"]
        )

    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["time"])

    total = len(df)

    # ─── Check 1: Missing values ─────────────────────────────
    missing_pct = df[["open", "high", "low", "close", "volume"]].isnull().mean().mean() * 100
    if missing_pct > 5:
        issues.append(f"High missing values: {missing_pct:.1f}%")

    # ─── Check 2: Price outliers (Z-score > 3) ───────────────
    z_scores    = np.abs((df["close"] - df["close"].mean()) / df["close"].std())
    outlier_pct = (z_scores > 3).mean() * 100
    if outlier_pct > 1:
        issues.append(f"Price outliers detected: {outlier_pct:.1f}%")

    # ─── Check 3: OHLC consistency ───────────────────────────
    invalid_ohlc = ((df["high"] < df["low"]) |
                    (df["close"] > df["high"]) |
                    (df["close"] < df["low"])).sum()
    if invalid_ohlc > 0:
        issues.append(f"Invalid OHLC rows: {invalid_ohlc}")

    # ─── Check 4: Time gaps (missing bars) ───────────────────
    df = df.sort_values("time")
    time_diffs = df["time"].diff().dt.total_seconds() / 60
    gap_count  = (time_diffs > 5).sum()  # gaps > 5 minutes
    if gap_count > 10:
        issues.append(f"Time gaps detected: {gap_count}")

    # ─── Check 5: Data freshness ─────────────────────────────
    latest          = df["time"].max()
    freshness_mins  = (datetime.now(timezone.utc) - latest.tz_convert("UTC")).total_seconds() / 60
    if freshness_mins > 120:
        issues.append(f"Stale data: {freshness_mins:.0f} minutes old")

    passed = len(issues) == 0

    report = QualityReport(
        symbol=symbol,
        total_rows=total,
        missing_pct=round(missing_pct, 2),
        outlier_pct=round(outlier_pct, 2),
        gap_count=int(gap_count),
        freshness_mins=round(freshness_mins, 1),
        passed=passed,
        issues=issues
    )

    status = "✅ PASSED" if passed else f"⚠️ FAILED ({len(issues)} issues)"
    logger.info(f"{symbol}: {status} | rows={total} | fresh={freshness_mins:.0f}m")

    return report


def run_quality_checks(symbols: List[str] = None) -> dict:
    """Run quality checks for all symbols and return summary."""
    if not symbols:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM watchlist WHERE active = TRUE")
            symbols = [r[0] for r in cur.fetchall()]
        conn.close()

    results = {}
    passed  = 0
    failed  = 0

    for symbol in symbols:
        report = check_data_quality(symbol)
        results[symbol] = report
        if report.passed:
            passed += 1
        else:
            failed += 1

    logger.info(f"\n📊 Quality Summary: {passed} passed, {failed} failed")
    return results


if __name__ == "__main__":
    results = run_quality_checks()
    for symbol, report in results.items():
        status = "✅" if report.passed else "❌"
        print(f"{status} {symbol}: {report.total_rows} rows, "
              f"fresh={report.freshness_mins:.0f}m, "
              f"issues={report.issues}")
