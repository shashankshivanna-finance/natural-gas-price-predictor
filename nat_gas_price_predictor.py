"""
╔══════════════════════════════════════════════════════════════╗
║          Natural Gas Price Predictor                         ║
║──────────────────────────────────────────────────────────────║
║  HOW TO RUN                                                  ║
║  1. Put this file and Nat_Gas.csv in the same folder         ║
║  2. Open a terminal / command prompt in that folder          ║
║  3. Install dependencies (one-time):                         ║
║       pip install pandas numpy scipy matplotlib              ║
║  4. Run:                                                     ║
║       python nat_gas_price_predictor.py                      ║
║                                                              ║
║  You can also pass a date directly:                          ║
║       python nat_gas_price_predictor.py 2025-06-15           ║
╚══════════════════════════════════════════════════════════════╝

MODEL DESCRIPTION
─────────────────
• Historical dates (Oct 2020 – Sep 2024):
    Cubic Spline Interpolation — fits a smooth curve through all
    49 monthly data points.

• Future dates (Oct 2024 – Sep 2025, one year ahead):
    Linear Trend + Monthly Seasonality — captures the long-term
    upward drift and repeating seasonal pattern (e.g. winter highs,
    spring/summer dips).

• Dates beyond Sep 2025 are rejected; too far to be reliable.
"""

# ── Auto-install missing packages ─────────────────────────────────────────────
import subprocess, sys

REQUIRED = ["pandas", "numpy", "scipy", "matplotlib"]
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILENAME = "Nat_Gas.csv"   # must be in the same folder as this script


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"\n  Cannot find '{path}'.")
        print("  Make sure Nat_Gas.csv is in the same folder as this script.")
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "price"}, inplace=True)
    df["date"]  = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PRICE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class NatGasPriceModel:
    def __init__(self, df: pd.DataFrame):
        self.df           = df
        self.first_date   = df["date"].iloc[0]
        self.last_date    = df["date"].iloc[-1]
        self.forecast_end = self.last_date + pd.DateOffset(years=1)

        t      = (df["date"] - self.first_date).dt.days.values.astype(float)
        prices = df["price"].values

        # Cubic spline for interpolation inside historical window
        self._spline = CubicSpline(t, prices, extrapolate=False)

        # Linear trend for extrapolation
        slope, intercept, *_ = linregress(t, prices)
        self._slope     = slope
        self._intercept = intercept

        # Monthly seasonal offsets (average residual per calendar month)
        residuals = prices - (slope * t + intercept)
        df2 = df.copy()
        df2["residual"] = residuals
        df2["month"]    = df2["date"].dt.month
        self._seasonal  = df2.groupby("month")["residual"].mean().to_dict()

    def predict(self, date) -> float:
        ts = pd.Timestamp(date)

        if ts > self.forecast_end:
            raise ValueError(
                f"  {ts.date()} is beyond the one-year extrapolation limit "
                f"({self.forecast_end.date()}).\n"
                "  Estimates that far ahead are not reliable."
            )

        t_val = float((ts - self.first_date).days)

        if self.first_date <= ts <= self.last_date:
            price = float(self._spline(t_val))
        else:
            trend    = self._slope * t_val + self._intercept
            seasonal = self._seasonal.get(ts.month, 0.0)
            price    = trend + seasonal

        return round(price, 4)

    def batch_predict(self, dates) -> pd.DataFrame:
        rows = []
        for d in dates:
            ts = pd.Timestamp(d)
            try:
                price = self.predict(ts)
                zone  = "Historical" if ts <= self.last_date else "Forecast"
                rows.append({"date": ts.date(), "price": price, "zone": zone})
            except ValueError as e:
                rows.append({"date": ts.date(), "price": None, "zone": str(e)})
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_and_save(model: NatGasPriceModel, save_path: str = "nat_gas_forecast.png"):
    all_dates  = pd.date_range(model.first_date, model.forecast_end, freq="D")
    all_prices = np.array([model.predict(d) for d in all_dates])
    hist_mask  = all_dates <= model.last_date

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(all_dates[hist_mask],  all_prices[hist_mask],
            color="#2563EB", lw=2, label="Interpolated (historical)")
    ax.plot(all_dates[~hist_mask], all_prices[~hist_mask],
            color="#F59E0B", lw=2, linestyle="--", label="Extrapolated (forecast)")
    ax.scatter(model.df["date"], model.df["price"],
               color="#1E3A5F", s=35, zorder=5, label="Monthly data points")

    ax.axvline(model.last_date, color="grey", linestyle=":", lw=1.2)
    ax.text(model.last_date + timedelta(days=8), ax.get_ylim()[0] + 0.15,
            "Forecast ->", fontsize=9, color="grey")

    ax.set_title("Natural Gas Prices — Historical Interpolation & 1-Year Forecast",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Price ($/MMBtu)", labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Chart saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE / CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(model: NatGasPriceModel):
    sample_dates = [
        "2020-11-15", "2021-06-10", "2022-01-20",
        "2023-03-01", "2024-04-15", "2024-09-30",
        "2024-11-15", "2025-01-31", "2025-06-15", "2025-09-30",
    ]
    results = model.batch_predict(sample_dates)
    print("\n" + "-" * 52)
    print(f"  {'Date':<14}  {'Price ($/MMBtu)':>16}  {'Zone'}")
    print("-" * 52)
    for _, row in results.iterrows():
        price_str = f"${row['price']:.4f}" if row["price"] else "N/A"
        print(f"  {str(row['date']):<14}  {price_str:>16}  {row['zone']}")
    print("-" * 52)


def interactive_loop(model: NatGasPriceModel):
    print("\n" + "=" * 58)
    print("  Natural Gas Price Estimator — Interactive Mode")
    print("=" * 58)
    print(f"  Historical data : {model.first_date.date()} -> {model.last_date.date()}")
    print(f"  Forecast range  : up to {model.forecast_end.date()}")
    print("  Type a date (YYYY-MM-DD) or 'quit' to exit.")
    print("=" * 58)

    while True:
        raw = input("\n  Enter date (YYYY-MM-DD): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if not raw:
            continue
        try:
            price = model.predict(raw)
            ts    = pd.Timestamp(raw)
            zone  = "Historical estimate" if ts <= model.last_date else "Forecast"
            print(f"  -> [{zone}]  {raw}  :  ${price:.4f} / MMBtu")
        except ValueError as e:
            print(e)
        except Exception:
            print("  Invalid date format. Please use YYYY-MM-DD (e.g. 2024-06-15).")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILENAME)

    df    = load_data(csv_path)
    model = NatGasPriceModel(df)

    # Command-line date argument: python script.py 2025-06-15
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        try:
            price = model.predict(date_str)
            ts    = pd.Timestamp(date_str)
            zone  = "Historical estimate" if ts <= model.last_date else "Forecast"
            print(f"\n  [{zone}]  {date_str}  ->  ${price:.4f} / MMBtu\n")
        except ValueError as e:
            print(e)
        return

    print_summary_table(model)
    plot_and_save(model, os.path.join(script_dir, "nat_gas_forecast.png"))
    interactive_loop(model)


if __name__ == "__main__":
    main()
