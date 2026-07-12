# Natural Gas Price Predictor

A Python tool that estimates the purchase price of natural gas for any given date,
using historical monthly spot price data (Oct 2020 – Sep 2024).

## Method
- **Historical dates (Oct 2020 – Sep 2024)** → Cubic Spline Interpolation across 49 monthly data points
- **Future dates (up to 1 year ahead)** → Linear trend + monthly seasonal adjustment

## Project Structure
natural-gas-price-predictor/
├── nat_gas_price_predictor.py
├── Nat_Gas.csv
├── README.md
└── .gitignore

## Installation
```bash
pip install pandas numpy scipy matplotlib
```

## Usage

```bash
# Interactive mode — prints a sample table, saves a chart, then prompts for dates
python nat_gas_price_predictor.py

# Single date lookup
python nat_gas_price_predictor.py 2025-06-15
```

## Example Output
Date            Price ($/MMBtu)   Zone
2020-11-15          $10.0264      Historical
2024-09-30          $11.8000      Historical
2025-01-31          $12.9464      Forecast
2025-06-15          $11.8519      Forecast
A chart (`nat_gas_forecast.png`) is also saved locally showing the full
historical interpolation and 1-year forecast.

## Skills Demonstrated
- Time-series interpolation (Cubic Spline)
- Trend extrapolation with seasonal adjustment
- Financial data analysis in Python
- Data visualisation with Matplotlib
