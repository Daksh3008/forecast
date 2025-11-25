"""
Wrapper script to forecast for a target date.

Usage:
    python scripts/forecast_date.py --target 2026-01-01
"""

import argparse
from forecast.inference.forecast_to_date import forecast_to_date

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    forecast_to_date(args.target)
