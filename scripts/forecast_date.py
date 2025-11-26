"""
Wrapper to call forecast.inference.forecast_to_date
"""

import argparse
from forecast.inference.forecast_to_date import forecast_to_date

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    out = forecast_to_date(args.target)
    print("Forecast:", out)
