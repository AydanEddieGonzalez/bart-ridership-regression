# Bart Ridership Regression

## Data

Raw data is not committed (too large). To reproduce `data/processed/bart_daily.csv`:

1. **BART ridership:** download `date-hour-soo-dest-2024.csv.gz` and
   `date-hour-soo-dest-2025.csv.gz` (hourly origin-destination data, linked from
   [BART Ridership Reports](https://www.bart.gov/about/reports/ridership)) into `data/raw/`.
2. **Weather:** run `python scripts/download_weather.py`. This pulls daily max temperature
   and precipitation for Oakland from the [Open-Meteo Historical API](https://open-meteo.com/).
3. **Holidays:** US federal holidays from the `holidays` Python package.
4. Run `notebooks/01_eda.ipynb` top to bottom.

**Final dataset:** daily exits for the 8 busiest stations, Jan 2024 to Dec 2025
(about 5,800 rows).