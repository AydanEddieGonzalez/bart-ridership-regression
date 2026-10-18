# Bart Ridership Regression

I am modeling how many people exit a BART station on a given day. Ridership depends on stuff like day of the week, weather, holidays, and where the station is. Let's say if it rains on a random Tuesday, I am trying to predict what effect that has on the amount of people exiting a certain station, if they will choose to exit sooner or later. I will do this through a multiple linear regression model.

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

### Stations

The 8 busiest stations by total exits over 2024 to 2025:

| Code | Station |
|---|---|
| `12TH` | 12th St / Oakland City Center |
| `16TH` | 16th St Mission |
| `24TH` | 24th St Mission |
| `CIVC` | Civic Center / UN Plaza |
| `DBRK` | Downtown Berkeley |
| `EMBR` | Embarcadero |
| `MONT` | Montgomery St |
| `POWL` | Powell St |

## Variables

Each row is one station on one day (8 stations × 731 days, 2024 to 2025).

### Target
| Variable | Description |
|---|---|
| `exits` | Total trips ending at the station that day (summed from BART hourly origin-destination data) |


### Features

I implemented a total of 27 features. I did not use `date` as a feature since that was used just to split the testing and training data

| Variable | Type | Description | Preprocessing |
|---|---|---|---|
| `temp_max_f` | Numeric | Daily max temperature (°F), Oakland | Standardized with train mean/std |
| `precip_in` | Numeric | Daily precipitation (inches), Oakland | Standardized with train mean/std |
| `is_holiday` | Binary | 1 if a US federal holiday, else 0 | None |
| `day_of_week_*` | One-hot (6 columns) | Day of week | Baseline: Friday |
| `month_*` | One-hot (11 columns) | Month (2 to 12) | Baseline: January |
| `station_*` | One-hot (7 columns) | BART station code | Baseline: [FIRST STATION CODE] |

## Preprocessing

All preprocessing is in `src/inreg/preprocessing.py` and follows a **fit/transform** pattern: anything the pipline learns (category lists, means, standard deviations) is learned from training set only, then applied to both train and test.
