# BART Ridership Regression

I am modeling how many people exit a BART station on a given day. Ridership depends on stuff like day of the week, weather, holidays, and where the station is. For example, if it rains on a random Tuesday, how many fewer people exit at a certain station? I will do this through a multiple linear regression model.

## Data

Raw data is not committed (too large). To reproduce `data/processed/bart_daily.csv`:

1. **BART ridership:** download `date-hour-soo-dest-2024.csv.gz` and
   `date-hour-soo-dest-2025.csv.gz` (hourly origin-destination data, linked from
   [BART Ridership Reports](https://www.bart.gov/about/reports/ridership)) into `data/raw/`.
2. **Weather:** run `python scripts/download_weather.py`. This pulls daily max temperature
   and precipitation for Oakland from the [Open-Meteo Historical API](https://open-meteo.com/).
3. **Holidays:** US federal holidays from the `holidays` Python package.
4. Run `notebooks/01_eda.ipynb` top to bottom.

**Final dataset:** daily exits for the 8 busiest stations, Jan 1, 2024 to Dec 31, 2025
(5,848 rows: 8 stations × 731 days, no missing values or duplicates).

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
| `station_*` | One-hot (7 columns) | BART station code | Baseline: `12TH` |

## Preprocessing

All preprocessing is in `src/linreg/preprocessing.py` and follows a **fit/transform** pattern: anything the pipeline learns (category lists, means, standard deviations) is learned from training set only, then applied to both train and test.

## The math

**Design matrix.** We use a matrix because there are multiple $b$ values in the linear regression (one for each feature), so instead of solving for each one separately, we solve for all of them at once to find the line of best fit. Each row of $X$ is one station-day and each column is one feature. The first column is all 1s, because that's the $a$ value (the intercept) in the linear regression. Multiplying it by 1 in every row adds it to every prediction.

$n$ is the number of equations we have, and $d$ is the number of variables. For the training data, $n = 366 \times 8 = 2928$ and $d = 27$, so with the column of 1s, $X$ is $2928 \times 28$ and $\theta$ is $28 \times 1$. The predictions are:

$$
\hat{y} = X\theta
$$

which gives one prediction per row.

**Gradient.** The loss is:

$$
J(\theta) = \frac{1}{2n}\left\| X\theta - y \right\|^2
$$

To find the gradient for one coefficient $\theta_j$, we use the chain rule, and the 2 from it cancels the $\frac{1}{2}$ in the loss. What's left is the error times the value of feature $j$, averaged over every row:

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{n}\sum_{i=1}^{n} \left(\hat{y}_i - y_i\right) x_{i,j}
$$

This gives one number per coefficient, and then $\theta_j$ is updated once. If the model is overestimating, $\theta_j$ decreases, and if it's underestimating, it increases. Doing this for every coefficient at once gives:

$$
\nabla J(\theta) = \frac{1}{n} X^\top \left(X\theta - y\right)
$$

The update is:

$$
\theta \leftarrow \theta - \alpha \nabla J(\theta)
$$

where the learning rate $\alpha$ is the step size. It stays fixed, but the steps get smaller as the errors shrink.

**Normal equation.** At the best fit, the gradient is zero, so:

$$
X^\top X \theta = X^\top y \quad \Longrightarrow \quad \theta = \left(X^\top X\right)^{-1} X^\top y
$$

This only works if $X$ has full column rank, meaning no free variables. If a column can be made from the others (multicollinearity), there are infinitely many answers and the inverse doesn't exist. That's why we dropped one dummy variable per category, since keeping all 7 weekday columns would add up to the column of 1s.

**Three types of gradient descent.** We're doing all three and comparing them with each other and with the normal equation, which gives the exact answer. The difference is how many rows are used for each update:

| Type | Rows per update | Behavior |
|---|---|---|
| Batch | All 2,928 | Smooth but slow |
| Stochastic (SGD) | 1 random row | Updates much more often, but noisy |
| Mini-batch | Small groups (e.g. 32) | In between |

They should all end up close to the normal equation's answer, so the comparison is about how fast and smoothly each one gets there.