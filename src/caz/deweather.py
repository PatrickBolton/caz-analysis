"""Weather normalisation ("deweathering") of hourly concentrations with a random forest.

Method of Grange & Carslaw (2019), as applied by Liu et al. (2023, pp. 208-209)
following Shi et al. (2021):
1. Train a random forest to predict the hourly concentration at one site from
   time variables (trend, day of year, weekday, hour) and weather variables.
2. For every hour, predict again `n_samples` times, each time with the weather
   variables replaced by those of hours drawn at random, without replacement,
   from the whole period; the time variables keep their real values. The mean
   of these predictions is the weather-normalised concentration: what it would
   have been under average weather.
Forest settings follow the rmweather/ranger defaults the paper used: 300 trees,
sqrt(p) variables tried per split, minimum 5 observations per leaf.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TIME_VARIABLES = ["date_unix", "day_julian", "weekday", "hour"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class Normalised:
    hourly: pd.DataFrame  # measured_at, observed, normalised
    r2_oob: float  # out-of-bag R^2 of the forest: how well it predicts unseen hours


def time_variables(measured_at: pd.Series) -> pd.DataFrame:
    """Time predictors, named as in Liu et al. (2023), Fig. 6."""
    return pd.DataFrame(
        {
            # seconds since 1970 (the trend); independent of the storage unit (ns or us)
            "date_unix": (measured_at - EPOCH) // pd.Timedelta(seconds=1),
            "day_julian": measured_at.dt.day_of_year,
            "weekday": measured_at.dt.day_of_week,  # Monday = 0
            "hour": measured_at.dt.hour,
        },
        index=measured_at.index,
    )


def normalise(
    hourly: pd.DataFrame,
    weather_columns: list[str],
    n_samples: int = 500,
    n_trees: int = 300,
    seed: int = 0,
) -> Normalised:
    """Weather-normalise `hourly` (columns measured_at, value, *weather_columns).

    Hours with a missing value or missing weather are not used to train, and get
    no normalised value; the weather that is resampled comes from all complete hours.
    """
    df = hourly.dropna(subset=["value", *weather_columns]).reset_index(drop=True)
    features = pd.concat([time_variables(df["measured_at"]), df[weather_columns]], axis=1)
    forest = RandomForestRegressor(
        n_estimators=n_trees,
        max_features="sqrt",
        min_samples_leaf=5,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(features.to_numpy(), df["value"].to_numpy())

    rng = np.random.default_rng(seed)
    weather = df[weather_columns].to_numpy()
    resampled = features.to_numpy().copy()
    weather_index = [features.columns.get_loc(c) for c in weather_columns]
    total = np.zeros(len(df))
    for _ in range(n_samples):
        # each hour gets the weather of another hour (a permutation = draw without replacement)
        resampled[:, weather_index] = weather[rng.permutation(len(df))]
        total += forest.predict(resampled)
    out = pd.DataFrame(
        {"measured_at": df["measured_at"], "observed": df["value"], "normalised": total / n_samples}
    )
    return Normalised(hourly=out, r2_oob=float(forest.oob_score_))


def weekly_means(hourly: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Weekly means of hourly columns; weeks start on Monday (as in Liu et al., footnote 11)."""
    week_start = (hourly["measured_at"].dt.tz_convert("UTC").dt.tz_localize(None)
                  .dt.to_period("W-SUN").dt.start_time)  # fmt: skip
    return hourly.assign(week_start=week_start).groupby("week_start")[columns].mean().reset_index()
