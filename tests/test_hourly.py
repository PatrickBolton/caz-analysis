"""Reading openair site-year files (no database or download needed)."""

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest
import rdata

from caz.hourly import FileContentError, read_site_year
from caz.openair import NETWORKS, site_year_url

JAN_1_2023 = 1672531200.0  # 2023-01-01 00:00 UTC, in seconds since 1970


def write_file(tmp_path: Path, table: dict, name: str = "X_2023") -> Path:
    """An .RData file shaped like openair's: a table named after the file."""
    path = tmp_path / f"{name}.RData"
    rdata.write_rda(path, {name: pd.DataFrame(table)})
    return path


def hours(n: int, start: float = JAN_1_2023) -> list[float]:
    return [start + 3600 * i for i in range(n)]


def test_columns_are_renamed_and_times_are_utc_hour_beginning(tmp_path: Path) -> None:
    path = write_file(
        tmp_path,
        {"date": hours(2), "NO2": [40.0, 41.0], "PM2.5": [5.0, None], "ws": [2.0, 3.0],
         "wd": [180.0, 190.0], "temp": [6.5, 6.0], "O3": [30.0, 31.0], "site": ["X", "X"]},
    )  # fmt: skip
    df = read_site_year(path, "X", 2023)
    assert list(df.columns) == ["measured_at", "no2", "pm25", "ws", "wd", "air_temp"]
    assert str(df["measured_at"].iloc[0]) == "2023-01-01 00:00:00+00:00"
    assert df["no2"].tolist() == [40.0, 41.0]
    assert pd.isna(df["pm25"].iloc[1])


def test_missing_columns_are_empty_and_empty_hours_dropped(tmp_path: Path) -> None:
    # LMAM files have no weather; an hour with no value at all is not stored
    path = write_file(tmp_path, {"date": hours(3), "NO2": [10.0, None, 12.0]})
    df = read_site_year(path, "X", 2023)
    assert len(df) == 2
    assert df["air_temp"].isna().all()


def test_file_without_the_hourly_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "X_2023.RData"
    rdata.write_rda(path, {"X_2023_daily_mean": pd.DataFrame({"date": hours(1)})})
    with pytest.raises(FileContentError, match="no table X_2023"):
        read_site_year(path, "X", 2023)


def test_hours_outside_the_year_are_rejected(tmp_path: Path) -> None:
    path = write_file(tmp_path, {"date": hours(2, start=JAN_1_2023 - 3600), "NO2": [1.0, 2.0]})
    with pytest.raises(FileContentError, match="outside 2023"):
        read_site_year(path, "X", 2023)


@pytest.mark.parametrize(
    "column,value",
    [("NO2", -9999.0), ("PM2.5", 5000.0), ("wd", 400.0), ("ws", -1.0), ("temp", 60.0)],
)
def test_implausible_values_are_rejected(tmp_path: Path, column: str, value: float) -> None:
    path = write_file(tmp_path, {"date": hours(1), column: [value]})
    with pytest.raises(pandera.errors.SchemaError):
        read_site_year(path, "X", 2023)


def test_duplicate_or_off_the_hour_times_are_rejected(tmp_path: Path) -> None:
    duplicate = write_file(tmp_path, {"date": [JAN_1_2023] * 2, "NO2": [1.0, 2.0]})
    with pytest.raises(pandera.errors.SchemaError):
        read_site_year(duplicate, "X", 2023)
    off_hour = write_file(tmp_path, {"date": [JAN_1_2023 + 1800], "NO2": [1.0]})
    with pytest.raises(pandera.errors.SchemaError):
        read_site_year(off_hour, "X", 2023)


def test_site_year_urls() -> None:
    assert site_year_url(NETWORKS["AURN"], "NEWC", 2023) == (
        "https://uk-air.defra.gov.uk/openair/R_data/NEWC_2023.RData"
    )
    # LMAM files sit in the provider's subfolder
    assert site_year_url(NETWORKS["LMAM"], "NCA5", 2023, "aqdm") == (
        "https://uk-air.defra.gov.uk/openair/LMAM/R_data/aqdm/NCA5_2023.RData"
    )
    with pytest.raises(ValueError, match="provider code"):
        site_year_url(NETWORKS["LMAM"], "NCA5", 2023)
