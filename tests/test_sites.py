"""Building the site tables from openair metadata (no database needed)."""

import pandas as pd
import pandera.errors
import pytest

from caz.sites import drop_conflicting_copies, site_tables


def metadata(rows: list[dict]) -> pd.DataFrame:
    """Metadata in the form caz.openair.read_metadata returns, with defaults."""
    defaults = {
        "network": "AURN",
        "site_name": "Somewhere",
        "location_type": "Urban Traffic",
        "latitude": 52.0,
        "longitude": -1.5,
        "parameter": "NO2",
        "start_date": "2015-01-01",
        "end_date": None,
        "provider_code": None,  # LMAM only
    }
    df = pd.DataFrame([defaults | row for row in rows])
    df["provider_code"] = df["provider_code"].astype("string")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def test_one_site_row_per_network_site_and_one_parameter_row_per_period() -> None:
    meta = metadata(
        [
            {"site_id": "A1", "parameter": "NO2"},
            {"site_id": "A1", "parameter": "PM2.5", "start_date": "2018-03-01"},
            {"site_id": "A1", "parameter": "NO2", "start_date": "2010-01-01",
             "end_date": "2014-12-31"},  # an earlier instrument
            {"site_id": "B2", "location_type": "Urban Background", "latitude": 52.1},
        ]
    )  # fmt: skip
    sites, parameters = site_tables(meta)
    assert list(sites["site_id"]) == ["A1", "B2"]
    assert list(sites["site_class"]) == ["roadside", "background"]
    assert len(parameters) == 4
    assert parameters["end_date"].isna().sum() == 3  # still running


def test_same_instrument_in_two_networks_points_to_the_kept_record() -> None:
    meta = metadata(
        [
            {"network": "AURN", "site_id": "ABD7"},
            {"network": "SAQN", "site_id": "ABD3", "latitude": 52.00005},  # ~6 m away
            {"network": "SAQN", "site_id": "FAR", "latitude": 52.5},
        ]
    )
    sites = site_tables(meta)[0].set_index(["network", "site_id"])
    assert tuple(sites.loc[("SAQN", "ABD3"), ["canon_network", "canon_site_id"]]) == (
        "AURN",
        "ABD7",
    )
    assert sites.loc[("AURN", "ABD7"), "canon_site_id"] == "ABD7"  # kept: points to itself
    assert sites.loc[("SAQN", "FAR"), "canon_site_id"] == "FAR"


def test_copy_with_unknown_location_type_is_dropped() -> None:
    # As LMAM EL3: listed twice, once without a location type
    meta = metadata(
        [
            {"site_id": "EL3", "site_name": "Hampton Court Parade"},
            {"site_id": "EL3", "site_name": "Hampton - Court Parade",
             "location_type": "unknown unknown", "latitude": 52.00003},
        ]
    )  # fmt: skip
    out = drop_conflicting_copies(meta)
    assert list(out["site_name"]) == ["Hampton Court Parade"]
    assert len(site_tables(meta)[1]) == 1  # no duplicate measurement period


def test_unresolvable_conflict_is_an_error() -> None:
    meta = metadata([{"site_id": "X"}, {"site_id": "X", "latitude": 53.0}])
    with pytest.raises(ValueError, match="conflicting details"):
        site_tables(meta)


def test_site_outside_the_british_isles_is_rejected() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        site_tables(metadata([{"site_id": "X", "latitude": 40.0}]))


def test_period_ending_before_it_starts_is_rejected() -> None:
    meta = metadata([{"site_id": "X", "start_date": "2020-01-01", "end_date": "2019-01-01"}])
    with pytest.raises(pandera.errors.SchemaError):
        site_tables(meta)


def test_lmam_sites_need_a_provider_code() -> None:
    with pytest.raises(pandera.errors.SchemaError):
        site_tables(metadata([{"network": "LMAM", "site_id": "X"}]))
    sites = site_tables(metadata([{"network": "LMAM", "site_id": "X", "provider_code": "aqdm"}]))[0]
    assert sites.loc[0, "provider_code"] == "aqdm"
