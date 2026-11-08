"""Tests for caz.feasibility and the offline parts of caz.openair."""

import datetime as dt

import pandas as pd
import pytest

from caz import feasibility as f
from caz.openair import NETWORKS, annual_summary_url, metadata_url, normalise_metadata


def make_metadata(rows: list[dict]) -> pd.DataFrame:
    """Metadata rows with sensible defaults, in normalised form."""
    defaults = {
        "network": "AURN",
        "site_name": "Test site",
        "location_type": "Urban Traffic",
        "latitude": 52.48,
        "longitude": -1.90,
        "parameter": "NO2",
        "start_date": pd.Timestamp("2010-01-01"),
        "end_date": pd.NaT,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# --- openair helpers -------------------------------------------------------


def test_urls_follow_openair_pattern() -> None:
    aurn = NETWORKS["AURN"]
    assert metadata_url(aurn) == "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
    assert annual_summary_url(NETWORKS["SAQN"], 2019).endswith("summary_annual_SCOT_2019.rds")


def test_normalise_metadata_handles_ongoing_and_case() -> None:
    raw = pd.DataFrame(
        {
            "site_id": ["bir1"],
            "site_name": ["Birmingham A"],
            "location_type": ["Urban Traffic"],
            "latitude": [52.4],
            "longitude": [-1.9],
            "parameter": ["NO2"],
            "start_date": ["2010-01-01"],
            "end_date": ["ongoing"],
        }
    )
    out = normalise_metadata(raw, "LMAM")
    assert out.loc[0, "site_id"] == "BIR1"
    assert out.loc[0, "network"] == "LMAM"
    assert pd.isna(out.loc[0, "end_date"])


# --- site selection --------------------------------------------------------


def test_site_class_mapping() -> None:
    assert f.site_class("Urban Traffic") == "roadside"
    assert f.site_class("Suburban Background") == "background"
    assert f.site_class("unknown unknown") == "other"


def test_no2_sites_keeps_only_no2_and_merges_instrument_records() -> None:
    ts = pd.Timestamp
    meta = make_metadata(
        [
            {"site_id": "A", "parameter": "PM10"},
            {"site_id": "B", "start_date": ts("2012-01-01"), "end_date": pd.NaT},
            {"site_id": "B", "start_date": ts("2008-01-01"), "end_date": ts("2011-12-31")},
            {"site_id": "C", "end_date": ts("2015-06-30")},
        ]
    )
    sites = f.no2_sites(meta).set_index("site_id")
    assert list(sites.index) == ["B", "C"]
    assert sites.loc["B", "no2_start"] == pd.Timestamp("2008-01-01")
    assert pd.isna(sites.loc["B", "no2_end"])  # one record still open -> still running
    assert sites.loc["C", "no2_end"] == pd.Timestamp("2015-06-30")
    assert sites.loc["B", "site_class"] == "roadside"


PRIORITY = ["AURN", "SAQN", "AQE", "LMAM"]
ABERDEEN = {"latitude": 57.1574, "longitude": -2.0943}


def test_same_instrument_in_two_networks_is_merged_even_with_different_codes() -> None:
    ts = pd.Timestamp
    sites = f.no2_sites(
        make_metadata(
            [
                # AURN stopped listing it in 2024; SAQN still runs it
                {"network": "AURN", "site_id": "ABD7", "end_date": ts("2024-12-31"), **ABERDEEN},
                {"network": "SAQN", "site_id": "ABD3", "end_date": pd.NaT, **ABERDEEN},
                # same code as nothing above, far away: kept
                {"network": "LMAM", "site_id": "ABD7", "latitude": 51.0, "longitude": -1.0},
            ]
        )
    )
    kept, mapping = f.merge_duplicate_sites(sites, PRIORITY)
    assert sorted(kept["network"]) == ["AURN", "LMAM"]
    aurn = kept[kept["network"] == "AURN"].iloc[0]
    assert pd.isna(aurn["no2_end"])  # period is the union of both copies
    saqn_row = mapping[mapping["network"] == "SAQN"].iloc[0]
    assert (saqn_row["canon_network"], saqn_row["canon_site_id"]) == ("AURN", "ABD7")


def test_nearby_sites_in_the_same_network_are_not_merged() -> None:
    sites = f.no2_sites(make_metadata([{"site_id": "A", **ABERDEEN}, {"site_id": "B", **ABERDEEN}]))
    kept, _ = f.merge_duplicate_sites(sites, PRIORITY)
    assert len(kept) == 2  # co-located instruments in one network are separate series


def bath_like_sites(new_start: str) -> pd.DataFrame:
    """An old roadside site closing mid-2019 and a new one ~65 m away."""
    ts = pd.Timestamp
    return f.no2_sites(
        make_metadata(
            [
                {"site_id": "OLD", "latitude": 51.3890, "longitude": -2.3540,
                 "start_date": ts("1996-11-18"), "end_date": ts("2019-06-11")},
                {"site_id": "NEW", "latitude": 51.3895, "longitude": -2.3536,
                 "start_date": ts(new_start), "end_date": pd.NaT},
            ]
        )
    )  # fmt: skip


def test_relocation_detected_when_new_site_opens_soon_after_old_closes() -> None:
    reloc = f.find_relocations(bath_like_sites("2019-10-24"))
    assert len(reloc) == 1
    r = reloc.iloc[0]
    assert (r["old_site_id"], r["new_site_id"]) == ("OLD", "NEW")
    assert 50 <= r["distance_m"] <= 80


def test_no_relocation_when_gap_is_too_long() -> None:
    assert f.find_relocations(bath_like_sites("2021-01-01")).empty


def test_chain_relocations_joins_pairs_into_one_series() -> None:
    reloc = f.find_relocations(bath_like_sites("2019-10-24"))
    keys = [("AURN", "OLD"), ("AURN", "NEW"), ("AURN", "OTHER")]
    series = f.chain_relocations(keys, reloc)
    assert sorted(map(len, series)) == [1, 2]


def test_sites_near_uses_radius_and_reports_distance() -> None:
    sites = f.no2_sites(
        make_metadata(
            [
                {"site_id": "NEAR", "latitude": 52.48, "longitude": -1.90},
                # 0.05 deg latitude ~ 5.6 km north: outside a 5 km radius
                {"site_id": "FAR", "latitude": 52.53, "longitude": -1.90},
            ]
        )
    )
    areas = pd.DataFrame(
        [{"area_id": "birmingham", "area_type": "zone", "lat": 52.48, "lon": -1.90}]
    )
    near = f.sites_near(sites, areas, radius_km=5.0)
    assert list(near["site_id"]) == ["NEAR"]
    assert near.loc[0, "distance_km"] == 0.0
    assert near.loc[0, "area_id"] == "birmingham"


# --- years and verdicts ----------------------------------------------------


def test_usable_years_applies_capture_threshold() -> None:
    annual = pd.DataFrame(
        {
            "network": "AURN",
            "site_id": "A",
            "year": [2017, 2018, 2019],
            "capture": [0.75, 0.74, 0.99],
        }
    )
    out = f.usable_years(annual)
    assert out.loc[0, "usable_years"] == [2017, 2019]  # exactly 75% counts, 74% does not


def test_year_containing_the_date_is_neither_before_nor_after() -> None:
    years = [2019, 2020, 2021, 2022]
    date = dt.date(2021, 3, 15)
    assert f.years_before(years, date) == 2
    assert f.years_after(years, date) == 1


@pytest.mark.parametrize(
    "years,expected",
    [
        ([2016, 2017, 2018, 2022], True),  # 3 before, 1 after
        ([2017, 2018, 2022], False),  # only 2 before
        ([2016, 2017, 2018, 2021], False),  # 2021 is the charging year: no after-years
    ],
)
def test_is_usable(years: list[int], expected: bool) -> None:
    assert f.is_usable(years, pre_end=dt.date(2019, 3, 5), post_start=dt.date(2021, 3, 15)) is (
        expected
    )


def test_verdict_thresholds() -> None:
    assert [f.verdict(n) for n in (0, 1, 2, 5)] == ["NONE", "THIN", "GO", "GO"]
