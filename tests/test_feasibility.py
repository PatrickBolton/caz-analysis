"""Tests for caz.feasibility and the offline parts of caz.openair."""

import datetime as dt

import pandas as pd
import pytest
from shapely.geometry import box

from caz import feasibility as f
from caz.openair import (
    NETWORKS,
    MetadataFormatError,
    annual_summary_url,
    metadata_url,
    normalise_metadata,
    parse_dates,
)


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


def test_parse_dates_accepts_ongoing_and_blank_end_dates() -> None:
    out = parse_dates(pd.Series(["2019-06-11", "ongoing", "Ongoing", "", None]), allow_open=True)
    assert out.iloc[0] == pd.Timestamp("2019-06-11")
    assert out.iloc[1:].isna().all()


@pytest.mark.parametrize("bad", ["11/06/2019", "2019-13-01", "closed"])
def test_parse_dates_rejects_unknown_formats(bad: str) -> None:
    # a closed site must never silently become "still running"
    with pytest.raises(MetadataFormatError):
        parse_dates(pd.Series(["2019-06-11", bad]), allow_open=True)


def test_parse_dates_rejects_missing_start_dates() -> None:
    with pytest.raises(MetadataFormatError):
        parse_dates(pd.Series(["2019-06-11", "ongoing"]), allow_open=False)


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


def test_copy_joins_the_nearest_kept_site_not_the_first() -> None:
    m = 1 / 60_600  # ~1 m of longitude at Aberdeen's latitude, in degrees
    lat, lon = ABERDEEN["latitude"], ABERDEEN["longitude"]
    sites = f.no2_sites(
        make_metadata(
            [
                {"network": "AURN", "site_id": "A1", "latitude": lat, "longitude": lon},
                {"network": "AURN", "site_id": "A2", "latitude": lat, "longitude": lon + 8 * m},
                # 7 m from A1, 1 m from A2: both within 10 m, A2 is nearer
                {"network": "SAQN", "site_id": "S", "latitude": lat, "longitude": lon + 7 * m},
            ]
        )
    )
    _, mapping = f.merge_duplicate_sites(sites, PRIORITY)
    s = mapping[mapping["network"] == "SAQN"].iloc[0]
    assert (s["canon_network"], s["canon_site_id"]) == ("AURN", "A2")


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


# --- combining the pieces ----------------------------------------------------


def test_pool_merged_annual_relabels_copies_and_drops_unknown_sites() -> None:
    mapping = pd.DataFrame(
        {
            "network": ["AURN", "SAQN"],
            "site_id": ["ABD7", "ABD3"],
            "canon_network": ["AURN", "AURN"],
            "canon_site_id": ["ABD7", "ABD7"],
        }
    )
    annual = pd.DataFrame(
        {
            "network": ["AURN", "SAQN", "SAQN", "LMAM"],
            "site_id": ["ABD7", "ABD3", "ABD3", "XYZ"],
            "year": [2019, 2019, 2020, 2020],
            "capture": [0.5, 0.9, 0.9, 0.9],
        }
    )
    pooled = f.pool_merged_annual(annual, mapping)
    assert set(zip(pooled["network"], pooled["site_id"], strict=True)) == {("AURN", "ABD7")}
    assert len(pooled) == 3  # XYZ has no metadata, so no location: dropped
    # 2019: the AURN copy is below 75% but the SAQN copy is not, so the year counts
    assert f.usable_years(pooled).loc[0, "usable_years"] == [2019, 2020]


ZONE_DATES = {"decision_date": dt.date(2019, 3, 5), "charging_start": dt.date(2021, 3, 15)}


def toy_world() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One zone at (52, -2), a donor city 50 km away and a too-close city 10 km away.

    Zone sites: GOOD (usable), OLD + NEW (a relocation, usable only if joined), SHORT.
    """
    areas = pd.DataFrame(
        [
            {"area_id": "zone", "area_type": "zone", "lat": 52.0, "lon": -2.0, **ZONE_DATES},
            {"area_id": "far", "area_type": "control", "lat": 52.45, "lon": -2.0},
            {"area_id": "near", "area_type": "control", "lat": 52.09, "lon": -2.0},
        ]
    )
    pairs = pd.DataFrame(
        [
            ("zone", "zone", "GOOD", "roadside"),
            ("zone", "zone", "OLD", "roadside"),
            ("zone", "zone", "NEW", "roadside"),
            ("zone", "zone", "SHORT", "background"),
            ("far", "control", "FAR1", "roadside"),
            ("near", "control", "NEAR1", "roadside"),
        ],
        columns=["area_id", "area_type", "site_id", "site_class"],
    ).assign(network="AURN", latitude=52.0, longitude=-2.0)
    # SHORT sits 0.02 deg (~2.2 km) north: outside the zone below
    pairs.loc[pairs["site_id"] == "SHORT", "latitude"] = 52.02
    usable = pd.DataFrame(
        {
            "network": "AURN",
            "site_id": ["GOOD", "OLD", "NEW", "SHORT", "FAR1", "NEAR1"],
            "usable_years": [
                [2016, 2017, 2018, 2019, 2020, 2022],
                [2015, 2016, 2017, 2018],  # closed before the zone
                [2022, 2023],  # opened after
                [2018, 2022],
                list(range(2012, 2024)),
                list(range(2012, 2024)),
            ],
        }
    )
    relocations = pd.DataFrame(
        [{"old_network": "AURN", "old_site_id": "OLD", "new_network": "AURN", "new_site_id": "NEW"}]
    )
    zone = box(-2.01, 51.99, -1.99, 52.01)  # ~1.4 x 2.2 km around the zone centre
    pairs = f.locate_sites(f.describe_pairs(pairs, usable, areas), {"zone": zone})
    return pairs, areas, relocations


def test_describe_pairs_counts_years_for_zones_only() -> None:
    pairs, _, _ = toy_world()
    good = pairs.set_index("site_id").loc["GOOD"]
    assert good["years_before_charging"] == 5  # 2016-2020 (2021 is the charging year)
    assert good["years_before_decision"] == 3  # 2016-2018 (2019 is the decision year)
    assert good["years_after_charging"] == 1
    assert bool(good["usable_vs_charging"]) and bool(good["usable_vs_decision"])
    controls = pairs[pairs["area_type"] == "control"]
    assert controls["usable_vs_charging"].isna().all()
    assert str(pairs["usable_vs_charging"].dtype) == "boolean"


def test_summarise_zones_counts_sites_relocations_and_donors() -> None:
    pairs, areas, relocations = toy_world()
    row = f.summarise_zones(pairs, areas, relocations).iloc[0]
    assert row["sites_within_radius"] == 4
    assert row["usable_vs_charging"] == 1  # only GOOD
    assert row["usable_roadside"] == 1 and row["usable_background"] == 0
    assert row["verdict_vs_charging"] == "THIN"
    assert row["usable_vs_charging_if_relocations_joined"] == 2  # GOOD + (OLD+NEW)
    assert row["donor_cities_vs_charging"] == 1  # "near" is under 15 km away
    assert (row["usable_inside"], row["usable_edge"], row["usable_outside"]) == (1, 0, 0)
    assert row["verdict_inside"] == "THIN"


def test_locate_sites_marks_zone_rows_and_leaves_controls_empty() -> None:
    pairs, _, _ = toy_world()
    by_site = pairs.set_index("site_id")
    assert by_site.loc["GOOD", "zone_location"] == "inside"
    assert by_site.loc["SHORT", "zone_location"] == "outside"
    assert by_site.loc["SHORT", "distance_to_zone_m"] > 1000
    assert pairs.loc[pairs["area_type"] == "control", "zone_location"].isna().all()


def test_split_by_location_without_boundary_reports_no_counts() -> None:
    out = f.split_by_location(pd.Series(["unknown", "unknown"]))
    assert out["verdict_inside"] == "NO BOUNDARY"
    assert pd.isna(out["usable_inside"])


def test_flag_relocations_lists_every_successor() -> None:
    pairs = pd.DataFrame({"network": ["AURN"], "site_id": ["OLD"]})
    relocations = pd.DataFrame(
        {
            "old_network": ["AURN", "AURN"],
            "old_site_id": ["OLD", "OLD"],
            "new_network": ["AURN", "LMAM"],
            "new_site_id": ["NEW", "NEW2"],
        }
    )
    mapping = pd.DataFrame(
        {
            "network": ["AURN"],
            "site_id": ["OLD"],
            "canon_network": ["AURN"],
            "canon_site_id": ["OLD"],
        }
    )
    out = f.flag_relocations(pairs, relocations, mapping).iloc[0]
    assert out["replaced_by"] == "AURN:NEW;LMAM:NEW2"
    assert out["merged_copies"] == "" and out["replaces"] == ""
