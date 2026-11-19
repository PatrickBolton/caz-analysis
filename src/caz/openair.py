"""Download and read the openair-format files published by UK air-quality networks.

Each network publishes, in R's file formats:
- a metadata file: one row per site x pollutant, with coordinates, site type and
  the dates each pollutant was measured;
- one small annual summary file per year: every site's annual mean and data capture
  for each pollutant.

These are the same files the R package `openair` (Carslaw & Ropkins, 2012) uses.
URL patterns are taken from openair's source (`R/importUKAQ-utils.R`).
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import rdata
import requests

TIMEOUT_S = 60  # seconds without any data before a request is abandoned
MAX_DOWNLOAD_S = 180  # total time allowed per file (a server can trickle data forever)
DATE_FORMAT = "%Y-%m-%d"  # dates in metadata files
OPEN_END_VALUES = {"ongoing", ""}  # end_date values meaning "still running"

# Older R files do not declare a text encoding; rdata then assumes ASCII and warns.
warnings.filterwarnings("ignore", message="Unknown encoding. Assumed ASCII.")


@dataclass(frozen=True)
class Network:
    """One openair network: where its files live and how they are named."""

    name: str  # our label, e.g. "AURN"
    base_url: str
    abbr: str  # file-name abbreviation used by the network


# Order matters: when the same site appears in several networks, the first wins
# (national network data is the most thoroughly quality-assured).
NETWORKS: dict[str, Network] = {
    n.name: n
    for n in [
        Network("AURN", "https://uk-air.defra.gov.uk/openair/R_data/", "AURN"),
        Network("SAQN", "https://www.scottishairquality.scot/openair/R_data/", "SCOT"),
        Network("AQE", "https://www.airqualityengland.co.uk/assets/openair/R_data/", "AQE"),
        Network("LMAM", "https://uk-air.defra.gov.uk/openair/LMAM/R_data/", "LMAM"),
    ]
}


class DownloadError(RuntimeError):
    """A file could not be downloaded or is not a valid R data file."""


class NotFoundOnServerError(DownloadError):
    """The server answered 404: the file does not exist (as opposed to an outage)."""


class MetadataFormatError(ValueError):
    """A metadata file contains values in a format this code does not understand."""


def metadata_url(network: Network) -> str:
    """URL of a network's site metadata file."""
    return f"{network.base_url}{network.abbr}_metadata.RData"


def site_year_url(
    network: Network, site_id: str, year: int, provider_code: str | None = None
) -> str:
    """URL of one site's hourly data file for one year.

    Locally managed (LMAM) files sit in a subfolder per data provider (the
    metadata's pcode); the other networks have one folder.
    """
    folder = f"{provider_code}/" if network.name == "LMAM" else ""
    if network.name == "LMAM" and not provider_code:
        raise ValueError(f"LMAM site {site_id} needs a provider code")
    return f"{network.base_url}{folder}{site_id}_{year}.RData"


def annual_summary_url(network: Network, year: int) -> str:
    """URL of a network's annual summary statistics file for one year."""
    return f"{network.base_url}summary_annual_{network.abbr}_{year}.rds"


def download(url: str, dest: Path, refresh: bool = False) -> Path:
    """Download `url` to `dest` unless it is already cached there.

    Raises DownloadError if the server does not return a real R data file (for
    example a redirect to a maintenance page, which returns HTTP 200 with HTML).
    """
    if dest.exists() and not refresh:
        return dest
    started = time.monotonic()
    try:
        with requests.get(url, timeout=TIMEOUT_S, stream=True) as response:
            if response.status_code == 404:
                raise NotFoundOnServerError(f"{url}: HTTP 404")
            if response.status_code != 200:
                raise DownloadError(f"{url}: HTTP {response.status_code}")
            chunks = []
            for chunk in response.iter_content(chunk_size=65536):
                chunks.append(chunk)
                if time.monotonic() - started > MAX_DOWNLOAD_S:
                    raise DownloadError(f"{url}: took longer than {MAX_DOWNLOAD_S} s")
            content, final_url = b"".join(chunks), response.url
    except requests.RequestException as e:
        raise DownloadError(f"{url}: {e}") from e
    # R data files are gzip-compressed (magic bytes 1f 8b); an HTML page is not.
    if not content.startswith(b"\x1f\x8b"):
        raise DownloadError(f"{url}: not an R data file (final URL {final_url})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


def read_metadata(path: Path, network: Network) -> pd.DataFrame:
    """Read a metadata .RData file into a tidy table (one row per site x pollutant)."""
    objects = rdata.read_rda(path)
    # Each network's file holds exactly one table; its name differs between networks.
    if len(objects) != 1:
        raise DownloadError(f"{path}: expected one table, found {list(objects)}")
    (table,) = objects.values()
    return normalise_metadata(pd.DataFrame(table), network.name)


def normalise_metadata(raw: pd.DataFrame, network_name: str) -> pd.DataFrame:
    """Standardise raw openair metadata columns and types."""
    df = pd.DataFrame(
        {
            "network": network_name,
            "site_id": raw["site_id"].astype(str).str.upper(),
            "site_name": raw["site_name"].astype(str),
            "location_type": raw["location_type"].astype(str),
            "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
            "parameter": raw["parameter"].astype(str),
            # LMAM only: the data provider, whose subfolder holds the site's files
            "provider_code": (
                raw["pcode"].astype("string") if "pcode" in raw else pd.Series(pd.NA, raw.index)
            ).astype("string"),
            "start_date": parse_dates(raw["start_date"], allow_open=False),
            # "ongoing" and blanks become NaT (= still running)
            "end_date": parse_dates(raw["end_date"], allow_open=True),
        }
    )
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def parse_dates(values: pd.Series, allow_open: bool) -> pd.Series:
    """Parse YYYY-MM-DD dates strictly.

    With allow_open, "ongoing" and blanks become NaT (= still running). Anything
    else that does not parse raises MetadataFormatError, so a change of date
    format in a network's files cannot silently make closed sites look open.
    """
    text = values.astype("string").str.strip()
    is_open = text.isna() | text.str.lower().isin(OPEN_END_VALUES)
    parsed = pd.to_datetime(text.where(~is_open), format=DATE_FORMAT, errors="coerce")
    bad = parsed.isna() & ~is_open
    if bad.any():
        raise MetadataFormatError(f"unrecognised dates: {sorted(set(text[bad]))[:5]}")
    if is_open.any() and not allow_open:
        raise MetadataFormatError(f"{int(is_open.sum())} missing or open-ended start dates")
    return parsed


def read_annual_summary(path: Path, network_name: str, pollutant: str = "NO2") -> pd.DataFrame:
    """Read an annual summary .rds file: one row per site with mean and data capture.

    Data capture is returned as a fraction (0-1) of hours with valid data.
    """
    raw = pd.DataFrame(rdata.read_rds(path))
    mean_col, capture_col = f"{pollutant}.mean", f"{pollutant}.capture"
    if mean_col not in raw.columns:  # network measured no such pollutant that year
        return pd.DataFrame(columns=["network", "site_id", "year", "mean", "capture"])
    df = pd.DataFrame(
        {
            "network": network_name,
            "site_id": raw["code"].astype(str).str.upper(),
            "year": pd.to_numeric(raw["Year"], errors="coerce").astype("Int64"),
            "mean": pd.to_numeric(raw[mean_col], errors="coerce"),
            "capture": pd.to_numeric(raw[capture_col], errors="coerce"),
        }
    )
    return df.dropna(subset=["capture"]).reset_index(drop=True)
