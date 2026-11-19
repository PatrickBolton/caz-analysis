"""Download and load hourly NO2, PM2.5 and modelled weather for the study sites.

Run from the project root, after scripts/build_database.py:
    uv run --env-file .env python scripts/load_hourly.py               # what is missing
    uv run --env-file .env python scripts/load_hourly.py --limit 5     # try a few first
    uv run --env-file .env python scripts/load_hourly.py --retry    # not_found and rejected
    uv run --env-file .env python scripts/load_hourly.py --reload      # everything again

The site-years come from the view aq.study_site_years. Each one is downloaded
once (cached in data/raw/openair/hourly/), validated, and loaded in its own
transaction, with the outcome recorded in obs.hourly_loads. A rerun skips
site-years already loaded, so an interrupted run can simply be started again.
Server outages are not recorded, so those site-years are retried next time.
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pandera.errors
from sqlalchemy import text

from caz.db import get_engine
from caz.hourly import FileContentError, load_site_year, read_site_year, record_outcome
from caz.openair import NETWORKS, DownloadError, NotFoundOnServerError, download, site_year_url

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "openair" / "hourly"
PAUSE_S = 0.2  # between downloads, to be polite to the servers

TODO = text("""
    SELECT y.network, y.site_id, y.provider_code, y.year
    FROM aq.study_site_years y
    LEFT JOIN obs.hourly_loads l USING (network, site_id, year)
    WHERE l.status IS NULL
       OR (:retry AND l.status IN ('not_found', 'rejected'))
       OR :reload
    ORDER BY y.network, y.site_id, y.year
""")

SUMMARY = """
    SELECT status, count(*) AS site_years, sum(n_hours) AS hours, sum(n_no2) AS no2_hours
    FROM obs.hourly_loads GROUP BY status ORDER BY status
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="load at most this many site-years")
    parser.add_argument(
        "--retry", action="store_true", help="try not_found and rejected site-years again"
    )
    parser.add_argument("--reload", action="store_true", help="re-download and reload all")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        params = {"retry": args.retry, "reload": args.reload}
        todo = pd.read_sql(TODO, conn, params=params)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} site-years to load")

    outcomes: Counter[str] = Counter()
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        network = NETWORKS[row.network]
        url = site_year_url(network, row.site_id, row.year, row.provider_code)
        dest = CACHE / row.network / f"{row.site_id}_{row.year}.RData"
        cached = dest.exists() and not args.reload
        try:
            download(url, dest, refresh=args.reload)
            hourly = read_site_year(dest, row.site_id, row.year)
        except NotFoundOnServerError as e:
            record_outcome(engine, row.network, row.site_id, row.year, "not_found", url, str(e))
            outcomes["not_found"] += 1
        except (FileContentError, pandera.errors.SchemaError, ValueError) as e:
            detail = str(e).splitlines()[0][:500]
            record_outcome(engine, row.network, row.site_id, row.year, "rejected", url, detail)
            outcomes["rejected"] += 1
            print(f"  rejected {row.network} {row.site_id} {row.year}: {detail}")
        except DownloadError as e:  # outage: not recorded, so retried next run
            outcomes["download_failed"] += 1
            print(f"  download failed {row.network} {row.site_id} {row.year}: {e}")
        else:
            load_site_year(engine, row.network, row.site_id, row.year, hourly, url)
            outcomes["loaded"] += 1
        if not cached:
            time.sleep(PAUSE_S)
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {dict(outcomes)}")

    with engine.connect() as conn:
        print("\nobs.hourly_loads:")
        print(pd.read_sql(SUMMARY, conn).to_string(index=False))


if __name__ == "__main__":
    main()
