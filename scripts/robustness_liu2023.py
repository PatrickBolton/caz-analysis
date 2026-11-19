"""Robustness checks for the Liu et al. (2023) replication (scripts/replicate_liu2023.py).

Run from the project root (database running, hourly data loaded):
    uv run --env-file .env python scripts/robustness_liu2023.py                 # all checks
    uv run --env-file .env python scripts/robustness_liu2023.py --checks loo placebo

Checks:
- seeds    Random-forest seed: weather normalisation redone with seeds 1-4
           (~10 min per seed the first time; cached).
- loo      Leave-one-donor-out: the estimate with each donor removed in turn.
- placebo  In-place placebo (paper Fig. 14): each donor in turn is treated as if
           it had a zone, with the other donors as its controls. Sites without a
           zone show how large "effects" get by chance, and how often the 95%
           intervals wrongly exclude zero.
- intime   In-time placebo (paper Fig. 13): the same design in 2019, with a fake
           zone from 27 May 2019. A real method should find no effect.
Results are printed and saved to data/processed/replication_liu2023/robustness_*.csv.
"""

import argparse

import pandas as pd

from caz.db import get_engine
from caz.replication import (
    OUT,
    PAPER_WINDOW,
    PLACEBO_2019,
    Window,
    deweathered,
    estimate,
    replication_sites,
    weekly_panel,
)

TREATED = {"BIRR": "roadside", "BMLD": "background"}
SEEDS = [0, 1, 2, 3, 4]


def panel_for(engine, sites: pd.DataFrame, window: Window, seed: int = 0) -> pd.DataFrame:
    hourly = {}
    for site in sites.itertuples(index=False):
        h = deweathered(engine, site, window, seed=seed)
        if h is not None:
            hourly[site.site_id] = h
    return weekly_panel(hourly, window)


def donors_of(sites: pd.DataFrame, site_class: str, panel: pd.DataFrame) -> list[str]:
    is_donor = (sites["role"] == "donor") & (sites["site_class"] == site_class)
    return [d for d in sites.loc[is_donor, "site_id"] if d in panel.columns]


def rows_for(label: dict, run: dict) -> list[dict]:
    return [
        label | {"window": w, "effect": e.absolute, "lower": e.lower, "upper": e.upper,
                 "percent": e.percent, "pre_rmse": run["pre_rmse"]}
        for w, e in run["effects"].items()
    ]  # fmt: skip


def check_seeds(engine, sites) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        print(f"  seed {seed} ...", flush=True)
        panel = panel_for(engine, sites, PAPER_WINDOW, seed)
        for treated, site_class in TREATED.items():
            run = estimate(panel, treated, donors_of(sites, site_class, panel), PAPER_WINDOW)
            rows += rows_for({"treated": treated, "seed": seed}, run)
    df = pd.DataFrame(rows)
    summary = df.groupby(["treated", "window"])["effect"].agg(["mean", "std", "min", "max"])
    print("\nRandom-forest seed (5 seeds), NO2 effect ug/m3:")
    print(summary.round(2).to_string())
    return df


def check_leave_one_out(sites, panel) -> pd.DataFrame:
    rows = []
    for treated, site_class in TREATED.items():
        donors = donors_of(sites, site_class, panel)
        for left_out in [None, *donors]:
            used = [d for d in donors if d != left_out]
            run = estimate(panel, treated, used, PAPER_WINDOW)
            rows += rows_for({"treated": treated, "left_out": left_out or "(none)"}, run)
    df = pd.DataFrame(rows)
    loo = df[df["left_out"] != "(none)"]
    full = df[df["left_out"] == "(none)"].set_index(["treated", "window"])["effect"]
    print("\nLeave-one-donor-out, NO2 effect ug/m3 (full donor pool, then range when each "
          "donor is removed; the most influential donor):")  # fmt: skip
    for (treated, window), g in loo.groupby(["treated", "window"]):
        biggest = g.loc[(g["effect"] - full[(treated, window)]).abs().idxmax()]
        print(f"  {treated} {window:8s} {full[(treated, window)]:+.2f}  range "
              f"{g['effect'].min():+.2f} to {g['effect'].max():+.2f}  (without "
              f"{biggest['left_out']}: {biggest['effect']:+.2f})")  # fmt: skip
    return df


def check_placebo(sites, panel, real: dict) -> pd.DataFrame:
    rows = []
    for _treated, site_class in TREATED.items():
        donors = donors_of(sites, site_class, panel)
        for fake in donors:
            if panel[fake].isna().mean() > 0.1:  # a fake treated site needs most weeks
                continue
            others = [d for d in donors if d != fake]
            run = estimate(panel, fake, others, PAPER_WINDOW)
            rows += rows_for({"site_class": site_class, "fake_treated": fake}, run)
    df = pd.DataFrame(rows)
    df["excludes_zero"] = (df["upper"] < 0) | (df["lower"] > 0)
    print("\nIn-place placebo (fake zone at each donor), NO2 'effect' ug/m3:")
    for (site_class, window), g in df.groupby(["site_class", "window"]):
        treated = next(t for t, c in TREATED.items() if c == site_class)
        r = real[(treated, window)]
        as_large = int((g["effect"].abs() >= abs(r)).sum())
        n, excl = len(g), int(g["excludes_zero"].sum())
        mean, sd = g["effect"].mean(), g["effect"].std()
        print(f"  {site_class:10s} {window:8s} {n} placebo sites: mean {mean:+.2f}, sd {sd:.2f}; "
              f"{excl} of {n} intervals exclude zero; "
              f"|effect| >= {treated}'s {abs(r):.2f} at {as_large} of {n}")  # fmt: skip
    return df


def check_in_time(engine, sites) -> pd.DataFrame:
    panel = panel_for(engine, sites, PLACEBO_2019)
    rows = []
    print("\nIn-time placebo (fake zone from 27 May 2019; 7 months), NO2 'effect' ug/m3:")
    for treated, site_class in TREATED.items():
        if treated not in panel.columns or panel[treated].notna().sum() < 30:
            print(f"  {treated}: no 2019 data (site opened later)")
            continue
        run = estimate(panel, treated, donors_of(sites, site_class, panel), PLACEBO_2019)
        e = run["effects"]["7 months"]
        n_donors, rmse = len(run["donors_used"]), run["pre_rmse"]
        print(f"  {treated}: {e.absolute:+.2f} ({e.lower:+.2f}, {e.upper:+.2f}), "
              f"{n_donors} donors, pre-period RMSE {rmse:.2f}")  # fmt: skip
        rows += rows_for({"treated": treated}, run)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choices = ["seeds", "loo", "placebo", "intime"]
    parser.add_argument("--checks", nargs="+", choices=choices, default=choices)
    args = parser.parse_args()

    engine = get_engine()
    sites = replication_sites(engine)
    panel = panel_for(engine, sites, PAPER_WINDOW)
    real = {}
    for treated, site_class in TREATED.items():
        run = estimate(panel, treated, donors_of(sites, site_class, panel), PAPER_WINDOW)
        real |= {(treated, w): e.absolute for w, e in run["effects"].items()}

    results = {}
    if "loo" in args.checks:
        results["loo"] = check_leave_one_out(sites, panel)
    if "placebo" in args.checks:
        results["placebo"] = check_placebo(sites, panel, real)
    if "intime" in args.checks:
        results["intime"] = check_in_time(engine, sites)
    if "seeds" in args.checks:
        results["seeds"] = check_seeds(engine, sites)
    for name, df in results.items():
        df.to_csv(OUT / f"robustness_{name}.csv", index=False)


if __name__ == "__main__":
    main()
