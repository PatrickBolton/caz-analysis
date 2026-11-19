"""Study windows of the Liu et al. (2023) replication (no database needed)."""

import pandas as pd

from caz.replication import PAPER_WINDOW, PLACEBO_2019


def test_paper_window_uses_complete_weeks() -> None:
    # 1 Jan 2021 is a Friday and 31 May 2022 a Tuesday
    assert PAPER_WINDOW.first_full_week == pd.Timestamp("2021-01-04")
    assert PAPER_WINDOW.last_full_week == pd.Timestamp("2022-05-23")
    assert PAPER_WINDOW.treatment_week.day_name() == "Monday"


def test_placebo_window_is_two_years_earlier_and_before_covid() -> None:
    assert PLACEBO_2019.first_full_week == pd.Timestamp("2019-01-07")
    assert PLACEBO_2019.last_full_week == pd.Timestamp("2019-12-23")
    assert PLACEBO_2019.treatment_week == pd.Timestamp("2019-05-27")
    assert PLACEBO_2019.end < pd.Timestamp("2020-03-01").date()
