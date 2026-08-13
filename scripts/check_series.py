"""Time-series integrity check.

Reports everything that stops a CSV from forming a clean, regular series:
missing timestamps, duplicate timestamps, out-of-order rows, off-grid
timestamps, NaNs, and constant runs (a common signature of forward-filled
sensor dropout that does not show up as a NaN).

    python check_series.py data/weather.csv --freq 10min
    python check_series.py data/weather.csv --freq 10min --time-col date

Nothing is modified. Fix the file yourself, or use `--suggest` to print the
Darts/pandas calls that would repair each problem found -- deciding how to fill
a gap is a modelling choice, not a cleanup detail, so it is not automated here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def find_time_col(df: pd.DataFrame, explicit: str | None) -> str:
    """Locate the timestamp column, or fail with the list of candidates."""
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(
                f"--time-col '{explicit}' not found. Columns: {list(df.columns)}"
            )
        return explicit

    for cand in ("date", "Date", "time", "Time", "datetime", "Datetime",
                 "DateTime", "timestamp", "Timestamp"):
        if cand in df.columns:
            return cand

    # Fall back to the first column that parses cleanly as datetime.
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
        except (ValueError, TypeError):
            continue
        if parsed.notna().all():
            return col

    raise SystemExit(
        f"No timestamp column found. Columns: {list(df.columns)}. "
        "Pass --time-col explicitly."
    )


def infer_freq(idx: pd.DatetimeIndex) -> tuple[pd.Timedelta, pd.Series]:
    """Modal spacing between consecutive timestamps, plus the full distribution.

    `pd.infer_freq` returns None on any irregular index, which is precisely the
    case being diagnosed here, so the modal difference is used instead.
    """
    deltas = pd.Series(idx[1:] - idx[:-1])
    counts = deltas.value_counts().sort_values(ascending=False)
    return counts.index[0], counts


def fmt_span(a: pd.Timestamp, b: pd.Timestamp, step: pd.Timedelta) -> str:
    n = int((b - a) / step) + 1
    return f"{a}  ->  {b}   ({n} slot{'s' if n != 1 else ''})"


def group_consecutive(missing: pd.DatetimeIndex,
                      step: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collapse a list of missing timestamps into contiguous runs, so that a
    single multi-day outage reports as one gap rather than thousands."""
    if len(missing) == 0:
        return []
    runs, start, prev = [], missing[0], missing[0]
    for t in missing[1:]:
        if t - prev == step:
            prev = t
            continue
        runs.append((start, prev))
        start = prev = t
    runs.append((start, prev))
    return runs


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--time-col", default=None)
    p.add_argument("--freq", default=None,
                   help="expected spacing, e.g. 10min, 15min, 1h. "
                        "Inferred from the modal gap when omitted.")
    p.add_argument("--max-report", type=int, default=25,
                   help="max gaps/duplicates to list individually")
    p.add_argument("--constant-run", type=int, default=6,
                   help="flag runs of >= N identical consecutive values "
                        "(forward-filled dropout often hides here)")
    p.add_argument("--suggest", action="store_true",
                   help="print the code that would fix each problem found")
    args = p.parse_args()

    df = pd.read_csv(args.path)
    tcol = find_time_col(df, args.time_col)
    print(f"file        : {args.path}")
    print(f"rows        : {len(df):,}")
    print(f"time column : {tcol}")
    print(f"value cols  : {len([c for c in df.columns if c != tcol])}")

    # --- parse timestamps --------------------------------------------------
    ts = pd.to_datetime(df[tcol], errors="coerce")
    n_unparsed = int(ts.isna().sum())
    if n_unparsed:
        print(f"\n[!] {n_unparsed:,} timestamps failed to parse. Examples:")
        for v in df.loc[ts.isna(), tcol].head(5):
            print(f"      {v!r}")
        df, ts = df[ts.notna()].copy(), ts[ts.notna()]

    df = df.assign(**{tcol: ts})
    ordered = df.sort_values(tcol)
    n_unsorted = int((df[tcol].values != ordered[tcol].values).sum())
    idx = pd.DatetimeIndex(ordered[tcol])

    print(f"\nrange       : {idx[0]}  ->  {idx[-1]}")
    print(f"span        : {idx[-1] - idx[0]}")

    # --- expected spacing ---------------------------------------------------
    modal, dist = infer_freq(idx)
    step = pd.Timedelta(args.freq) if args.freq else modal
    print(f"modal gap   : {modal}")
    print(f"using step  : {step}" + ("  (from --freq)" if args.freq else "  (inferred)"))

    print("\nSpacing distribution (top 8):")
    total = int(dist.sum())
    for delta, n in dist.head(8).items():
        marker = "  <-- expected" if delta == step else ""
        print(f"  {str(delta):>20}  {n:>9,}  ({100 * n / total:5.2f}%){marker}")

    problems = []

    # --- ordering -----------------------------------------------------------
    print("\n" + "=" * 68)
    if n_unsorted:
        print(f"[!] {n_unsorted:,} rows are not in chronological order")
        problems.append("unsorted")
    else:
        print("[ok] rows are in chronological order")

    # --- duplicates ---------------------------------------------------------
    dup_mask = idx.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        dup_times = idx[dup_mask].unique()
        print(f"[!] {n_dup_rows:,} rows share {len(dup_times):,} duplicated timestamps")
        problems.append("duplicates")

        vals = [c for c in ordered.columns if c != tcol]
        sub = ordered[dup_mask]
        # Identical duplicates are safe to drop; conflicting ones need a rule.
        conflicting = 0
        for _, grp in sub.groupby(tcol):
            if len(grp[vals].drop_duplicates()) > 1:
                conflicting += 1
        print(f"      identical rows   : {len(dup_times) - conflicting:,}  (safe to drop)")
        print(f"      CONFLICTING rows : {conflicting:,}  (same time, different values)")
        for t in dup_times[: args.max_report]:
            print(f"      {t}")
        if len(dup_times) > args.max_report:
            print(f"      ... and {len(dup_times) - args.max_report:,} more")
    else:
        print("[ok] no duplicate timestamps")

    # --- off-grid timestamps ------------------------------------------------
    origin = idx[0]
    offsets = (idx - origin) % step
    off_grid = idx[offsets != pd.Timedelta(0)]
    if len(off_grid):
        print(f"[!] {len(off_grid):,} timestamps are off the {step} grid "
              f"(anchored at {origin})")
        problems.append("off-grid")
        for t in off_grid[: args.max_report]:
            print(f"      {t}")
        if len(off_grid) > args.max_report:
            print(f"      ... and {len(off_grid) - args.max_report:,} more")
    else:
        print(f"[ok] all timestamps sit on the {step} grid")

    # --- missing timestamps -------------------------------------------------
    full = pd.date_range(idx[0], idx[-1], freq=step)
    missing = full.difference(idx.unique())
    expected, present = len(full), len(idx.unique())
    if len(missing):
        runs = group_consecutive(missing, step)
        print(f"[!] {len(missing):,} missing timestamps in {len(runs):,} gap(s) "
              f"-- {100 * len(missing) / expected:.3f}% of the expected grid")
        problems.append("missing")
        runs_sorted = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)
        print(f"\n      Largest gaps:")
        for a, b in runs_sorted[: args.max_report]:
            print(f"      {fmt_span(a, b, step)}")
        if len(runs) > args.max_report:
            print(f"      ... and {len(runs) - args.max_report:,} more gaps")
    else:
        print(f"[ok] no missing timestamps ({expected:,} slots, all present)")

    print(f"\n      expected slots : {expected:,}")
    print(f"      present        : {present:,}")
    print(f"      coverage       : {100 * present / expected:.4f}%")

    # --- NaNs ---------------------------------------------------------------
    vals = [c for c in ordered.columns if c != tcol]
    nan_counts = ordered[vals].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols):
        print(f"\n[!] NaNs present in {len(nan_cols)} of {len(vals)} value columns")
        problems.append("nans")
        for c, n in nan_cols.sort_values(ascending=False).head(args.max_report).items():
            print(f"      {c:<28} {n:>9,}  ({100 * n / len(ordered):5.2f}%)")
    else:
        print("\n[ok] no NaNs in the value columns")

    # --- constant runs ------------------------------------------------------
    # A sensor that drops out and is forward-filled leaves no NaN, only a flat
    # line. Worth surfacing because it corrupts a forecast silently.
    flagged = []
    for c in vals:
        s = ordered[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        grp = (s != s.shift()).cumsum()
        longest = int(s.groupby(grp).transform("size").max())
        if longest >= args.constant_run:
            flagged.append((c, longest))
    if flagged:
        print(f"\n[?] longest run of identical consecutive values "
              f"(>= {args.constant_run} flagged):")
        for c, n in sorted(flagged, key=lambda x: -x[1])[: args.max_report]:
            print(f"      {c:<28} {n:>9,} steps")
        print("      Long flat runs often mean forward-filled sensor dropout;")
        print("      they will not appear as missing data but do distort training.")
    else:
        print(f"\n[ok] no constant runs of {args.constant_run}+ steps")

    # --- verdict ------------------------------------------------------------
    print("\n" + "=" * 68)
    if not problems:
        print("No structural problems found. A regular series can be built directly.")
        sys.exit(0)

    print(f"Problems found: {', '.join(problems)}")

    if args.suggest:
        print("\nSuggested repairs (review each -- these are modelling decisions):\n")
        if "unsorted" in problems:
            print("  # chronological order")
            print(f"  df = df.sort_values('{tcol}')\n")
        if "duplicates" in problems:
            print("  # drop exact duplicates, then decide how to resolve conflicts")
            print("  df = df.drop_duplicates()")
            print(f"  df = df.groupby('{tcol}', as_index=False).mean(numeric_only=True)")
            print("  # ...or .first() / .last() -- averaging conflicting readings is")
            print("  # a choice that must be stated in the paper\n")
        if "off-grid" in problems:
            print("  # snap to the grid, or resample")
            print(f"  df['{tcol}'] = df['{tcol}'].dt.round('{args.freq or step}')\n")
        if "missing" in problems or "nans" in problems:
            print("  # build a regular series; Darts fills gaps with NaN, which is")
            print("  # what you want -- then choose an explicit imputation")
            print("  from darts import TimeSeries")
            print("  from darts.dataprocessing.transformers import MissingValuesFiller")
            print("  series = TimeSeries.from_dataframe(")
            print(f"      df, time_col='{tcol}', value_cols=[c for c in df.columns"
                  f" if c != '{tcol}'],")
            print(f"      fill_missing_dates=True, freq='{args.freq or step}',")
            print("  )")
            print("  series = MissingValuesFiller().transform(series)  # linear interp")
            print("  # For gaps of many hours, interpolation invents data. Consider")
            print("  # truncating to the longest clean stretch instead, and report")
            print("  # which you did.\n")
    else:
        print("Re-run with --suggest to print repair code for each.")

    sys.exit(1)


if __name__ == "__main__":
    main()
