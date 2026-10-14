"""
Derive on-track overtakes per corner for Circuit of the Americas (US GP)
from F1 timing/position data via FastF1, and write overtakes-cota.json.

Method (one race):
  1. Reference lap (fastest lap telemetry) gives the track line: X, Y and
     distance-from-line in metres. Corner apexes come from FastF1 circuit info.
  2. Every car's ~4 Hz position samples are snapped to the nearest point on
     the reference line -> distance into the lap. Unwrapping that across laps
     gives cumulative race progress P(t) in metres.
  3. All cars are resampled onto a common 0.5 s grid. A pass is a sign change
     of P_a - P_b for a pair of cars (same-lap only: lapping a backmarker is
     P_a - P_b crossing a full lap, not zero, so it is never counted).
  4. Flicker (order changing and changing back within 3 s) is discarded.
  5. Excluded: either car in the pit window, under SC/VSC/red flag, the passed
     car retiring within 90 s, or after the winner takes the flag.
     Kept but tagged: lap 1 ("lap1") and the lap after a SC/VSC ends ("restart").
  6. Each pass is attributed to a corner: the corner whose apex it completed
     within EXIT_WINDOW metres after, otherwise the next corner ahead
     (a DRS pass finished on the straight counts for the corner it was into).

Usage: python scripts/overtakes.py [--years 2022 2023 2024 2025] [--out overtakes-cota.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np

EVENT = "United States Grand Prix"
GRID_DT = 0.5          # s, common time grid
FLICKER_S = 3.0        # s, order swaps reverted within this are noise
PIT_BUFFER_IN = 15.0   # s before pit entry treated as pit-affected
PIT_BUFFER_OUT = 20.0  # s after pit exit treated as pit-affected
RETIRE_WINDOW = 90.0   # s, passing a car about to retire is not a pass
EXIT_WINDOW = 80.0     # m past an apex still credited to that corner
NEUTRAL = {"4", "5", "6", "7"}  # SC, red, VSC, VSC ending


# ---------------------------------------------------------------- pure logic

def unwrap_progress(d: np.ndarray, t: np.ndarray, L: float) -> np.ndarray:
    """Cumulative progress (m) from distance-into-lap samples d in [0, L).

    Starts negative for cars gridded behind the line. Jumps that are
    physically impossible (snap glitches between nearby track sections) are
    replaced by the previous step's speed.
    """
    p = np.empty_like(d, dtype=float)
    p[0] = d[0] - L if d[0] > L / 2 else d[0]
    v_prev = 0.0
    for k in range(1, len(d)):
        dt = max(t[k] - t[k - 1], 1e-3)
        step = d[k] - d[k - 1]
        if step < -L / 2:
            step += L
        elif step > L / 2:
            step -= L
        if abs(step) > 110.0 * dt + 30.0 or step < -30.0:   # >~400 km/h or big reverse
            step = v_prev * dt
        p[k] = p[k - 1] + step
        v_prev = step / dt
    return p


def find_swaps(t: np.ndarray, pa: np.ndarray, pb: np.ndarray, flicker=FLICKER_S):
    """Return list of (t_cross, overtaker) where overtaker is 'a' or 'b'.

    pa/pb may contain NaN where a car has no data; those samples break runs.
    """
    diff = pa - pb
    ok = ~np.isnan(diff)
    events = []
    last = None            # index of last non-zero sample in the current unbroken run
    for k in range(len(diff)):
        if not ok[k]:
            last = None    # data gap breaks the run
            continue
        if diff[k] == 0:
            continue       # level: wait for the next non-zero sample
        if last is not None and np.sign(diff[k]) != np.sign(diff[last]):
            d0, d1 = diff[last], diff[k]
            tc = t[last] + d0 / (d0 - d1) * (t[k] - t[last])
            events.append((tc, "a" if d1 > 0 else "b"))
        last = k
    # drop flicker: a swap undone within `flicker` seconds
    out, i = [], 0
    while i < len(events):
        if i + 1 < len(events) and events[i + 1][0] - events[i][0] < flicker \
                and events[i + 1][1] != events[i][1]:
            i += 2
            continue
        out.append(events[i])
        i += 1
    return out


def assign_corner(d: float, apex: list[float], labels: list[str], L: float,
                  exit_window=EXIT_WINDOW) -> str:
    """Corner credited for a pass completed at lap distance d (m)."""
    # completed shortly after an apex -> that corner
    best_after = None
    for a, lab in zip(apex, labels):
        after = (d - a) % L
        gap_to_next = min(((b - a) % L) or L for b in apex)
        win = min(exit_window, 0.5 * gap_to_next)
        if after <= win and (best_after is None or after < best_after[0]):
            best_after = (after, lab)
    if best_after:
        return best_after[1]
    # otherwise the next corner ahead
    ahead = min(zip(apex, labels), key=lambda al: (al[0] - d) % L)
    return ahead[1]


def pit_windows(ins: list[float], outs: list[float]) -> list[tuple[float, float]]:
    """Pair each pit entry with the next pit exit; buffer both ends."""
    iv = []
    for pin in ins:
        pout = next((o for o in outs if o > pin), float("inf"))
        iv.append((pin - PIT_BUFFER_IN, pout + PIT_BUFFER_OUT))
    return iv


def resample(grid: np.ndarray, t: np.ndarray, v: np.ndarray, max_gap=5.0) -> np.ndarray:
    """Linear resample, NaN outside the data and across gaps > max_gap s."""
    g = np.interp(grid, t, v, left=np.nan, right=np.nan)
    j = np.searchsorted(t, grid)
    j = np.clip(j, 1, len(t) - 1)
    g[(t[j] - t[j - 1]) > max_gap] = np.nan
    return g


def lap_chart_swaps(pos: dict, grid_pos: dict, skip: dict, retired_from: dict):
    """Passes from the official lap chart.

    pos[drv][lap] = official position at the end of that lap.
    grid_pos[drv] = starting position (order before lap 1).
    skip[drv] = laps where that car's position change is pit-induced (in/out laps).
    retired_from[drv] = first lap on which a non-finisher was slowing to retire.
    Returns [(lap, winner, loser)] for every pair whose relative order flipped
    between consecutive laps. Lapping a backmarker never flips position order,
    so it is never counted.
    """
    out = []
    laps = sorted({n for d in pos.values() for n in d})
    for n in laps:
        prev = grid_pos if n == 1 else {d: p[n - 1] for d, p in pos.items() if n - 1 in p}
        cur = {d: p[n] for d, p in pos.items() if n in p}
        live = sorted(d for d in cur if d in prev)
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if (prev[a] < prev[b]) == (cur[a] < cur[b]):
                    continue
                win, lose = (a, b) if cur[a] < cur[b] else (b, a)
                if n in skip.get(a, ()) or n in skip.get(b, ()):
                    continue
                if n >= retired_from.get(lose, 10 ** 9):
                    continue
                out.append((n, win, lose))
    return out


def in_intervals(x: float, iv: list[tuple[float, float]]) -> bool:
    return any(a <= x <= b for a, b in iv)


# ---------------------------------------------------------- FastF1 plumbing

def td(x) -> float:
    return x.total_seconds() if x is not None and not (isinstance(x, float) and math.isnan(x)) \
        and str(x) != "NaT" else float("nan")


def analyse_year(year: int) -> dict:
    import fastf1
    from scipy.spatial import cKDTree

    import time
    for attempt in range(2):
        s = fastf1.get_session(year, EVENT, "R")
        s.load(laps=True, telemetry=True, weather=False, messages=True)
        try:
            if len(s.laps) and s.pos_data:
                break
        except Exception:
            pass
        print(f"{year}: data incomplete, retry {attempt + 1} in 60s", flush=True)
        time.sleep(60)
    else:
        raise RuntimeError(f"{year}: timing data unavailable after retries")
    ci = s.get_circuit_info()

    ref = s.laps.pick_fastest().get_telemetry()
    rx, ry, rd = ref["X"].to_numpy(float), ref["Y"].to_numpy(float), ref["Distance"].to_numpy(float)
    L = float(rd[-1])
    tree = cKDTree(np.c_[rx, ry])

    corners = ci.corners.sort_values("Distance")
    apex = [float(v) for v in corners["Distance"]]
    labels = [f"{int(n)}{l or ''}" for n, l in zip(corners["Number"], corners["Letter"])]

    laps = s.laps
    res = s.results
    status = {str(r.DriverNumber): str(r.Status) for r in res.itertuples()}
    abbr = {str(r.DriverNumber): str(r.Abbreviation) for r in res.itertuples()}

    winner = str(res.sort_values("Position").iloc[0].DriverNumber)
    t_end = np.nanmax([td(x) for x in laps.pick_drivers(winner)["Time"]])
    t_start = np.nanmin([td(x) for x in laps[laps["LapNumber"] == 1]["LapStartTime"]])
    med_lap = float(np.nanmedian([td(x) for x in laps["LapTime"]]))

    # neutralisation and restart windows from track status
    ts = s.track_status
    neutral, restart = [], []
    rows = [(td(r.Time), str(r.Status)) for r in ts.itertuples()]
    for i, (t0, st) in enumerate(rows):
        t1 = rows[i + 1][0] if i + 1 < len(rows) else t_end
        if st in NEUTRAL:
            neutral.append((t0, t1))
            nxt = rows[i + 1][1] if i + 1 < len(rows) else "1"
            if nxt not in NEUTRAL:
                restart.append((t1, t1 + med_lap))

    # per-driver pit windows, lap-1 end, last data time
    pit, lap1_end, last_t = {}, {}, {}
    for drv in status:
        dl = laps.pick_drivers(drv)
        ins = sorted(x for x in (td(v) for v in dl["PitInTime"]) if not math.isnan(x))
        outs = sorted(x for x in (td(v) for v in dl["PitOutTime"]) if not math.isnan(x) and x > t_start)
        pit[drv] = pit_windows(ins, outs)
        l1 = dl[dl["LapNumber"] == 1]
        lap1_end[drv] = td(l1.iloc[0].Time) if len(l1) else t_start + med_lap
        last_t[drv] = np.nanmax([td(x) for x in dl["Time"]]) if len(dl) else t_start

    grid = np.arange(t_start, t_end, GRID_DT)
    P, D = {}, {}
    for drv, pos in s.pos_data.items():
        drv = str(drv)
        if drv not in status:
            continue
        pt = pos["SessionTime"].dt.total_seconds().to_numpy()
        m = (pt >= t_start - 5) & (pt <= t_end + 5) & (pos["Status"].to_numpy() == "OnTrack")
        if m.sum() < 50:
            continue
        pt = pt[m]
        _, idx = tree.query(np.c_[pos["X"].to_numpy(float)[m], pos["Y"].to_numpy(float)[m]])
        d = rd[idx]
        prog = unwrap_progress(d, pt, L)
        g = resample(grid, pt, prog)
        g[grid > last_t[drv] + 5] = np.nan
        P[drv] = g

    # ---- count passes from the official lap chart
    pos, skip, retired_from, lap_t = {}, {}, {}, {}
    for drv in status:
        dl = laps.pick_drivers(drv)
        pos[drv], skip[drv], lap_t[drv] = {}, set(), {}
        for r in dl.itertuples():
            n = int(r.LapNumber)
            if not (isinstance(r.Position, float) and math.isnan(r.Position)):
                pos[drv][n] = float(r.Position)
            if not math.isnan(td(r.PitInTime)):
                skip[drv].add(n)
            if not math.isnan(td(r.PitOutTime)) and n > 1:
                skip[drv].add(n)
            lap_t[drv][n] = (td(r.LapStartTime), td(r.Time))
        st = status[drv]
        if not (st == "Finished" or "Lap" in st) and pos[drv]:
            retired_from[drv] = max(pos[drv])          # its final, slowing lap
    grid_pos = {}
    for r in res.itertuples():
        g = float(r.GridPosition) if not math.isnan(float(r.GridPosition)) else 0.0
        grid_pos[str(r.DriverNumber)] = g if g > 0 else 99.0   # pit-lane starters at the back

    passes, dropped, unlocated = [], {"neutral": 0}, 0
    for n, win, lose in lap_chart_swaps(pos, grid_pos, skip, retired_from):
        # ---- locate it on track with position data
        spans = [lap_t[d].get(n, (np.nan, np.nan)) for d in (win, lose)]
        lo = np.nanmin([s0 for s0, _ in spans] + [np.inf]) - 5
        hi = np.nanmax([s1 for _, s1 in spans] + [-np.inf]) + 5
        tc, d = None, None
        if win in P and lose in P and np.isfinite(lo) and np.isfinite(hi):
            m = (grid >= lo) & (grid <= hi)
            ev = [e for e in find_swaps(grid[m], P[win][m], P[lose][m], flicker=0.0) if e[1] == "a"]
            if ev:
                tc = ev[-1][0]
                d = float(np.interp(tc, grid, P[win])) % L
        t_ref = tc if tc is not None else spans[0][0]
        if t_ref is not None and not math.isnan(t_ref) and in_intervals(t_ref, neutral):
            dropped["neutral"] += 1
            continue
        if d is None:
            unlocated += 1
        cat = "lap1" if n == 1 else \
              "restart" if (t_ref is not None and in_intervals(t_ref, restart)) else "green"
        passes.append({"lap": n, "turn": assign_corner(d, apex, labels, L) if d is not None else "?",
                       "cat": cat, "by": abbr.get(win, win), "on": abbr.get(lose, lose),
                       "d": round(d) if d is not None else None})
    dropped["unlocated"] = unlocated

    passes.sort(key=lambda p: (p["lap"], p["d"] if p["d"] is not None else -1))
    rot = math.radians(float(ci.rotation))
    def rotxy(x, y):
        return [round(x * math.cos(rot) - y * math.sin(rot)), round(x * math.sin(rot) + y * math.cos(rot))]
    step = max(1, len(rx) // 300)
    return {
        "lapLength": round(L),
        "track": [rotxy(x, y) for x, y in zip(rx[::step], ry[::step])],
        "corners": [{"turn": lab, "d": round(a), "xy": rotxy(float(x), float(y))}
                    for lab, a, x, y in zip(labels, apex, corners["X"], corners["Y"])],
        "passes": passes,
        "dropped": dropped,
    }


def combine(results: dict, out_path: str) -> int:
    """Merge per-year results into one file and print the summary tables."""
    import os
    out = {"circuit": "Circuit of the Americas", "event": EVENT,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "method": "Passes counted from the official lap chart (relative order flips between "
                     "laps); located on track with FastF1 position data. Pit in/out laps, "
                     "SC/VSC/red and retiring cars excluded; lap 1 and restart laps tagged.",
           "races": {}}
    for y in sorted(results):
        r = dict(results[y])
        if "track" not in out:
            out["track"], out["corners"], out["lapLength"] = r["track"], r["corners"], r["lapLength"]
        for k in ("track", "corners", "lapLength"):
            r.pop(k, None)
        out["races"][str(y)] = r

    n_races = len(out["races"])
    if not n_races:
        print("No races analysed.")
        return 1
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    lines = ["| Year | Green | Lap 1 | Restart | Under SC (dropped) / no location |", "|---|---|---|---|---|"]
    by_turn, by_turn_all, tot_g, tot_all = {}, {}, 0, 0
    for y, r in out["races"].items():
        c = {k: sum(p["cat"] == k for p in r["passes"]) for k in ("green", "lap1", "restart")}
        dr = r["dropped"]
        lines.append(f"| {y} | {c['green']} | {c['lap1']} | {c['restart']} | {dr['neutral']} / {dr['unlocated']} |")
        tot_g += c["green"]; tot_all += sum(c.values())
        for p in r["passes"]:
            by_turn_all[p["turn"]] = by_turn_all.get(p["turn"], 0) + 1
            if p["cat"] == "green":
                by_turn[p["turn"]] = by_turn.get(p["turn"], 0) + 1
    lines += ["", f"Avg per race ({n_races} races): {tot_g / n_races:.1f} green-flag, "
                  f"{tot_all / n_races:.1f} incl. lap 1 and restarts", "",
              "| Turn | Green total | Green avg/race | All avg/race |", "|---|---|---|---|"]
    lines += [f"| T{t} | {by_turn.get(t, 0)} | {by_turn.get(t, 0) / n_races:.1f} | {n / n_races:.1f} |"
              for t, n in sorted(by_turn_all.items(), key=lambda kv: -by_turn.get(kv[0], 0))]
    summary = "\n".join(lines)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## COTA overtakes\n\n" + summary + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[])
    ap.add_argument("--part", help="write a single-year result here instead of combining")
    ap.add_argument("--combine", nargs="+", help="merge these part files")
    ap.add_argument("--out", default="overtakes-cota.json")
    ap.add_argument("--cache", default=".fastf1-cache")
    args = ap.parse_args(argv)

    if args.combine:
        results = {}
        for path in args.combine:
            with open(path) as f:
                part = json.load(f)
            results[int(part["year"])] = part["result"]
        return combine(results, args.out)

    import os, fastf1
    os.makedirs(args.cache, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    results = {}
    for y in args.years:
        try:
            results[y] = analyse_year(y)
        except Exception:
            import traceback
            print(f"=== {y} FAILED ===", flush=True)
            traceback.print_exc(file=sys.stdout)
    if args.part:
        if len(results) != 1:
            return 1
        (y, r), = results.items()
        with open(args.part, "w") as f:
            json.dump({"year": y, "result": r}, f, separators=(",", ":"))
        return 0
    return combine(results, args.out)


if __name__ == "__main__":
    sys.exit(main())
