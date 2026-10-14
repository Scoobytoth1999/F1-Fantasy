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


# ------------------------------------------------------------ analysis core
#
# Both data sources are normalised into one `race` dict:
#   ref_x, ref_y   reference lap line (same units as position samples)
#   ref_d          metres along that line
#   corners        [(label, x, y)] in the same coordinate frame
#   rotation       degrees, for display only
#   abbr           {drv: "VER"}         finished {drv: bool}
#   samples        {drv: (t, x, y)}     on-track position samples, t in seconds
#   laps           {drv: {lap: (start_t, end_t, official_position|None, pit_lap: bool)}}
#   grid_pos       {drv: grid slot}     neutral / restart [(t0, t1)]
#   t_start, t_end race start and winner's finish

def densify(x, y, step_m=1.0, units_per_m=10.0):
    """Resample a polyline to ~1 m spacing; returns x, y, distance (m)."""
    seg = np.hypot(np.diff(x), np.diff(y)) / units_per_m
    d = np.r_[0, np.cumsum(seg)]
    keep = np.r_[True, seg > 0]
    x, y, d = x[keep], y[keep], d[keep]
    dd = np.arange(0, d[-1], step_m)
    return np.interp(dd, d, x), np.interp(dd, d, y), dd


def analyse(race: dict) -> dict:
    from scipy.spatial import cKDTree
    rx, ry, rd = race["ref_x"], race["ref_y"], race["ref_d"]
    L = float(rd[-1])
    tree = cKDTree(np.c_[rx, ry])

    cor = []
    for lab, x, y in race["corners"]:
        _, i = tree.query([x, y])
        cor.append((float(rd[i]), lab, x, y))
    cor.sort()
    apex = [c[0] for c in cor]
    labels = [c[1] for c in cor]

    t_start, t_end = race["t_start"], race["t_end"]
    grid = np.arange(t_start, t_end, GRID_DT)
    last_t = {drv: max((e for _, e, _, _ in lp.values() if e == e), default=t_start)
              for drv, lp in race["laps"].items()}
    P = {}
    for drv, (pt, px, py) in race["samples"].items():
        m = (pt >= t_start - 5) & (pt <= t_end + 5)
        if m.sum() < 50:
            continue
        pt = pt[m]
        _, idx = tree.query(np.c_[px[m], py[m]])
        g = resample(grid, pt, unwrap_progress(rd[idx], pt, L))
        g[grid > last_t.get(drv, t_end) + 5] = np.nan
        P[drv] = g

    pos, skip, retired_from, lap_t = {}, {}, {}, {}
    for drv, lp in race["laps"].items():
        pos[drv], skip[drv], lap_t[drv] = {}, set(), {}
        for n, (s0, s1, p, pit) in lp.items():
            if p is not None:
                pos[drv][n] = float(p)
            if pit:
                skip[drv].update({n, n + 1})      # in-lap and out-lap
            lap_t[drv][n] = (s0, s1)
        if not race["finished"].get(drv, True) and pos[drv]:
            retired_from[drv] = max(pos[drv])
    abbr = race["abbr"]

    passes, dropped, unlocated = [], {"neutral": 0}, 0
    for n, win, lose in lap_chart_swaps(pos, race["grid_pos"], skip, retired_from):
        spans = [lap_t[d].get(n, (np.nan, np.nan)) for d in (win, lose)]
        lo = np.nanmin([s0 for s0, _ in spans] + [np.inf]) - 5
        hi = np.nanmax([s1 for _, s1 in spans] + [-np.inf]) + 5
        if n == 1:
            lo = t_start - 5
        tc, d = None, None
        if win in P and lose in P and np.isfinite(lo) and np.isfinite(hi):
            m = (grid >= lo) & (grid <= hi)
            ev = [e for e in find_swaps(grid[m], P[win][m], P[lose][m], flicker=0.0) if e[1] == "a"]
            if ev:
                tc = ev[-1][0]
                d = float(np.interp(tc, grid, P[win])) % L
        t_ref = tc if tc is not None else spans[0][0]
        has_t = t_ref is not None and not math.isnan(t_ref)
        if has_t and in_intervals(t_ref, race["neutral"]):
            dropped["neutral"] += 1
            continue
        if d is None:
            unlocated += 1
        cat = "lap1" if n == 1 else "restart" if has_t and in_intervals(t_ref, race["restart"]) else "green"
        passes.append({"lap": n, "turn": assign_corner(d, apex, labels, L) if d is not None else "?",
                       "cat": cat, "by": abbr.get(win, win), "on": abbr.get(lose, lose),
                       "d": round(d) if d is not None else None})
    dropped["unlocated"] = unlocated
    passes.sort(key=lambda p: (p["lap"], p["d"] if p["d"] is not None else -1))

    rot = math.radians(float(race["rotation"]))
    def rotxy(x, y):
        return [round(x * math.cos(rot) - y * math.sin(rot)), round(x * math.sin(rot) + y * math.cos(rot))]
    step = max(1, len(rx) // 300)
    return {
        "lapLength": round(L),
        "track": [rotxy(x, y) for x, y in zip(rx[::step], ry[::step])],
        "corners": [{"turn": lab, "d": round(a), "xy": rotxy(float(x), float(y))} for a, lab, x, y in cor],
        "passes": passes,
        "dropped": dropped,
        "source": race.get("source", ""),
    }


def neutral_windows(events, t_end, med_lap):
    """events: [(t, kind)] with kind 'start' or 'end'. Returns neutral, restart windows."""
    neutral, restart, open_t = [], [], None
    for t, kind in sorted(events):
        if kind == "start" and open_t is None:
            open_t = t
        elif kind == "end" and open_t is not None:
            neutral.append((open_t, t)); restart.append((t, t + med_lap)); open_t = None
    if open_t is not None:
        neutral.append((open_t, t_end))
    return neutral, restart


# ---------------------------------------------------------- source: FastF1

def td(x) -> float:
    return x.total_seconds() if x is not None and not (isinstance(x, float) and math.isnan(x)) \
        and str(x) != "NaT" else float("nan")


def load_fastf1(year: int) -> dict:
    import fastf1
    s = fastf1.get_session(year, EVENT, "R")
    s.load(laps=True, telemetry=True, weather=False, messages=True)
    if not (len(s.laps) and s.pos_data):
        raise RuntimeError(f"{year}: FastF1 timing data unavailable")
    ci = s.get_circuit_info()
    ref = s.laps.pick_fastest().get_telemetry()
    rx, ry, rd = densify(ref["X"].to_numpy(float), ref["Y"].to_numpy(float))
    res, laps = s.results, s.laps
    abbr = {str(r.DriverNumber): str(r.Abbreviation) for r in res.itertuples()}
    finished = {str(r.DriverNumber): (str(r.Status) == "Finished" or "Lap" in str(r.Status))
                for r in res.itertuples()}
    winner = str(res.sort_values("Position").iloc[0].DriverNumber)
    t_end = np.nanmax([td(x) for x in laps.pick_drivers(winner)["Time"]])
    t_start = np.nanmin([td(x) for x in laps[laps["LapNumber"] == 1]["LapStartTime"]])
    med_lap = float(np.nanmedian([td(x) for x in laps["LapTime"]]))

    ev, prev = [], "1"
    for r in s.track_status.itertuples():
        st = str(r.Status)
        if st in NEUTRAL and prev not in NEUTRAL:
            ev.append((td(r.Time), "start"))
        elif st not in NEUTRAL and prev in NEUTRAL:
            ev.append((td(r.Time), "end"))
        prev = st
    neutral, restart = neutral_windows(ev, t_end, med_lap)

    lp = {}
    for drv in abbr:
        lp[drv] = {}
        for r in laps.pick_drivers(drv).itertuples():
            p = None if (isinstance(r.Position, float) and math.isnan(r.Position)) else float(r.Position)
            lp[drv][int(r.LapNumber)] = (td(r.LapStartTime), td(r.Time), p, not math.isnan(td(r.PitInTime)))
    grid_pos = {}
    for r in res.itertuples():
        g = float(r.GridPosition) if not math.isnan(float(r.GridPosition)) else 0.0
        grid_pos[str(r.DriverNumber)] = g if g > 0 else 99.0
    samples = {}
    for drv, pos in s.pos_data.items():
        m = pos["Status"].to_numpy() == "OnTrack"
        samples[str(drv)] = (pos["SessionTime"].dt.total_seconds().to_numpy()[m],
                             pos["X"].to_numpy(float)[m], pos["Y"].to_numpy(float)[m])
    corners = [(f"{int(n)}{l or ''}", float(x), float(y))
               for n, l, x, y in zip(ci.corners["Number"], ci.corners["Letter"], ci.corners["X"], ci.corners["Y"])]
    return dict(ref_x=rx, ref_y=ry, ref_d=rd, corners=corners, rotation=float(ci.rotation),
                abbr=abbr, finished=finished, samples=samples, laps=lp, grid_pos=grid_pos,
                neutral=neutral, restart=restart, t_start=t_start, t_end=t_end, source="FastF1")


# ---------------------------------------------------------- source: OpenF1

OPENF1 = "https://api.openf1.org/v1/"
MULTIVIEWER = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"


def get_json(url: str, tries=5):
    import time, urllib.request, urllib.error
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1-fantasy-overtakes/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                time.sleep(0.4)          # stay under OpenF1's free rate limit
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(5 * (i + 1)); continue
            raise
        except urllib.error.URLError:
            if i < tries - 1:
                time.sleep(5 * (i + 1)); continue
            raise


def iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_openf1(year: int) -> dict:
    sess = get_json(f"{OPENF1}sessions?year={year}&session_name=Race&location=Austin")
    if not sess:
        raise RuntimeError(f"{year}: no OpenF1 race session for Austin")
    k = sess[0]["session_key"]
    circ = get_json(MULTIVIEWER.format(key=sess[0]["circuit_key"], year=year))

    drivers = get_json(f"{OPENF1}drivers?session_key={k}")
    abbr = {str(d["driver_number"]): d["name_acronym"] for d in drivers}
    results = get_json(f"{OPENF1}session_result?session_key={k}")
    finished = {str(r["driver_number"]): not (r.get("dnf") or r.get("dns") or r.get("dsq")) for r in results}
    laps_raw = get_json(f"{OPENF1}laps?session_key={k}")
    pits = get_json(f"{OPENF1}pit?session_key={k}")
    posr = get_json(f"{OPENF1}position?session_key={k}")
    rc = get_json(f"{OPENF1}race_control?session_key={k}")

    # lap windows per driver
    by_drv = {}
    for l in laps_raw:
        by_drv.setdefault(str(l["driver_number"]), {})[int(l["lap_number"])] = l
    durs = [l["lap_duration"] for l in laps_raw if l.get("lap_duration") and not l.get("is_pit_out_lap")]
    med_lap = float(np.median(durs))
    lap_win = {}
    for drv, ls in by_drv.items():
        lap_win[drv] = {}
        for n, l in ls.items():
            s0 = iso(l["date_start"]) if l.get("date_start") else None
            nxt = ls.get(n + 1)
            if nxt and nxt.get("date_start"):
                s1 = iso(nxt["date_start"])
            elif s0 is not None and l.get("lap_duration"):
                s1 = s0 + l["lap_duration"]
            else:
                s1 = None
            if s0 is None and s1 is not None:      # lap 1 often has no start stamp
                s0 = s1 - (l.get("lap_duration") or med_lap * 1.1)
            if s0 is not None and s1 is not None:
                lap_win[drv][n] = (s0, s1)

    # official position at each lap end = latest position update at or before it
    timeline = {}
    for p in sorted(posr, key=lambda p: p["date"]):
        timeline.setdefault(str(p["driver_number"]), []).append((iso(p["date"]), p["position"]))
    grid_pos = {drv: float(tl[0][1]) for drv, tl in timeline.items()}

    def pos_at(drv, t):
        tl = timeline.get(drv, [])
        i = int(np.searchsorted([x[0] for x in tl], t + 1.0, side="right")) - 1
        return float(tl[i][1]) if i >= 0 else None

    pit_laps = {}
    for p in pits:
        if p.get("lap_number"):
            pit_laps.setdefault(str(p["driver_number"]), set()).add(int(p["lap_number"]))

    lp = {}
    for drv, wins in lap_win.items():
        lp[drv] = {n: (s0, s1, pos_at(drv, s1), n in pit_laps.get(drv, ())) for n, (s0, s1) in wins.items()}

    t_start = min(w[1][0] for w in ((d, lap_win[d].get(1)) for d in lap_win) if w[1])
    winner = next(str(r["driver_number"]) for r in results if r.get("position") == 1)
    t_end = max(s1 for _, s1 in lap_win[winner].values())

    ev = []
    for m in rc:
        msg, t = (m.get("message") or "").upper(), iso(m["date"])
        if ("SAFETY CAR DEPLOYED" in msg) or (m.get("flag") == "RED"):
            ev.append((t, "start"))
        elif "VIRTUAL SAFETY CAR ENDING" in msg or "VSC ENDING" in msg:
            ev.append((t + 10, "end"))
        elif "SAFETY CAR IN THIS LAP" in msg:
            ev.append((t + med_lap * 0.8, "end"))
        elif "RESUME" in msg and "RACE" in msg:
            ev.append((t, "end"))
    neutral, restart = neutral_windows(ev, t_end, med_lap)

    # reference line: fastest clean lap's location samples
    fl = min((l for l in laps_raw if l.get("lap_duration") and l.get("date_start") and not l.get("is_pit_out_lap")),
             key=lambda l: l["lap_duration"])
    a = fl["date_start"]
    b = datetime.fromtimestamp(iso(a) + fl["lap_duration"], tz=timezone.utc).isoformat()
    loc = get_json(f"{OPENF1}location?session_key={k}&driver_number={fl['driver_number']}"
                   f"&date>={a}&date<={b}")
    xs = np.array([p["x"] for p in loc], float); ys = np.array([p["y"] for p in loc], float)
    rx, ry, rd = densify(xs, ys)

    samples = {}
    t0 = datetime.fromtimestamp(t_start - 120, tz=timezone.utc).isoformat()
    t1 = datetime.fromtimestamp(t_end + 60, tz=timezone.utc).isoformat()
    for drv in abbr:
        loc = get_json(f"{OPENF1}location?session_key={k}&driver_number={drv}&date>={t0}&date<={t1}")
        pts = [(iso(p["date"]), p["x"], p["y"]) for p in loc if p["x"] or p["y"]]
        if pts:
            arr = np.array(pts, float)
            samples[drv] = (arr[:, 0], arr[:, 1], arr[:, 2])

    corners = [(f"{c['number']}{c.get('letter') or ''}", float(c["trackPosition"]["x"]), float(c["trackPosition"]["y"]))
               for c in circ["corners"]]
    return dict(ref_x=rx, ref_y=ry, ref_d=rd, corners=corners, rotation=float(circ.get("rotation", 0)),
                abbr=abbr, finished=finished, samples=samples, laps=lp, grid_pos=grid_pos,
                neutral=neutral, restart=restart, t_start=t_start, t_end=t_end, source="OpenF1")


def analyse_year(year: int) -> dict:
    """FastF1 for 2022 (OpenF1 starts in 2023); OpenF1 from 2023 on."""
    race = load_fastf1(year) if year < 2023 else load_openf1(year)
    return analyse(race)


def combine(results: dict, out_path: str, previous: str | None = None) -> int:
    """Merge per-year results into one file and print the summary tables.

    Years missing from `results` are carried over from `previous` (the last
    committed file), so one failed download doesn't erase a year.
    """
    import os
    prev = {}
    if previous and os.path.exists(previous):
        with open(previous) as f:
            prev = json.load(f)
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
    for y, r in prev.get("races", {}).items():
        if y not in out["races"] and r.get("passes") is not None:
            print(f"{y}: not re-fetched this run; keeping previous result")
            out["races"][y] = r
            if "track" not in out and "track" in prev:
                out["track"], out["corners"], out["lapLength"] = prev["track"], prev["corners"], prev["lapLength"]
    out["races"] = dict(sorted(out["races"].items()))

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
    ap.add_argument("--previous", help="earlier output to carry missing years over from")
    ap.add_argument("--cache", default=".fastf1-cache")
    args = ap.parse_args(argv)

    if args.combine:
        results = {}
        for path in args.combine:
            with open(path) as f:
                part = json.load(f)
            results[int(part["year"])] = part["result"]
        return combine(results, args.out, args.previous)

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
    return combine(results, args.out, args.previous)


if __name__ == "__main__":
    sys.exit(main())
