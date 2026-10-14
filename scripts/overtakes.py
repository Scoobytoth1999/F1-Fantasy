"""
On-track overtakes for every Grand Prix since 2023, from OpenF1 timing data.

Counting (per race):
  * A pass is a change in the official running order between two cars from
    one lap to the next (lap 1 compares against the grid). Lapping a
    backmarker never changes running order, so it is never counted.
  * Excluded: pit in/out laps for either car, passes under SC / VSC / red
    flag, and passes on a car in its retirement lap.
  * Tagged, shown only with the app's "include" toggle: lap 1 ("lap1"), the
    lap after a safety car ends ("restart"), and passes on a car that lost 4+
    places in one lap - a spin or problem ("incident"). Everything else is
    "green": a racing pass.

Locating (per pass):
  * Each car's ~4 Hz X/Y position is snapped to a reference lap line to get
    distance-into-lap; the moment the two cars' race distance crosses is the
    pass point. It is credited to the corner it completed within 80 m after,
    otherwise the corner it was heading into (so a DRS pass finished on a
    straight counts for the braking zone it was into).

Usage:
  python scripts/overtakes.py --plan [--years 2025 2026] [--force]   # matrix JSON
  python scripts/overtakes.py --race '<matrix item JSON>'              # one race
  python scripts/overtakes.py --build                                  # merge
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np

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


INCIDENT_MIN = 4   # a car losing this many places in one lap had a spin/off/problem


def tag_incidents(passes: list[dict]) -> list[dict]:
    """Re-tag passes on a car that lost INCIDENT_MIN+ places in one lap as 'incident'.

    A spun or stricken car being driven past by half the field is a position
    change, not racing; lap-1 passes keep their own tag.
    """
    from collections import Counter
    n = Counter((p["lap"], p["on"]) for p in passes if p["cat"] != "lap1")
    for p in passes:
        if p["cat"] != "lap1" and n[(p["lap"], p["on"])] >= INCIDENT_MIN:
            p["cat"] = "incident"
    return passes


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


def neutral_events(rc: list[dict], med_lap: float) -> list[tuple[float, str]]:
    """SC / VSC / red-flag start and end times from OpenF1 race-control messages."""
    ev, red = [], False
    for m in sorted(rc, key=lambda m: m["date"]):
        msg, flag, t = (m.get("message") or "").upper(), m.get("flag"), iso(m["date"])
        if "SAFETY CAR DEPLOYED" in msg:                 # also matches VIRTUAL SAFETY CAR DEPLOYED
            ev.append((t, "start"))
        elif flag == "RED" and "PIT EXIT" not in msg:
            ev.append((t, "start")); red = True
        elif "VIRTUAL SAFETY CAR ENDING" in msg or "VSC ENDING" in msg:
            ev.append((t + 10, "end")); red = False
        elif "SAFETY CAR IN THIS LAP" in msg:
            ev.append((t + med_lap * 0.8, "end")); red = False
        elif red and (flag == "GREEN" or ("RESUME" in msg and "RACE" in msg)):
            ev.append((t + med_lap, "end")); red = False   # rolling/standing restart lap
    return ev


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


# ---------------------------------------------------------- source: OpenF1

OPENF1 = "https://api.openf1.org/v1/"
MULTIVIEWER = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"


def get_json(url: str, tries=6):
    """GET JSON from OpenF1/MultiViewer, staying under OpenF1's free limit (30/min)."""
    import time, urllib.request, urllib.error
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1-fantasy-overtakes/1.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.load(r)
            time.sleep(2.1)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []                 # OpenF1 answers "no rows" with 404
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                wait = float(e.headers.get("Retry-After") or 0) or 15 * (i + 1)
                time.sleep(min(wait, 90)); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if i < tries - 1:
                time.sleep(10 * (i + 1)); continue
            raise


def iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class NoRaceData(Exception):
    pass


def load_openf1(k: int, circuit_key: int, year: int) -> dict:
    circ = get_json(MULTIVIEWER.format(key=circuit_key, year=year))

    drivers = get_json(f"{OPENF1}drivers?session_key={k}")
    abbr = {str(d["driver_number"]): d["name_acronym"] for d in drivers}
    results = get_json(f"{OPENF1}session_result?session_key={k}")
    finished = {str(r["driver_number"]): not (r.get("dnf") or r.get("dns") or r.get("dsq")) for r in results}
    laps_raw = get_json(f"{OPENF1}laps?session_key={k}")
    if not laps_raw or not results:
        raise NoRaceData("no lap or result data (race cancelled or not yet published)")
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

    ev = neutral_events(rc, med_lap)
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
                neutral=neutral, restart=restart, t_start=t_start, t_end=t_end, source="OpenF1",
                rc_counts={"sc_or_vsc": sum(1 for m in rc if "SAFETY CAR DEPLOYED" in (m.get("message") or "").upper()),
                           "red": sum(1 for m in rc if m.get("flag") == "RED")})


# --------------------------------------------------------------- orchestration
#
#   --plan   list finished Grands Prix (2023+) not yet analysed -> GitHub matrix JSON
#   --race   analyse one race (by OpenF1 session key) -> overtakes/races/<year>-<slug>.json
#   --build  merge every race file -> overtakes-<slug>.json for MAP_CIRCUITS
#            + overtakes-f1.json (per-race totals for the whole grid, for F1 averages)

RACES_DIR = "overtakes/races"
FIRST_YEAR = 2023                      # OpenF1 coverage starts here
# Circuits that get a passing map in the app (OpenF1 circuit_short_name -> file slug)
MAP_CIRCUITS = {
    "Austin": "austin", "Mexico City": "mexico-city", "Interlagos": "interlagos",
    "Las Vegas": "las-vegas", "Lusail": "lusail", "Yas Marina Circuit": "yas-marina",
}


def slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def race_file(year: int, circuit: str) -> str:
    return f"{RACES_DIR}/{year}-{slugify(circuit)}.json"


def plan(years: list[int], force: bool = False) -> list[dict]:
    import os
    now = datetime.now(timezone.utc).timestamp()
    todo = []
    for y in years:
        meetings = {m["meeting_key"]: m for m in get_json(f"{OPENF1}meetings?year={y}")}
        for s in get_json(f"{OPENF1}sessions?year={y}&session_name=Race"):
            if s.get("session_type") != "Race" or not s.get("date_end"):
                continue
            if iso(s["date_end"]) > now - 3 * 3600:          # not finished (or data still settling)
                continue
            circuit = s["circuit_short_name"]
            if not force and os.path.exists(race_file(y, circuit)):
                continue
            m = meetings.get(s["meeting_key"], {})
            todo.append({"key": s["session_key"], "circuit_key": s["circuit_key"], "year": y,
                         "circuit": circuit, "meeting": m.get("meeting_name", circuit),
                         "date": s["date_start"][:10]})
    return todo


def run_race(item: dict) -> str:
    import os
    out = {k: item[k] for k in ("key", "year", "circuit", "meeting", "date")}
    os.makedirs(RACES_DIR, exist_ok=True)
    path = race_file(item["year"], item["circuit"])
    try:
        race = load_openf1(int(item["key"]), int(item["circuit_key"]), int(item["year"]))
    except NoRaceData as e:
        # Recorded so it isn't retried every week; ignored by --build.
        out["no_data"] = str(e)
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"{item['year']} {item['meeting']}: {e}")
        return path
    r = analyse(race)
    r["rc_counts"] = race.get("rc_counts", {})
    out["result"] = r
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    c = {k: sum(p["cat"] == k for p in r["passes"]) for k in ("green", "lap1", "restart")}
    print(f"{item['year']} {item['meeting']}: {c} dropped={r['dropped']} rc={r['rc_counts']}")
    return path


def build(out_dir: str = ".") -> int:
    import glob, os
    races = []
    for path in sorted(glob.glob(f"{RACES_DIR}/*.json")):
        with open(path) as f:
            rc = json.load(f)
        if rc.get("result"):
            races.append(rc)
    if not races:
        print("No race files."); return 1
    for rc in races:
        tag_incidents(rc["result"]["passes"])

    # whole-grid summary: one row per Grand Prix
    rows = []
    for rc in sorted(races, key=lambda r: r["date"]):
        cnt = {k: sum(p["cat"] == k for p in rc["result"]["passes"]) for k in ("green", "lap1", "restart", "incident")}
        rows.append({"year": rc["year"], "date": rc["date"], "meeting": rc["meeting"],
                     "circuit": rc["circuit"], **cnt})
    f1 = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "method": "On-track passes per Grand Prix from OpenF1 lap-by-lap positions; green = racing passes "
                    "excluding lap 1, safety-car restarts, pit stops and passes on spun/stricken cars.",
          "races": rows}
    with open(os.path.join(out_dir, "overtakes-f1.json"), "w") as f:
        json.dump(f1, f, separators=(",", ":"))

    # per-circuit passing-map files
    for circuit, slug in MAP_CIRCUITS.items():
        mine = sorted((r for r in races if r["circuit"] == circuit), key=lambda r: r["year"])
        if not mine:
            print(f"{circuit}: no races yet"); continue
        latest = mine[-1]["result"]
        out = {"circuit": circuit, "meeting": mine[-1]["meeting"],
               "generated": f1["generated"], "method": f1["method"],
               "lapLength": latest["lapLength"], "track": latest["track"], "corners": latest["corners"],
               "races": {str(r["year"]): {"passes": r["result"]["passes"], "dropped": r["result"]["dropped"]}
                         for r in mine}}
        with open(os.path.join(out_dir, f"overtakes-{slug}.json"), "w") as f:
            json.dump(out, f, separators=(",", ":"))

    # summary for the Action log
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r["green"])
    lines = ["| Year | GPs | Avg green-flag passes |", "|---|---|---|"]
    lines += [f"| {y} | {len(v)} | {sum(v) / len(v):.1f} |" for y, v in sorted(by_year.items())]
    allg = [r["green"] for r in rows]
    lines += ["", f"All {len(allg)} GPs: {sum(allg) / len(allg):.1f} green-flag passes per race", "",
              "| Circuit | GPs | Avg green | Rank |", "|---|---|---|---|"]
    per_c = {}
    for r in rows:
        per_c.setdefault(r["circuit"], []).append(r["green"])
    ranked = sorted(per_c.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    lines += [f"| {c}{' *' if c in MAP_CIRCUITS else ''} | {len(v)} | {sum(v) / len(v):.1f} | {i + 1} |"
              for i, (c, v) in enumerate(ranked)]
    summary = "\n".join(lines)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## F1 overtakes\n\n" + summary + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--force", action="store_true", help="re-analyse races that already have a file")
    ap.add_argument("--race", help="JSON matrix item (from --plan) to analyse")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)

    if args.plan:
        years = args.years or list(range(FIRST_YEAR, datetime.now(timezone.utc).year + 1))
        todo = plan(years, args.force)
        print(json.dumps(todo, separators=(",", ":")))
        return 0
    if args.race:
        run_race(json.loads(args.race))
        return 0
    if args.build:
        return build()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
