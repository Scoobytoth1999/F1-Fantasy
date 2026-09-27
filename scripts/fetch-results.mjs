// Pulls the 2026 sprint + GP classifications from Jolpica-F1 (the Ergast
// successor) and writes results-2026.json at the repo root, which the league
// app reads. Run by .github/workflows/results.yml — no secrets involved, and
// nothing here talks to Firestore.
//
// The file is only rewritten when the results themselves change, so the bot
// doesn't create a commit on every scheduled run.
import { readFile, writeFile } from "node:fs/promises";

const SEASON = 2026;
const BASE = `https://api.jolpi.ca/ergast/f1/${SEASON}`;
const OUT = new URL(`../results-${SEASON}.json`, import.meta.url);
const SCHEDULE_OUT = new URL(`../schedule-${SEASON}.json`, import.meta.url);
// Jolpica asks every client to identify itself with a custom User-Agent.
const HEADERS = {
  "User-Agent": "F1FantasyLeague/1.0 (+https://github.com/Scoobytoth1999/F1-Fantasy)",
  Accept: "application/json",
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function getJSON(url, tries = 4) {
  for (let i = 1; ; i++) {
    const res = await fetch(url, { headers: HEADERS });
    if (res.ok) return res.json();
    const retryable = res.status === 429 || res.status >= 500;
    if (!retryable || i >= tries) throw new Error(`${res.status} ${res.statusText} — ${url}`);
    await sleep(3000 * i);
  }
}

// Ergast-style endpoints page by result ROW (max 100 per page), so one race can
// straddle two pages. Collect everything, then group by round.
async function fetchAll(path, key) {
  const byRound = {};
  let offset = 0, total = Infinity;
  while (offset < total) {
    const { MRData } = await getJSON(`${BASE}/${path}.json?limit=100&offset=${offset}`);
    total = Number(MRData.total) || 0;
    for (const race of MRData.RaceTable.Races) {
      const r = (byRound[race.round] ||= { round: Number(race.round), name: race.raceName, date: race.date, rows: [] });
      r.rows.push(...(race[key] || []));
    }
    offset += Number(MRData.limit) || 100;
    await sleep(500);   // stay well inside the rate limit
  }
  return byRound;
}

const toRow = x => {
  const classified = /^\d+$/.test(x.positionText || "");
  return {
    pos: classified ? Number(x.positionText) : null,
    classified,
    name: `${x.Driver.givenName} ${x.Driver.familyName}`,
    code: x.Driver.code || null,
    status: x.status || null,
  };
};
const rowsOf = r => r.rows
  .slice()
  .sort((a, b) => Number(a.position) - Number(b.position))
  .map(toRow);

// ── Schedule: official session start times (UTC), for the app's Detroit-time display.
// Written to its own file with its own change check, so a schedule tweak never
// touches results-2026.json and vice versa.
{
  const { MRData } = await getJSON(`${BASE}/races.json?limit=100`);
  const slim = x => (x && x.date ? { date: x.date, time: x.time || null } : undefined);
  const races = {};
  for (const r of MRData.RaceTable.Races) {
    races[r.round] = {
      round: Number(r.round), name: r.raceName, date: r.date, time: r.time || null,
      qualifying: slim(r.Qualifying), sprint: slim(r.Sprint), sprintQualifying: slim(r.SprintQualifying),
    };
  }
  let prevSched = null;
  try { prevSched = JSON.parse(await readFile(SCHEDULE_OUT, "utf8")); } catch { /* first run */ }
  if (Object.keys(races).length === 0) {
    console.log("Schedule empty from API — keeping the existing file.");
  } else if (prevSched && JSON.stringify(prevSched.races) === JSON.stringify(races)) {
    console.log("Schedule unchanged.");
  } else {
    await writeFile(SCHEDULE_OUT, JSON.stringify({ season: SEASON, source: "api.jolpi.ca (Jolpica-F1)",
      updatedAt: new Date().toISOString(), races }, null, 2) + "\n");
    console.log(`Wrote schedule: ${Object.keys(races).length} rounds.`);
  }
  await sleep(500);
}

const gp     = await fetchAll("results", "Results");
const sprint = await fetchAll("sprint", "SprintResults");

const races = {};
for (const r of Object.values(gp)) {
  races[r.round] = { round: r.round, name: r.name, date: r.date, gp: rowsOf(r) };
}
for (const r of Object.values(sprint)) {
  races[r.round] ||= { round: r.round, name: r.name, date: r.date };
  races[r.round].sprint = rowsOf(r);
}

let prev = null;
try { prev = JSON.parse(await readFile(OUT, "utf8")); } catch { /* first run */ }

if (prev && JSON.stringify(prev.races) === JSON.stringify(races)) {
  console.log("Results unchanged — nothing to write.");
  process.exit(0);
}

// Guard against an API hiccup replacing good data with a partial response.
const rowCount = o => Object.values(o || {}).reduce((n, r) => n + (r.gp?.length || 0) + (r.sprint?.length || 0), 0);
if (prev && rowCount(races) < rowCount(prev.races) * 0.9) {
  throw new Error(`Refusing to write: ${rowCount(races)} rows vs ${rowCount(prev.races)} before.`);
}

await writeFile(OUT, JSON.stringify({
  season: SEASON,
  source: "api.jolpi.ca (Jolpica-F1)",
  updatedAt: new Date().toISOString(),
  races,
}, null, 2) + "\n");
console.log(`Wrote ${Object.keys(races).length} rounds (${rowCount(races)} result rows).`);
