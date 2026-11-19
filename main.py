#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a target instant.

Method (counts, snapshots, refresh/dilution, honest spread):

* Work with counts.  A rate r (percent) over N tracked players means
  E = r/100 * N earners.  Forecast E and N separately, rate = E/N.
* Deduplicate the history into distinct snapshots: consecutive identical
  reads are one stale page, not new information.  Trends are fitted on
  snapshots only.
* Per trophy, fit logit(rate) against log(1 + hours since release) by
  recency-weighted least squares when there are enough snapshots;
  otherwise the trend is flat.
* Model refreshes.  Exophase refreshes its counts on its own schedule:
  the page can sit unchanged for hours, then a refresh jumps N and
  dilutes almost every rate, because new players have earned few
  trophies.  From the snapshots, detect jumps in players_tracked,
  estimate the typical gap between refreshes and the typical jump
  size, and project N upward to the target.  When no jump has been
  seen yet, use a prior (refreshes roughly every few hours, N
  multiplying a few-fold per refresh) sharpened by how stale the
  current page already is: a page flat for most of a refresh interval
  is due one.  Earners lag: new players earn trophies only gradually
  (early/common trophies faster, rare/late ones ~never in the window),
  so E grows only a fraction of what the new cohort would earn at the
  incumbents' rates.  The 100% trophy stays at ~100%.
* Spread.  Quantiles are logit-normal around the median with a spread
  that grows with the horizon, with the expected number of refreshes
  before the target, with jump-size uncertainty and with the per-trophy
  fit residual, and never below a floor: a refresh moves rates by
  ~1.8 logit, so a narrow band is dishonest whenever one is due.

Pure standard library, deterministic, no network, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- priors (fallbacks only; fitted values win whenever data exists) ---
PRIOR_REFRESH_GAP_H = 4.5      # hours between Exophase refreshes
PRIOR_JUMP_MULT = 3.5          # players_tracked multiplier per refresh
PRIOR_JUMP_LOG_STD = 0.7       # spread of log jump size
BASE_LOGIT_SPREAD = 0.35       # irreducible logit spread
MIN_LOGIT_SPREAD = 0.80
MAX_LOGIT_SPREAD = 2.30
KAPPA_BASE = 0.05              # new-cohort earn fraction, rare trophy
KAPPA_TOP = 0.25               # new-cohort earn fraction, common trophy


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """logit of a percent rate, clamped to the scoring range."""
    p = clamp(rate, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    l = clamp(l, -9.0, 9.0)
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(text):
    """Parse an ISO-8601 timestamp to epoch seconds; None on failure."""
    if not isinstance(text, str) or not text:
        return None
    try:
        t = text.strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def parse_release(game):
    """Release instant (epoch seconds).  A bare date means midnight UTC."""
    if not isinstance(game, dict):
        return None
    raw = game.get("release_date")
    if isinstance(raw, str):
        ts = parse_time(raw)
        if ts is not None:
            return ts
        try:
            dt = datetime.strptime(raw.strip(), "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def hours_since_release(ts, release_ts):
    if ts is None or release_ts is None:
        return 0.0
    return max(0.0, (ts - release_ts) / 3600.0)


def dedup_snapshots(history):
    """Collapse consecutive identical reads into distinct snapshots.

    Returns a list of dicts {t, n, rates} where t is the epoch of the
    first read of the snapshot, n the players tracked, rates the list.
    A snapshot changes when N or any rate changes.
    """
    snaps = []
    prev_key = None
    for read in history or []:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            rates = []
        n = read.get("players_tracked")
        try:
            n = float(n)
        except (TypeError, ValueError):
            n = None
        t = parse_time(read.get("at"))
        if t is None:
            t = parse_time(read.get("slot"))
        key = (n, tuple(rates))
        if key == prev_key and snaps:
            continue
        prev_key = key
        snaps.append({"t": t, "n": n, "rates": list(rates)})
    return snaps


def last_known_rates(snaps, ntrophies):
    """Last non-null rate per trophy over the snapshots (oldest first)."""
    last = [None] * ntrophies
    for snap in snaps:
        for i, r in enumerate(snap["rates"][:ntrophies]):
            if isinstance(r, (int, float)) and math.isfinite(r):
                last[i] = float(r)
    return last


def weighted_trend(snaps, idx, release_ts, target_ts):
    """Recency-weighted least squares of logit(rate) vs log1p(hours).

    Returns (shift, resid): the predicted logit change from the last
    snapshot to the target, and the residual RMS of the fit.  Returns
    (0.0, 0.0) when there are too few usable snapshots.
    """
    pts = []
    for snap in snaps:
        if snap["t"] is None:
            continue
        rates = snap["rates"]
        if idx >= len(rates):
            continue
        r = rates[idx]
        if not isinstance(r, (int, float)) or not math.isfinite(r):
            continue
        if r <= 0.0 or r >= 100.0:
            continue
        x = math.log1p(hours_since_release(snap["t"], release_ts))
        pts.append((snap["t"], x, logit_pct(r)))
    if len(pts) < 3:
        return 0.0, 0.0
    t_last = pts[-1][0]
    # Recency weights: half-life 12 h, so launch-night drift matters
    # less than the most recent day of play.
    ws = [math.exp(-max(0.0, t_last - t) / (12.0 * 3600.0)) for t, _, _ in pts]
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0
    mx = sum(w * x for w, (_, x, _) in zip(ws, pts)) / sw
    my = sum(w * y for w, (_, _, y) in zip(ws, pts)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, (_, x, _) in zip(ws, pts))
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, (_, x, y) in zip(ws, pts))
    slope = sxy / sxx
    slope = clamp(slope, -3.0, 3.0)
    x_target = math.log1p(hours_since_release(target_ts, release_ts))
    x_last = pts[-1][1]
    shift = slope * (x_target - x_last)
    shift = clamp(shift, -1.5, 1.5)
    resid2 = sum(w * (y - (my + slope * (x - mx))) ** 2
                 for w, (_, x, y) in zip(ws, pts)) / sw
    return shift, math.sqrt(max(0.0, resid2))


def refresh_model(snaps, checkpoint_ts, target_ts):
    """Estimate refresh behaviour and the expected N growth to target.

    Returns (exp_refreshes, log_jump, jump_log_std, stale_hours):
    how many refreshes are expected before the target, the typical
    log jump in players_tracked per refresh, its spread, and how long
    the current snapshot has already been flat.
    """
    jumps = []   # (gap_hours, log_jump)
    prev = None
    for snap in snaps:
        if prev is not None and snap["n"] and prev["n"]:
            if snap["n"] > prev["n"] * 1.01 and snap["t"] and prev["t"]:
                gap = (snap["t"] - prev["t"]) / 3600.0
                if gap > 0:
                    jumps.append((gap, math.log(snap["n"] / prev["n"])))
        prev = snap
    stale = 0.0
    if snaps and snaps[-1]["t"] is not None and checkpoint_ts is not None:
        # snaps[-1]["t"] is the FIRST read showing the current page
        # (dedup keeps the first read of each distinct snapshot),
        # so this is how long the page has already been flat.
        stale = max(0.0, (checkpoint_ts - snaps[-1]["t"]) / 3600.0)
    horizon_h = 0.0
    if checkpoint_ts is not None and target_ts is not None:
        horizon_h = max(0.0, (target_ts - checkpoint_ts) / 3600.0)

    if jumps:
        gaps = sorted(g for g, _ in jumps)
        logs = sorted(l for _, l in jumps)
        gap_med = gaps[len(gaps) // 2]
        jump_med = logs[len(logs) // 2]
        if len(logs) >= 2:
            mean_l = sum(logs) / len(logs)
            var_l = sum((l - mean_l) ** 2 for l in logs) / len(logs)
            jump_std = math.sqrt(max(0.04, var_l))
        else:
            jump_std = PRIOR_JUMP_LOG_STD
        # A page flat for most of a refresh interval is due one: the
        # refresh expected inside the horizon counts the time already
        # waited as well as the horizon itself.
        waited = min(stale, gap_med)
        exp_ref = (horizon_h + 0.5 * waited) / max(gap_med, 0.5)
        exp_ref = clamp(exp_ref, 0.0, 40.0)
        return exp_ref, jump_med, jump_std, stale

    # No refresh seen yet: fall back to the prior, but a page that has
    # already been flat for close to a full prior interval is due a
    # refresh inside the horizon, not merely horizon/interval of one.
    gap = PRIOR_REFRESH_GAP_H
    jump = math.log(PRIOR_JUMP_MULT)
    if stale + horizon_h >= gap:
        exp_ref = 1.0 + max(0.0, (stale + horizon_h - gap)) / gap
    else:
        exp_ref = horizon_h / gap
    # Same sanity cap as the empirical branch: a page flat far past
    # one interval is overdue for one refresh, not for many at once.
    exp_ref = clamp(exp_ref, 0.0, 40.0)
    return exp_ref, jump, PRIOR_JUMP_LOG_STD, stale


def forecast_case(case):
    trophies = case.get("trophies") or []
    ntr = len(trophies)
    if ntr == 0:
        return {"trophies": []}
    history = case.get("history") or []
    snaps = dedup_snapshots(history)
    release_ts = parse_release(case.get("game"))
    checkpoint_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    horizon_min = case.get("horizon_minutes")
    if target_ts is None and checkpoint_ts is not None and horizon_min:
        try:
            target_ts = checkpoint_ts + float(horizon_min) * 60.0
        except (TypeError, ValueError):
            target_ts = checkpoint_ts
    if target_ts is None:
        target_ts = checkpoint_ts
    if checkpoint_ts is None:
        checkpoint_ts = target_ts

    horizon_days = 0.0
    if checkpoint_ts is not None and target_ts is not None:
        horizon_days = max(0.0, (target_ts - checkpoint_ts) / 86400.0)

    last = last_known_rates(snaps, ntr)
    n_last = None
    for snap in reversed(snaps):
        if snap["n"]:
            n_last = snap["n"]
            break

    exp_ref, jump_log, jump_std, _stale = refresh_model(
        snaps, checkpoint_ts, target_ts)
    # Expected log growth of players tracked by the target.
    exp_log_n = exp_ref * jump_log
    exp_log_n = clamp(exp_log_n, 0.0, math.log(60.0))
    n_mult = math.exp(exp_log_n)

    # The most common non-100 rate anchors how fast new players earn.
    known = [r for r in last if r is not None and 0.0 < r < 100.0]
    r_top = max(known) if known else 1.0
    r_top = max(r_top, 0.05)

    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id", i + 1)
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            tid = i + 1
        r_last = last[i]
        tier = str(trophy.get("tier", "")).lower()

        if r_last is None:
            # Never seen: a rare trophy, most likely still ~unearned.
            med = 0.05
            spread = 1.6
            rows.append((tid, med, spread))
            continue

        if r_last >= 99.99:
            # Everybody holds it, including every newcomer: it stays.
            rows.append((tid, None, None))
            continue

        # Earners lag the tracked count: new players earn a trophy at
        # only a fraction kappa of the incumbents' rate inside the
        # window.  Platinum is effectively unearned by newcomers;
        # common bronze the fastest.
        if tier == "platinum":
            kappa = 0.0
        elif tier == "gold":
            kappa = 0.05
        elif tier == "silver":
            kappa = 0.10
        else:
            kappa = KAPPA_BASE + (KAPPA_TOP - KAPPA_BASE) * clamp(
                r_last / r_top, 0.0, 1.0)
        # Median rate at target: E grows by kappa on the new cohort,
        # N grows by n_mult.
        rate_factor = (1.0 + kappa * (n_mult - 1.0)) / n_mult
        rate_factor = clamp(rate_factor, 0.005, 1.5)
        med_rate = r_last * rate_factor

        # Trend on the logit scale, from the snapshots.
        shift, resid = weighted_trend(snaps, i, release_ts, target_ts)
        med_logit = logit_pct(med_rate) + shift
        med = inv_logit_pct(med_logit)
        med = clamp(med, 0.01, 99.9)

        # Spread: horizon random walk + refresh uncertainty (each
        # expected refresh contributes jump-size uncertainty) + fit
        # residual, with a floor so a refresh due is never claimed
        # precisely, and a cap so the band stays informative.
        var = BASE_LOGIT_SPREAD ** 2
        var += (0.5 * math.sqrt(horizon_days)) ** 2
        var += exp_ref * (jump_std ** 2)
        var += resid ** 2
        spread = math.sqrt(var)
        spread = clamp(spread, MIN_LOGIT_SPREAD, MAX_LOGIT_SPREAD)
        rows.append((tid, med, spread))

    out_rows = []
    for tid, med, spread in rows:
        if med is None:
            # The 100% trophy: tight band just under 100.
            out_rows.append({"id": tid,
                             "q": [99.0, 99.5, 99.9, 99.97, 99.99]})
            continue
        centre = logit_pct(med)
        qs = []
        prev = None
        for z in Z:
            q = inv_logit_pct(centre + z * spread)
            q = clamp(q, 0.01, 99.99)
            q = round(q, 4)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out_rows.append({"id": tid, "q": qs})
    return {"trophies": out_rows}


def fallback_forecast(case):
    """Last-rate forecast with a modest spread; never crashes."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    snaps = dedup_snapshots(history)
    last = last_known_rates(snaps, len(trophies))
    rows = []
    for i, trophy in enumerate(trophies):
        try:
            tid = int(trophy.get("id", i + 1))
        except (TypeError, ValueError):
            tid = i + 1
        r = last[i]
        if r is None:
            r = 0.5
        if r >= 99.99:
            rows.append({"id": tid, "q": [99.0, 99.5, 99.9, 99.97, 99.99]})
            continue
        centre = logit_pct(r)
        qs = []
        prev = None
        for z in Z:
            q = clamp(round(inv_logit_pct(centre + z * 0.8), 4), 0.01, 99.99)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def process_case(case):
    try:
        return forecast_case(case)
    except Exception:
        try:
            return fallback_forecast(case)
        except Exception:
            trophies = case.get("trophies") or []
            rows = []
            for i, trophy in enumerate(trophies):
                try:
                    tid = int(trophy.get("id", i + 1))
                except (TypeError, ValueError):
                    tid = i + 1
                rows.append({"id": tid, "q": [0.01, 0.1, 0.5, 1.0, 5.0]})
            return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(in_dir, name), "r",
                      encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            case = {"id": case_id, "trophies": [], "history": []}
        if not isinstance(case, dict):
            case = {"id": case_id, "trophies": [], "history": []}
        result = process_case(case)
        payload = json.dumps(result, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        if len(payload) > 65536:  # pragma: no cover - safety valve
            payload = json.dumps(
                {"trophies": [{"id": row["id"],
                               "q": [round(v, 2) for v in row["q"]]}
                              for row in result["trophies"]]},
                separators=(",", ":")).encode("utf-8")
        out_path = os.path.join(out_dir, case_id + ".out")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
        os.replace(tmp_path, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
