"""Free batter projection baseline from MLB StatsAPI season rates.

Unlocks the batter Pick6 markets (hits, total bases, home runs, RBI, runs) that
RotoWire paywalls. For each batter: season per-AB / per-PA rates x projected
plate appearances -> lambda for the market's distribution (markets.py).

    lambda_hits = (H/AB) * expected_AB ,  lambda_hr = (HR/AB) * expected_AB
    lambda_tb   = (TB/AB) * expected_AB
    lambda_rbi  = (RBI/PA) * expected_PA ,  lambda_runs = (R/PA) * expected_PA

When `project()` is given the game date, the season-rate lambda is adjusted for
the matchup (all free StatsAPI data, every factor degrades to 1.0 on any miss):
  - OPPOSING STARTER: his season rates-against vs league average (BA-against for
    hits, SLG-against for TB, HR/BF for HR, OPS-against for RBI/runs).
  - PLATOON: the batter's split vs the starter's throwing hand relative to his
    overall rate (hits/TB/HR only; needs MIN_SPLIT_AB to count).
Both factors are damped by STARTER_WEIGHT (the batter only faces the starter for
~60% of PAs; the bullpen is unknown) and clamped, so a brutal matchup can dent a
projection but never zero it.

*** STILL LOWER CONFIDENCE than the strikeout model: no fitted dispersion, no
park factor, no recent form. markets.BASELINE_P_CAP caps what these legs may
claim. RotoWire's free tb/runs cross-check remains the second opinion. ***
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from feed import norm

# Typical plate appearances by batting-order slot (1-9); default for unknown slot.
PA_BY_SLOT = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.1, 6: 4.0, 7: 3.9, 8: 3.8, 9: 3.7}
DEFAULT_PA = 4.2

# our market -> how to project it: (rate stat, denom stat, denom kind)
_RECIPE = {
    "hits":        ("hits", "atBats", "ab"),
    "total_bases": ("totalBases", "atBats", "ab"),
    "home_runs":   ("homeRuns", "atBats", "ab"),
    "rbi":         ("rbi", "plateAppearances", "pa"),
    "runs":        ("runs", "plateAppearances", "pa"),
}

_id_cache: dict[str, int | None] = {}
_stat_cache: dict[tuple, dict | None] = {}

# ---- matchup adjustment ------------------------------------------------------
STARTER_WEIGHT = 0.6          # share of PAs a batter sees vs the starter
FACTOR_CLAMP = (0.75, 1.30)   # per-factor clamp (raw rate ratios are noisy)
TOTAL_CLAMP = (0.65, 1.35)    # combined opponent x platoon clamp
MIN_SPLIT_AB = 40             # platoon split sample floor
MIN_OPP_BF = 50               # opposing starter sample floor (batters faced)

# market -> which rate-against of the opposing starter scales it
_OPP_RATE = {"hits": "avg", "total_bases": "slg", "rbi": "ops", "runs": "ops"}

_sched_cache: dict[str, dict[int, int]] = {}
_person_cache: dict[int, dict | None] = {}
_pitching_cache: dict[tuple, dict | None] = {}
_split_cache: dict[tuple, dict] = {}
_league_cache: dict[int, dict | None] = {}


def _get(url):
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.load(r)


def player_id(name: str) -> int | None:
    key = norm(name)
    if key in _id_cache:
        return _id_cache[key]
    try:
        s = _get("https://statsapi.mlb.com/api/v1/people/search?names="
                 + urllib.parse.quote(name))
        ppl = s.get("people", [])
        # prefer an exact accent-folded name match
        pid = next((p["id"] for p in ppl if norm(p.get("fullName", "")) == key),
                   ppl[0]["id"] if ppl else None)
    except Exception:
        pid = None
    _id_cache[key] = pid
    return pid


def season_hitting(pid: int, season: int) -> dict | None:
    ck = (pid, season)
    if ck in _stat_cache:
        return _stat_cache[ck]
    try:
        st = _get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                  f"?stats=season&group=hitting&season={season}")
        splits = st.get("stats", [{}])[0].get("splits", [])
        stat = splits[0]["stat"] if splits else None
    except Exception:
        stat = None
    _stat_cache[ck] = stat
    return stat


def _person(pid: int) -> dict | None:
    if pid in _person_cache:
        return _person_cache[pid]
    try:
        p = _get(f"https://statsapi.mlb.com/api/v1/people/{pid}?hydrate=currentTeam")
        out = p.get("people", [None])[0]
    except Exception:
        out = None
    _person_cache[pid] = out
    return out


def _opponent_starters(date: str) -> dict[int, int]:
    """teamId -> opposing probable starter's player id, for every game on date."""
    if date in _sched_cache:
        return _sched_cache[date]
    out: dict[int, int] = {}
    try:
        s = _get("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date="
                 + date + "&hydrate=probablePitcher")
        for d in s.get("dates", []):
            for g in d.get("games", []):
                home, away = g["teams"]["home"], g["teams"]["away"]
                hp = (home.get("probablePitcher") or {}).get("id")
                ap = (away.get("probablePitcher") or {}).get("id")
                if ap:
                    out[home["team"]["id"]] = ap
                if hp:
                    out[away["team"]["id"]] = hp
    except Exception:
        pass
    _sched_cache[date] = out
    return out


def _season_pitching(pid: int, season: int) -> dict | None:
    ck = (pid, season)
    if ck in _pitching_cache:
        return _pitching_cache[ck]
    try:
        st = _get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                  f"?stats=season&group=pitching&season={season}")
        splits = st.get("stats", [{}])[0].get("splits", [])
        stat = splits[0]["stat"] if splits else None
    except Exception:
        stat = None
    _pitching_cache[ck] = stat
    return stat


def _platoon_splits(pid: int, season: int) -> dict[str, dict]:
    """'L'/'R' (opposing pitcher hand) -> the batter's hitting stat in that split."""
    ck = (pid, season)
    if ck in _split_cache:
        return _split_cache[ck]
    out: dict[str, dict] = {}
    try:
        st = _get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                  f"?stats=statSplits&group=hitting&season={season}&sitCodes=vl,vr")
        for spl in st.get("stats", [{}])[0].get("splits", []):
            code = spl.get("split", {}).get("code", "")
            if code in ("vl", "vr"):
                out[code[-1].upper()] = spl["stat"]
    except Exception:
        pass
    _split_cache[ck] = out
    return out


def _league_rates(season: int) -> dict | None:
    """League-average avg / slg / ops / hr_bf from summed team hitting stats."""
    if season in _league_cache:
        return _league_cache[season]
    try:
        st = _get(f"https://statsapi.mlb.com/api/v1/teams/stats"
                  f"?season={season}&group=hitting&stats=season&sportIds=1")
        ab = h = tb = hr = pa = obp_n = 0.0
        for spl in st.get("stats", [{}])[0].get("splits", []):
            s = spl["stat"]
            ab += float(s.get("atBats", 0) or 0)
            h += float(s.get("hits", 0) or 0)
            tb += float(s.get("totalBases", 0) or 0)
            hr += float(s.get("homeRuns", 0) or 0)
            pa += float(s.get("plateAppearances", 0) or 0)
            obp_n += float(s.get("plateAppearances", 0) or 0) * float(s.get("obp", 0) or 0)
        if ab <= 0 or pa <= 0:
            raise ValueError("empty league stats")
        out = {"avg": h / ab, "slg": tb / ab, "hr_bf": hr / pa,
               "ops": obp_n / pa + tb / ab}
    except Exception:
        out = None
    _league_cache[season] = out
    return out


def _fnum(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    return max(lo_hi[0], min(lo_hi[1], x))


def _damp(f: float) -> float:
    """Shrink a factor toward 1.0 by the starter's share of PAs."""
    return 1.0 + STARTER_WEIGHT * (_clamp(f, FACTOR_CLAMP) - 1.0)


def matchup_factor(pid: int, market: str, season: int, date: str) -> float:
    """Multiplier on the season-rate lambda for tonight's matchup (1.0 = neutral).

    opponent factor: opposing starter's rate-against / league rate
    platoon factor:  batter's rate vs the starter's hand / his overall rate
    Every lookup failure or thin sample silently contributes 1.0.
    """
    try:
        me = _person(pid)
        team = (me or {}).get("currentTeam", {}).get("id")
        opp_pid = _opponent_starters(date).get(team) if team else None
        if not opp_pid:
            return 1.0
        lg = _league_rates(season)
        opp = _season_pitching(opp_pid, season)
        f_opp = 1.0
        if lg and opp and float(opp.get("battersFaced", 0) or 0) >= MIN_OPP_BF:
            if market == "home_runs":
                rate = float(opp.get("homeRuns", 0) or 0) / float(opp["battersFaced"])
                f_opp = rate / lg["hr_bf"] if lg["hr_bf"] > 0 else 1.0
            else:
                key = _OPP_RATE.get(market, "ops")
                num, den = _fnum(opp.get(key)), lg.get(key)
                if num is not None and den:
                    f_opp = num / den

        f_pl = 1.0
        if market in ("hits", "total_bases", "home_runs"):
            hand = ((_person(opp_pid) or {}).get("pitchHand") or {}).get("code")
            split = _platoon_splits(pid, season).get(hand or "", None)
            overall = season_hitting(pid, season)
            s_ab = float((split or {}).get("atBats", 0) or 0)
            o_ab = float((overall or {}).get("atBats", 0) or 0)
            if split and overall and s_ab >= MIN_SPLIT_AB and o_ab > 0:
                key = {"hits": "hits", "total_bases": "totalBases",
                       "home_runs": "homeRuns"}[market]
                s_rate = float(split.get(key, 0) or 0) / s_ab
                o_rate = float(overall.get(key, 0) or 0) / o_ab
                if o_rate > 0:
                    f_pl = s_rate / o_rate

        return _clamp(_damp(f_opp) * _damp(f_pl), TOTAL_CLAMP)
    except Exception:
        return 1.0


def project(name: str, market: str, season: int, slot: int | None = None,
            date: str | None = None) -> float | None:
    """lambda for a batter market, or None if unavailable / insufficient sample.

    With `date`, the season-rate lambda is scaled by matchup_factor() —
    opposing starter quality + platoon split for that day's game."""
    recipe = _RECIPE.get(market)
    if recipe is None:
        return None
    pid = player_id(name)
    if pid is None:
        return None
    stat = season_hitting(pid, season)
    if not stat:
        return None
    num_key, den_key, kind = recipe
    ab = float(stat.get("atBats", 0) or 0)
    pa = float(stat.get("plateAppearances", 0) or 0)
    if pa < 30:  # too small to trust a rate
        return None
    exp_pa = PA_BY_SLOT.get(slot, DEFAULT_PA)
    num = float(stat.get(num_key, 0) or 0)
    if kind == "ab":
        if ab <= 0:
            return None
        exp_ab = exp_pa * (ab / pa)          # expected at-bats this game
        lam = (num / ab) * exp_ab
    else:
        lam = (num / pa) * exp_pa            # per-PA markets (rbi, runs)
    if date:
        lam *= matchup_factor(pid, market, season, date)
    return lam
