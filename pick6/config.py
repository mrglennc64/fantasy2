"""DraftKings Pick6 payout model + per-leg breakeven math.

Pick6 is a pick'em: DK sets ONE projection number per player, you choose
More/Less, and a *power play* pays a fixed multiplier only if EVERY leg hits
(all-or-nothing). Some legs carry a per-leg "flex boost" (e.g. 1.1x More on a
line DK wants action on, 0.9x on one they think is likely) that multiplies into
the entry payout.

There is NO sportsbook vig here — the edge is entirely (your calibrated
P(side)) vs (DK's line + the breakeven the multiplier demands). See
[[strike-app-layout]]: the single-leg model over-projects, so calibration is
the whole game before any of these breakevens can be trusted.

*** VERIFY the multipliers against the live DK Pick6 board before staking.
DK changes them and runs promos; these are the commonly published power-play
values as of 2026-07. ***
"""
from __future__ import annotations

# Per-platform "power play" (all-legs-must-hit) payout multiplier by pick count.
# Each pick'em ENTRY lives on ONE platform; the projection is platform-agnostic
# but the LINE and the PAYOUT differ, so line-shopping across apps is real edge.
# DK Pick6 is just one app — PrizePicks is the biggest; the pick'em ecosystem
# also includes Underdog, Sleeper, Betr, ParlayPlay, Dabble, Chalkboard, etc.
# *** VERIFY LIVE — these are APPROXIMATE power-play tables. Apps change them,
# run flex/insured/partial variants (not modeled here), and vary by promo. Fix
# each app's real numbers before trusting its breakevens. ***
PLATFORMS = {
    "prizepicks": {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 37.5},   # Power Play
    "underdog":   {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 35.0},
    "dk_pick6":   {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 35.0},
    "sleeper":    {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 25.0},
    "betr":       {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 35.0},
    "parlayplay": {2: 3.0, 3: 6.0, 4: 10.0, 5: 25.0, 6: 50.0},
}
DEFAULT_PLATFORM = "dk_pick6"  # only a fallback for board rows with no platform
BASE_MULTIPLIER = PLATFORMS[DEFAULT_PLATFORM]  # back-compat alias

MIN_PICKS = 2
MAX_PICKS = 6
ENTRY_SIZES = [2, 3, 4, 5]  # pick counts the picker builds/compares


def entry_multiplier(n_picks: int, leg_boosts: list[float] | None = None,
                     platform: str = DEFAULT_PLATFORM) -> float:
    """Total payout multiplier for an n-pick power play on `platform`, incl. boosts.

    entry pays  base(platform, n) * product(leg_boosts)  on a win, 0 on a loss.
    """
    table = PLATFORMS.get(platform)
    if table is None:
        raise ValueError(f"unknown platform {platform!r} (have {list(PLATFORMS)})")
    if n_picks not in table:
        raise ValueError(f"{platform}: unsupported pick count {n_picks}")
    m = table[n_picks]
    if leg_boosts:
        for b in leg_boosts:
            m *= b
    return m


def breakeven_per_leg(n_picks: int, leg_boosts: list[float] | None = None,
                      platform: str = DEFAULT_PLATFORM) -> float:
    """Per-leg true win prob an entry must average to be +EV: p = (1/M)**(1/n)."""
    m = entry_multiplier(n_picks, leg_boosts, platform)
    return (1.0 / m) ** (1.0 / n_picks)


def entry_ev(leg_probs: list[float], leg_boosts: list[float] | None = None,
             platform: str = DEFAULT_PLATFORM) -> float:
    """Expected profit per $1 staked (legs independent): (prod p) * M - 1."""
    n = len(leg_probs)
    p_all = 1.0
    for p in leg_probs:
        p_all *= p
    return p_all * entry_multiplier(n, leg_boosts, platform) - 1.0


if __name__ == "__main__":  # breakeven-per-leg reference across platforms
    print(f"  {'picks':>5}", *(f"{p:>12}" for p in PLATFORMS))
    for n in ENTRY_SIZES:
        cells = []
        for p in PLATFORMS:
            m = PLATFORMS[p][n]
            cells.append(f"{m:.0f}x/{breakeven_per_leg(n, platform=p)*100:.1f}%")
        print(f"  {n:>5}", *(f"{c:>12}" for c in cells))
