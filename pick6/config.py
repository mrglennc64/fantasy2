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

# Base power-play multiplier by number of picks (all legs must hit).
# TODO: confirm live — DK also runs "flex"/partial-win variants not modeled here.
BASE_MULTIPLIER = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
    6: 35.0,
}

MIN_PICKS = 2
MAX_PICKS = 6


def entry_multiplier(n_picks: int, leg_boosts: list[float] | None = None) -> float:
    """Total payout multiplier for an n-pick power play, including per-leg boosts.

    entry pays  base(n) * product(leg_boosts)  on a win, 0 on a loss.
    """
    if n_picks not in BASE_MULTIPLIER:
        raise ValueError(f"unsupported pick count {n_picks} (support {MIN_PICKS}-{MAX_PICKS})")
    m = BASE_MULTIPLIER[n_picks]
    if leg_boosts:
        for b in leg_boosts:
            m *= b
    return m


def breakeven_per_leg(n_picks: int, leg_boosts: list[float] | None = None) -> float:
    """The per-leg true win probability an entry must average to be +EV.

    Solve  p**n * M = 1  ->  p = (1/M) ** (1/n).  With heterogeneous boosts the
    geometric-mean leg probability must clear this.
    """
    m = entry_multiplier(n_picks, leg_boosts)
    return (1.0 / m) ** (1.0 / n_picks)


def entry_ev(leg_probs: list[float], leg_boosts: list[float] | None = None) -> float:
    """Expected profit per $1 staked on a power-play entry (legs independent).

    EV = P(all hit) * (M - 1) - P(any miss) * 1 = (prod p) * M - 1.
    """
    n = len(leg_probs)
    p_all = 1.0
    for p in leg_probs:
        p_all *= p
    m = entry_multiplier(n, leg_boosts)
    return p_all * m - 1.0


if __name__ == "__main__":  # quick reference table
    print("picks  base_mult  breakeven_per_leg")
    for n in range(MIN_PICKS, MAX_PICKS + 1):
        print(f"  {n}      {BASE_MULTIPLIER[n]:5.1f}x        {breakeven_per_leg(n)*100:5.1f}%")
