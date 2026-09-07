"""Battery sizing and a rules-based grid-interactive dispatch.

One flat net-load target per building-month, found by bisection as the lowest level
the battery can hold all month. That single target caps every Rate II register at
once. Inside the DRV window the battery runs at full power and exports instead.
"""
import numpy as np
import pandas as pd

from data import CFG

TARGET = "target"
DRV = "drv"
BISECT_ITERS = 40


def charge_efficiency():
    return CFG["battery"]["roundtrip_efficiency_ac"] ** 0.5


def discharge_efficiency():
    return CFG["battery"]["roundtrip_efficiency_ac"] ** 0.5


def size_battery(peaks):
    """Battery sizing and cost basis, one row per building."""
    b = CFG["battery"]
    out = pd.DataFrame({"peak_kw": peaks})
    out["p_ac_kw"] = out.peak_kw * b["sizing_multiplier"]
    out["p_dc_kw"] = out.p_ac_kw / b["inverter_ac_dc_conversion"]
    out["e_dc_kwh"] = out.p_dc_kw * b["duration_h"]
    out["e_ac_kwh"] = out.e_dc_kwh * discharge_efficiency()   # deliverable, one direction
    out["capex_usd"] = out.p_dc_kw * b["capex_usd_per_kw_dc"]
    out["om_usd_yr"] = out.p_dc_kw * b["om_usd_per_kw_dc_yr"]
    out.index.name = "bldg_id"
    return out


def size_gas_engine(peaks):
    """Engine plant for full defection, sized 2N."""
    g = CFG["gas_engine"]
    out = pd.DataFrame({"peak_kw": peaks})
    out["firm_capacity_kw"] = out.peak_kw * g["sizing_multiplier"]
    out["installed_capacity_kw"] = out.firm_capacity_kw * g["redundancy_factor"]
    out["capex_usd"] = out.installed_capacity_kw * g["capex_usd_per_kw_ac"]
    out.index.name = "bldg_id"
    return out


def in_drv_window(ts):
    """Eligible for the DRV credit: summer weekdays, 11:00-15:00, minus two holidays."""
    d = CFG["drv"]
    if ts.weekday() >= 5:
        return False
    sm, sd = (int(x) for x in d["season_start"].split("-"))
    em, ed = (int(x) for x in d["season_end"].split("-"))
    if not ((sm, sd) <= (ts.month, ts.day) <= (em, ed)):
        return False
    if (ts.month, ts.day) == (7, 4):                          # Independence Day
        return False
    if ts.month == 9 and ts.weekday() == 0 and ts.day <= 7:   # Labor Day
        return False
    return d["call_window_start_hour"] <= ts.hour < d["call_window_end_hour"]


def drv_rate_usd_per_kwh():
    d = CFG["drv"]
    rate = d["rate_usd_per_kwh"]
    if d["loss_adjustment_applied"]:
        rate *= CFG["electric"]["rider_m_loss_factor"]
    return rate


def _simulate(load, ceiling, target, p_kw, e_kwh, soc, full_power, allow_export, hold):
    """One pass over a net-load target.

    `ceiling` is the 15-minute maximum within each hour and `target` is per hour.
    Returns charge, discharge, state of charge, and whether the target held.
    """
    eta_c, eta_d = charge_efficiency(), discharge_efficiency()
    n = len(load)
    chg, dis, soc_trace = np.zeros(n), np.zeros(n), np.zeros(n)
    held = True
    for i in range(n):
        avail = max(0.0, soc - hold[i])
        if full_power[i]:
            d = min(p_kw, avail * eta_d)
            if not allow_export[i]:
                d = min(d, load[i])
        else:
            d = min(max(0.0, ceiling[i] - target[i]), p_kw, avail * eta_d, load[i])
        if d > 0:
            dis[i] = d
            soc -= d / eta_d
        elif not full_power[i]:
            headroom = target[i] - ceiling[i]
            if headroom > 0:
                c = min(headroom, p_kw, (e_kwh - soc) / eta_c)
                chg[i] = c
                soc += c * eta_c
        if ceiling[i] + chg[i] - dis[i] > target[i] + 1e-6:
            held = False
        soc_trace[i] = soc
    return chg, dis, soc_trace, held


def monthly_targets(load_hourly, ceiling_hourly, p_kw, e_kwh, full_power, allow_export, hold):
    """Lowest sustainable flat target per month, by bisection.

    Each month is judged alone, starting from a full battery, with the committed
    blocks already accounted for.
    """
    out = {}
    for m, grp in load_hourly.groupby(load_hourly.index.month):
        sel = ceiling_hourly.index.month == m
        arr, ceil_m = grp.to_numpy(), ceiling_hourly[sel].to_numpy()
        lo, hi = 0.0, float(grp.max())
        for _ in range(BISECT_ITERS):
            mid = (lo + hi) / 2
            _, _, _, held = _simulate(arr, ceil_m, np.full(len(arr), mid), p_kw,
                                      e_kwh, e_kwh, full_power[sel],
                                      allow_export[sel], hold[sel])
            if held:
                hi = mid
            else:
                lo = mid
        out[m] = hi
    return pd.Series(out)


def dispatch(load_hourly, ceiling_hourly, p_kw, e_kwh, peak_hour, variant=DRV):
    """Run the schedule for one building over one year, hourly.

    The ICAP block's energy is reserved first; the target is set knowing that.
    """
    assert variant in (TARGET, DRV)
    eta_d = discharge_efficiency()
    idx = load_hourly.index
    hour = np.asarray(idx.hour)
    lo, hi = CFG["capacity"]["icap_block_hours"]

    on_peak_day = np.array([t.date() == peak_hour.date() for t in idx])
    in_drv = np.array([in_drv_window(t) for t in idx])
    # On the peak day the ICAP block replaces the DRV block.
    icap_block = on_peak_day & (hour >= lo) & (hour < hi)
    drv_block = (in_drv & ~on_peak_day) if variant == DRV else np.zeros(len(idx), bool)

    full_power = drv_block | icap_block
    allow_export = drv_block

    # Energy owed to ICAP-block hours still ahead, withheld from every other use.
    icap_req = np.where(icap_block, np.minimum(p_kw, load_hourly.to_numpy()) / eta_d, 0.0)
    hold = np.zeros(len(idx))
    owed_later = 0.0
    for i in range(len(idx) - 1, -1, -1):
        hold[i] = owed_later if on_peak_day[i] else 0.0
        owed_later += icap_req[i]

    targets = monthly_targets(load_hourly, ceiling_hourly, p_kw, e_kwh,
                              full_power, allow_export, hold)
    tgt = np.asarray(pd.Series(idx.month, index=idx).map(targets))

    L, C = load_hourly.to_numpy(), ceiling_hourly.to_numpy()
    chg, dis, soc_trace, _ = _simulate(L, C, tgt, p_kw, e_kwh, e_kwh,
                                       full_power, allow_export, hold)

    out = pd.DataFrame({"load_kw": L, "charge_kw": chg, "discharge_kw": dis,
                        "target_kw": tgt, "soc_kwh": soc_trace}, index=idx)
    out["net_kw"] = out.load_kw + out.charge_kw - out.discharge_kw
    out["export_kw"] = (-out.net_kw).clip(lower=0)
    out["drv_export_kwh"] = np.where(in_drv, out.export_kw, 0.0)
    return out


def fleet_dispatch(loads, hourly, peak_hour, variant=DRV):
    """Dispatch every building. Ceilings are the 15-minute maximum within each hour."""
    bat = size_battery(loads.max())
    ceilings = loads.resample("h").max()
    return {b: dispatch(hourly[b], ceilings[b], bat.p_ac_kw[b], bat.e_dc_kwh[b],
                        peak_hour, variant) for b in hourly.columns}


def net_loads_15min(loads, runs):
    """Hourly battery output expanded back onto the 15-minute load series."""
    out = {}
    for b, r in runs.items():
        delta = (r.charge_kw - r.discharge_kw).reindex(loads.index, method="ffill")
        out[b] = loads[b] + delta
    return pd.DataFrame(out)


def reserve_revenue(run, p_kw, asp):
    """10-minute spinning reserve, which pays for availability rather than energy.

        r = min(P - discharge - charge,  eta_d * SOC / h)

    Offerable power is what is unused, capped by what the stored energy can actually
    deliver for an hour.
    """
    d = CFG["der"]
    if asp is None:
        return 0.0
    price = asp[d["reserve_product"]].reindex(run.index) / 1000.0
    headroom = (p_kw - run.discharge_kw - run.charge_kw).clip(lower=0)
    sustainable = run.soc_kwh / d["reserve_sustain_hours"]
    offered = pd.concat([headroom, sustainable], axis=1).min(axis=1).clip(lower=0)
    return float((offered * price).sum())


def ecbl(metered_kw, add_back=None):
    """Economic Customer Baseline Load, M-38 section 7.5.1.

    For each weekday hour: take the same hour on the ten previous weekdays, sort,
    and average the 5th and 6th. Dispatched intervals enter the window at metered
    load plus the reduction. Net injections floor at zero.
    """
    n = CFG["der"]["ecbl_like_weekdays"]
    contrib = metered_kw.clip(lower=0.0)
    if add_back is not None:
        contrib = (metered_kw + add_back.reindex(metered_kw.index).fillna(0.0)).clip(lower=0.0)

    # hourly profile of each weekday: {date: {hour: kW}}
    profile = {}
    for stamp, value in contrib.items():
        if stamp.weekday() < 5:
            profile.setdefault(stamp.date(), {})[stamp.hour] = value
    weekdays = sorted(profile)

    baseline = pd.Series(np.nan, index=metered_kw.index)
    for i in range(n, len(weekdays)):
        window = weekdays[i - n:i]
        for hour in range(24):
            ten = sorted(profile[day].get(hour, 0.0) for day in window)
            mid_range = (ten[n // 2 - 1] + ten[n // 2]) / 2
            stamp = pd.Timestamp(weekdays[i]) + pd.Timedelta(hours=hour)
            if stamp in baseline.index:
                baseline[stamp] = mid_range
    return baseline


def demand_reduction(metered_kw, price, offer_usd_per_mwh):
    """Paid demand reduction, kWh per hour, credited only when the offer clears.

    The baseline is seeded without an add-back, then re-solved with it.
    """
    cleared = price.reindex(metered_kw.index) > offer_usd_per_mwh
    cleared |= pd.Series([in_drv_window(t) for t in metered_kw.index],
                         index=metered_kw.index)

    baseline = ecbl(metered_kw)
    for _ in range(CFG["der"]["ecbl_iterations"]):
        reduction = (baseline - metered_kw).clip(lower=0).where(cleared, 0.0).fillna(0.0)
        baseline = ecbl(metered_kw, reduction)

    dispatched = cleared & baseline.notna()
    dr = (baseline - metered_kw).clip(lower=0).where(dispatched, 0.0).fillna(0.0)
    return pd.DataFrame({"dr_kwh": dr, "price": price.reindex(metered_kw.index)})
