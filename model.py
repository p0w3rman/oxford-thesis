#!/usr/bin/env python
"""The three configurations, their annual cost, and whole-life value.

    grid reliance   the electric bill plus the cost of unserved energy
    full defection  a gas bill plus engine O&M, on plant sized 2N
    cooperation     a reduced electric bill less market revenue, plus battery O&M

All figures are nominal. Streams escalate at CPI; the DRV credit is held flat.

    python model.py
"""
import numpy as np
import pandas as pd

import battery as bat
import data
import tariff
from data import CFG, load_rate_components


def voll_usd_per_kwh(duration_h):
    """Cost per unserved kWh at a given duration, 2026 dollars.

    Interpolated between the published points and held flat outside them.
    """
    table = CFG["reliability"]["voll_usd_per_kwh"]
    keys = sorted(table)
    return float(np.interp(duration_h, keys, [table[k] for k in keys]))


def outage_cost(unserved_kwh_per_event):
    r = CFG["reliability"]
    return float(r["saifi_per_yr"] * unserved_kwh_per_event * voll_usd_per_kwh(r["caidi_hours"]))


def unserved_grid(load_hourly):
    """Unserved kWh per outage with no on-site generation: mean load x duration."""
    return float(load_hourly.mean() * CFG["reliability"]["caidi_hours"])


def unserved_with_storage(load_hourly, soc_kwh, p_kw):
    """Unserved kWh per outage, ridden through from the charge held when it starts."""
    d = CFG["reliability"]["caidi_hours"]
    deliverable = np.minimum(soc_kwh * bat.discharge_efficiency(), p_kw * d)
    return float((load_hourly * d - deliverable).clip(lower=0).mean())


def export_threshold_usd_per_mwh(dam, rate=None):
    """Day-ahead price at which an export covers the delivered cost of stored energy.

        (mean daily minimum price x loss factor + flat adders) x tax / round-trip
    """
    c = load_rate_components(rate or CFG["electric"]["rate"])
    charge_price = float(dam.groupby(dam.index.date).min().mean())
    flat = float(c[c.charge_class.isin(["delivery", "supply"])
                   & (c.billing_basis == "energy")].rate.sum()) * 1000.0
    pct = c[c.charge_class == "tax"].set_index("rate_name").rate
    tax = 1 + (pct["TAX: Delivery Rates and Charges"] + pct["TAX: Sales Tax"]) / 100
    lf = CFG["electric"]["rider_m_loss_factor"]
    return (charge_price * lf + flat) * tax / CFG["battery"]["roundtrip_efficiency_ac"]


def cooperation(loads, nyiso, tags, baseline, runs, spot_price, asp):
    """What the cooperative configuration earns and costs, per building.

    The dispatched net load is scored through the same bill engine as the baseline.
    """
    cap = CFG["capacity"]
    dam = nyiso.dam_lbmp_usd_mwh
    net15 = bat.net_loads_15min(loads, runs)
    net_h = net15.resample("h").mean()
    new_tags = net_h.loc[nyiso.nyca_load_mw.idxmax()].clip(lower=0)

    scen = tariff.fleet_bills(net15, new_tags, dam)
    sized = bat.size_battery(loads.max())
    offer = export_threshold_usd_per_mwh(dam)
    excess = (sized.p_ac_kw - tags).clip(lower=0)      # capacity above the host's tag

    out = pd.DataFrame({
        "baseline_bill_usd": baseline.total,
        "cooperative_bill_usd": scen.total,
        "icap_tag_base_kw": tags,
        "icap_tag_new_kw": new_tags,
    })
    out["bill_saving_usd"] = out.baseline_bill_usd - out.cooperative_bill_usd
    out["drv_revenue_usd"] = pd.Series(
        {b: r.drv_export_kwh.sum() * bat.drv_rate_usd_per_kwh() for b, r in runs.items()})
    # exports settle at the busbar, so no loss gross-up
    out["export_energy_revenue_usd"] = pd.Series(
        {b: float((r.export_kw * (dam.reindex(r.index) / 1000.0)).sum())
         for b, r in runs.items()})
    out["capacity_revenue_usd"] = (excess * cap["duration_factor"]
                                   * cap["availability_factor"] * spot_price * 12)
    dr = {b: bat.demand_reduction(net_h[b], dam, offer) for b in net_h.columns}
    out["dr_energy_revenue_usd"] = pd.Series(
        {b: float((d.dr_kwh * d.price / 1000.0).sum()) for b, d in dr.items()})
    out["reserve_revenue_usd"] = pd.Series(
        {b: bat.reserve_revenue(r, sized.p_ac_kw[b], asp) for b, r in runs.items()})
    out["battery_om_usd"] = sized.om_usd_yr
    out["battery_capex_usd"] = sized.capex_usd
    out["net_annual_benefit_usd"] = (
        out.bill_saving_usd + out.drv_revenue_usd + out.export_energy_revenue_usd
        + out.dr_energy_revenue_usd + out.reserve_revenue_usd
        + out.capacity_revenue_usd - out.battery_om_usd)
    out.index.name = "bldg_id"
    return out


def islanded(loads, gas_commodity_usd_per_therm):
    """Annual cost of running each building on its own engines."""
    g = CFG["gas_engine"]
    rows = {}
    for b in loads.columns:
        kwh = float(loads[b].sum()) / data.INTERVALS_PER_HOUR
        rows[b] = {"gas_bill_usd": tariff.gas_bill(loads[b], gas_commodity_usd_per_therm).total,
                   "engine_om_usd": kwh * g["om_usd_per_kwh_ac"]}
    out = pd.DataFrame(rows).T
    out["total_operating_usd"] = out.gas_bill_usd + out.engine_om_usd
    out["engine_capex_usd"] = bat.size_gas_engine(loads.max()).capex_usd
    out.index.name = "bldg_id"
    return out


def _years():
    return np.arange(1, CFG["finance"]["analysis_years"] + 1)


def cashflow(streams, capex):
    """A scenario's yearly operating streams and its year-0 capital.

    Year 1 is the base amount unescalated.
    """
    esc = CFG["finance"]["escalation"]
    years = _years()
    by_year = pd.DataFrame(index=years)
    by_year.index.name = "year"
    for name, base in streams.items():
        rate = esc["drv_rate"] if name == "drv_revenue" else esc["cpi"]
        by_year[name] = base * (1 + rate) ** (years - 1)
    by_year["operating_cost"] = by_year.sum(axis=1)
    return {"by_year": by_year, "capex": capex}


def present_value(scenario, case="central"):
    """Present value of total cost, capital included at year 0."""
    r = CFG["finance"]["discount_rate_nominal"][case]
    operating = scenario["by_year"].operating_cost.to_numpy()
    return float((operating * (1 + r) ** -_years()).sum()) + scenario["capex"]


def _annual_saving(alt, base):
    return (base["by_year"].operating_cost - alt["by_year"].operating_cost).to_numpy()


def irr_against(alt, base):
    """Rate at which `alt` and `base` cost the same. NaN if it never pays back."""
    extra_capital = alt["capex"] - base["capex"]
    saving = _annual_saving(alt, base)
    if extra_capital <= 0:
        return float("inf")
    if saving.sum() < extra_capital:
        return float("nan")
    low, high = -0.99, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        if float((saving * (1 + mid) ** -_years()).sum()) > extra_capital:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def payback_years(alt, base):
    """Years until cumulative saving covers the extra capital."""
    extra_capital = alt["capex"] - base["capex"]
    cumulative = np.cumsum(_annual_saving(alt, base))
    reached = np.flatnonzero(cumulative >= extra_capital)
    if not len(reached):
        return float("nan")
    i = reached[0]
    before = cumulative[i - 1] if i else 0.0
    return float(i + (extra_capital - before) / (cumulative[i] - before))


def summarize(scenarios, baseline_name):
    """Capital, year-1 cost, present value at each discount rate, IRR and payback."""
    base = scenarios[baseline_name]
    rows = []
    for name, s in scenarios.items():
        row = {"scenario": name, "capex_usd": s["capex"],
               "year1_operating_usd": s["by_year"].operating_cost.iloc[0]}
        for case in CFG["finance"]["discount_rate_nominal"]:
            row[f"pv_{case}"] = present_value(s, case)
        if name != baseline_name:
            row["pv_saving_vs_baseline"] = present_value(base) - present_value(s)
            row["irr"] = irr_against(s, base)
            row["payback_yr"] = payback_years(s, base)
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def main():
    loads = data.load_loads()
    hourly = data.load_hourly(loads)
    nyiso = data.load_nyiso()
    dam = nyiso.dam_lbmp_usd_mwh
    peak_hour = data.nyca_peak_hour(nyiso)
    tags = data.icap_tags(hourly, nyiso)
    asp = pd.read_csv(data.path("asp"), index_col=0, parse_dates=True)
    spot = float(pd.read_csv(data.path("icap_spot")).nyc_usd_per_kw_month.mean())

    mandatory = tariff.rate_ii_is_mandatory(loads)
    print(f"{len(loads.columns)} buildings, {loads.sum().sum() / 4 / 1e6:,.1f} GWh, "
          f"peak hour {peak_hour}")
    print(f"Rate II mandatory for {int(mandatory.sum())} of {len(mandatory)}; "
          f"billed uniformly on {CFG['electric']['rate']}\n")

    baseline = tariff.fleet_bills(loads, tags, dam)
    runs = bat.fleet_dispatch(loads, hourly, peak_hour, bat.DRV)
    sized = bat.size_battery(loads.max())
    coop = cooperation(loads, nyiso, tags, baseline, runs, spot, asp)
    isl = islanded(loads, CFG["gas"]["commodity_usd_per_therm"])

    outage_grid = sum(outage_cost(unserved_grid(hourly[b])) for b in hourly.columns)
    outage_coop = sum(outage_cost(unserved_with_storage(hourly[b], runs[b].soc_kwh,
                                                        sized.p_ac_kw[b]))
                      for b in hourly.columns)
    market = (coop.export_energy_revenue_usd + coop.dr_energy_revenue_usd
              + coop.reserve_revenue_usd + coop.capacity_revenue_usd).sum()

    scenarios = {
        "grid reliance": cashflow(
            {"electric_bill": baseline.total.sum(), "outage_cost": outage_grid},
            capex=0.0),
        "full defection": cashflow(
            {"gas_bill": isl.gas_bill_usd.sum(), "engine_om": isl.engine_om_usd.sum()},
            capex=isl.engine_capex_usd.sum()),
        "grid interactivity": cashflow(
            {"electric_bill": coop.cooperative_bill_usd.sum(),
             "battery_om": coop.battery_om_usd.sum(),
             "outage_cost": outage_coop,
             "drv_revenue": -coop.drv_revenue_usd.sum(),
             "market_revenue": -market},
            capex=coop.battery_capex_usd.sum()),
    }
    summary = summarize(scenarios, "grid reliance")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.0f}".format)
    print(f"Whole-life cost, {CFG['finance']['analysis_years']} years, nominal\n")
    print(summary.drop(columns=["irr"]).to_string())
    print("\nIRR vs grid reliance:")
    for name in ("full defection", "grid interactivity"):
        print(f"  {name:20} {summary.loc[name, 'irr']:.1%}")

    print("\nCooperative revenue stack, fleet, year 1")
    for k in ("bill_saving_usd", "drv_revenue_usd", "export_energy_revenue_usd",
              "dr_energy_revenue_usd", "reserve_revenue_usd", "capacity_revenue_usd",
              "battery_om_usd"):
        print(f"  {k:28} {coop[k].sum():>14,.0f}")
    print(f"  {'net annual benefit':28} {coop.net_annual_benefit_usd.sum():>14,.0f}")

    summary.to_csv("results_summary.csv")
    coop.to_csv("results_cooperation.csv")
    baseline.to_csv("results_baseline_bills.csv")
    print("\nwrote results_*.csv")


if __name__ == "__main__":
    main()
