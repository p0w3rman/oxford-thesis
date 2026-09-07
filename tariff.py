"""Con Edison SC 9 electric bill and SC 2 gas bill.

Demand is billed on the highest integrated 30-minute value. Delivery uses the 2026
schedule; supply energy uses 2018 day-ahead LBMP.
"""
import numpy as np
import pandas as pd

from data import CFG, INTERVALS_PER_HOUR, load_gas_tariff, load_rate_components

RATE_I = "Rate I"
RATE_II = "Rate II"
DEMAND_DELIVERY = "Demand Delivery Charge"


def billing_demand(load_15min):
    """15-minute kW rolled to the 30-minute demand the tariff bills on."""
    n = CFG["electric"]["demand_intervals_15min"]
    if n == 1:
        return load_15min
    return load_15min.rolling(n, min_periods=n).mean()


def register_mask(index, spec):
    """Mask for one time-of-day register. Hours are half-open, [lo, hi)."""
    summer = CFG["electric"]["summer_months"]
    m = np.ones(len(index), dtype=bool)
    if spec["months"] == "summer":
        m &= index.month.isin(summer)
    elif spec["months"] == "winter":
        m &= ~index.month.isin(summer)
    if spec["weekdays"]:
        m &= index.weekday < 5
    lo, hi = spec["hours"]
    if (lo, hi) != (0, 24):
        m &= (index.hour >= lo) & (index.hour < hi)
    return m


def monthly_register_peaks(load_15min, specs):
    """Monthly maximum billing demand per register, kW. An empty register bills 0."""
    demand = billing_demand(load_15min)
    out = {}
    for name, spec in specs.items():
        masked = demand.where(register_mask(demand.index, spec))
        out[name] = masked.groupby(masked.index.month).max()
    return pd.DataFrame(out).reindex(range(1, 13)).fillna(0.0)


def _sum_rate(comp, charge_class, basis, exclude=()):
    sel = comp[(comp.charge_class == charge_class) & (comp.billing_basis == basis)]
    for name in exclude:
        sel = sel[~sel.rate_name.str.startswith(name)]
    return float(sel.rate.sum())


def demand_delivery(load_15min, rate):
    """Delivery demand for the year. Rate II registers overlap and their charges add."""
    e = CFG["electric"]
    if rate == RATE_II:
        peaks = monthly_register_peaks(load_15min, e["registers"])
        charges = peaks.mul(pd.Series(e["rate_ii_demand_usd_per_kw"]))
    else:
        peaks = monthly_register_peaks(load_15min, {"monthly_max": e["rate_i_register"]})
        r = e["rate_i_demand_usd_per_kw"]
        seasonal = pd.Series([r["summer"] if m in e["summer_months"] else r["winter"]
                              for m in range(1, 13)], index=range(1, 13))
        charges = peaks.mul(seasonal, axis=0)
    return float(charges.to_numpy().sum())


def demand_adders(load_15min, rate, comp):
    """Per-kW General Rule 26 surcharges, billed on their own determinant."""
    e = CFG["electric"]
    per_kw = _sum_rate(comp, "delivery", "demand", exclude=(DEMAND_DELIVERY,))
    spec = e["adder_register"] if rate == RATE_II else e["rate_i_register"]
    peaks = monthly_register_peaks(load_15min, {"adder": spec})["adder"]
    return float(peaks.sum() * per_kw)


def supply_capacity(icap_tag_kw, rate, comp):
    """Market Supply Charge, billed on the ICAP tag.

    The tag is inclusive of losses, so no gross-up is applied.
    """
    is_capacity = (comp.rate_name.str.startswith("Market Supply Charge - Capacity")
                   | comp.rate_name.str.startswith("Capacity Component"))
    sel = comp[is_capacity]
    assert not sel.empty, f"{rate}: no capacity supply component found"
    if rate == RATE_II:
        n_summer = len(CFG["electric"]["summer_months"])
        by_period = sel.groupby(sel.period.str.startswith("Jun-Sep")).rate.sum()
        months = pd.Series({True: n_summer, False: 12 - n_summer})
        return float(icap_tag_kw * (by_period * months).sum())
    return float(icap_tag_kw * 12 * sel.rate.sum())


def energy_charges(kwh_hourly, dam_usd_mwh, comp):
    """Per-kWh delivery and supply charges, and the Rider M commodity."""
    e = CFG["electric"]
    annual_kwh = float(kwh_hourly.sum())
    vendor = comp.loc[comp.rate_type == "LossFactor", "rate"]
    assert len(vendor) == 1 and float(vendor.iloc[0]) == e["rateacuity_loss_row_pct"], \
        "the vendor loss row has changed; recheck it against General Rule 25.1"
    dam = dam_usd_mwh.reindex(kwh_hourly.index)
    assert dam.notna().all(), "load hours not covered by the DAM series"
    return {
        "delivery_energy": annual_kwh * _sum_rate(comp, "delivery", "energy"),
        "supply_energy_adders": annual_kwh * _sum_rate(comp, "supply", "energy"),
        "supply_commodity": float((kwh_hourly * dam / 1000.0).sum() * e["rider_m_loss_factor"]),
    }


def taxes(delivery, commodity, comp):
    """GRT by base, then sales tax on the grossed-up total."""
    pct = comp[comp.charge_class == "tax"].set_index("rate_name").rate / 100.0
    grt = (delivery * (pct["TAX: Delivery Rates and Charges"]
                       + pct["TAX: Additional percentage increase to Delivery Rates and Charges"])
           + commodity * pct["TAX: Commodity Rates and Charges"])
    sales = (delivery + commodity + grt) * pct["TAX: Sales Tax"]
    return {"gross_receipts_tax": float(grt), "sales_tax": float(sales)}


def annual_bill(load_15min, rate, icap_tag_kw, dam_usd_mwh):
    """One building's annual electric bill, itemized, in nominal 2026 dollars."""
    comp = load_rate_components(rate)
    kwh_hourly = load_15min.resample("h").mean()
    energy = energy_charges(kwh_hourly, dam_usd_mwh, comp)

    items = {
        "delivery_demand": demand_delivery(load_15min, rate),
        "delivery_demand_adders": demand_adders(load_15min, rate, comp),
        "delivery_energy": energy["delivery_energy"],
        "delivery_fixed": 12.0 * _sum_rate(comp, "delivery", "fixed"),
        "supply_capacity_icap": supply_capacity(icap_tag_kw, rate, comp),
        "supply_commodity": energy["supply_commodity"],
        "supply_energy_adders": energy["supply_energy_adders"],
    }
    delivery = sum(v for k, v in items.items() if k.startswith("delivery"))
    commodity = sum(v for k, v in items.items() if k.startswith("supply"))
    items.update(taxes(delivery, commodity, comp))
    items["total"] = delivery + commodity + items["gross_receipts_tax"] + items["sales_tax"]
    items["annual_kwh"] = float(kwh_hourly.sum())
    return pd.Series(items)


def fleet_bills(loads, tags, dam_usd_mwh, rate=None):
    """Every building's annual bill on the billed rate."""
    rate = rate or CFG["electric"]["rate"]
    out = {b: annual_bill(loads[b], rate, float(tags[b]), dam_usd_mwh) for b in loads.columns}
    df = pd.DataFrame(out).T
    df.index.name = "bldg_id"
    return df


def rate_ii_is_mandatory(loads):
    """Rate II is mandatory above 1,500 kW of billing demand in any month."""
    thresh = CFG["electric"]["rate_ii_mandatory_above_kw"]
    demand = pd.DataFrame({b: billing_demand(loads[b]) for b in loads.columns})
    return (demand.resample("MS").max() > thresh).any()


# ------------------------------------------------------------------ gas, SC 2
def therms_per_kwh():
    g = CFG["gas_engine"]
    return g["heat_rate_btu_per_kwh_hhv"] / g["btu_per_therm"]


def sc2_block_cost(therms, rate_option, tariff):
    """Delivery cost for one meter-month under the declining-block base rate."""
    key = "base_delivery_rate_I" if rate_option == RATE_I else "base_delivery_rate_II"
    blocks = next(c for c in tariff["charges"] if c["id"] == key)["blocks"]
    total, remaining = 0.0, therms
    for b in blocks:
        lo, hi = b["from_therms"], b["to_therms"]
        width = (hi - lo) if hi is not None else float("inf")
        take = min(remaining, width)
        if take <= 0:
            break
        total += b["fixed_charge"] if b["fixed_charge"] is not None else take * b["rate"]
        remaining -= take
    return total


def _charge(cid, tariff):
    return next(c for c in tariff["charges"] if c["id"] == cid)


def gas_bill(load_15min, gas_commodity_usd_per_therm, rate_option=None):
    """Annual SC 2 gas bill. Delivery blocks are assessed per meter-month."""
    rate_option = rate_option or CFG["gas"]["rate"]
    t = load_gas_tariff()
    opt = "rate_I" if rate_option == RATE_I else "rate_II"

    monthly_kwh = load_15min.resample("MS").sum() / INTERVALS_PER_HOUR
    monthly_therms = monthly_kwh * therms_per_kwh()
    therms = float(monthly_therms.sum())

    adders = t["aggregates_aug_2026"]["delivery_per_therm_adders_total_sc2"][opt]
    mfc = _charge("mfc", t).get("rate")
    if isinstance(mfc, dict) or mfc is None:
        mfc = _charge("mfc", t)["rate_by_option"][opt]

    items = {
        "gas_delivery_blocks": float(sum(sc2_block_cost(x, rate_option, t)
                                         for x in monthly_therms)),
        "gas_delivery_adders": therms * adders,
        "gas_billing_charge": 12.0 * float(
            _charge("billing_payment_processing", t)["rate_by_case"]["single_service_gas"]),
        "gas_commodity": therms * gas_commodity_usd_per_therm,
        "gas_merchant_function": therms * float(mfc),
    }
    delivery = (items["gas_delivery_blocks"] + items["gas_delivery_adders"]
                + items["gas_billing_charge"])
    commodity = items["gas_commodity"] + items["gas_merchant_function"]

    grt = (delivery * float(_charge("grt_delivery_nonresidential", t)["rate"]) / 100.0
           + commodity * float(_charge("grt_commodity_nonresidential", t)["rate"]) / 100.0)
    sales = (delivery + commodity + grt) * float(_charge("sales_tax", t)["rate"]) / 100.0
    items["gas_gross_receipts_tax"] = grt
    items["gas_sales_tax"] = sales
    items["total"] = delivery + commodity + grt + sales
    items["therms"] = therms
    return pd.Series(items)
