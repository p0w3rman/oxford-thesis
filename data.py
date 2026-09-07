"""Loading the load profiles, the NYISO series and the tariff component tables."""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "params.yaml").read_text())

INTERVALS_PER_HOUR = 4
N_INTERVALS = 35040
N_HOURS = 8760


def path(key):
    return ROOT / CFG["paths"][key]


def load_population():
    """All-electric large offices with an on-site data center."""
    p = CFG["population"]
    df = pd.read_csv(path("population"))
    assert len(df) == p["n_source"], f"expected {p['n_source']} source models, got {len(df)}"
    mask = (df["subtype"] == p["subtype"]) & (df["heating_fuel"] == p["heating_fuel"])
    if p["exclude_district_energy"]:
        mask &= ~df["district_energy"]
    out = df[mask].set_index("bldg_id").sort_index()
    assert len(out) == p["n_expected"], f"expected {p['n_expected']} buildings, got {len(out)}"
    return out


def load_loads():
    """15-minute average kW, interval-beginning EST, one column per building."""
    pop = load_population()
    df = pd.read_parquet(path("loads")).astype("float64")
    df.columns = df.columns.astype(int)
    df = df[pop.index.to_list()]
    assert df.shape == (N_INTERVALS, CFG["population"]["n_expected"]), df.shape
    assert not df.isna().any().any(), "null values in the load matrix"
    peak_err = (df.max() - pop["annual_peak_kw"]).abs().max()
    assert peak_err < 0.01, f"computed peaks disagree with the shipped metadata by {peak_err}"
    return df


def load_hourly(loads):
    return loads.resample("h").mean()


def load_nyiso():
    """Zone J and NYCA hourly load plus day-ahead LBMP, hour-beginning EST."""
    df = pd.read_csv(path("nyiso"), index_col=0, parse_dates=True)
    assert len(df) == N_HOURS, len(df)
    assert not df.isna().any().any()
    # Gold Book cross-checks
    assert round(df.nyca_load_mw.max()) == 31861
    assert round(df.zonej_load_mw.max()) == 11070
    return df


def load_rate_components(rate):
    """Every priced component of one SC 9 schedule: one effective date's rows."""
    key = {"Rate I": "rate_i_components", "Rate II": "rate_ii_components"}[rate]
    as_of = pd.Timestamp(CFG["electric"]["components_as_of"])
    df = pd.read_csv(path(key), parse_dates=["effective_date"])
    eligible = df.effective_date[df.effective_date <= as_of]
    assert not eligible.empty, f"no {rate} vintage effective on or before {as_of.date()}"
    df = df[df.effective_date == eligible.max()].copy()
    df["rate"] = df["rate"].astype("float64")
    assert df.rate.notna().all(), "null rate in the component table"
    for cls in ("delivery", "supply", "tax"):
        assert (df.charge_class == cls).any(), f"{rate}: no {cls} components"
    return df.reset_index(drop=True)


def load_gas_tariff():
    return json.loads(path("gas_tariff").read_text())


def nyca_peak_hour(nyiso):
    """Hour-beginning EST of the NYCA coincident peak. Sets the ICAP tag."""
    return nyiso.nyca_load_mw.idxmax()


def icap_tags(hourly, nyiso):
    """Each building's demand in the NYCA coincident peak hour, kW."""
    return hourly.loc[nyca_peak_hour(nyiso)]
