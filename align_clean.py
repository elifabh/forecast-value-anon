import pandas as pd

PRICE     = "./data/price_dayahead_ember.csv"   # datetime_utc, price_eur_mwh (hourly, UTC)
DEMAND    = "./data/demand_ie_entsoe.csv"       # datetime_utc, demand_mw   (hourly, UTC) - switched from Power Stats
WINDSOLAR = "./data/eirgrid_wind_solar.csv"     # datetime, wind_mw, solar_mw (15-min)
WIND_SOLAR_TZ = "Europe/Dublin"   # assumption; verified by the solar-hour check
OUT       = "./data/aligned_30min.csv"
MAX_FILL  = 4                      # interpolate at most 4 consecutive steps (=2 hours); long gaps stay NaN


def _utc(s):
    # pin every series to tz-aware UTC (assume UTC if naive; convert to UTC if aware)
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
    return s

def load_series():
    p = pd.read_csv(PRICE, parse_dates=["datetime_utc"]).set_index("datetime_utc")["price_eur_mwh"].sort_index()
    d = pd.read_csv(DEMAND, parse_dates=["datetime_utc"]).set_index("datetime_utc")["demand_mw"].sort_index()
    ws = pd.read_csv(WINDSOLAR, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if ws.index.tz is None:
        ws.index = ws.index.tz_localize(WIND_SOLAR_TZ, ambiguous="NaT", nonexistent="shift_forward")
    ws.index = ws.index.tz_convert("UTC")
    return _utc(p), _utc(d), ws

def to_30min(p, d, ws):
    w30 = ws["wind_mw"].resample("30min").mean()
    s30 = ws["solar_mw"].resample("30min").mean()
    p30 = p.resample("30min").ffill(limit=1)
    d30 = d.resample("30min").ffill(limit=1)
    return pd.concat({"demand_mw": d30, "wind_mw": w30,
                      "solar_mw": s30, "price_eur_mwh": p30}, axis=1)

def main():
    p, d, ws = load_series()
    df = to_30min(p, d, ws)
    df["solar_mw"] = df["solar_mw"].clip(lower=0)   # <-- add this line
    df = df.interpolate(method="time", limit=MAX_FILL, limit_area="inside")
    joint = df.dropna()
    df.to_csv(OUT); joint.to_csv(OUT.replace(".csv", "_joint.csv"))
    print("Full grid:", df.shape, "|", df.index.min(), "->", df.index.max())
    print("\nCoverage (filled %):\n" + (df.notna().mean()*100).round(1).to_string())
    print("\nJOINT window (all 4 series filled):", joint.shape)
    if len(joint):
        print("  ", joint.index.min(), "->", joint.index.max())
        print("\nStatistics (joint):\n" + joint.describe().round(1).to_string())
    sol = df["solar_mw"].dropna()
    if len(sol):
        bh = sol.groupby(sol.index.hour).mean().round(1)
        print("\nSolar mean by UTC hour (peak expected ~11-12):\n" + bh.to_string())

if __name__ == "__main__":
    main()