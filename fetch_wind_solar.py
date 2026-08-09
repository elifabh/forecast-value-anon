import requests, time, os
import pandas as pd

URL = "https://www.smartgriddashboard.com/api/chart/"
# ---------- AYARLAR ----------
START  = "2023-01-01"   # uzun geçmiş istersen "2019-01-01" yap (daha uzun sürer)
END    = "2026-06-01"
REGION = "ROI"
OUT    = "./data/eirgrid_wind_solar.csv"
VARS   = {"wind": "windactual", "solar": "solaractual"}  # demand YOK (o ENTSO-E'den)

def fetch_day(chart, area, day):
    p = {"region": REGION, "chartType": chart, "dateRange": "day",
         "dateFrom": day, "dateTo": day, "areas": area}
    r = requests.get(URL, params=p, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    rows = r.json().get("Rows", [])
    return [(x.get("EffectiveTime"), x.get("Value")) for x in rows]

def main():
    days = pd.date_range(START, END, freq="D")
    store = {v: {} for v in VARS}
    for i, d in enumerate(days):
        dstr = d.strftime("%d-%b-%Y")
        for v, area in VARS.items():
            try:
                for t, val in fetch_day(v, area, dstr):
                    store[v][t] = val
            except Exception as e:
                print(f"  {dstr} {v} HATA: {e}")
            time.sleep(0.15)
        if i % 30 == 0:
            print(f"{dstr} ... ({i+1}/{len(days)} gün)")
    df = pd.concat([pd.Series(store["wind"], name="wind_mw"),
                    pd.Series(store["solar"], name="solar_mw")], axis=1)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index(); df.index.name = "datetime"
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    df.to_csv(OUT)
    print(f"\nBitti: {len(df)} satır -> {OUT}")
    print("wind dolu:", int(df['wind_mw'].notna().sum()),
          "| solar dolu:", int(df['solar_mw'].notna().sum()))

if __name__ == "__main__":
    main()