import os
import pandas as pd
from entsoe import EntsoePandasClient

def fetch_dayahead_price_entsoe(start, end, api_key=None,
                                out_csv="./data/price_dayahead_entsoe.csv"):
    """İrlanda (SEM) gün-öncesi fiyatını ENTSO-E'den çeker.
    start/end: 'YYYY-MM-DD'. NOT: end gününün BİR SONRASINI ver (bitiş hariç tutulur).
    Çıktı: datetime_utc, price_eur_mwh."""
    api_key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError("ENTSO-E token yok: ENTSOE_API_KEY ayarla ya da api_key ver.")
    client = EntsoePandasClient(api_key=api_key)
    start_ts = pd.Timestamp(start, tz="Europe/Dublin")
    end_ts   = pd.Timestamp(end,   tz="Europe/Dublin")
    series = client.query_day_ahead_prices("IE_SEM", start=start_ts, end=end_ts)  # EUR/MWh
    df = series.rename("price_eur_mwh").to_frame()
    df.index = df.index.tz_convert("UTC"); df.index.name = "datetime_utc"
    df = df.reset_index()
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[ENTSO-E] {len(df)} satır -> {out_csv}")
    return df

# kullanım (ornek): fetch_dayahead_price_entsoe("2024-01-01", "2025-01-01")