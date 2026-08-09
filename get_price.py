import os, zipfile, glob
import pandas as pd

def load_ireland_price_ember(zip_path, out_csv="./data/price_dayahead_ember.csv",
                             country="Ireland"):
    """Ember 'European wholesale price - hourly' ZIP'inden İrlanda gün-öncesi fiyatını
    çıkarır. Kolon adlarını OTOMATİK tespit eder (şema değişse de kırılmaz)."""
    ext = os.path.splitext(zip_path)[0] + "_extracted"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ext)
    csvs = glob.glob(os.path.join(ext, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("ZIP içinde CSV yok.")
    print("CSV'ler:", [os.path.basename(c) for c in csvs])
    # Önce İrlanda'ya özel dosya var mı; yoksa derlenmiş dosyayı oku ve filtrele
    cand = [c for c in csvs if "irela" in os.path.basename(c).lower()
            or os.path.basename(c).lower() in ("ie.csv", "irl.csv")]
    frames = []
    for f in (cand if cand else csvs):
        df = pd.read_csv(f)
        ccol = next((col for col in df.columns
                     if df[col].astype(str).str.contains(country, case=False, na=False).any()),
                    None)
        if ccol is not None:
            df = df[df[ccol].astype(str).str.contains(country, case=False, na=False)]
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError(f"'{country}' satırı bulunamadı.")
    raw = pd.concat(frames, ignore_index=True)
    low = {c.lower(): c for c in raw.columns}
    dcol = next((low[c] for c in low if "date" in c or "time" in c), None)
    pcol = next((low[c] for c in low if "price" in c or "eur" in c), None)
    if not dcol or not pcol:
        raise ValueError(f"Tarih/fiyat kolonu bulunamadı. Kolonlar: {list(raw.columns)}")
    out = (pd.DataFrame({
            "datetime_utc": pd.to_datetime(raw[dcol], utc=True, errors="coerce"),
            "price_eur_mwh": pd.to_numeric(raw[pcol], errors="coerce")})
           .dropna().sort_values("datetime_utc").reset_index(drop=True))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[Ember] {len(out)} satır -> {out_csv}")
    return out


if __name__ == "__main__":
    load_ireland_price_ember("european_wholesale_electricity_price_data_hourly.zip")