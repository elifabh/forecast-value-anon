import io, os
import pandas as pd

def _read_one(path, country):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    lines = [ln[1:-1] if (ln.startswith('"') and ln.endswith('"')) else ln for ln in lines]
    header = lines[0]
    sep = "\t" if "\t" in header else (";" if ";" in header else ",")
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["CountryCode"] = df["CountryCode"].astype(str).str.strip()
    df = df[df["CountryCode"] == country].copy()
    df["datetime_utc"] = pd.to_datetime(df["DateUTC"], dayfirst=True, errors="coerce", utc=True)
    df["demand_mw"]   = pd.to_numeric(df["Value"], errors="coerce")
    return df[["datetime_utc", "demand_mw"]]

def load_ireland_demand(paths, out_csv="./data/demand_ie_powerstats.csv", country="IE"):
    out = pd.concat([_read_one(p, country) for p in paths], ignore_index=True)
    out = (out.dropna().sort_values("datetime_utc")
              .drop_duplicates("datetime_utc").reset_index(drop=True))
    if out.empty:
        raise ValueError(f"'{country}' icin satir yok.")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[demand] {len(out)} satir -> {out_csv} ({out.datetime_utc.min()} .. {out.datetime_utc.max()})")
    return out

if __name__ == "__main__":
    load_ireland_demand([
        "monthly_hourly_load_values_2022.csv",
        "monthly_hourly_load_values_2023.csv",
        "monthly_hourly_load_values_2024.csv",
        "monthly_hourly_load_values_2025.csv",
    ])  # istersen basa "..._2022.csv" ekle