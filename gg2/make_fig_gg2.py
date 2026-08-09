#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_gg2.py (v2) — GridGuard-2 boundary-evidence figures for the GG-1 paper.

v2 CHANGES vs v1
  - Fig. C is now a LINE/SCATTER plot (alpha is a continuous axis, not a category).
  - Fig. C reads the 5-point CSV with a 'tip' column: anchor vs sweep points are
    drawn differently (anchors are pre-registered constants, NOT outputs of the
    sweep run -- open markers; sweep points -- filled markers).
  - No "pass = green" colouring. alpha=0.25 is unreachable; green would mislead.
  - Kill thresholds are CLI arguments, never hard-coded, and never derived from
    the CSV anchors. (Deliberate: the threshold uses V_ORACLE=8.73 (full oracle),
    while the alpha=0 anchor is 8.69 (solar-only oracle). Different numbers.
    Deriving the threshold from the CSV would silently produce 9.66, not 9.68.)

INPUTS (read-only)
  gg2_C_alpha_sweep_TAM.csv : columns [alpha, ihlal_pct, tip]
                              tip in {anchor, sweep}; if the column is absent,
                              every row is treated as 'sweep' (with a warning).
  gg2_A_scale_sweep.csv     : columns [bara, H, wall_med_ms]  (long format)

OUTPUTS (to --outdir, default "figs")
  fig_gg2_alpha.png / .pdf
  fig_gg2_scale.png / .pdf

SAFETY
  - Existing outputs are NOT overwritten unless --force is given.
  - Importing this module has no side effects (main-guard).
  - Every plotted DATA value comes from the CSVs. Every CRITERION value
    (thresholds, band bounds) comes from the command line and is echoed to
    stdout so it can be audited against the pre-registration.

USAGE (MINGW64, .venv active) -- run from the folder holding the CSVs:
  python make_fig_gg2.py --datadir . --outdir figs \
      --thr 9.68 --nwp-lo 0.50 --nwp-hi 0.65 \
      --kill-ms 2000 --step-ms 1800000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------ style
COL_LINE = "#4C72B0"     # the alpha response curve / neutral series colour
COL_TARGET = "#C44E52"   # kill-threshold lines
COL_BAND = "#8172B3"     # NWP-achievable band
COL_SERIES = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]


def apply_style() -> None:
    """Paper style: NeurIPS/CCAI single column (5.5 in text width), vector-safe."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ------------------------------------------------------------------ io
def _die(msg: str) -> None:
    sys.exit(f"[HATA] {msg}")


def load_c(path: Path) -> pd.DataFrame:
    """Aday C: solar forecast alpha sweep (5 points: 2 anchors + 3 sweep)."""
    if not path.exists():
        _die(f"bulunamadi: {path}")
    # index_col=False: bir hucrede tirnaklanmamis virgul varsa pandas ilk kolonu
    # index sanip TUM kolonlari kaydirir (alpha <- ihlal_pct ...). Bu onu engeller.
    df = pd.read_csv(path, index_col=False)

    missing = {"alpha", "ihlal_pct"} - set(df.columns)
    if missing:
        _die(f"{path.name}: eksik kolon(lar) {sorted(missing)} | bulunan: {list(df.columns)}")

    if "tip" not in df.columns:
        print(f"[UYARI] {path.name}: 'tip' kolonu yok -> tum noktalar 'sweep' sayildi.")
        df["tip"] = "sweep"

    df = df[["alpha", "ihlal_pct", "tip"]].dropna(subset=["alpha", "ihlal_pct"]).copy()

    # Kolon kaymasi tespiti: kayma olursa alpha/ihlal_pct sayisal olmaz ya da
    # alpha [0,1] disina cikar. Sessiz yanlis figur uretmektense burada dur.
    for c in ("alpha", "ihlal_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            _die(f"{path.name}: '{c}' kolonunda sayisal olmayan deger var -> muhtemelen "
                 f"kolon kaymasi (bir hucrede tirnaklanmamis virgul?). CSV'yi kontrol et.")
    if not df["alpha"].between(0.0, 1.0).all():
        _die(f"{path.name}: alpha [0,1] disinda: {sorted(df['alpha'].tolist())} -> "
             f"muhtemelen kolon kaymasi. CSV'yi kontrol et.")

    df["tip"] = df["tip"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["tip"]) - {"anchor", "sweep"})
    if bad:
        _die(f"{path.name}: tanimsiz 'tip' degeri {bad} (beklenen: anchor / sweep)")
    if df.empty:
        _die(f"{path.name}: veri satiri yok.")
    if df["alpha"].duplicated().any():
        _die(f"{path.name}: ayni alpha birden fazla satirda.")

    return df.sort_values("alpha", ascending=False).reset_index(drop=True)


def load_a(path: Path) -> pd.DataFrame:
    """Aday A: bus x horizon solve-time sweep (long format -> wide)."""
    if not path.exists():
        _die(f"bulunamadi: {path}")
    df = pd.read_csv(path, index_col=False)

    missing = {"bara", "H", "wall_med_ms"} - set(df.columns)
    if missing:
        _die(f"{path.name}: eksik kolon(lar) {sorted(missing)} | bulunan: {list(df.columns)}")

    df = df[["bara", "H", "wall_med_ms"]].dropna()
    if df.empty:
        _die(f"{path.name}: veri satiri yok.")
    if df.duplicated(subset=["bara", "H"]).any():
        _die(f"{path.name}: ayni (bara, H) icin birden fazla satir var.")

    wide = df.pivot(index="bara", columns="H", values="wall_med_ms").sort_index()
    return wide[sorted(wide.columns)]


def save(fig, outdir: Path, stem: str, force: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    targets = [outdir / f"{stem}.png", outdir / f"{stem}.pdf"]
    existing = [str(t) for t in targets if t.exists()]
    if existing and not force:
        plt.close(fig)
        _die("cikti dosyasi zaten var, ezilmedi: " + ", ".join(existing) + "  (--force ile ez)")
    for t in targets:
        fig.savefig(t)
    plt.close(fig)
    print("  yazildi: " + " , ".join(str(t) for t in targets))


# ------------------------------------------------------------------ figures
def plot_c(df: pd.DataFrame, thr: float, nwp_lo, nwp_hi):
    """Response curve: violation % vs solar-forecast error multiplier alpha."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    d = df.sort_values("alpha")  # ascending, for a clean polyline

    if nwp_lo is not None and nwp_hi is not None:
        lo, hi = sorted((float(nwp_lo), float(nwp_hi)))
        ax.axvspan(lo, hi, color=COL_BAND, alpha=0.13, lw=0, zorder=1,
                   label=f"NWP-achievable ($\\alpha$ {lo:g}–{hi:g})")

    ax.axhline(thr, color=COL_TARGET, ls="--", lw=1.2, zorder=2,
               label=f"kill threshold ({thr:.2f}%)")

    ax.plot(d["alpha"], d["ihlal_pct"], "-", color=COL_LINE, lw=1.4, zorder=3)

    sw = d[d["tip"] == "sweep"]
    an = d[d["tip"] == "anchor"]
    if not sw.empty:
        ax.plot(sw["alpha"], sw["ihlal_pct"], "o", ms=4.6, color=COL_LINE,
                ls="none", zorder=4, label="sweep point (this run)")
    if not an.empty:
        ax.plot(an["alpha"], an["ihlal_pct"], "s", ms=4.6, mfc="white",
                mec=COL_LINE, mew=1.2, ls="none", zorder=4,
                label="anchor (pre-registered)")

    vals = d["ihlal_pct"].tolist()
    lo_y, hi_y = min(min(vals), thr), max(max(vals), thr)
    pad = (hi_y - lo_y) * 0.22
    for a, v in zip(d["alpha"], d["ihlal_pct"]):
        ax.annotate(f"{v:.2f}", (a, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)

    ax.set_xlabel(r"Solar forecast error multiplier $\alpha$")
    ax.set_ylabel("Voltage violations (%)")
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlim(-0.06, 1.06)
    ax.invert_xaxis()  # alpha=1 (today's forecast) on the left -> descending curve
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
              ncol=2, borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_a(wide: pd.DataFrame, kill_ms):
    """Median solve time vs buses, one series per horizon, with kill threshold."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    buses = wide.index.tolist()

    for i, h in enumerate(wide.columns):
        ax.plot(buses, wide[h], marker="o", ms=3.8,
                color=COL_SERIES[i % len(COL_SERIES)], label=f"$H$ = {int(h)}", zorder=3)

    top = float(wide.to_numpy().max())
    if kill_ms is not None:
        ax.axhline(float(kill_ms), color=COL_TARGET, ls="--", lw=1.2, zorder=2,
                   label=f"kill threshold ({float(kill_ms):g} ms)")
        top = max(top, float(kill_ms))

    ax.set_xlabel("Number of buses")
    ax.set_ylabel("Median solve time (ms)")
    ax.set_xticks(buses)
    ax.set_ylim(0, top * 1.12)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30),
              ncol=2, borderaxespad=0.0)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description="GG-2 boundary-evidence figures (v2).")
    p.add_argument("--datadir", default=".", help="CSV klasoru (varsayilan: gecerli klasor)")
    p.add_argument("--outdir", default="figs", help="cikti klasoru (varsayilan: figs)")
    p.add_argument("--c-csv", default="gg2_C_alpha_sweep_TAM.csv", help="Aday C CSV adi")
    p.add_argument("--a-csv", default="gg2_A_scale_sweep.csv", help="Aday A CSV adi")
    p.add_argument("--force", action="store_true", help="mevcut cikti dosyalarini ez")
    p.add_argument("--thr", type=float, required=True,
                   help="ZORUNLU. Aday C on-kayitli kill esigi, ihlal %% (or: 9.68). "
                        "CSV'den TURETILMEZ: 8.73 (tam oracle) ile 8.69 (solar oracle) farkli.")
    p.add_argument("--nwp-lo", type=float, default=None, help="NWP bandi alt alpha (or: 0.50)")
    p.add_argument("--nwp-hi", type=float, default=None, help="NWP bandi ust alpha (or: 0.65)")
    p.add_argument("--kill-ms", type=float, default=None,
                   help="Aday A on-kayitli kill esigi, medyan ms (or: 2000)")
    p.add_argument("--step-ms", type=float, default=None,
                   help="Kontrol adimi suresi ms (or: 1800000 = 30 dk). Yalniz stdout istatistigi.")
    args = p.parse_args()

    if (args.nwp_lo is None) != (args.nwp_hi is None):
        _die("--nwp-lo ve --nwp-hi birlikte verilmeli (ya ikisi de, ya hicbiri).")

    datadir, outdir = Path(args.datadir), Path(args.outdir)
    apply_style()

    print("KRITER DEGERLERI (komut satirindan, on-kayit ile karsilastir):")
    print(f"  Aday C kill esigi : {args.thr:.2f} %")
    print(f"  NWP bandi         : {args.nwp_lo} – {args.nwp_hi}"
          if args.nwp_lo is not None else "  NWP bandi         : (cizilmedi)")
    print(f"  Aday A kill esigi : {args.kill_ms} ms"
          if args.kill_ms is not None else "  Aday A kill esigi : (cizilmedi)")

    # --- Aday C -------------------------------------------------------
    c_df = load_c(datadir / args.c_csv)
    print(f"\n[C] {args.c_csv} (CSV ile karsilastir):")
    for a, v, t in zip(c_df["alpha"], c_df["ihlal_pct"], c_df["tip"]):
        mark = "<= esik" if v <= args.thr else "       "
        print(f"  alpha={a:<5g} ihlal={v:6.2f}%  [{t:<6}] {mark}")
    under = c_df[c_df["ihlal_pct"] <= args.thr]
    if under.empty:
        print("  -> hicbir nokta esigin altinda degil.")
    else:
        a_star = float(under["alpha"].max())
        row = under[under["alpha"] == a_star].iloc[0]
        print(f"  -> esigin altina inen en yuksek alpha: a*={a_star:g} "
              f"({row['ihlal_pct']:.2f}%, tip={row['tip']})")
        if args.nwp_lo is not None:
            inside = float(args.nwp_lo) <= a_star <= float(args.nwp_hi)
            print(f"     NWP bandi icinde mi? {'EVET' if inside else 'HAYIR -> ulasilamaz'}")
    fig = plot_c(c_df, args.thr, args.nwp_lo, args.nwp_hi)
    save(fig, outdir, "fig_gg2_alpha", args.force)

    # --- Aday A -------------------------------------------------------
    a_wide = load_a(datadir / args.a_csv)
    print(f"\n[A] {args.a_csv}, medyan wall-clock ms (CSV ile karsilastir):")
    print(a_wide.to_string(float_format=lambda v: f"{v:9.1f}"))
    first, last = a_wide.index[0], a_wide.index[-1]
    print(f"  bus-eksenli buyume (bara orani x{last / first:.2f}):")
    for h in a_wide.columns:
        r = float(a_wide[h].loc[last]) / float(a_wide[h].loc[first])
        print(f"    H={int(h):<3d}  {first}->{last} bara : x{r:.2f}")
    worst = float(a_wide.to_numpy().max())
    print(f"  en kotu medyan: {worst:.1f} ms")
    if args.kill_ms is not None:
        print(f"    kill esiginin ({args.kill_ms:g} ms) %{100 * worst / args.kill_ms:.1f}'i "
              f"-> {'ALTINDA (gecti)' if worst < args.kill_ms else 'USTUNDE (tikandi)'}")
    if args.step_ms is not None:
        print(f"    kontrol adiminin ({args.step_ms:g} ms) %{100 * worst / args.step_ms:.3f}'i")
    fig = plot_a(a_wide, args.kill_ms)
    save(fig, outdir, "fig_gg2_scale", args.force)

    print("\nBitti.")


if __name__ == "__main__":
    main()