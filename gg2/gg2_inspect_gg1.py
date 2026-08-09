#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_inspect_gg1.py
GridGuard-2 fizibilite / Aday C on-testi hazirligi.
GG-1 deposunun SALT-OKUNUR envanterini cikarir. HICBIR DOSYAYA YAZMAZ.

Kullanim (MINGW64, .venv aktif):
    python gg2_inspect_gg1.py <GG1_repo_yolu>
Yol verilmezse bulundugun dizin taranir.
Ciktinin TAMAMINI kopyala-yapistir ile Claude'a ver.
"""

import sys
import json
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".idea", ".vscode",
             ".mypy_cache", ".pytest_cache", "node_modules"}
LIST_EXTS = {".py", ".csv", ".npz", ".json", ".yaml", ".yml", ".md", ".txt"}
CODE_KEYWORDS = ("mpc", "safe", "forecast", "sac", "bc", "lindist", "dist",
                 "main", "run", "train", "eval", "baseline", "backoff",
                 "oracle", "env", "battery")
RESULT_KEYWORDS = ("result", "metric", "summary", "viol", "profit",
                   "final", "eval", "table", "report")
SMALL_JSON_LIMIT = 20_000   # bayt
CSV_PREVIEW_ROWS = 10


def fmt_size(n):
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return "%.0f%s" % (n, unit)
        n /= 1024.0
    return "%.1fGB" % n


def iter_files(root):
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in LIST_EXTS:
            yield p


def section(title):
    print("\n" + "=" * 70)
    print("== " + title)
    print("=" * 70)


def show_tree(files, root):
    section("1) DOSYA AGACI (ilgili uzantilar)")
    for p in files:
        print("%-70s %8s" % (str(p.relative_to(root)), fmt_size(p.stat().st_size)))



def show_npz(files, root):
    section("2) NPZ DOSYALARI (anahtar / sekil / dtype)")
    try:
        import numpy as np
    except ImportError:
        print("!! numpy import edilemedi (.venv aktif mi?) — bolum atlandi")
        return
    npzs = [p for p in files if p.suffix.lower() == ".npz"]
    if not npzs:
        print("(npz bulunamadi)")
    for p in npzs:
        print("\n-- %s" % p.relative_to(root))
        try:
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    print("   %-30s shape=%-20s dtype=%s"
                          % (k, str(a.shape), a.dtype))
        except Exception as e:
            print("   !! okunamadi: %r" % e)


def count_lines(p):
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
    return n


def show_csv_headers(files, root):
    section("3) CSV BASLIKLARI (baslik + ilk satir + satir sayisi)")
    for p in [f for f in files if f.suffix.lower() == ".csv"]:
        print("\n-- %s  (%s)" % (p.relative_to(root), fmt_size(p.stat().st_size)))
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().rstrip("\n")
                first = f.readline().rstrip("\n")
            print("   baslik : %s" % header[:400])
            print("   ilk satir: %s" % first[:400])
            print("   satir sayisi (baslik dahil): %d" % count_lines(p))
        except Exception as e:
            print("   !! okunamadi: %r" % e)


def looks_like_result(p):
    name = p.name.lower()
    return any(k in name for k in RESULT_KEYWORDS)


def show_results(files, root):
    section("4) SONUC-BENZERI DOSYALAR (json tam/anahtar, csv ilk %d satir)"
            % CSV_PREVIEW_ROWS)
    hits = [p for p in files
            if looks_like_result(p) and p.suffix.lower() in (".json", ".csv")]
    if not hits:
        print("(isminde %s geceni bulamadim)" % (RESULT_KEYWORDS,))
    for p in hits:
        print("\n-- %s  (%s)" % (p.relative_to(root), fmt_size(p.stat().st_size)))
        try:
            if p.suffix.lower() == ".json":
                raw = p.read_text(encoding="utf-8", errors="replace")
                if len(raw) <= SMALL_JSON_LIMIT:
                    obj = json.loads(raw)
                    print(json.dumps(obj, indent=2, ensure_ascii=False)[:SMALL_JSON_LIMIT])
                else:
                    obj = json.loads(raw)
                    keys = list(obj.keys()) if isinstance(obj, dict) else \
                        "liste, uzunluk=%d" % len(obj)
                    print("   buyuk json — ust duzey anahtarlar: %s" % keys)
            else:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i > CSV_PREVIEW_ROWS:
                            print("   ... (devami kesildi)")
                            break
                        print("   %s" % line.rstrip("\n")[:400])
        except Exception as e:
            print("   !! okunamadi: %r" % e)


def show_py_api(files, root):
    section("5) PY API TASLAGI (def/class satirlari; anahtar-kelimeli dosyalar)")
    for p in [f for f in files if f.suffix.lower() == ".py"]:
        if not any(k in p.name.lower() for k in CODE_KEYWORDS):
            continue
        print("\n-- %s" % p.relative_to(root))
        try:
            has_guard = False
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    s = line.rstrip()
                    t = s.lstrip()
                    if t.startswith("def ") or t.startswith("class "):
                        print("   L%-5d %s" % (i, s[:160]))
                    if t.startswith("if __name__"):
                        has_guard = True
            print("   main-guard: %s" % ("VAR" if has_guard else "YOK"))
        except Exception as e:
            print("   !! okunamadi: %r" % e)


def show_env():
    section("6) ORTAM")
    print("python : %s" % sys.version.replace("\n", " "))
    for mod in ("numpy", "pandas", "pandapower"):
        try:
            m = __import__(mod)
            print("%-11s: %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("%-11s: YOK" % mod)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    root = root.resolve()
    if not root.is_dir():
        print("HATA: dizin bulunamadi: %s" % root)
        sys.exit(1)
    print("GG-1 SALT-OKUNUR ENVANTER — kok: %s" % root)
    files = list(iter_files(root))
    show_tree(files, root)
    show_npz(files, root)
    show_csv_headers(files, root)
    show_results(files, root)
    show_py_api(files, root)
    show_env()
    print("\nBITTI — ciktinin tamamini Claude'a yapistir.")


if __name__ == "__main__":
    main()