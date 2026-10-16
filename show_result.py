# -*- coding: utf-8 -*-
"""
show_result.py
==============
result_<便名>.pkl（run_dp_one.py が保存）から、燃料・ETA・距離・船速・区間ごとの内訳を表示する。

使い方:
  python3 show_result.py result_20080129.pkl                 # 全 ETA の一覧
  python3 show_result.py result_20080129.pkl --eta 106.0     # その ETA の区間表も出す
  python3 show_result.py result_20080129.pkl --eta 106.0 --csv   # 航路を CSV に書き出す
  python3 show_result.py result_20080129.pkl --keys          # pkl の中身（キー名）だけ見る
  python3 show_result.py result_20080424_2800teu.pkl --paper --fuel-scale 0.868
      # 論文（Wang et al. 2019, Table 3）と比べる。--fuel-scale は燃料に一律にかける係数（較正用）。
      # 燃料に一律の係数をかけても最適な航路・船速は変わらないので、DP を回し直さずに済む。
"""
import argparse
import csv
import pickle
import re

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")

# Wang et al. (2019) Table 3（西行き）：便名 -> {区分: (ETA h, 燃料 t, 距離 km)}
PAPER = {
    "20080129": {"大圏": (105.6, 304.5, 3168.1), "3DDA": (106.0, 286.2, 3270.9)},
    "20080218": {"大圏": (94.5, 272.0, 3121.3), "3DDA": (94.5, 269.0, 3142.0)},
    "20080424": {"大圏": (92.5, 246.1, 3129.9), "3DDA": (92.5, 243.8, 3136.4)},
    "20081214": {"大圏": (105.3, 272.5, 3114.5), "3DDA": (105.0, 262.2, 3193.4)},
}


def leg_table(points, speeds):
    """経由点と区間ごとの船速から、区間の距離[nm]・所要時間[h]・累積時間[h]を求める。"""
    n = len(points) - 1
    sp = [s for s in speeds][:n]
    rows, t = [], 0.0
    for k in range(n):
        (la1, lo1), (la2, lo2) = points[k], points[k + 1]
        az, _, d_m = _GEOD.inv(lo1, la1, lo2, la2)
        d_nm = d_m / 1852.0
        v = sp[k] if k < len(sp) and sp[k] else float("nan")
        dt = d_nm / v if v == v else float("nan")
        rows.append(dict(leg=k + 1, lat=la1, lon=lo1, heading=az % 360.0, dist_nm=d_nm,
                         speed_kn=v, hours=dt, t_start_h=t))
        t += dt
    return rows, t


def summarize(label, r):
    pts, sp = r.get("points"), r.get("speeds")
    line = f"{label:<14} ETA {r['eta']:6.1f} h  燃料 {r['fuel']:7.1f} t  距離 {r['dist_km']:6.0f} km"
    if pts and sp:
        v = np.array([s for s in sp if s], dtype=float)
        line += f"  船速 平均 {v.mean():4.1f} / 最小 {v.min():4.1f} / 最大 {v.max():4.1f} kn"
    print(line)


ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("--eta", type=float, default=None, help="区間表を出す ETA（pkl のキーに最も近いもの）")
ap.add_argument("--csv", action="store_true", help="区間表を CSV に書き出す")
ap.add_argument("--keys", action="store_true", help="pkl の中のキーだけ表示する")
ap.add_argument("--fuel-scale", type=float, default=1.0, help="燃料に一律にかける係数（較正用）")
ap.add_argument("--paper", action="store_true", help="論文 Table 3 の値と比べる（便名はファイル名から読む）")
a = ap.parse_args()

with open(a.pkl, "rb") as f:
    pk = pickle.load(f)

if a.fuel_scale != 1.0:
    def _scale(r):
        if isinstance(r, dict) and "fuel" in r:
            r["fuel"] = r["fuel"] * a.fuel_scale
    _scale(pk.get("gc"))
    for _k in ("d3s", "gc_same"):
        for _r in (pk.get(_k) or {}).values():
            _scale(_r)
    print(f"※ 燃料に係数 {a.fuel_scale} をかけて表示しています")

if a.keys:
    print("トップ:", list(pk))
    for name in ("gc", "d3s", "gc_same"):
        v = pk.get(name)
        if isinstance(v, dict) and "fuel" in v:
            print(f"{name}: {list(v)}")
        elif isinstance(v, dict) and v:
            first = next(iter(v.values()))
            print(f"{name}: ETA キー {list(v)}  / 1 件の中身 {list(first) if isinstance(first, dict) else type(first)}")
        elif isinstance(v, dict):
            print(f"{name}: 空")
        else:
            print(f"{name}: {list(v) if hasattr(v, 'keys') else type(v)}")
    raise SystemExit

print("=== 全体 ===")
if isinstance(pk.get("gc"), dict) and "fuel" in pk["gc"]:
    summarize("大圏(Vf一定)", pk["gc"])
for eta in sorted(pk["d3s"]):
    print(f"\n--- 目標 ETA {eta} h ---")
    g, d = pk["gc_same"][eta], pk["d3s"][eta]
    summarize("大圏(同ETA)", g)
    summarize("3DDA", d)
    print(f"  削減 {g['fuel'] - d['fuel']:.1f} t（{100 * (1 - d['fuel'] / g['fuel']):.1f} %）"
          f"  / 距離の増加 {d['dist_km'] - g['dist_km']:+.0f} km")

if a.paper:
    m = re.search(r"(\d{8})", a.pkl.replace("\\", "/").split("/")[-1])
    ref = PAPER.get(m.group(1)) if m else None
    if ref is None:
        print("\n（ファイル名から便名が読めない、または論文の表にない便です）")
    else:
        eta0 = min(pk["d3s"], key=lambda e: abs(e - ref["3DDA"][0]))
        g, d = pk["gc_same"][eta0], pk["d3s"][eta0]
        print(f"\n=== 論文 Table 3 との比較（{m.group(1)}、ETA {eta0} h）===")
        print(f"{'':<8}{'燃料 本研究':>12}{'燃料 論文':>10}{'比':>7}{'距離 本研究':>12}{'距離 論文':>10}")
        for label, mine, key in (("大圏", g, "大圏"), ("3DDA", d, "3DDA")):
            pe, pf, pd = ref[key]
            print(f"{label:<8}{mine['fuel']:>12.1f}{pf:>10.1f}{mine['fuel'] / pf:>7.2f}"
                  f"{mine['dist_km']:>12.0f}{pd:>10.1f}")
        red_m = 100 * (1 - d["fuel"] / g["fuel"])
        red_p = 100 * (1 - ref["3DDA"][1] / ref["大圏"][1])
        print(f"削減率   本研究 {red_m:.1f} %  /  論文 {red_p:.1f} %")

if a.eta is not None:
    eta = min(pk["d3s"], key=lambda e: abs(e - a.eta))
    r = pk["d3s"][eta]
    rows, total_t = leg_table(r["points"], r["speeds"])
    print(f"\n=== 3DDA の区間表（ETA {eta} h）===")
    print(f"{'区間':>4}{'緯度':>8}{'経度':>9}{'針路':>7}{'距離nm':>8}{'船速kn':>8}{'所要h':>7}{'累積h':>7}")
    for w in rows:
        print(f"{w['leg']:>4}{w['lat']:>8.2f}{w['lon']:>9.2f}{w['heading']:>7.0f}{w['dist_nm']:>8.1f}"
              f"{w['speed_kn']:>8.1f}{w['hours']:>7.2f}{w['t_start_h'] + w['hours']:>7.1f}")
    print(f"合計 {sum(w['dist_nm'] for w in rows):.0f} nm / {total_t:.1f} h")
    if a.csv:
        out = f"{a.pkl.rsplit('.', 1)[0]}_eta{eta}_legs.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{out} を保存しました")