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
"""
import argparse
import csv
import pickle

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


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
a = ap.parse_args()

with open(a.pkl, "rb") as f:
    pk = pickle.load(f)

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