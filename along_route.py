# -*- coding: utf-8 -*-
"""
hs_along_route.py
=================
保存した航路（大圏と 3DDA）に沿って、船が通る時刻・場所での Hs を表示する。
論文の図（0129 の大圏は Hs 8 m 超に遭遇）と、手元の ERA5 の気象を比べるためのもの。

使い方:
  python3 hs_along_route.py 20080129 result_20080129.pkl --eta 106.0
  python3 hs_along_route.py 20080129 result_20080129.pkl --eta 106.0 --offset 0 6 12   # 出発時刻をずらす [h]
"""
import argparse
import glob
import pickle
from datetime import datetime, timedelta

import numpy as np
from pyproj import Geod

from era5_weather import ERA5Weather

_GEOD = Geod(ellps="WGS84")


def timeline(points, speeds):
    """各経由点の通過時刻 [h] を返す（区間ごとの船速から）。"""
    t = [0.0]
    for k in range(len(points) - 1):
        (la1, lo1), (la2, lo2) = points[k], points[k + 1]
        d_nm = _GEOD.inv(lo1, la1, lo2, la2)[2] / 1852.0
        t.append(t[-1] + d_nm / speeds[k])
    return np.array(t)


ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("pkl")
ap.add_argument("--eta", type=float, default=None)
ap.add_argument("--offset", type=float, nargs="*", default=[0.0],
                help="出発時刻のずらし [h]（正で遅く）。ERA5 の範囲内に限る")
a = ap.parse_args()

paths = sorted(glob.glob(f"era5_{a.name}_*.nc"))
t0 = datetime(int(a.name[:4]), int(a.name[4:6]), int(a.name[6:8]))
weather = ERA5Weather(paths, t0)
with open(a.pkl, "rb") as f:
    pk = pickle.load(f)
eta = min(pk["d3s"], key=lambda e: abs(e - (a.eta if a.eta is not None else min(pk["d3s"]))))
routes = {"大圏": pk["gc_same"][eta], "3DDA": pk["d3s"][eta]}

for off in a.offset:
    print(f"\n===== 出発 {t0 + timedelta(hours=off):%m-%d %H:%M} UTC（+{off:g} h）=====")
    for label, r in routes.items():
        pts, sp = r["points"], r["speeds"]
        t = timeline(pts, sp) + off
        hs = np.array([weather(la, lo, tt).Hs for (la, lo), tt in zip(pts, t)])
        k = int(np.argmax(hs))
        print(f"{label:<5} Hs 最大 {hs[k]:.1f} m（経度 {pts[k][1]:.1f}°、出発から {t[k] - off:.0f} h）"
              f"  平均 {hs.mean():.1f} m")
    if a.offset == [0.0]:
        print(f"\n{'経度':>7}{'大圏 Hs':>9}{'3DDA Hs':>9}")
        g, d = routes["大圏"], routes["3DDA"]
        tg, td = timeline(g["points"], g["speeds"]), timeline(d["points"], d["speeds"])
        for i in range(0, len(g["points"]), 3):
            la, lo = g["points"][i]
            j = min(range(len(d["points"])), key=lambda m: abs(d["points"][m][1] - lo))
            print(f"{lo:>7.1f}{weather(la, lo, tg[i]).Hs:>9.1f}"
                  f"{weather(*d['points'][j], td[j]).Hs:>9.1f}")