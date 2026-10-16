# scan_departure.py
# 論文 Fig.10-12 の「大圏の Hs（経度ごと）」に ERA5 が最もよく合う出発時刻を探す。
# 事前に era5x_<航海名>_*.nc（航海日の6日前から取得）が必要（下記の取得手順を参照）。
# 使い方: python3 scan_departure.py 20080129 20080424 20081214
import glob
import sys
from datetime import datetime

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")
NM = 1852.0

# 論文の大圏(青破線)の Hs を目視で読んだ代表点 (経度[deg], Hs[m])。読み取り誤差は ±0.5〜0.7 m 程度。
PAPER_HS = {
    "20080129": [(-10, 2.9), (-20, 5.3), (-30, 7.5), (-38.5, 8.5), (-45, 4.0), (-48, 3.0)],
    "20080424": [(-10, 3.5), (-16, 5.5), (-25, 3.9), (-38, 2.1), (-48, 4.0)],
    "20081214": [(-10, 2.4), (-20, 3.1), (-30, 3.8), (-38.5, 6.9), (-47, 6.0)],
}
OFFSETS_H = np.arange(-144, 73, 3)       # 航海日 00:00 UTC からの出発のずれ [h]


def gc_track(start, end, speed_kn, n_seg=27):
    """大圏を n_seg 等分した点の (経度, 緯度, 通過時刻[h])。船速は一定。"""
    pts = [start] + [(la, lo) for lo, la in
                     _GEOD.npts(start[1], start[0], end[1], end[0], n_seg - 1)] + [end]
    lat = np.array([p[0] for p in pts]); lon = np.array([p[1] for p in pts])
    d = np.concatenate([[0.0], np.cumsum([_GEOD.inv(lon[i], lat[i], lon[i + 1], lat[i + 1])[2]
                                          for i in range(len(pts) - 1)])]) / NM
    return lon, lat, d / speed_kn


def hs_profile(weather, lon, lat, t_ship, offset, lon_query):
    hs = np.array([weather(la, lo, offset + t).Hs for lo, la, t in zip(lon, lat, t_ship)])
    order = np.argsort(lon)                       # np.interp は昇順が必要
    return np.interp(lon_query, lon[order], hs[order]), hs.max()


def scan(weather, lon, lat, t_ship, target, offsets=OFFSETS_H):
    lq = np.array([p[0] for p in target]); hq = np.array([p[1] for p in target])
    rows = []
    for off in offsets:
        h, hmax = hs_profile(weather, lon, lat, t_ship, off, lq)
        rows.append((off, float(np.sqrt(np.mean((h - hq) ** 2))), hmax, h))
    return rows


if __name__ == "__main__":
    from reproduce_wang2019 import PAPER, get_endpoints
    from era5_weather import ERA5Weather
    for name in (sys.argv[1:] or list(PAPER_HS)):
        ref = PAPER[name]
        start, end = get_endpoints(ref["dir"])
        dnm = _GEOD.inv(start[1], start[0], end[1], end[0])[2] / NM
        vf = dnm / ref["actual_eta"]
        lon, lat, t_ship = gc_track(start, end, vf)
        paths = sorted(glob.glob(f"era5x_{name}_*.nc"))
        if not paths:
            print(f"{name}: era5x_{name}_*.nc がありません（先に取得してください）"); continue
        t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]))
        weather = ERA5Weather(paths, t0)
        rows = scan(weather, lon, lat, t_ship, PAPER_HS[name])
        r0 = next(r for r in rows if r[0] == 0)
        print(f"\n=== {name}  Vf {vf:.2f} kn  論文の代表点 {PAPER_HS[name]}")
        print(f"  ずれ 0 h: RMS {r0[1]:.2f} m  Hs最大 {r0[2]:.1f} m")
        print(f"  {'ずれ[h]':>8}{'RMS[m]':>8}{'Hs最大':>8}   各代表点の ERA5 Hs")
        for off, rms, hmax, h in sorted(rows, key=lambda r: r[1])[:6]:
            print(f"  {off:>+8.0f}{rms:>8.2f}{hmax:>8.1f}   " + " ".join(f"{x:4.1f}" for x in h))