# check_hs_domain.py
# ERA5 の Hs を、(1) 領域全体の時系列最大、(2) 大圏経路の各点での全時刻最大、で調べる。
# 使い方: python3 check_hs_domain.py 20080129 result_20080129.pkl --eta 106.0
import argparse
import glob
import pickle
from datetime import datetime

import numpy as np
import xarray as xr

from era5_weather import _open

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("pkl")
ap.add_argument("--eta", type=float, default=None)
ap.add_argument("--hours", type=float, default=130.0, help="出発から何hまで見るか")
a = ap.parse_args()

paths = sorted(glob.glob(f"era5_{a.name}_*.nc"))
t0 = np.datetime64(datetime(int(a.name[:4]), int(a.name[4:6]), int(a.name[6:8])))
ds = xr.concat([_open(p) for p in paths], dim="time").sortby("time")
ds = ds.sortby("latitude").sortby("longitude")

th = ((ds["time"].values - t0) / np.timedelta64(1, "h")).astype(float)
keep = (th >= 0) & (th <= a.hours)
swh = ds["swh"].isel(time=np.where(keep)[0])
th = th[keep]
lat, lon = ds["latitude"].values, ds["longitude"].values

# ---- (1) 領域全体 ----
print(f"領域: 緯度 {lat.min():.1f}〜{lat.max():.1f}  経度 {lon.min():.1f}〜{lon.max():.1f}  "
      f"格子 {len(lat)}x{len(lon)}  時刻 {th[0]:.0f}〜{th[-1]:.0f} h")
print("\n--- 領域全体の Hs 最大（12 h ごと）---")
print(f"{'t[h]':>6}{'Hs最大':>8}{'緯度':>7}{'経度':>8}")
for i in range(0, len(th), 12):
    f = swh.isel(time=i).values
    j = np.unravel_index(np.nanargmax(f), f.shape)
    print(f"{th[i]:>6.0f}{f[j]:>8.1f}{lat[j[0]]:>7.1f}{lon[j[1]]:>8.1f}")
allmax = swh.values
k = np.unravel_index(np.nanargmax(allmax), allmax.shape)
print(f"全期間の最大 {allmax[k]:.1f} m  （t={th[k[0]]:.0f} h, 緯度 {lat[k[1]]:.1f}, 経度 {lon[k[2]]:.1f}）")

# ---- (2) 大圏経路上、各点で全時刻の最大 ----
with open(a.pkl, "rb") as f:
    pk = pickle.load(f)
eta = min(pk["gc_same"], key=lambda e: abs(e - (a.eta if a.eta is not None else min(pk["gc_same"]))))
pts = pk["gc_same"][eta]["points"]

print("\n--- 大圏経路の各点での Hs（全時刻の最大。船が通る時刻とは無関係）---")
print(f"{'経度':>7}{'緯度':>7}{'Hs最大':>8}{'その時刻[h]':>12}")
for la, lo in pts[::3]:
    i, j = np.abs(lat - la).argmin(), np.abs(lon - lo).argmin()
    series = swh.values[:, i, j]
    t_i = int(np.nanargmax(series))
    print(f"{lo:>7.1f}{la:>7.1f}{series[t_i]:>8.1f}{th[t_i]:>12.0f}")