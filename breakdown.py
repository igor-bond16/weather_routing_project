# breakdown.py
# 保存した経路（大圏 or 3DDA）に沿って、各区間の抵抗の内訳（静水中・波浪中付加・風圧）と
# 成分ごとの燃料寄与を出す。FuelModel.rate と同じ式を、量子化あり／なしの両方で計算する。
# 使い方: python3 breakdown.py 20080129 result_20080129.pkl --eta 106.0 [--route 3dda]
import argparse
import glob
import pickle
from datetime import datetime

import numpy as np

from build_grid import _GEOD, NM_TO_M
from dijkstra_3d import relative_angle, KT_TO_MS
from fuel_consumption import (SeaState, added_resistance_irregular,
                              true_to_apparent_wind)
from wind_resistance import wind_resistance
from era5_weather import ERA5Weather
from reproduce_wang2019 import make_fuel_model

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("pkl")
ap.add_argument("--eta", type=float, default=None)
ap.add_argument("--route", choices=["gc", "3dda"], default="gc")
a = ap.parse_args()

t0 = datetime(int(a.name[:4]), int(a.name[4:6]), int(a.name[6:8]))
weather = ERA5Weather(sorted(glob.glob(f"era5_{a.name}_*.nc")), t0)
fm = make_fuel_model()

with open(a.pkl, "rb") as f:
    pk = pickle.load(f)
src = pk["gc_same"] if a.route == "gc" else pk["d3s"]
eta = min(src, key=lambda e: abs(e - (a.eta if a.eta is not None else min(src))))
r = src[eta]
pts, spd = r["points"], [s for s in r["speeds"] if s]


def components(v_kt, sea, quantize):
    """(R_calm, R_aw, R_aa) [N]。quantize=True なら FuelModel.rate と同じ丸めを使う。"""
    if quantize:
        vk = round(v_kt * 10)
        q = SeaState(Hs=round(sea.Hs * 10) / 10.0, Tp=max(float(round(sea.Tp)), 1.0),
                     wave_dir_rel_deg=round(sea.wave_dir_rel_deg / 5) * 5.0,
                     wind_speed_true=float(round(sea.wind_speed_true)),
                     wind_dir_rel_deg=round(sea.wind_dir_rel_deg / 5) * 5.0)
        v_ms = (vk / 10.0) * KT_TO_MS
    else:
        q, v_ms = sea, v_kt * KT_TO_MS
        vk = round(v_kt * 10)
    r_calm = fm._r_calm(vk, v_ms)
    r_aw = (added_resistance_irregular(fm.added, q, v_ms, n_points=fm.n_omega)
            if q.Hs >= 0.25 else 0.0)
    u_a, psi_a = true_to_apparent_wind(v_ms, q.wind_speed_true, q.wind_dir_rel_deg)
    r_aa = float(np.atleast_1d(wind_resistance(psi_a, u_a, fm.wind_params)[0])[0])
    return r_calm, r_aw, r_aa


t = 0.0
rows = []
print(f"{'t[h]':>6}{'経度':>7}{'v[kn]':>7}{'Hs':>5}{'T1':>5}{'波相対':>6}{'風m/s':>6}"
      f"{'Rcalm':>8}{'Raw量子':>8}{'Raw厳密':>8}{'Raa':>7}  [kN]")
for k in range(len(pts) - 1):
    (la1, lo1), (la2, lo2) = pts[k], pts[k + 1]
    az, _, d_m = _GEOD.inv(lo1, la1, lo2, la2)
    w = weather(la1, lo1, t)
    hd = az % 360.0
    sea = SeaState(Hs=w.Hs, Tp=w.Tp,
                   wave_dir_rel_deg=relative_angle(w.wave_from_deg, hd),
                   wind_speed_true=w.wind_speed,
                   wind_dir_rel_deg=relative_angle(w.wind_from_deg, hd))
    v = spd[k]
    dur = d_m / NM_TO_M / v
    cq = components(v, sea, True)
    ce = components(v, sea, False)
    rows.append((dur, v, cq, ce))
    if k % 3 == 0:
        print(f"{t:>6.1f}{lo1:>7.1f}{v:>7.1f}{w.Hs:>5.1f}{w.Tp:>5.1f}{sea.wave_dir_rel_deg:>6.0f}"
              f"{w.wind_speed:>6.1f}{cq[0]/1e3:>8.0f}{cq[1]/1e3:>8.0f}{ce[1]/1e3:>8.0f}{cq[2]/1e3:>7.0f}")
    t += dur

# 燃料 ∝ R_total * v * 時間（推進効率・SFOC は一定なので比で見る）
for label, idx in (("量子化あり", 2), ("量子化なし", 3)):
    e = np.array([[x[0] * x[1] * c for c in x[idx]] for x in rows])
    tot = e.sum()
    print(f"\n[{label}] 燃料への寄与: 静水中 {100*e[:,0].sum()/tot:.1f}%  "
          f"波浪中付加 {100*e[:,1].sum()/tot:.1f}%  風圧 {100*e[:,2].sum()/tot:.1f}%")