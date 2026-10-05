# -*- coding: utf-8 -*-
"""
run_dp_one.py
=============
1 便・1 ETA だけ 3DDA を回し、結果を result_<name>.pkl に保存して、そのまま
make_weather_gif.py で「大圏 vs 3DDA」の気象 GIF を作る。
reproduce_wang2019.py の関数をそのまま使うので、同じフォルダに置くこと。

使い方:
  python3 -u run_dp_one.py 20081224                   # ETA = 実績 ETA × 1.05
  python3 -u run_dp_one.py 20081224 --eta 99
  python3 -u run_dp_one.py 20081224 --dt 0.5 --lat-div 1    # 軽くしたいとき
  python3 -u run_dp_one.py 20081224 --field resist          # 抵抗の地図で GIF を作る
"""
import argparse
import glob
import pickle
import subprocess
import sys
from datetime import datetime

import reproduce_wang2019 as R

ap = argparse.ArgumentParser()
ap.add_argument("name", help="便の名前（例 20081224）")
ap.add_argument("--eta", type=float, default=None, help="目標 ETA [h]（既定: 実績 ETA × 1.05）")
ap.add_argument("--dt", type=float, default=None, help="時間刻み [h]（既定は reproduce_wang2019.py の DT_TIME_H）")
ap.add_argument("--lat-div", type=int, default=None, help="横間隔の分割数（reproduce 側に LAT_DIV がある場合）")
ap.add_argument("--half", type=int, default=None, help="横方向の最大半幅（ノード数）")
ap.add_argument("--field", choices=["hs", "resist"], default="hs", help="GIF に出す量")
ap.add_argument("--no-gif", action="store_true")
a = ap.parse_args()

# --- 設定の上書き（reproduce_wang2019.py のモジュール変数を書き換える）---
if a.dt is not None:
    R.DT_TIME_H = a.dt
if a.lat_div is not None:
    if hasattr(R, "LAT_DIV"):
        R.LAT_DIV = a.lat_div
    else:
        print("注意: reproduce_wang2019.py に LAT_DIV がないので --lat-div は無視します")
if a.half is not None:
    for attr in ("HALF_MAX_NODES", "HALF_MAX"):
        if hasattr(R, attr):
            setattr(R, attr, a.half)
            break

name = a.name
ref = R.PAPER[name]
t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]), 0, 0)

east_pt, west_lat = (49.3, -10.0), 46.5
west_pt = (west_lat, R.calibrate_west_lon(east_pt, west_lat, ref["GC"][2]))
start, end = (west_pt, east_pt) if ref["dir"] == "E" else (east_pt, west_pt)

grid, dgc_nm = R.build_grids(start, end)
vf = dgc_nm / ref["actual_eta"]
print(f"{name}: ノード数 {sum(grid.nodes_per_stage)}  Vf {vf:.2f} kn  DT {R.DT_TIME_H} h", flush=True)

paths = sorted(glob.glob(f"era5_{name}_*.nc")) or R.download_era5(
    t0, hours=ref["actual_eta"] + 24, area=R.AREA, out_prefix=f"era5_{name}")
weather = R.ERA5Weather(paths, t0)
fm = R.make_fuel_model()

gc = R.simulate_great_circle(grid, weather, fm, vf)

eta = a.eta if a.eta is not None else round(ref["actual_eta"] * 1.05, 1)
print(f"3DDA 開始: 目標 ETA {eta} h（大圏の実績 ETA {ref['actual_eta']} h）", flush=True)
d3s, res = R.run_3dda_multi(grid, weather, fm, vf, [eta], eta_max=eta + R.DT_TIME_H)
d3 = d3s[eta]

# 比較用：3DDA と同じ所要時間になる定速の大圏
gc_same = {eta: R.simulate_great_circle(grid, weather, fm, dgc_nm / d3["eta"])}

with open(f"result_{name}.pkl", "wb") as f:
    pickle.dump(dict(gc=gc, d3s=d3s, gc_same=gc_same), f)

g = gc_same[eta]
print(f"\n定速の大圏: ETA {g['eta']:.1f} h  燃料 {g['fuel']:.1f} t  距離 {g['dist_km']:.0f} km")
print(f"3DDA      : ETA {d3['eta']:.1f} h  燃料 {d3['fuel']:.1f} t  距離 {d3['dist_km']:.0f} km"
      f"  （削減 {100 * (1 - d3['fuel'] / g['fuel']):.1f}%）", flush=True)
print(f"result_{name}.pkl を保存しました", flush=True)

if not a.no_gif:
    cmd = [sys.executable, "make_weather_gif.py", name, "--pkl", f"result_{name}.pkl",
           "--eta", repr(eta), "--field", a.field]
    print("GIF 作成:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)