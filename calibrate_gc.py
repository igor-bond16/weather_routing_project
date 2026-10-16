# -*- coding: utf-8 -*-
"""
calibrate_gc.py
===============
大圏・定速の燃料だけを使って、(1) 波浪増加抵抗の倍率 k_aw と (2) 燃料の一律係数 s を、
論文 Table 3 の「大圏」の燃料に合わせて探す。大圏のシミュレーションだけなので DP より桁違いに速い。

  燃料(本研究, k_aw) × s  ≈  論文の大圏の燃料        （便ごとに比べる）

使い方（船を 2800TEU にするなら SHIP=2800teu を付ける）:
  SHIP=2800teu python3 calibrate_gc.py
  SHIP=2800teu python3 calibrate_gc.py --names 20080129 20080424 20081214 --k 1.0 1.5 2.0 2.5 3.0

注意
  * k_aw を変えると最適な航路・船速も変わる。ここで決めた値で DP を回し直して確認すること。
  * 大圏が出力の上限で減速して ETA がずれる便（0218 など）は、合わせる対象から外すか、
    ETA のずれを見て判断する（表の ETA 列）。
  * 実船の計測で補正した論文のモデルに合わせる、モデルの較正である。何をどう合わせたかを、
    結果と一緒に必ず書くこと。
"""
import argparse
import glob
from datetime import datetime

import numpy as np

import dijkstra_3d as D
import reproduce_wang2019 as R

PAPER_GC = {"20080129": 304.5, "20080218": 272.0, "20080424": 246.1, "20081214": 272.5}

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="*", default=list(PAPER_GC), help="対象の便")
ap.add_argument("--k", nargs="*", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0],
                help="波浪増加抵抗にかける倍率 k_aw")
ap.add_argument("--exclude", nargs="*", default=[], help="合わせる対象から外す便（表には出す）")
a = ap.parse_args()

orig_aw = D.added_resistance_irregular


def setup(name):
    """run_dp_one.py と同じ手順で、格子・気象・Vf を作る。"""
    ref = R.PAPER[name]
    t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]), 0, 0)
    start, end = R.get_endpoints(ref["dir"])
    grid, dgc_nm = R.build_grids(start, end)
    vf = dgc_nm / ref["actual_eta"]
    paths = sorted(glob.glob(f"era5_{name}_*.nc")) or R.download_era5(
        t0, hours=ref["actual_eta"] + 24, area=R.AREA, out_prefix=f"era5_{name}")
    return grid, R.ERA5Weather(paths, t0), vf, ref["actual_eta"]


fuel = {k: {} for k in a.k}
eta = {k: {} for k in a.k}
for name in a.names:
    grid, weather, vf, eta_target = setup(name)
    for k in a.k:
        D.added_resistance_irregular = lambda *args, _k=k, **kw: _k * orig_aw(*args, **kw)
        gc = R.simulate_great_circle(grid, weather, R.make_fuel_model(), vf)   # k ごとに新しいモデル（キャッシュを分ける）
        fuel[k][name], eta[k][name] = gc["fuel"], gc["eta"]
        print(f"  {name}  k_aw={k:<4}  燃料 {gc['fuel']:6.1f} t  ETA {gc['eta']:6.1f} h（目標 {eta_target}）", flush=True)
D.added_resistance_irregular = orig_aw

use = [n for n in a.names if n not in a.exclude]
print("\n=== 大圏の燃料（本研究 × s）と論文との比。s は『使う便』の対数平均で決める ===")
best = None
for k in a.k:
    ratio = np.array([PAPER_GC[n] / fuel[k][n] for n in use])
    s = float(np.exp(np.log(ratio).mean()))
    resid = {n: 100 * (s * fuel[k][n] / PAPER_GC[n] - 1) for n in a.names}
    sse = float(np.sum([resid[n] ** 2 for n in use]))
    best = (sse, k, s) if best is None or sse < best[0] else best
    cells = "  ".join(f"{n[4:]}:{resid[n]:+5.1f}%" + ("*" if n in a.exclude else "") for n in a.names)
    print(f"k_aw={k:<4} s={s:.3f}  残差 {cells}   二乗和 {sse:7.1f}")
print(f"\n最小は k_aw={best[1]}, s={best[2]:.3f}（残差の二乗和 {best[0]:.1f}）。* は合わせる対象から外した便。")
print("この k_aw と s で DP を回し直して、削減率・航路が論文に近づくか確認すること。")