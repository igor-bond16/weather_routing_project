# reproduce_wang2019.py
import glob
import math
from datetime import datetime
from unicodedata import name

import numpy as np

from build_grid import build_routing_grid, _GEOD, NM_TO_M
from dijkstra_3d import (build_spatial_edges, FuelModel, make_fuel_cost,
                         solve_3d_dijkstra, relative_angle, KT_TO_MS)
from Fuel_consumption import PropulsionModel, SeaState
from Calm_Water_Resistance import KCS as CalmWaterKCS
from Added_Resistance import KCS as AddedResistanceKCS
from era5_weather import ERA5Weather, download_era5
import time
_T0 = time.time()

def log(msg):
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)

# ---------------- 論文の条件 ----------------
V_SERVICE_KT = 21.0                 # Table 1
STAGE_NM = V_SERVICE_KT * 3.0       # Eq.(5): 63 nm（ステージ間隔・横間隔とも）
HALF_MAX = 8                        # 最大 2*8+1 = 17 ノード（Fig.2）
DT_TIME_H = 0.25                    # 時間刻み（論文は 12〜30 分）
SPEED_RATIO = (0.5, 1.2)            # Eq.(19)
MAX_INDEX_DIFF = 1                  # 針路変更の制限（仮定）
SERVICE_LOAD = 0.85                 # 通常運航 = 85% MCR（2020年論文の記述。最大出力の仮定に使用）
AREA = [62, -70, 35, 5]             # ERA5 取得範囲 [N, W, S, E]

# 論文 Table 2, 3: actual=実測ETA（Vf の算出に使用）, GC=大圏航行, 3DDA
PAPER = {
    # 東航
    "20080117": dict(dir="E", actual_eta=90.8,  GC=(90.6, 235.8, 3130.4), D3=(91.0, 230.6, 3136.9)),
    "20080523": dict(dir="E", actual_eta=88.7,  GC=(88.6, 220.5, 3127.2), D3=(89.0, 220.4, 3135.3)),
    "20081224": dict(dir="E", actual_eta=94.7,  GC=(94.6, 218.6, 3132.1), D3=(94.5, 216.7, 3132.1)),
    # 西航
    "20080129": dict(dir="W", actual_eta=105.8, GC=(105.6, 304.5, 3168.1), D3=(106.0, 286.2, 3270.9)),
    "20080218": dict(dir="W", actual_eta=94.5,  GC=(94.5, 272.0, 3121.3), D3=(94.5, 269.0, 3142.0)),
    "20080424": dict(dir="W", actual_eta=92.5,  GC=(92.5, 246.1, 3129.9), D3=(92.5, 243.8, 3136.4)),
    "20081214": dict(dir="W", actual_eta=104.8, GC=(105.3, 272.5, 3114.5), D3=(105.0, 262.2, 3193.4)),
}  # タプルは (ETA[h], 燃料[ton], 距離[km])


# ---------------- 補助関数 ----------------
def calibrate_west_lon(east_pt, west_lat, target_km):
    """大圏距離が target_km になるよう北米側の経度を二分法で決める。"""
    lo, hi = -75.0, -35.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        d = _GEOD.inv(east_pt[1], east_pt[0], mid, west_lat)[2] / 1000.0
        if d > target_km:
            lo = mid          # 遠すぎる → 東へ（経度を大きく）
        else:
            hi = mid
    return 0.5 * (lo + hi)


def path_km(points):
    return sum(_GEOD.inv(a[1], a[0], b[1], b[0])[2]
               for a, b in zip(points[:-1], points[1:])) / 1000.0


def to_sea(w, heading):
    return SeaState(Hs=w.Hs, Tp=w.Tp,
                    wave_dir_rel_deg=relative_angle(w.wave_from_deg, heading),
                    wind_speed_true=w.wind_speed,
                    wind_dir_rel_deg=relative_angle(w.wind_from_deg, heading))


def make_fuel_model():
    calm = CalmWaterKCS(is_full_scale=True)
    prop = PropulsionModel()                      # あなたの Fuel_consumption.py のもの
    v = V_SERVICE_KT * KT_TO_MS
    p_service = prop.power_and_fuel(float(calm.calc_calm_water_resistance(v)), v)["P_S [kW]"]
    mcr = p_service / SERVICE_LOAD                # 静水中21ktの出力が85%MCRとなるよう逆算
    print(f"最大出力(仮定) {mcr:.0f} kW  / 21kt静水中 {p_service:.0f} kW")
    return FuelModel(calm, AddedResistanceKCS(is_full_scale=True), prop, max_power_kw=mcr)


LAT_DIV = 3            # 横間隔を 1/3 にする
HALF_MAX_NODES = 12    # 横方向の最大半幅（ノード数）= 12 × 21 nm ≒ 252 nm

def build_grids(start, end):
    dgc_nm = _GEOD.inv(start[1], start[0], end[1], end[0])[2] / NM_TO_M
    N = max(3, round(dgc_nm / STAGE_NM))
    hw = [0] * (N + 1)
    for i in range(1, N):
        hw[i] = min(LAT_DIV * (i + 1), LAT_DIV * (N - i + 1), HALF_MAX_NODES)
    grid = build_routing_grid(start, end, num_stages=N,
                              dy_nm=STAGE_NM / LAT_DIV, half_widths=hw)
    return grid, dgc_nm

def simulate_great_circle(grid, weather, fm, v_kt):
    """大圏航路を固定速度 Vf で航行。主機出力を超える場合は 0.5kt ずつ減速（論文の非自発的減速）。"""
    nodes = [next(n for n in row if n.index == 0) for row in grid.stages]
    t = fuel = 0.0
    pts, spd = [(nodes[0].lat, nodes[0].lon)], []
    for a, b in zip(nodes[:-1], nodes[1:]):
        az, _, d_m = _GEOD.inv(a.lon, a.lat, b.lon, b.lat)
        d_nm = d_m / NM_TO_M
        sea = to_sea(weather(a.lat, a.lon, t), az % 360.0)
        v = v_kt
        r = fm.rate(v, sea)
        while r is None and v > 3.0:
            v -= 0.5
            r = fm.rate(v, sea)
        if r is None:
            raise RuntimeError("この海象では航行不能（出力制限）")
        dur = d_nm / v
        fuel += r[0] * dur / 24.0
        t += dur
        pts.append((b.lat, b.lon))
        spd.append(v)
    return dict(eta=t, fuel=fuel, dist_km=path_km(pts), points=pts, speeds=spd)


# def run_3dda(grid, weather, fm, vf, eta_target):
#     edges = build_spatial_edges(grid, max_index_diff=MAX_INDEX_DIFF)
#     res = solve_3d_dijkstra(grid, edges, weather, make_fuel_cost(fm),
#                             service_speed_kt=vf, dt_h=DT_TIME_H, speed_ratio=SPEED_RATIO,
#                             max_eta_h=eta_target + DT_TIME_H)
#     r = res.route(eta_h=eta_target)
#     pts = [(L.lat, L.lon) for L in r["legs"]]
#     return dict(eta=r["eta_h"], fuel=r["fuel_ton"], dist_km=path_km(pts), points=pts,
#                 speeds=[L.speed_next_kt for L in r["legs"]], times=[L.t_h for L in r["legs"]],
#                 pareto=res.pareto_front())
def simulate_offset_path(grid, weather, fm, v_kt, offset):
    """index=offset に最も近い平行経路（両端は index 0）を固定速度で航行。"""
    N = len(grid.stages)

    def pick(s):
        k = 0 if s in (0, N - 1) else offset
        row = [n for n in grid.stages[s] if n.is_ocean]
        return min(row, key=lambda n: abs(n.index - k))

    nodes = [pick(s) for s in range(N)]
    t = fuel = 0.0
    pts = [(nodes[0].lat, nodes[0].lon)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        az, _, d_m = _GEOD.inv(a.lon, a.lat, b.lon, b.lat)
        d_nm = d_m / NM_TO_M
        sea = to_sea(weather(a.lat, a.lon, t), az % 360.0)
        v = v_kt
        r = fm.rate(v, sea)
        while r is None and v > 3.0:
            v -= 0.5
            r = fm.rate(v, sea)
        if r is None:
            return None
        dur = d_nm / v
        fuel += r[0] * dur / 24.0
        t += dur
        pts.append((b.lat, b.lon))
    return dict(eta=t, fuel=fuel, dist_km=path_km(pts))

def run_3dda_multi(grid, weather, fm, vf, etas, eta_max):
    edges = build_spatial_edges(grid, max_index_diff=MAX_INDEX_DIFF)
    res = solve_3d_dijkstra(grid, edges, weather, make_fuel_cost(fm),
                            service_speed_kt=vf, dt_h=DT_TIME_H,
                            speed_ratio=SPEED_RATIO, max_eta_h=eta_max)
    out = {}
    for e in etas:
        r = res.route(eta_h=e)
        pts = [(L.lat, L.lon) for L in r["legs"]]
        out[e] = dict(eta=r["eta_h"], fuel=r["fuel_ton"], dist_km=path_km(pts),
                      points=pts,
                      speeds=[L.speed_next_kt for L in r["legs"]],
                      times=[L.t_h for L in r["legs"]])
    return out, res


def run_voyage(name, plot=True):
    ref = PAPER[name]
    t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]), 0, 0)

    east_pt, west_lat = (49.3, -10.0), 46.5
    west_pt = (west_lat, calibrate_west_lon(east_pt, west_lat, ref["GC"][2]))
    start, end = (west_pt, east_pt) if ref["dir"] == "E" else (east_pt, west_pt)

    grid, dgc_nm = build_grids(start, end)
    log(f"{name}: グリッド生成完了（ノード数 {sum(grid.nodes_per_stage)}）")
    vf = dgc_nm / ref["actual_eta"]                      # Eq.(18)
    print(f"\n=== {name} ({'東航' if ref['dir']=='E' else '西航'}) ===")
    print(f"start {start}  end ({end[0]:.1f}, {end[1]:.2f})  stages {grid.num_stages}  "
          f"nodes/stage {grid.nodes_per_stage}  Vf {vf:.2f} kt")

    paths = sorted(glob.glob(f"era5_{name}_*.nc")) or download_era5(
        t0, hours=ref["actual_eta"] + 24, area=AREA, out_prefix=f"era5_{name}")
    weather = ERA5Weather(paths, t0)
    log(f"{name}: ERA5 読み込み完了")
    fm = make_fuel_model()

    gc = simulate_great_circle(grid, weather, fm, vf)
    log(f"{name}: 大圏航行の計算完了")

    # 平行経路（固定速度）の確認
    for k in (-4, -2, -1, 0, 1, 2, 4):
        r = simulate_offset_path(grid, weather, fm, vf, k)
        print(f"offset {k:+d}: " + ("航行不能" if r is None else
              f"ETA {r['eta']:.1f} h  fuel {r['fuel']:.1f} t  dist {r['dist_km']:.0f} km"))
    log(f"{name}: 平行経路の計算完了")

    # 3DDA（複数 ETA）
    etas = [ref["D3"][0], 110.0, 115.0, 120.0]

    log(f"{name}: 3DDA 開始（ETA {etas}）")
    d3s, res = run_3dda_multi(grid, weather, fm, vf, etas, eta_max=max(etas) + DT_TIME_H)
    log(f"{name}: 3DDA 完了")

    gc_same = {}
    for e, d in d3s.items():
        gc_same[e] = simulate_great_circle(grid, weather, fm, dgc_nm / d["eta"])
        print(f"ETA {e:5.1f}: 3DDA {d['fuel']:.1f} t ({d['dist_km']:.0f} km) / "
              f"GC {gc_same[e]['fuel']:.1f} t "
              f"-> 削減 {100*(1-d['fuel']/gc_same[e]['fuel']):.1f}%")
    log(f"{name}: ETA 比較の計算完了")

    d3 = d3s[etas[0]]
    gc_ref = gc_same[etas[0]]

    print(f"{'':8}{'ETA[h]':>18}{'燃料[t]':>20}{'距離[km]':>20}      (本実装 / 論文)")
    for label, mine, key in (("GC固定速", gc, "GC"), ("3DDA", d3, "D3")):
        p = ref[key]
        print(f"{label:8}{mine['eta']:9.1f}/{p[0]:<8.1f}{mine['fuel']:10.1f}/{p[1]:<9.1f}"
              f"{mine['dist_km']:10.0f}/{p[2]:<9.1f}")
    mine_s = 100 * (1 - d3["fuel"] / gc_ref["fuel"])
    paper_s = 100 * (1 - ref["D3"][1] / ref["GC"][1])
    print(f"3DDAの燃料削減率（対大圏）: 本実装 {mine_s:.1f}%  論文 {paper_s:.1f}%")

    if plot:
        plot_result(name, grid, weather, gc, d3)
        for e in (115.0, 120.0):
            plot_result(f"{name}_eta{int(e)}", grid, weather, gc, d3s[e])
    return gc, d3

def plot_result(name, grid, weather, gc, d3s):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for lab, r, col in (("Great circle", gc, "tab:blue"), ("3DDA", d3s, "tab:red")):
        lons = [p[1] for p in r["points"]]
        t = np.concatenate([[0.0], np.cumsum([_GEOD.inv(a[1], a[0], b[1], b[0])[2] / NM_TO_M / s
                                              for a, b, s in zip(r["points"][:-1], r["points"][1:],
                                                                  [s for s in r["speeds"] if s])])])
        hs = [weather(p[0], p[1], tt).Hs for p, tt in zip(r["points"], t)]
        ax[0].plot(lons[:-1], [s for s in r["speeds"] if s], "-o", ms=3, c=col, label=lab)
        ax[1].plot(lons, hs, "-o", ms=3, c=col, label=lab)
        ax[2].plot(lons, [p[0] for p in r["points"]], "-", c=col, label=lab)
    ax[0].set_ylabel("Speed [kn]"); ax[1].set_ylabel("Hs [m]"); ax[2].set_ylabel("Lat [deg]")
    ax[2].set_xlabel("Longitude [deg]"); ax[0].legend(); ax[0].set_title(f"Voyage {name}")
    plt.tight_layout(); plt.savefig(f"result_{name}.png", dpi=150)

    try:                                            # 地図に重ねる（plot_grid.py がある場合）
        import cartopy.crs as ccrs
        from plot_grid import plot_grid
        axm = plot_grid(grid)
        for r, col, lab in ((gc, "tab:blue", "great circle"), (d3s, "m", "3DDA")):
            axm.plot([p[1] for p in r["points"]], [p[0] for p in r["points"]], "-",
                     color=col, lw=2, transform=ccrs.PlateCarree(), zorder=7, label=lab)
        axm.legend(loc="lower left")
        plt.savefig(f"map_{name}.png", dpi=150)
    except ImportError:
        pass
    plt.close("all")


# if __name__ == "__main__":
#     run_voyage("20081224")          # まず1便だけ。全便は for name in PAPER: run_voyage(name)
#     for name in PAPER:
#         run_voyage(name)

# if __name__ == "__main__":
#     for name in PAPER:
#         try:
#             run_voyage(name)
#         except Exception as e:
#             print(f"!! {name} 失敗: {type(e).__name__}: {e}")

if __name__ == "__main__":
    for i, name in enumerate(PAPER, 1):
        log(f"=== {name} 開始（{i}/{len(PAPER)}）===")
        try:
            run_voyage(name)
        except Exception as e:
            log(f"!! {name} 失敗: {type(e).__name__}: {e}")