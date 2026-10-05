# # -*- coding: utf-8 -*-
# """
# make_weather_gif.py
# ===================
# 3DDA（reproduce_wang2019.py）を回さずに、実際の ERA5 データだけから気象の GIF を作る単体スクリプト。

# 使い方（プロジェクトのフォルダで）:
#   # 1) 有義波高 Hs の地図（いちばん軽い。ERA5 の .nc だけで動く）
#   python3 make_weather_gif.py 20081224

#   # 2) 抵抗が大きい場所：波＋風による抵抗増加を、静水中抵抗に対する % で表示
#   python3 make_weather_gif.py 20081224 --field resist --speed 17.9 --heading 90

#   # 3) 大圏航路の船を重ねる（reproduce_wang2019.py の設定から大圏を作る）
#   python3 make_weather_gif.py 20081224 --field resist --ship

#   # 4) 計算済みの 3DDA 結果（result_<name>.pkl）を重ねる
#   python3 make_weather_gif.py 20081224 --pkl result_20081224.pkl --eta 115

# 必要なファイル: era5_weather.py, animate_weather.py（同じフォルダ）、era5_<name>_*.nc
# --field resist のときだけ、Calm_Water_Resistance / Added_Resistance / Fuel_consumption /
# wind_resistance（これまでのモデル）を使う。
# """
# import argparse
# import glob
# import pickle
# from datetime import datetime, timedelta

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
# from scipy.interpolate import RegularGridInterpolator
# from pyproj import Geod

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from era5_weather import ERA5Weather
# from animate_weather import EXTENT, route_timeline, position_at, speed_at

# KT_TO_MS = 0.51444
# _GEOD = Geod(ellps="WGS84")


# # ----------------------------------------------------------------------
# # 角度・気象場
# # ----------------------------------------------------------------------
# def fold(a):
#     """角度差を 0〜180° に畳む（0 = 正面から）。"""
#     return np.abs(((a + 180.0) % 360.0) - 180.0)


# def raw_fields(weather, t_h, lat2d, lon2d):
#     """ERA5Weather の内部補間器で、格子全体を一括評価する。"""
#     lon_q = lon2d % 360.0 if weather._lon_360 else ((lon2d + 180.0) % 360.0) - 180.0
#     pts = np.column_stack([np.full(lat2d.size, t_h), lat2d.ravel(), lon_q.ravel()])
#     d = weather._interp(pts)                        # 列: swh, pp1d, sin(mwd), cos(mwd), u10, v10
#     sh = lat2d.shape
#     hs, tp = d[:, 0].reshape(sh), d[:, 1].reshape(sh)
#     s, c = d[:, 2].reshape(sh), d[:, 3].reshape(sh)
#     u, v = d[:, 4].reshape(sh), d[:, 5].reshape(sh)
#     return dict(
#         hs=hs, tp=tp,
#         wave_from=np.degrees(np.arctan2(s, c)) % 360.0,          # 波が来る方向
#         wind_speed=np.hypot(u, v),
#         wind_from=(180.0 + np.degrees(np.arctan2(u, v))) % 360.0,  # 風が吹いてくる方向
#         land=hs <= 0.0,
#     )


# # ----------------------------------------------------------------------
# # 抵抗モデル（--field resist のときだけ）
# # ----------------------------------------------------------------------
# class ResistanceMap:
#     """指定した船速で、(Hs, Tp, 波向, 風) から  (R_AW + R_AA) / R_calm [%]  を一括計算する。"""

#     def __init__(self, speed_kt):
#         from Calm_Water_Resistance import KCS as CalmWaterKCS
#         from Added_Resistance import KCS as AddedResistanceKCS
#         from Fuel_consumption import SeaState, added_resistance_irregular
#         from wind_resistance import wind_resistance, KCS_FULL_LOAD

#         self.v = speed_kt * KT_TO_MS
#         self.wind_resistance, self.wind_params = wind_resistance, KCS_FULL_LOAD
#         self.r_calm = float(CalmWaterKCS(is_full_scale=True).calc_calm_water_resistance(self.v))

#         # 付加抵抗は Hs^2 に比例するので、Hs = 1 m の値を (Tp, 波向) の表にしておく
#         added = AddedResistanceKCS(is_full_scale=True)
#         self.tps = np.arange(4.0, 21.0, 1.0)
#         self.rels = np.arange(0.0, 181.0, 15.0)
#         tab = np.zeros((len(self.tps), len(self.rels)))
#         for i, tp in enumerate(self.tps):
#             for j, rel in enumerate(self.rels):
#                 sea = SeaState(Hs=1.0, Tp=float(tp), wave_dir_rel_deg=float(rel),
#                                wind_speed_true=0.0, wind_dir_rel_deg=0.0)
#                 tab[i, j] = added_resistance_irregular(added, sea, self.v)
#         self.table = RegularGridInterpolator((self.tps, self.rels), tab,
#                                              bounds_error=False, fill_value=None)
#         print(f"静水中抵抗 {self.r_calm / 1e3:.0f} kN（{speed_kt:.1f} kn）", flush=True)

#     def percent(self, f, heading):
#         """heading: 度（スカラー、または格子と同じ形の配列）。"""
#         hs = np.nan_to_num(f["hs"])
#         tp = np.clip(f["tp"], self.tps[0], self.tps[-1])
#         rel_w = fold(f["wave_from"] - heading)
#         k = self.table(np.column_stack([tp.ravel(), rel_w.ravel()])).reshape(hs.shape)
#         r_aw = hs ** 2 * k

#         rel_wind = np.radians(fold(f["wind_from"] - heading))
#         ux = self.v + f["wind_speed"] * np.cos(rel_wind)
#         uy = f["wind_speed"] * np.sin(rel_wind)
#         u_a = np.hypot(ux, uy)
#         psi_a = np.degrees(np.arctan2(uy, ux))
#         raa, _ = self.wind_resistance(psi_a.ravel(), u_a.ravel(), self.wind_params)
#         r_aa = np.asarray(raa, dtype=float).reshape(hs.shape)
#         return 100.0 * (r_aw + r_aa) / self.r_calm


# # ----------------------------------------------------------------------
# # 航路（重ねる場合）
# # ----------------------------------------------------------------------
# def make_heading_fn(points):
#     """航路の各区間の方位を、経度の関数として返す（格子上の各点の針路に使う）。"""
#     lon_mid, az = [], []
#     for a, b in zip(points[:-1], points[1:]):
#         az.append(_GEOD.inv(a[1], a[0], b[1], b[0])[0] % 360.0)
#         lon_mid.append(0.5 * (a[1] + b[1]))
#     lon_mid, az = np.array(lon_mid), np.unwrap(np.radians(az))
#     order = np.argsort(lon_mid)
#     lon_mid, az = lon_mid[order], np.degrees(az[order])
#     return lambda lon2d: np.interp(lon2d, lon_mid, az) % 360.0


# def great_circle_from_paper(name):
#     """reproduce_wang2019.py の設定から、大圏の点列と船速 Vf を作る。"""
#     from reproduce_wang2019 import PAPER, calibrate_west_lon
#     ref = PAPER[name]
#     east_pt, west_lat = (49.3, -10.0), 46.5
#     west_pt = (west_lat, calibrate_west_lon(east_pt, west_lat, ref["GC"][2]))
#     start, end = (west_pt, east_pt) if ref["dir"] == "E" else (east_pt, west_pt)
#     mid = _GEOD.npts(start[1], start[0], end[1], end[0], 26)          # (lon, lat)
#     pts = [start] + [(la, lo) for lo, la in mid] + [end]
#     dist_nm = _GEOD.inv(start[1], start[0], end[1], end[0])[2] / 1852.0
#     vf = dist_nm / ref["actual_eta"]
#     return dict(points=pts, speeds=[vf] * (len(pts) - 1))


# # ----------------------------------------------------------------------
# # GIF 作成
# # ----------------------------------------------------------------------
# def make_gif(name, weather, field, out, hours, step_h, fps, dpi, vmax,
#              routes, resist=None, heading=None, arrow_step=6, land=True):
#     t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]))
#     t_max = float(weather._interp.grid[0][-1])
#     hours = min(hours, t_max)
#     frames = np.append(np.arange(0.0, hours, step_h), hours)

#     lons = np.arange(EXTENT[0], EXTENT[1] + 1e-6, 0.25)
#     lats = np.arange(EXTENT[2], EXTENT[3] + 1e-6, 0.25)
#     lon2d, lat2d = np.meshgrid(lons, lats)
#     heading_arr = heading(lon2d) if callable(heading) else heading

#     def get_field(t):
#         f = raw_fields(weather, t, lat2d, lon2d)
#         if field == "resist":
#             val = resist.percent(f, heading_arr)
#         else:
#             val = f["hs"]
#         return f, np.where(f["land"], np.nan, val)

#     fig = plt.figure(figsize=(9.5, 7.2), dpi=dpi)
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#     ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

#     f0, v0 = get_field(0.0)
#     cmap = plt.get_cmap("turbo" if field == "hs" else "YlOrRd").copy()
#     cmap.set_bad(alpha=0.0)
#     im = ax.imshow(v0, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
#                    transform=ccrs.PlateCarree(), cmap=cmap, vmin=0.0, vmax=vmax,
#                    interpolation="bilinear", zorder=1)
#     cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
#     cb.set_label("Significant wave height Hs [m]" if field == "hs"
#                  else "Extra resistance (waves + wind) / calm-water resistance [%]")

#     if land:
#         ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e4dccb",
#                        edgecolor="0.3", linewidth=0.5, zorder=3)
#     gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle=":")
#     gl.top_labels = False
#     gl.right_labels = False

#     qs = (slice(None, None, arrow_step), slice(None, None, arrow_step))

#     def arrows(f):                                   # 波の進行方向（来る方向の反対）
#         th = np.radians(f["wave_from"] + 180.0)
#         U, V = np.sin(th), np.cos(th)
#         return np.where(f["land"], np.nan, U)[qs], np.where(f["land"], np.nan, V)[qs]

#     U0, V0 = arrows(f0)
#     quiv = ax.quiver(lon2d[qs], lat2d[qs], U0, V0, transform=ccrs.PlateCarree(),
#                      angles="xy", scale_units="xy", scale=1.1, width=0.0018,
#                      color="k" if field == "resist" else "white", alpha=0.45, zorder=2)

#     ships = []
#     for r in routes:
#         pts = r["points"]
#         ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color=r["color"], lw=1.0,
#                 alpha=0.6, transform=ccrs.PlateCarree(), zorder=5)
#         tl, sp = route_timeline(pts, r["speeds"])
#         trail, = ax.plot([], [], "-", color=r["color"], lw=2.5, transform=ccrs.PlateCarree(), zorder=6)
#         dot, = ax.plot([], [], "o", ms=10, mfc=r["color"], mec="k", mew=1.5,
#                        transform=ccrs.PlateCarree(), zorder=7, label=r["label"])
#         ships.append((r, tl, sp, trail, dot))
#     if ships:
#         ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

#     info = ax.text(0.99, 0.015, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
#                    bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9), zorder=8)
#     title = ax.set_title("")
#     state = {"cs": None}

#     def clear_contour():
#         cs = state["cs"]
#         if cs is None:
#             return
#         try:
#             cs.remove()
#         except AttributeError:                       # matplotlib < 3.8
#             for c in cs.collections:
#                 c.remove()
#         state["cs"] = None

#     def update(t):
#         f, val = get_field(t)
#         im.set_data(val)
#         U, V = arrows(f)
#         quiv.set_UVC(U, V)
#         clear_contour()
#         if np.nanmax(f["hs"]) > 5.0:                 # Hs = 5 m の等値線
#             state["cs"] = ax.contour(lons, lats, np.nan_to_num(f["hs"]), levels=[5.0],
#                                      colors="k", linewidths=1.0,
#                                      transform=ccrs.PlateCarree(), zorder=2)
#         lines = []
#         for r, tl, sp, trail, dot in ships:
#             la, lo = position_at(t, tl, r["points"])
#             dot.set_data([lo], [la])
#             idx = int(np.searchsorted(tl, t, side="right"))
#             trail.set_data([p[1] for p in r["points"][:idx]] + [lo],
#                            [p[0] for p in r["points"][:idx]] + [la])
#             lines.append(f'{r["label"]}: {speed_at(t, tl, sp):4.1f} kn, Hs {weather(la, lo, t).Hs:3.1f} m')
#         if field == "resist" and not lines:
#             lines.append(f"Vf {resist.v / KT_TO_MS:.1f} kn")
#         info.set_text("\n".join(lines))
#         title.set_text(f"{name}   {t0 + timedelta(hours=float(t)):%Y-%m-%d %H:%M} UTC   (t = {t:5.1f} h)")
#         return ()

#     anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
#     anim.save(out, writer=PillowWriter(fps=fps))
#     plt.close(fig)
#     print(f"保存しました: {out}（{len(frames)} フレーム）", flush=True)


# # ----------------------------------------------------------------------
# def main():
#     ap = argparse.ArgumentParser(description="ERA5 の気象 GIF（3DDA は回さない）")
#     ap.add_argument("name", help="便の名前（例 20081224）。era5_<name>_*.nc を読む")
#     ap.add_argument("--field", choices=["hs", "resist"], default="hs")
#     ap.add_argument("--hours", type=float, default=108.0, help="アニメの長さ [h]")
#     ap.add_argument("--step", type=float, default=3.0, help="フレーム間隔 [h]")
#     ap.add_argument("--fps", type=int, default=6)
#     ap.add_argument("--dpi", type=int, default=80)
#     ap.add_argument("--vmax", type=float, default=None, help="色の上限（Hs: 8 m、resist: 200 %）")
#     ap.add_argument("--speed", type=float, default=None, help="resist の船速 [kn]")
#     ap.add_argument("--heading", type=float, default=None, help="resist の針路 [度]（航路があれば自動）")
#     ap.add_argument("--ship", action="store_true", help="大圏航路の船を重ねる")
#     ap.add_argument("--pkl", default=None, help="run_voyage が保存した result_<name>.pkl")
#     ap.add_argument("--eta", type=float, default=None, help="--pkl のうち使う ETA [h]")
#     ap.add_argument("--arrows", type=int, default=6, help="波向き矢印の間引き（大きいほど疎）")
#     ap.add_argument("--no-land", action="store_true", help="陸地の描画を省く")
#     ap.add_argument("--out", default=None)
#     a = ap.parse_args()

#     paths = sorted(glob.glob(f"era5_{a.name}_*.nc"))
#     if not paths:
#         raise SystemExit(f"era5_{a.name}_*.nc が見つかりません")
#     t0 = datetime(int(a.name[:4]), int(a.name[4:6]), int(a.name[6:8]))
#     weather = ERA5Weather(paths, t0)
#     print("ERA5 読み込み完了", flush=True)

#     routes = []
#     if a.pkl:
#         with open(a.pkl, "rb") as f:
#             r = pickle.load(f)
#         eta = a.eta if a.eta is not None else min(r["d3s"])
#         routes = [dict(label="Great circle", color="tab:blue", **{k: r["gc_same"][eta][k] for k in ("points", "speeds")}),
#                   dict(label="3DDA", color="magenta", **{k: r["d3s"][eta][k] for k in ("points", "speeds")})]
#     elif a.ship:
#         g = great_circle_from_paper(a.name)
#         routes = [dict(label="Great circle", color="tab:blue", **g)]

#     resist, heading = None, None
#     if a.field == "resist":
#         speed = a.speed
#         if speed is None:
#             speed = [s for s in routes[0]["speeds"] if s][0] if routes else 17.0
#             print(f"船速の指定がないので {speed:.1f} kn を使います（--speed で変更）", flush=True)
#         if a.heading is not None:
#             heading = a.heading
#         elif routes:
#             heading = make_heading_fn(routes[0]["points"])
#         else:
#             raise SystemExit("--heading（針路[度]。東航は約 90、西航は約 270）か --ship / --pkl を指定してください")
#         resist = ResistanceMap(speed)

#     vmax = a.vmax if a.vmax is not None else (8.0 if a.field == "hs" else 200.0)
#     out = a.out or f"weather_{a.name}_{a.field}.gif"
#     make_gif(a.name, weather, a.field, out, a.hours, a.step, a.fps, a.dpi, vmax,
#              routes, resist=resist, heading=heading, arrow_step=a.arrows, land=not a.no_land)


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
"""
make_weather_gif.py
===================
3DDA（reproduce_wang2019.py）を回さずに、実際の ERA5 データだけから気象の GIF を作る単体スクリプト。

使い方（プロジェクトのフォルダで）:
  # 1) 有義波高 Hs の地図（いちばん軽い。ERA5 の .nc だけで動く）
  python3 make_weather_gif.py 20081224

  # 2) 抵抗が大きい場所：波＋風による抵抗増加を、静水中抵抗に対する % で表示
  python3 make_weather_gif.py 20081224 --field resist --speed 17.9 --heading 90

  # 3) 大圏航路の船を重ねる（reproduce_wang2019.py の設定から大圏を作る）
  python3 make_weather_gif.py 20081224 --field resist --ship

  # 4) 計算済みの 3DDA 結果（result_<name>.pkl）を重ねる
  python3 make_weather_gif.py 20081224 --pkl result_20081224.pkl --eta 115

必要なファイル: era5_weather.py, animate_weather.py（同じフォルダ）、era5_<name>_*.nc
--field resist のときだけ、Calm_Water_Resistance / Added_Resistance / Fuel_consumption /
wind_resistance（これまでのモデル）を使う。
"""
import argparse
import glob
import pickle
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator
from pyproj import Geod

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from era5_weather import ERA5Weather
from animate_weather import EXTENT, route_timeline, position_at, speed_at

KT_TO_MS = 0.51444
_GEOD = Geod(ellps="WGS84")


# ----------------------------------------------------------------------
# 角度・気象場
# ----------------------------------------------------------------------
def fold(a):
    """角度差を 0〜180° に畳む（0 = 正面から）。"""
    return np.abs(((a + 180.0) % 360.0) - 180.0)


def raw_fields(weather, t_h, lat2d, lon2d):
    """ERA5Weather の内部補間器で、格子全体を一括評価する。"""
    lon_q = lon2d % 360.0 if weather._lon_360 else ((lon2d + 180.0) % 360.0) - 180.0
    pts = np.column_stack([np.full(lat2d.size, t_h), lat2d.ravel(), lon_q.ravel()])
    d = weather._interp(pts)                        # 列: swh, pp1d, sin(mwd), cos(mwd), u10, v10
    sh = lat2d.shape
    hs, tp = d[:, 0].reshape(sh), d[:, 1].reshape(sh)
    s, c = d[:, 2].reshape(sh), d[:, 3].reshape(sh)
    u, v = d[:, 4].reshape(sh), d[:, 5].reshape(sh)
    return dict(
        hs=hs, tp=tp,
        wave_from=np.degrees(np.arctan2(s, c)) % 360.0,          # 波が来る方向
        wind_speed=np.hypot(u, v),
        wind_from=(180.0 + np.degrees(np.arctan2(u, v))) % 360.0,  # 風が吹いてくる方向
        land=hs <= 0.0,
    )


# ----------------------------------------------------------------------
# 抵抗モデル（--field resist のときだけ）
# ----------------------------------------------------------------------
class ResistanceMap:
    """指定した船速で、(Hs, Tp, 波向, 風) から  (R_AW + R_AA) / R_calm [%]  を一括計算する。"""

    def __init__(self, speed_kt):
        from Calm_Water_Resistance import KCS as CalmWaterKCS
        from Added_Resistance import KCS as AddedResistanceKCS
        from Fuel_consumption import SeaState, added_resistance_irregular
        from wind_resistance import wind_resistance, KCS_FULL_LOAD

        self.v = speed_kt * KT_TO_MS
        self.wind_resistance, self.wind_params = wind_resistance, KCS_FULL_LOAD
        self.r_calm = float(CalmWaterKCS(is_full_scale=True).calc_calm_water_resistance(self.v))

        # 付加抵抗は Hs^2 に比例するので、Hs = 1 m の値を (Tp, 波向) の表にしておく
        added = AddedResistanceKCS(is_full_scale=True)
        self.tps = np.arange(4.0, 21.0, 1.0)
        self.rels = np.arange(0.0, 181.0, 15.0)
        tab = np.zeros((len(self.tps), len(self.rels)))
        for i, tp in enumerate(self.tps):
            for j, rel in enumerate(self.rels):
                sea = SeaState(Hs=1.0, Tp=float(tp), wave_dir_rel_deg=float(rel),
                               wind_speed_true=0.0, wind_dir_rel_deg=0.0)
                tab[i, j] = added_resistance_irregular(added, sea, self.v)
        self.table = RegularGridInterpolator((self.tps, self.rels), tab,
                                             bounds_error=False, fill_value=None)
        print(f"静水中抵抗 {self.r_calm / 1e3:.0f} kN（{speed_kt:.1f} kn）", flush=True)

    def percent(self, f, heading):
        """heading: 度（スカラー、または格子と同じ形の配列）。"""
        hs = np.nan_to_num(f["hs"])
        tp = np.clip(f["tp"], self.tps[0], self.tps[-1])
        rel_w = fold(f["wave_from"] - heading)
        k = self.table(np.column_stack([tp.ravel(), rel_w.ravel()])).reshape(hs.shape)
        r_aw = hs ** 2 * k

        rel_wind = np.radians(fold(f["wind_from"] - heading))
        ux = self.v + f["wind_speed"] * np.cos(rel_wind)
        uy = f["wind_speed"] * np.sin(rel_wind)
        u_a = np.hypot(ux, uy)
        psi_a = np.degrees(np.arctan2(uy, ux))
        raa, _ = self.wind_resistance(psi_a.ravel(), u_a.ravel(), self.wind_params)
        r_aa = np.asarray(raa, dtype=float).reshape(hs.shape)
        return 100.0 * (r_aw + r_aa) / self.r_calm


# ----------------------------------------------------------------------
# 航路（重ねる場合）
# ----------------------------------------------------------------------
def make_heading_fn(points):
    """航路の各区間の方位を、経度の関数として返す（格子上の各点の針路に使う）。"""
    lon_mid, az = [], []
    for a, b in zip(points[:-1], points[1:]):
        az.append(_GEOD.inv(a[1], a[0], b[1], b[0])[0] % 360.0)
        lon_mid.append(0.5 * (a[1] + b[1]))
    lon_mid, az = np.array(lon_mid), np.unwrap(np.radians(az))
    order = np.argsort(lon_mid)
    lon_mid, az = lon_mid[order], np.degrees(az[order])
    return lambda lon2d: np.interp(lon2d, lon_mid, az) % 360.0


def load_route_csv(path):
    """CSV（緯度, 経度, 速度[kn] の順。ヘッダー行は読み飛ばす）から航路を作る。
    速度は区間ごとの値で、空欄は最後の行だけに許す。"""
    pts, sp = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            try:
                la, lo = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue
            pts.append((la, lo))
            sp.append(float(parts[2]) if len(parts) > 2 and parts[2] else None)
    if len(pts) < 2 or not any(sp):
        raise SystemExit(f"{path}: 緯度,経度,速度[kn] の列が 2 行以上必要です")
    return dict(points=pts, speeds=sp)


def great_circle_from_paper(name):
    """reproduce_wang2019.py の設定から、大圏の点列と船速 Vf を作る。"""
    from reproduce_wang2019 import PAPER, calibrate_west_lon
    ref = PAPER[name]
    east_pt, west_lat = (49.3, -10.0), 46.5
    west_pt = (west_lat, calibrate_west_lon(east_pt, west_lat, ref["GC"][2]))
    start, end = (west_pt, east_pt) if ref["dir"] == "E" else (east_pt, west_pt)
    mid = _GEOD.npts(start[1], start[0], end[1], end[0], 26)          # (lon, lat)
    pts = [start] + [(la, lo) for lo, la in mid] + [end]
    dist_nm = _GEOD.inv(start[1], start[0], end[1], end[0])[2] / 1852.0
    vf = dist_nm / ref["actual_eta"]
    return dict(points=pts, speeds=[vf] * (len(pts) - 1))


# ----------------------------------------------------------------------
# GIF 作成
# ----------------------------------------------------------------------
def make_gif(name, weather, field, out, hours, step_h, fps, dpi, vmax,
             routes, resist=None, heading=None, arrow_step=6, land=True, tick_h=24.0):
    t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]))
    t_max = float(weather._interp.grid[0][-1])
    hours = min(hours, t_max)
    frames = np.append(np.arange(0.0, hours, step_h), hours)

    lons = np.arange(EXTENT[0], EXTENT[1] + 1e-6, 0.25)
    lats = np.arange(EXTENT[2], EXTENT[3] + 1e-6, 0.25)
    lon2d, lat2d = np.meshgrid(lons, lats)
    heading_arr = heading(lon2d) if callable(heading) else heading

    def get_field(t):
        f = raw_fields(weather, t, lat2d, lon2d)
        if field == "resist":
            val = resist.percent(f, heading_arr)
        else:
            val = f["hs"]
        return f, np.where(f["land"], np.nan, val)

    fig = plt.figure(figsize=(9.5, 7.2), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    f0, v0 = get_field(0.0)
    cmap = plt.get_cmap("turbo" if field == "hs" else "YlOrRd").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(v0, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=0.0, vmax=vmax,
                   interpolation="bilinear", zorder=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Significant wave height Hs [m]" if field == "hs"
                 else "Extra resistance (waves + wind) / calm-water resistance [%]")

    if land:
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e4dccb",
                       edgecolor="0.3", linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    qs = (slice(None, None, arrow_step), slice(None, None, arrow_step))

    def arrows(f):                                   # 波の進行方向（来る方向の反対）
        th = np.radians(f["wave_from"] + 180.0)
        U, V = np.sin(th), np.cos(th)
        return np.where(f["land"], np.nan, U)[qs], np.where(f["land"], np.nan, V)[qs]

    U0, V0 = arrows(f0)
    quiv = ax.quiver(lon2d[qs], lat2d[qs], U0, V0, transform=ccrs.PlateCarree(),
                     angles="xy", scale_units="xy", scale=1.1, width=0.0018,
                     color="k" if field == "resist" else "white", alpha=0.45, zorder=2)

    ships = []
    for r in routes:
        pts = r["points"]
        ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color=r["color"], lw=1.0,
                alpha=0.6, transform=ccrs.PlateCarree(), zorder=5)
        tl, sp = route_timeline(pts, r["speeds"])
        if tick_h and tick_h > 0:                    # 軌跡上に経過時間の目盛りを描く
            dy = 0.9 if len(ships) % 2 == 0 else -1.4
            for th_ in np.arange(tick_h, tl[-1], tick_h):
                la_t, lo_t = position_at(th_, tl, pts)
                ax.plot([lo_t], [la_t], "s", ms=4, mfc="white", mec=r["color"], mew=1.3,
                        transform=ccrs.PlateCarree(), zorder=6)
                ax.text(lo_t, la_t + dy, f"{th_:.0f} h", color=r["color"], fontsize=7,
                        ha="center", va="center", transform=ccrs.PlateCarree(), zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        trail, = ax.plot([], [], "-", color=r["color"], lw=2.5, transform=ccrs.PlateCarree(), zorder=6)
        dot, = ax.plot([], [], "o", ms=10, mfc=r["color"], mec="k", mew=1.5,
                       transform=ccrs.PlateCarree(), zorder=7, label=r["label"])
        ships.append((r, tl, sp, trail, dot))
    if ships:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    info = ax.text(0.99, 0.015, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                   bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9), zorder=8)
    title = ax.set_title("")
    state = {"cs": None}

    def clear_contour():
        cs = state["cs"]
        if cs is None:
            return
        try:
            cs.remove()
        except AttributeError:                       # matplotlib < 3.8
            for c in cs.collections:
                c.remove()
        state["cs"] = None

    def update(t):
        f, val = get_field(t)
        im.set_data(val)
        U, V = arrows(f)
        quiv.set_UVC(U, V)
        clear_contour()
        if np.nanmax(f["hs"]) > 5.0:                 # Hs = 5 m の等値線
            state["cs"] = ax.contour(lons, lats, np.nan_to_num(f["hs"]), levels=[5.0],
                                     colors="k", linewidths=1.0,
                                     transform=ccrs.PlateCarree(), zorder=2)
        lines = []
        for r, tl, sp, trail, dot in ships:
            la, lo = position_at(t, tl, r["points"])
            dot.set_data([lo], [la])
            idx = int(np.searchsorted(tl, t, side="right"))
            trail.set_data([p[1] for p in r["points"][:idx]] + [lo],
                           [p[0] for p in r["points"][:idx]] + [la])
            lines.append(f'{r["label"]}: {speed_at(t, tl, sp):4.1f} kn, Hs {weather(la, lo, t).Hs:3.1f} m')
        if field == "resist" and not lines:
            lines.append(f"Vf {resist.v / KT_TO_MS:.1f} kn")
        info.set_text("\n".join(lines))
        title.set_text(f"{name}   {t0 + timedelta(hours=float(t)):%Y-%m-%d %H:%M} UTC   (t = {t:5.1f} h)")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"保存しました: {out}（{len(frames)} フレーム）", flush=True)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ERA5 の気象 GIF（3DDA は回さない）")
    ap.add_argument("name", help="便の名前（例 20081224）。era5_<name>_*.nc を読む")
    ap.add_argument("--field", choices=["hs", "resist"], default="hs")
    ap.add_argument("--hours", type=float, default=108.0, help="アニメの長さ [h]")
    ap.add_argument("--step", type=float, default=3.0, help="フレーム間隔 [h]")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--vmax", type=float, default=None, help="色の上限（Hs: 8 m、resist: 200 %）")
    ap.add_argument("--speed", type=float, default=None, help="resist の船速 [kn]")
    ap.add_argument("--heading", type=float, default=None, help="resist の針路 [度]（航路があれば自動）")
    ap.add_argument("--ship", action="store_true", help="大圏航路の船を重ねる")
    ap.add_argument("--pkl", default=None, help="run_voyage が保存した result_<name>.pkl")
    ap.add_argument("--eta", type=float, default=None, help="--pkl のうち使う ETA [h]")
    ap.add_argument("--arrows", type=int, default=6, help="波向き矢印の間引き（大きいほど疎）")
    ap.add_argument("--route", action="append", default=None,
                    help="重ねる航路の CSV（緯度,経度,速度[kn]）。複数回指定できる")
    ap.add_argument("--ticks", type=float, default=24.0,
                    help="軌跡上に経過時間を描く間隔 [h]（0 で描かない）")
    ap.add_argument("--no-land", action="store_true", help="陸地の描画を省く")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(f"era5_{a.name}_*.nc"))
    if not paths:
        raise SystemExit(f"era5_{a.name}_*.nc が見つかりません")
    t0 = datetime(int(a.name[:4]), int(a.name[4:6]), int(a.name[6:8]))
    weather = ERA5Weather(paths, t0)
    print("ERA5 読み込み完了", flush=True)

    routes = []
    if a.pkl:
        with open(a.pkl, "rb") as f:
            r = pickle.load(f)
        eta = a.eta if a.eta is not None else min(r["d3s"])
        routes = [dict(label="Great circle", color="tab:blue", **{k: r["gc_same"][eta][k] for k in ("points", "speeds")}),
                  dict(label="3DDA", color="magenta", **{k: r["d3s"][eta][k] for k in ("points", "speeds")})]
    elif a.ship:
        g = great_circle_from_paper(a.name)
        routes = [dict(label="Great circle", color="tab:blue", **g)]

    palette = ["tab:green", "tab:orange", "tab:purple", "tab:brown"]
    for i, path in enumerate(a.route or []):
        r = load_route_csv(path)
        label = path.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        routes.append(dict(label=label, color=palette[i % len(palette)], **r))

    resist, heading = None, None
    if a.field == "resist":
        speed = a.speed
        if speed is None:
            speed = [s for s in routes[0]["speeds"] if s][0] if routes else 17.0
            print(f"船速の指定がないので {speed:.1f} kn を使います（--speed で変更）", flush=True)
        if a.heading is not None:
            heading = a.heading
        elif routes:
            heading = make_heading_fn(routes[0]["points"])
        else:
            raise SystemExit("--heading（針路[度]。東航は約 90、西航は約 270）か --ship / --pkl を指定してください")
        resist = ResistanceMap(speed)

    vmax = a.vmax if a.vmax is not None else (8.0 if a.field == "hs" else 200.0)
    out = a.out or f"weather_{a.name}_{a.field}.gif"
    make_gif(a.name, weather, a.field, out, a.hours, a.step, a.fps, a.dpi, vmax,
             routes, resist=resist, heading=heading, arrow_step=a.arrows, land=not a.no_land, tick_h=a.ticks)


if __name__ == "__main__":
    main()