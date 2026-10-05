# -*- coding: utf-8 -*-
"""
animate_weather.py
==================
ERA5 の有義波高 Hs を地図上にアニメーション（GIF）で描き、大圏航路の船と 3DDA 航路の船を
同じ時刻で動かして、「どの時点で・どこの荒天を避けたか」が分かるようにする。

使い方（reproduce_wang2019.py の run_voyage 内など）:
    from animate_weather import animate_voyage
    animate_voyage(name, weather, gc, d3, grid=grid)          # -> anim_<name>.gif

引数:
    weather : ERA5Weather（era5_weather.py）。内部の _interp と _lon_360 を使う
    gc, d3  : run_voyage が作る辞書。必要なキーは points（[(lat, lon), ...]）と
              speeds（区間ごとの速度[kt]。末尾に None があってもよい）
    grid    : 指定するとノードを薄い点で重ねる（省略可）
"""
import math
from datetime import datetime, timedelta

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")
NM_TO_M = 1852.0

# 既存の地図と同じ範囲 [西端, 東端, 南端, 北端]
EXTENT = [-55.0, -7.5, 38.0, 62.0]


# ----------------------------------------------------------------------
# 航路 -> 時刻・位置・速度
# ----------------------------------------------------------------------
def route_timeline(points, speeds):
    """各ノード通過時刻 [h]（出発 = 0）。speeds は区間ごとの速度[kt]。"""
    sp = [s for s in speeds if s]
    t = [0.0]
    for a, b, v in zip(points[:-1], points[1:], sp):
        d_nm = _GEOD.inv(a[1], a[0], b[1], b[0])[2] / NM_TO_M
        t.append(t[-1] + d_nm / v)
    return np.array(t), sp


def position_at(t, tl, points):
    lat = np.interp(t, tl, [p[0] for p in points[:len(tl)]])
    lon = np.interp(t, tl, [p[1] for p in points[:len(tl)]])
    return float(lat), float(lon)


def speed_at(t, tl, sp):
    i = int(np.searchsorted(tl, t, side="right") - 1)
    i = min(max(i, 0), len(sp) - 1)
    return sp[i]


# ----------------------------------------------------------------------
# 気象場（ERA5Weather の内部補間器を直接使って一括評価）
# ----------------------------------------------------------------------
def wave_field(weather, t_h, lat2d, lon2d):
    """戻り値: Hs[m]（陸は NaN）, 波の進行方向の東成分 U, 北成分 V（単位ベクトル）"""
    lon_q = lon2d % 360.0 if weather._lon_360 else ((lon2d + 180.0) % 360.0) - 180.0
    pts = np.column_stack([np.full(lat2d.size, t_h), lat2d.ravel(), lon_q.ravel()])
    d = weather._interp(pts)                       # 列: swh, pp1d, sin(mwd), cos(mwd), u10, v10
    hs = d[:, 0].reshape(lat2d.shape)
    s = d[:, 2].reshape(lat2d.shape)
    c = d[:, 3].reshape(lat2d.shape)
    n = np.hypot(s, c) + 1e-9
    # mwd は「波が来る方向」。矢印は進む向き（来る方向の反対）で描く
    U, V = -s / n, -c / n
    land = hs <= 0.0
    hs = np.where(land, np.nan, hs)
    U = np.where(land, np.nan, U)
    V = np.where(land, np.nan, V)
    return hs, U, V


# ----------------------------------------------------------------------
# 本体
# ----------------------------------------------------------------------
def animate_voyage(name, weather, gc, d3, grid=None, out=None,
                   step_h=3.0, fps=6, dpi=80, hs_max=8.0,
                   field_res=0.25, arrow_step=3, land=True):
    """GIF を書き出す。戻り値は出力ファイル名。"""
    out = out or f"anim_{name}.gif"

    tl_gc, sp_gc = route_timeline(gc["points"], gc["speeds"])
    tl_d3, sp_d3 = route_timeline(d3["points"], d3["speeds"])
    T = float(max(tl_gc[-1], tl_d3[-1]))
    frames = np.append(np.arange(0.0, T, step_h), T)

    try:                                            # 表示用の日付（name の先頭 8 桁が YYYYMMDD）
        t0 = datetime(int(name[:4]), int(name[4:6]), int(name[6:8]))
    except ValueError:
        t0 = None

    # --- 補助：船位置での Hs の時系列（下段の図用）---
    t_ser = np.arange(0.0, T + 0.5, 0.5)

    def hs_series(tl, points):
        out_hs = []
        for t in t_ser:
            la, lo = position_at(t, tl, points)
            out_hs.append(weather(la, lo, t).Hs)
        return np.array(out_hs)

    hs_gc_ser = hs_series(tl_gc, gc["points"])
    hs_d3_ser = hs_series(tl_d3, d3["points"])

    def speed_series(tl, sp):
        return np.array([speed_at(t, tl, sp) for t in t_ser])

    v_gc_ser = speed_series(tl_gc, sp_gc)
    v_d3_ser = speed_series(tl_d3, sp_d3)

    # --- 図の骨組み ---
    lons = np.arange(EXTENT[0], EXTENT[1] + 1e-6, field_res)
    lats = np.arange(EXTENT[2], EXTENT[3] + 1e-6, field_res)
    lon2d, lat2d = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(9.5, 9.5), dpi=dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.4, 1, 1], hspace=0.30)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_v = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2], sharex=ax_v)

    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    hs0, U0, V0 = wave_field(weather, 0.0, lat2d, lon2d)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(hs0, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=0.0, vmax=hs_max,
                   interpolation="bilinear", zorder=1)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cb.set_label("Significant wave height Hs [m]")

    if land:
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e4dccb",
                       edgecolor="0.3", linewidth=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    qs = (slice(None, None, arrow_step), slice(None, None, arrow_step))
    quiv = ax.quiver(lon2d[qs], lat2d[qs], U0[qs], V0[qs], transform=ccrs.PlateCarree(),
                     angles="xy", scale_units="xy", scale=1.1, width=0.0018,
                     color="white", alpha=0.7, zorder=2)

    if grid is not None:
        gx = [n.lon for row in grid.stages for n in row]
        gy = [n.lat for row in grid.stages for n in row]
        ax.plot(gx, gy, ".", ms=1.5, color="k", alpha=0.25, transform=ccrs.PlateCarree(), zorder=4)

    col_gc, col_d3 = "tab:blue", "magenta"
    for pts, col in ((gc["points"], col_gc), (d3["points"], col_d3)):
        ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color=col, lw=1.0, alpha=0.5,
                transform=ccrs.PlateCarree(), zorder=5)
    trail_gc, = ax.plot([], [], "-", color=col_gc, lw=2.5, transform=ccrs.PlateCarree(), zorder=6)
    trail_d3, = ax.plot([], [], "-", color=col_d3, lw=2.5, transform=ccrs.PlateCarree(), zorder=6)
    ship_gc, = ax.plot([], [], "o", ms=10, mfc=col_gc, mec="k", mew=1.5,
                       transform=ccrs.PlateCarree(), zorder=7, label="Great circle (constant speed)")
    ship_d3, = ax.plot([], [], "o", ms=10, mfc=col_d3, mec="k", mew=1.5,
                       transform=ccrs.PlateCarree(), zorder=7, label="3DDA")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    info = ax.text(0.99, 0.015, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                   bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9), zorder=8)
    title = ax.set_title("")

    # 下段：速度と、船位置の Hs の時系列（現在時刻に縦線）
    ax_v.plot(t_ser, v_gc_ser, color=col_gc, lw=1.5, label="Great circle")
    ax_v.plot(t_ser, v_d3_ser, color=col_d3, lw=1.5, label="3DDA")
    ax_v.set_ylabel("Speed [kn]")
    ax_v.legend(loc="upper right", fontsize=7, ncol=2)
    ax_v.grid(alpha=0.3)
    plt.setp(ax_v.get_xticklabels(), visible=False)
    ax_h.plot(t_ser, hs_gc_ser, color=col_gc, lw=1.5)
    ax_h.plot(t_ser, hs_d3_ser, color=col_d3, lw=1.5)
    ax_h.set_ylabel("Hs at ship [m]")
    ax_h.set_xlabel("Elapsed time [h]")
    ax_h.set_xlim(0, T)
    ax_h.grid(alpha=0.3)
    vl_v = ax_v.axvline(0, color="k", lw=1)
    vl_h = ax_h.axvline(0, color="k", lw=1)

    state = {"contour": None}

    def _clear_contour():
        cs = state["contour"]
        if cs is None:
            return
        try:
            cs.remove()
        except AttributeError:                       # matplotlib < 3.8
            for c in cs.collections:
                c.remove()
        state["contour"] = None

    def update(t):
        hs, U, V = wave_field(weather, t, lat2d, lon2d)
        im.set_data(hs)
        quiv.set_UVC(U[qs], V[qs])
        _clear_contour()
        if np.nanmax(hs) > 5.0:                      # Hs = 5 m の等値線（荒天域の目安）
            state["contour"] = ax.contour(lons, lats, np.nan_to_num(hs), levels=[5.0],
                                          colors="k", linewidths=1.0,
                                          transform=ccrs.PlateCarree(), zorder=2)
        lines = []
        for tl, pts, sp, ship, trail in ((tl_gc, gc["points"], sp_gc, ship_gc, trail_gc),
                                         (tl_d3, d3["points"], sp_d3, ship_d3, trail_d3)):
            la, lo = position_at(t, tl, pts)
            ship.set_data([lo], [la])
            idx = int(np.searchsorted(tl, t, side="right"))
            xs = [p[1] for p in pts[:idx]] + [lo]
            ys = [p[0] for p in pts[:idx]] + [la]
            trail.set_data(xs, ys)
            lines.append((la, lo, speed_at(t, tl, sp)))
        (la1, lo1, v1), (la2, lo2, v2) = lines
        h1 = weather(la1, lo1, t).Hs
        h2 = weather(la2, lo2, t).Hs
        info.set_text(f"Great circle: {v1:4.1f} kn, Hs {h1:3.1f} m\n"
                      f"3DDA:         {v2:4.1f} kn, Hs {h2:3.1f} m")
        stamp = f"{t0 + timedelta(hours=float(t)):%Y-%m-%d %H:%M} UTC  " if t0 else ""
        title.set_text(f"Voyage {name}   {stamp}(t = {t:5.1f} h)")
        vl_v.set_xdata([t, t])
        vl_h.set_xdata([t, t])
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"アニメーションを {out} に保存しました（{len(frames)} フレーム）", flush=True)
    return out