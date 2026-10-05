# plot_grid.py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from build_grid import build_routing_grid, edge_crosses_land

PC = ccrs.PlateCarree()  # データ座標（lon, lat）の座標系


def _unwrap(lon, ref):
    """日付変更線をまたぐ場合に備え、ref 経度の近くに連続化する。"""
    return ref + ((lon - ref + 180.0) % 360.0) - 180.0


def plot_grid(grid, draw_edges=False, max_index_diff=1,
              margin_deg=3.0, resolution="50m", ax=None):
    """resolution: Natural Earth の解像度 '110m' / '50m' / '10m'（細かいほど重い）"""
    ref = grid.start[1]
    all_nodes = [n for row in grid.stages for n in row]
    lats = np.array([n.lat for n in all_nodes])
    lons = np.array([_unwrap(n.lon, ref) for n in all_nodes])

    lat_min, lat_max = max(lats.min() - margin_deg, -85), min(lats.max() + margin_deg, 85)
    lon_min, lon_max = lons.min() - margin_deg, lons.max() + margin_deg
    mid_lon = (lon_min + lon_max) / 2

    if ax is None:
        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude=mid_lon))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=PC)

    # --- 背景地図 ---
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor="#dcecf7", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(resolution),
                   facecolor="#e6dfd0", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale(resolution),
                   facecolor="#dcecf7", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution),
                   edgecolor="#555555", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution),
                   edgecolor="#aaaaaa", linewidth=0.4, linestyle="--", zorder=2)

    gl = ax.gridlines(crs=PC, draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False

    # --- エッジ ---
    if draw_edges:
        for s in range(grid.num_stages):
            for a in grid.stages[s]:
                if not a.is_ocean:
                    continue
                for b in grid.stages[s + 1]:
                    if not b.is_ocean or abs(a.index - b.index) > max_index_diff:
                        continue
                    if edge_crosses_land(a, b):
                        continue
                    ax.plot([_unwrap(a.lon, ref), _unwrap(b.lon, ref)],
                            [a.lat, b.lat], transform=PC,
                            color="tab:blue", lw=0.4, alpha=0.5, zorder=3)

    # --- 基準線（大圏航路）: index 0 のノードを結ぶ ---
    center = [n for row in grid.stages for n in row if n.index == 0]
    ax.plot([_unwrap(n.lon, ref) for n in center], [n.lat for n in center],
            "r-", lw=1.5, transform=PC, label="great circle (baseline)", zorder=4)

    # --- ノード ---
    ocean = [n for n in all_nodes if n.is_ocean]
    bad = [n for n in all_nodes if not n.is_ocean]
    ax.scatter([_unwrap(n.lon, ref) for n in ocean], [n.lat for n in ocean],
               s=14, c="tab:blue", transform=PC, label="ocean node", zorder=5)
    if bad:
        ax.scatter([_unwrap(n.lon, ref) for n in bad], [n.lat for n in bad],
                   s=24, c="tab:red", marker="x", transform=PC,
                   label="land node (excluded)", zorder=5)

    # --- 出発地・目的地 ---
    ax.scatter(_unwrap(grid.start[1], ref), grid.start[0], s=90, c="lime",
               edgecolors="k", transform=PC, label="start", zorder=6)
    ax.scatter(_unwrap(grid.end[1], ref), grid.end[0], s=90, c="gold",
               edgecolors="k", transform=PC, label="end", zorder=6)

    ax.set_title(f"Routing grid ({grid.num_stages} stages, dy={grid.dy_nm:g} nm)")
    ax.legend(loc="lower left")
    return ax


if __name__ == "__main__":
    grid = build_routing_grid(
        start=(47.5, -10.0), end=(46.0, -55.0),
        num_stages=18, dy_nm=60.0, theta_max_deg=25.0, half_max=8,
    )
    plot_grid(grid, draw_edges=False)
    plt.tight_layout()
    plt.savefig("routing_grid.png", dpi=150)
    plt.show()