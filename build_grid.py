"""
build_grid.py
=============
大圏航路（基準線）直交グリッド生成法の実装（ステージごとにノード数が変わる版）。

Step 1: 出発地〜目的地の大圏航路（基準線）を生成
Step 2: 基準線を進行方向に ΔX 間隔でステージ分割
Step 3: 各ステージ基準点で進行方向に直角な方向へ ΔY 間隔でノードを展開
        （幅は到達可能性から決め、両端は狭く中間が広い紡錘形にする）
Step 4: 陸地と重なるノード／横切るエッジを除外

座標はすべて (lat, lon) の順（度単位、WGS84）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyproj import Geod
from land_mask import globe

_GEOD = Geod(ellps="WGS84")
NM_TO_M = 1852.0


@dataclass
class Node:
    stage: int
    index: int          # 中央=0、進行方向右が正、左が負
    lat: float
    lon: float
    is_ocean: bool = True


@dataclass
class RoutingGrid:
    start: tuple[float, float]
    end: tuple[float, float]
    num_stages: int
    half_widths: list[int]   # ステージごとの片側ノード数（ノード数 = 2*half + 1）
    dy_nm: float
    stages: list[list[Node]] = field(default_factory=list)
    total_distance_nm: float = 0.0
    stage_distance_nm: float = 0.0

    @property
    def nodes_per_stage(self) -> list[int]:
        return [2 * h + 1 for h in self.half_widths]


def generate_stage_baseline(start, end, num_stages):
    """大圏航路上に距離等間隔なステージ基準点を生成する。"""
    lat1, lon1 = start
    lat2, lon2 = end
    if num_stages < 1:
        raise ValueError("num_stages は 1 以上にしてください")

    _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    total_distance_nm = dist_m / NM_TO_M

    if num_stages == 1:
        interior = []
    else:
        interior = _GEOD.npts(lon1, lat1, lon2, lat2, num_stages - 1)
        interior = [(lat, lon) for lon, lat in interior]

    return [start] + interior + [end], total_distance_nm


def _stage_headings(stage_points):
    """各ステージ基準点の進行方位角（deg）。内部点は前後区間の円環平均。"""
    n = len(stage_points)
    fwd_az = []
    for i in range(n - 1):
        lat1, lon1 = stage_points[i]
        lat2, lon2 = stage_points[i + 1]
        az12, _, _ = _GEOD.inv(lon1, lat1, lon2, lat2)
        fwd_az.append(az12 % 360.0)

    headings = []
    for i in range(n):
        if i == 0:
            headings.append(fwd_az[0])
        elif i == n - 1:
            headings.append(fwd_az[-1])
        else:
            a1, a2 = math.radians(fwd_az[i - 1]), math.radians(fwd_az[i])
            x = math.cos(a1) + math.cos(a2)
            y = math.sin(a1) + math.sin(a2)
            headings.append(math.degrees(math.atan2(y, x)) % 360.0)
    return headings


def compute_half_widths(num_stages, stage_dist_nm, dy_nm,
                        theta_max_deg=25.0, half_max=8):
    """各ステージの片側ノード数（長さ num_stages + 1）。

    出発地から i ステージ進んで基準線から離れられる横距離は
    i * ΔX * sin(theta_max)、目的地へ戻れる範囲は (N - i) * ΔX * sin(theta_max)。
    小さい方を dy で割り、half_max で頭打ちにする。両端は 0（ノード1個）になる。
    """
    if dy_nm <= 0:
        raise ValueError("dy_nm は正の値にしてください")
    sin_t = math.sin(math.radians(theta_max_deg))
    widths = []
    for i in range(num_stages + 1):
        reach_nm = min(i, num_stages - i) * stage_dist_nm * sin_t
        widths.append(min(int(reach_nm // dy_nm), half_max))
    return widths


def generate_perpendicular_nodes(stage_points, half_widths, dy_nm):
    """各基準点から進行方向に直角な左右へノードを展開する。"""
    if len(half_widths) != len(stage_points):
        raise ValueError("half_widths の長さは stage_points と一致させてください")

    headings = _stage_headings(stage_points)
    stages = []
    for s_idx, ((lat, lon), heading) in enumerate(zip(stage_points, headings)):
        row = []
        half = half_widths[s_idx]
        for k in range(-half, half + 1):
            if k == 0:
                nlat, nlon = lat, lon
            else:
                az = (heading + 90.0) if k > 0 else (heading - 90.0)
                nlon, nlat, _ = _GEOD.fwd(lon, lat, az, abs(k) * dy_nm * NM_TO_M)
            row.append(Node(stage=s_idx, index=k, lat=nlat, lon=nlon))
        stages.append(row)
    return stages


def filter_land_nodes(stages):
    """各ノードが海上か判定し Node.is_ocean を更新する（in-place）。"""
    for row in stages:
        for node in row:
            lon_norm = ((node.lon + 180.0) % 360.0) - 180.0
            node.is_ocean = bool(globe.is_ocean(node.lat, lon_norm))


def _samples_on_geodesic(lat1, lon1, lat2, lon2, n_samples=6):
    pts = _GEOD.npts(lon1, lat1, lon2, lat2, n_samples)
    return [(lat, lon) for lon, lat in pts]


def edge_crosses_land(node_a, node_b, n_samples=6):
    """2ノードを結ぶ大圏区間の途中に陸地があるか判定する。"""
    for lat, lon in _samples_on_geodesic(node_a.lat, node_a.lon,
                                         node_b.lat, node_b.lon, n_samples):
        lon_norm = ((lon + 180.0) % 360.0) - 180.0
        if globe.is_land(lat, lon_norm):
            return True
    return False


def build_routing_grid(start, end, num_stages=20, dy_nm=30.0,
                       theta_max_deg=25.0, half_max=16, half_widths=None):
    """Step1〜4を通しで実行し、陸地フィルタ済みの RoutingGrid を返す。

    half_widths（長さ num_stages + 1）を渡せば幅を手動指定できる。
    省略時は compute_half_widths で自動決定する。
    """
    stage_points, total_distance_nm = generate_stage_baseline(start, end, num_stages)
    stage_distance_nm = total_distance_nm / num_stages

    if half_widths is None:
        half_widths = compute_half_widths(
            num_stages, stage_distance_nm, dy_nm,
            theta_max_deg=theta_max_deg, half_max=half_max,
        )

    stages = generate_perpendicular_nodes(stage_points, half_widths, dy_nm)
    filter_land_nodes(stages)

    # 出発地・目的地（港湾）は陸地判定されることがあるため常に有効にする
    for row in (stages[0], stages[-1]):
        for node in row:
            if node.index == 0:
                node.is_ocean = True

    return RoutingGrid(
        start=start, end=end, num_stages=num_stages,
        half_widths=half_widths, dy_nm=dy_nm, stages=stages,
        total_distance_nm=total_distance_nm,
        stage_distance_nm=stage_distance_nm,
    )


if __name__ == "__main__":
    grid = build_routing_grid(
        start=(49.0, -10.0), end=(46.0, -55.0),
        num_stages=18, dy_nm=60.0, theta_max_deg=25.0, half_max=8,
    )
    print(f"大圏距離: {grid.total_distance_nm:.1f} nm, ΔX: {grid.stage_distance_nm:.1f} nm")
    print("ステージごとのノード数:", grid.nodes_per_stage)