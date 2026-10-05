# -*- coding: utf-8 -*-
"""
dijkstra_3d.py
==============
Wang et al. (2019) "A Three-Dimensional Dijkstra's algorithm for multi-objective
ship voyage optimization" の 3D Dijkstra 法の実装。

  ノード : 船舶状態 P_{i,j,k} = (ステージ i, 横位置 j, 通過時刻 k)   ... 論文 Eq.(8)
  エッジ : 隣接ステージのノード間。航海時間 Δt から速度が決まり、
           速度範囲 [0.5 Vf, 1.2 Vf] を満たすものだけ張る               ... Sec.2.2 step 2,4
  コスト : 燃料消費量 = f(U, W) * Δt (天候は各エッジの始点ノード・時刻で取得)  ... Eq.(6),(9)
  探索   : 出発地から Dijkstra。到着ステージの各時刻 k が「ETA ごとの最小燃料」に対応 ... Eq.(10),(11)

論文はグラフを先に全生成してから探索しますが、ここでは
  (1) 空間エッジ（陸地チェック・距離・方位）だけ先に作る
  (2) 時間方向の展開とコスト計算は探索中に必要な分だけ行う（遅延生成）
としています。結果（ETA ごとの最小燃料経路）は同じで、メモリと計算量を抑えられます。
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
import time
from typing import Callable, Optional

import numpy as np

from build_grid import (
    RoutingGrid, Node, build_routing_grid, edge_crosses_land, _GEOD, NM_TO_M,
)
from Fuel_consumption import (
    SeaState, PropulsionModel, added_resistance_irregular, true_to_apparent_wind,
)
from Calm_Water_Resistance import KCS as CalmWaterKCS
from Added_Resistance import KCS as AddedResistanceKCS
from wind_resistance import wind_resistance, KCS_FULL_LOAD

KT_TO_MS = 0.51444


# ======================================================================
# 1. 気象データのインターフェース
# ======================================================================
@dataclass
class WeatherPoint:
    """ある位置・時刻の海象・気象（すべて「絶対方位・来る方向」基準）"""
    Hs: float               # 有義波高 [m]
    Tp: float               # ピーク波周期 [s]
    wave_from_deg: float    # 波が来る方向（北=0, 時計回り）[deg]
    wind_speed: float       # 真風速 [m/s]
    wind_from_deg: float    # 風が吹いてくる方向（北=0, 時計回り）[deg]


# weather(lat, lon, t_h) -> WeatherPoint  の形の関数を渡す。
# 実データ（ERA5 等）を使う場合は、この形の関数を自作して差し替える。
WeatherFn = Callable[[float, float, float], WeatherPoint]


class SyntheticStorm:
    """動作確認用の人工的な嵐（東へ移動するガウス型の高波域）。実データに差し替えて使う。"""

    def __init__(self, center=(48.0, -30.0), radius_deg=7.0, hs_base=1.5,
                 hs_peak=7.0, drift_deg_per_h=0.15):
        self.center, self.radius = center, radius_deg
        self.hs_base, self.hs_peak, self.drift = hs_base, hs_peak, drift_deg_per_h

    def __call__(self, lat, lon, t_h):
        clat, clon = self.center[0], self.center[1] + self.drift * t_h
        d = math.hypot(lat - clat, (lon - clon) * math.cos(math.radians(lat)))
        hs = self.hs_base + (self.hs_peak - self.hs_base) * math.exp(-(d / self.radius) ** 2)
        return WeatherPoint(Hs=hs, Tp=6.0 + 0.8 * hs, wave_from_deg=270.0,
                            wind_speed=5.0 + 2.2 * hs, wind_from_deg=270.0)


def relative_angle(from_deg: float, heading_deg: float) -> float:
    """「来る方向(絶対方位)」を船首基準の相対角 0〜180° に変換（0=船首方向から=向波/向かい風）。
    左右対称と仮定して折り返す。"""
    return abs(((from_deg - heading_deg + 180.0) % 360.0) - 180.0)


# ======================================================================
# 2. 燃料モデル（Fuel_consumption.py のラッパ＋キャッシュ）
# ======================================================================
class FuelModel:
    """速度・相対海象 -> 燃料消費率 [ton/day]。

    不規則波の付加抵抗積分が重いため、入力を量子化して結果をキャッシュする。
    量子化幅: 速度0.1kt, Hs 0.5m, Tp 1s, 角度15°, 風速1m/s
    """

    def __init__(self, calm_model, added_model, propulsion: PropulsionModel,
                 wind_params=None, max_power_kw: Optional[float] = None, n_omega: int = 80):
        self.calm, self.added, self.prop = calm_model, added_model, propulsion
        self.wind_params = wind_params or KCS_FULL_LOAD
        self.max_power_kw = max_power_kw   # 主機の最大軸出力（超過する速度は不可）
        self.n_omega = n_omega
        self._calm_cache: dict = {}
        self._rate_cache: dict = {}

    def _r_calm(self, vkey: int, v_ms: float) -> float:
        if vkey not in self._calm_cache:
            self._calm_cache[vkey] = float(self.calm.calc_calm_water_resistance(v_ms))
        return self._calm_cache[vkey]

    def rate(self, v_kt: float, sea: SeaState):
        """戻り値: (燃料 [ton/day], 軸馬力 [kW])。出力上限超過なら None。"""
        key = (round(v_kt * 10), round(sea.Hs * 10), round(sea.Tp),
               round(sea.wave_dir_rel_deg / 5), round(sea.wind_speed_true),
               round(sea.wind_dir_rel_deg / 5))       
        if key in self._rate_cache:
            return self._rate_cache[key]

        v_ms = (key[0] / 10.0) * KT_TO_MS
        q = SeaState(Hs=key[1] / 10.0, Tp=max(float(key[2]), 1.0),
                     wave_dir_rel_deg=key[3] * 5.0,
                     wind_speed_true=float(key[4]), wind_dir_rel_deg=key[5] * 5.0)        
        r_calm = self._r_calm(key[0], v_ms)
        r_aw = (added_resistance_irregular(self.added, q, v_ms, n_points=self.n_omega)
                if q.Hs >= 0.25 else 0.0)
        u_a, psi_a = true_to_apparent_wind(v_ms, q.wind_speed_true, q.wind_dir_rel_deg)
        r_aa, _ = wind_resistance(psi_a, u_a, self.wind_params)
        r_aa = float(np.atleast_1d(r_aa)[0])

        r_total = max(r_calm + r_aw + r_aa, 0.0)
        out = self.prop.power_and_fuel(r_total, v_ms)
        p_s = out["P_S [kW]"]
        result = None if (self.max_power_kw and p_s > self.max_power_kw) \
            else (out["Fuel [ton/day]"], p_s)
        self._rate_cache[key] = result
        return result


# ======================================================================
# 3. 空間エッジ（陸地チェック済み）
# ======================================================================
@dataclass
class SpatialEdge:
    a: Node
    b: Node
    dist_nm: float
    heading_deg: float


def build_spatial_edges(grid: RoutingGrid, max_index_diff: int = 2, n_samples: int = 6):
    """隣接ステージの海上ノード同士を結ぶ。

    max_index_diff : 横方向の index 差の上限（論文の「大きな針路変更は不可」に相当）
    陸地を横切る辺は除外（論文の制約「land crossing is not allowed」）。
    戻り値: {(stage, index): [SpatialEdge, ...]}
    """
    valid = [[n for n in row if n.is_ocean] for row in grid.stages]
    edges: dict = {}
    for s in range(grid.num_stages):
        for a in valid[s]:
            lst = []
            for b in valid[s + 1]:
                if abs(a.index - b.index) > max_index_diff:
                    continue
                if edge_crosses_land(a, b, n_samples):
                    continue
                az, _, d_m = _GEOD.inv(a.lon, a.lat, b.lon, b.lat)
                lst.append(SpatialEdge(a, b, d_m / NM_TO_M, az % 360.0))
            edges[(s, a.index)] = lst
    return edges


# ======================================================================
# 4. エッジコスト（差し替え可能）
# ======================================================================
@dataclass
class EdgeContext:
    edge: SpatialEdge
    t_h: float               # エッジ始点の時刻 [h]
    weather: WeatherPoint    # 始点ノード・始点時刻の気象
    sea: SeaState            # 上記を船首基準の相対角に変換したもの


def make_fuel_cost(fuel_model: FuelModel):
    """コスト = 燃料消費率 * 航海時間 （論文 Eq.(6),(9)）。不可なら None。"""
    def cost(ctx: EdgeContext, v_kt: float, dur_h: float):
        r = fuel_model.rate(v_kt, ctx.sea)
        return None if r is None else r[0] * dur_h / 24.0
    return cost

# 疲労損傷など別の目的関数を使う場合も、同じ (ctx, v_kt, dur_h) -> float の形で書けばよい。


# ======================================================================
# 5. 3D Dijkstra
# ======================================================================
@dataclass
class Leg:
    stage: int
    index: int
    lat: float
    lon: float
    t_h: float
    speed_next_kt: Optional[float]   # 次のステージへの速度（最終点は None）


class DijkstraResult:
    def __init__(self, grid, dt_h, dist, prev):
        self.grid, self.dt_h, self.dist, self.prev = grid, dt_h, dist, prev
        self._node = {(n.stage, n.index): n for row in grid.stages for n in row}
        self.final_stage = grid.num_stages

    def final_states(self):
        return sorted(st for st in self.dist if st[0] == self.final_stage)

    def eta_fuel(self):
        """到着可能な ETA [h] とその最小燃料 [ton] の一覧（論文 Fig.14 左に相当）"""
        return [(st[2] * self.dt_h, self.dist[st]) for st in self.final_states()]

    def pareto_front(self):
        """ETA と燃料の非劣解のみ（早くて燃料も少ない点が残る）"""
        front, best = [], math.inf
        for eta, fuel in sorted(self.eta_fuel()):
            if fuel < best - 1e-9:
                front.append((eta, fuel))
                best = fuel
        return front

    def route(self, eta_h: Optional[float] = None):
        """指定 ETA に最も近い到着時刻の最小燃料経路。eta_h=None なら全体で最小燃料の経路。"""
        finals = self.final_states()
        if not finals:
            raise RuntimeError("目的地に到達できるノードがありません（制約が厳しすぎる可能性）")
        if eta_h is None:
            st = min(finals, key=lambda s: self.dist[s])
        else:
            st = min(finals, key=lambda s: abs(s[2] * self.dt_h - eta_h))

        legs: list[Leg] = []
        speed_next = None
        while True:
            n = self._node[(st[0], st[1])]
            legs.append(Leg(st[0], st[1], n.lat, n.lon, st[2] * self.dt_h, speed_next))
            if st not in self.prev:
                break
            st, v_kt, _ = self.prev[st]
            speed_next = v_kt
        legs.reverse()
        return {"eta_h": legs[-1].t_h, "fuel_ton": self.dist[
            (self.final_stage, 0, int(round(legs[-1].t_h / self.dt_h)))], "legs": legs}


def solve_3d_dijkstra(
    grid: RoutingGrid,
    spatial_edges: dict,
    weather: WeatherFn,
    cost_fn,
    service_speed_kt: float,
    dt_h: float = 0.5,
    speed_ratio: tuple[float, float] = (0.5, 1.2),
    departure_h: float = 0.0,
    max_eta_h: Optional[float] = None,
    hs_limit: Optional[float] = None,
    progress: bool = True,
    progress_interval_s: float = 10.0,
) -> DijkstraResult:
    """
    service_speed_kt : 基準速度 Vf。速度範囲は [0.5 Vf, 1.2 Vf]（論文 Sec.2.2）
    dt_h             : 時間刻み Δt（論文は 12〜30 分）。小さいほど精密だが重い。
    max_eta_h        : 出発からの最大許容航海時間。指定すると間に合わない状態を枝刈りでき、大幅に速くなる。
    hs_limit         : この有義波高を超える点から出るエッジは不可（論文の荒天制約）。
    """
    N = grid.num_stages
    v_min, v_max = speed_ratio[0] * service_speed_kt, speed_ratio[1] * service_speed_kt
    rem_nm = [(N - s) * grid.stage_distance_nm for s in range(N + 1)]  # 残距離の下界

    counter = itertools.count()
    start_state = (0, 0, 0)
    dist = {start_state: 0.0}
    prev: dict = {}
    heap = [(0.0, next(counter), start_state)]

    t_start = time.time()
    t_last = t_start
    n_pop = 0
    max_stage = 0

    while heap:
        c, _, st = heapq.heappop(heap)
        if c > dist.get(st, math.inf):
            continue
        s, idx, m = st
        n_pop += 1
        if s > max_stage:
            max_stage = s
        if progress and time.time() - t_last >= progress_interval_s:
            t_last = time.time()
            print(f"  [3DDA] 最大到達ステージ {max_stage}/{N}  処理済み {n_pop:,}  "
                  f"ラベル数 {len(dist):,}  キュー {len(heap):,}  "
                  f"経過 {t_last - t_start:.0f}s", flush=True)
        if s == N:
            continue

        t_h = m * dt_h
        for e in spatial_edges.get((s, idx), []):
            w = weather(e.a.lat, e.a.lon, departure_h + t_h)
            if hs_limit is not None and w.Hs > hs_limit:
                continue
            sea = SeaState(
                Hs=w.Hs, Tp=w.Tp,
                wave_dir_rel_deg=relative_angle(w.wave_from_deg, e.heading_deg),
                wind_speed_true=w.wind_speed,
                wind_dir_rel_deg=relative_angle(w.wind_from_deg, e.heading_deg),
            )
            ctx = EdgeContext(e, departure_h + t_h, w, sea)

            # 航海時間を Δt の整数倍 n*Δt とし、速度が範囲内になる n だけ列挙
            n_min = math.ceil(e.dist_nm / (v_max * dt_h) - 1e-9)
            n_max = math.floor(e.dist_nm / (v_min * dt_h) + 1e-9)
            for n in range(max(n_min, 1), n_max + 1):
                dur_h = n * dt_h
                t_next = t_h + dur_h
                if max_eta_h is not None and t_next + rem_nm[s + 1] / v_max > max_eta_h:
                    continue
                v_kt = e.dist_nm / dur_h
                cost = cost_fn(ctx, v_kt, dur_h)
                if cost is None:
                    continue
                nst = (s + 1, e.b.index, m + n)
                nc = c + cost
                if nc < dist.get(nst, math.inf):
                    dist[nst] = nc
                    prev[nst] = (st, v_kt, cost)
                    heapq.heappush(heap, (nc, next(counter), nst))

    if progress:
        print(f"  [3DDA] 完了: 処理済み {n_pop:,}  ラベル数 {len(dist):,}  "
              f"{time.time() - t_start:.0f}s", flush=True)
    return DijkstraResult(grid, dt_h, dist, prev)

# ======================================================================
# 6. 使用例
# ======================================================================
if __name__ == "__main__":
    # --- 探索空間 ---
    grid = build_routing_grid(
        start=(49.0, -10.0), end=(46.0, -55.0),
        num_stages=18, dy_nm=60.0, theta_max_deg=25.0, half_max=8,
    )
    edges = build_spatial_edges(grid, max_index_diff=2)

    # --- 性能モデル・気象 ---
    fuel_model = FuelModel(
        CalmWaterKCS(is_full_scale=True), AddedResistanceKCS(is_full_scale=True),
        PropulsionModel(), max_power_kw=None,   # 主機の最大出力があれば kW で指定
    )
    weather = SyntheticStorm()      # 実データを使う場合は WeatherFn を自作して差し替え

    # --- 探索 ---
    Vf = 18.0                                   # 基準速度 [kt]
    t_ref = grid.total_distance_nm / Vf         # 基準速度で大圏航行した場合の所要時間 [h]
    result = solve_3d_dijkstra(
        grid, edges, weather, make_fuel_cost(fuel_model),
        service_speed_kt=Vf, dt_h=0.5,
        max_eta_h=t_ref * 1.15,                 # 枝刈り。外すと非常に重くなる
        hs_limit=10.0,
    )

    # --- ETA と燃料のトレードオフ ---
    print("ETA[h]  Fuel[ton]  (非劣解)")
    for eta, fuel in result.pareto_front()[::5]:
        print(f"{eta:7.1f} {fuel:9.1f}")

    # --- 指定 ETA の最適経路 ---
    route = result.route(eta_h=t_ref * 1.05)
    print(f"\nETA {route['eta_h']:.1f} h, 燃料 {route['fuel_ton']:.1f} ton")
    print("stage idx    lat     lon    t[h]  speed[kt]")
    for L in route["legs"]:
        sp = "" if L.speed_next_kt is None else f"{L.speed_next_kt:6.1f}"
        print(f"{L.stage:5d} {L.index:3d} {L.lat:7.2f} {L.lon:8.2f} {L.t_h:7.1f} {sp:>8}")

    # --- 経路を地図に重ねる（plot_grid.py を使用）---
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from plot_grid import plot_grid
        ax = plot_grid(grid)
        ax.plot([L.lon for L in route["legs"]], [L.lat for L in route["legs"]],
                "m-o", lw=2, ms=4, transform=ccrs.PlateCarree(), zorder=7, label="optimal route")
        ax.legend(loc="lower left")
        plt.savefig("optimal_route.png", dpi=150)
        plt.show()
    except ImportError:
        pass