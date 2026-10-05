# -*- coding: utf-8 -*-
"""
kcs_total_resistance.py
========================
KCSの (1) 静水中抵抗 (Holtrop-Mennen, Calm_Water_Resistance.py)
      (2) 波浪中付加抵抗 (Added_Resistance.py)
      (3) 風圧抵抗 (wind_resistance.py)
を統合し、実海域(実船スケール・不規則波・真風→見かけ風変換)条件下での
有効馬力 P_E・軸馬力 P_S・燃料消費量を算出するモジュール。

前提として要修正・要確認の点は本ファイル末尾の "実装ノート" と、
チャット回答本文を参照してください。特に:
  - Calm_Water_Resistance.py 側は is_full_scale=True で実船用の(1+k1), Ca に
    切り替わるが、ship_params.model1_params の中身(L_wl, rho_water,
    Viscosity 等)自体が実船値になっているかは要確認。
  - Added_Resistance.py の C_aw テーブルは水槽試験(模型)データから
    導出されているが、λ/L_pp という無次元比の関数として実装されているため
    フルード相似則により模型/実船のどちらのスケールでも同じ値を返す。
    ただし補間範囲(λ/L=0.65〜1.95)外は一定値でクランプされており、
    実海域の広い波スペクトルを積分する際はこの範囲外の挙動が物理的に
    正しくない可能性が高い(要拡張)。
  - wind_resistance.py は最初から実船ジオメトリ(KCS_FULL_LOAD)を使っており
    スケールの問題はない。ただし「真風→見かけ風」の変換が抜けているため
    本ファイルで追加した。
"""

import numpy as np
from dataclasses import dataclass

from Calm_Water_Resistance import KCS as CalmWaterKCS
from Added_Resistance import KCS as AddedResistanceKCS
from wind_resistance import wind_resistance, KCS_FULL_LOAD

G = 9.80665


# ----------------------------------------------------------------------
# 1. 実海域の海象・気象条件
# ----------------------------------------------------------------------
@dataclass
class SeaState:
    Hs: float             # 有義波高 [m]
    Tp: float              # ピーク波周期 [s]
    wave_dir_rel_deg: float   # 波向き(船首基準の相対角)[deg] 0=向波(船首から)
    wind_speed_true: float    # 真風速(海面上10m基準)[m/s]
    wind_dir_rel_deg: float   # 真風向(船首基準の相対角)[deg] 0=向かい風


def bretschneider_spectrum(omega, Hs, T1):
    """
    ITTC二助変数(Bretschneider)波スペクトル S(omega) [m^2 s]
    T1: 平均波周期。ピーク周期Tpしかない場合は T1 ≈ 0.834*Tp で近似
        (JONSWAPを仮定するならこの係数を変更すること。要検証)。
    """
    omega = np.asarray(omega, dtype=float)
    omega = np.where(omega <= 1e-6, 1e-6, omega)
    A = 173.0 * Hs ** 2 / T1 ** 4
    B = 691.0 / T1 ** 4
    return A * omega ** -5 * np.exp(-B * omega ** -4)


def added_resistance_irregular(added_model, sea_state, v_ms,
                                omega_min=0.25, omega_max=1.8, n_points=200):
    """
    不規則波中の平均付加抵抗 [N]:
        R_AW_bar = 2 * ∫ [R_AW(zeta_a=1m, omega) ] * S(omega) domega
    Added_Resistance.KCS.calc_total_R_AW は zeta_a^2 に比例するので、
    振幅1mの応答を計算し、スペクトル密度を掛けて周波数積分する。
    深海波の分散関係 lambda = 2*pi*g/omega^2 で波長へ変換。

    NOTE: 積分範囲(omega_min, omega_max)はC_aw補間テーブルの校正範囲
    (lambda/Lpp = 0.65〜1.95)に対応するようデフォルト値を選んでいるが、
    実海域のスペクトルはこの範囲外にもエネルギーを持つため、
    範囲外を無視すると短波・長波成分の付加抵抗を過小評価する。
    C_aw テーブルを Faltinsen短波近似などで拡張してから範囲を広げること。
    """
    omega = np.linspace(omega_min, omega_max, n_points)
    T1 = 0.834 * sea_state.Tp
    S = bretschneider_spectrum(omega, sea_state.Hs, T1)

    wave_length = 2 * np.pi * G / omega ** 2
    Rw_unit = np.array([
        added_model.calc_total_R_AW(2.0, wl, sea_state.wave_dir_rel_deg, v_ms)
        for wl in wave_length
    ])  # wave_height=2.0 -> zeta_a=1m の応答

    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(2.0 * trapz_fn(Rw_unit * S, omega))


# ----------------------------------------------------------------------
# 2. 真風 -> 見かけ風(相対風)変換 (ITTC 7.5-04-01-01.2 準拠)
# ----------------------------------------------------------------------
def true_to_apparent_wind(v_ms, wind_speed_true, wind_dir_true_rel_deg):
    """
    U_A*cos(psi_A) = V_s + U_T*cos(psi_T)
    U_A*sin(psi_A) = U_T*sin(psi_T)
    戻り値: (見かけ風速 U_A [m/s], 見かけ風向 psi_A [deg, 0-180])
    """
    psi_T = np.deg2rad(wind_dir_true_rel_deg)
    Ux = v_ms + wind_speed_true * np.cos(psi_T)
    Uy = wind_speed_true * np.sin(psi_T)
    U_A = float(np.hypot(Ux, Uy))
    psi_A = np.degrees(np.arctan2(Uy, Ux)) % 360.0
    if psi_A > 180.0:
        psi_A = 360.0 - psi_A
    return U_A, float(psi_A)


# ----------------------------------------------------------------------
# 3. 全抵抗成分の統合
# ----------------------------------------------------------------------
def total_resistance(v_ms, sea_state: SeaState,
                      calm_model: CalmWaterKCS, added_model: AddedResistanceKCS,
                      wind_params=None):
    """3成分を積み上げ、内訳と合計 [N] を辞書で返す"""
    wind_params = wind_params or KCS_FULL_LOAD

    R_calm = float(calm_model.calc_calm_water_resistance(v_ms))
    R_aw = added_resistance_irregular(added_model, sea_state, v_ms)

    U_A, psi_A = true_to_apparent_wind(v_ms, sea_state.wind_speed_true,
                                        sea_state.wind_dir_rel_deg)
    R_aa, _ = wind_resistance(psi_A, U_A, wind_params)
    R_aa = float(np.atleast_1d(R_aa)[0])

    R_total = R_calm + R_aw + R_aa
    return {
        "V_s [m/s]": v_ms, "R_calm [N]": R_calm, "R_AW [N]": R_aw,
        "R_AA [N]": R_aa, "U_A [m/s]": U_A, "psi_A [deg]": psi_A,
        "R_total [N]": R_total,
    }


# ----------------------------------------------------------------------
# 4. 馬力・燃料消費量への変換
# ----------------------------------------------------------------------
@dataclass
class PropulsionModel:
    """
    プロペラ・軸系の効率と機関特性。
    ここに置いた数値は暫定デフォルト(典型的なコンテナ船オーダー)であり、
    実際は Holtrop-Mennen の伴流率(w)・推力減少率(t)の回帰式や
    プロペラ単独性能曲線、主機の燃料消費率(SFOC)マップに置き換える必要がある。
    """
    eta_D: float = 0.70    # 推進性能係数 (eta_H * eta_O * eta_R)
    eta_S: float = 0.98    # 軸系伝達効率
    sfoc_g_per_kwh: float = 175.0  # 比燃料消費率 [g/kWh] (負荷帯で概ね一定と仮定)
    fuel_density_kg_per_l: float = 0.98   # 燃料密度[kg/L] (HFO目安。MGOなら0.85程度に変更)
    rpm_ref: float = 90.0                  # 基準回転数[rpm] (定格MCR点や海上試運転点)
    P_D_ref_kw: float = 25000.0            # 上のrpm_refに対応する送達馬力[kW]

    def power_and_fuel(self, R_total_N, v_ms):
        P_E = R_total_N * v_ms                     # 有効馬力 [W]
        P_D = P_E / self.eta_D                      # 推進馬力(プロペラ供給馬力)[W]
        P_S = P_D / self.eta_S                       # 軸馬力(主機出力=制動馬力P_B相当)[W]

        fuel_rate_ton_per_day = (P_S / 1000.0) * self.sfoc_g_per_kwh * 24.0 / 1.0e6
        fuel_rate_ton_per_h = fuel_rate_ton_per_day / 24.0
        fuel_rate_kg_per_h = fuel_rate_ton_per_h * 1000.0
        fuel_rate_l_per_min = fuel_rate_kg_per_h / self.fuel_density_kg_per_l / 60.0

        # プロペラ回転数の簡易推定(3乗則): N/N_ref = (P_D/P_D_ref)^(1/3)
        # 一定の前進係数J付近を仮定した近似。プロペラ単独性能曲線(Kt-Kq-J)が
        # あるならそちらに置き換えた方が精度が上がる。
        rpm = self.rpm_ref * (P_D / (self.P_D_ref_kw * 1000.0)) ** (1.0 / 3.0)

        return {
            "P_E [kW]": P_E / 1000.0, "P_D [kW]": P_D / 1000.0,
            "P_S [kW]": P_S / 1000.0,
            "Fuel [ton/day]": fuel_rate_ton_per_day,
            "Fuel [ton/h]": fuel_rate_ton_per_h,
            "Fuel [kg/h]": fuel_rate_kg_per_h,
            "Fuel [L/min]": fuel_rate_l_per_min,
            "RPM": rpm,
        }


# ----------------------------------------------------------------------
# 5. 使用例 (3時間ごとの航海時系列)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    calm_model = CalmWaterKCS(is_full_scale=True)
    added_model = AddedResistanceKCS(is_full_scale=True)
    propulsion = PropulsionModel()

    # 3時間刻みの気象海象・船速の想定時系列。
    # 実運用ではERA5等から作成したCSV/DataFrameを読み込んでこのリストに
    # 詰め替えるだけでよい(各要素の並びは下のfor文と対応させること)。
    # (時刻[h], 船速[kt], Hs[m], Tp[s], 波向き[deg], 真風速[m/s], 真風向[deg])
    forecast = [
        (0, 20.0, 2.0, 8.0, 20.0, 8.0, 20.0),
        (3, 20.0, 2.5, 8.5, 25.0, 10.0, 25.0),
        (6, 20.0, 3.0, 9.0, 30.0, 12.0, 30.0),
        (9, 20.0, 3.2, 9.2, 35.0, 13.0, 35.0),
    ]

    dt_h = 3.0
    cumulative_foc_ton = 0.0
    results = []

    for t_h, v_kt, Hs, Tp, wave_dir, wind_speed, wind_dir in forecast:
        v_ms = v_kt * 0.51444
        sea = SeaState(Hs=Hs, Tp=Tp, wave_dir_rel_deg=wave_dir,
                       wind_speed_true=wind_speed, wind_dir_rel_deg=wind_dir)

        r = total_resistance(v_ms, sea, calm_model, added_model)
        p = propulsion.power_and_fuel(r["R_total [N]"], v_ms)

        # この3時間区間で消費した燃料を積算(区間先頭の瞬時消費率で代表させる近似)
        cumulative_foc_ton += p["Fuel [ton/h]"] * dt_h

        results.append({
            "time [h]": t_h,
            "R_total [kN]": r["R_total [N]"] / 1e3,
            "P_B [kW]": p["P_S [kW]"],
            "RPM": p["RPM"],
            "Fuel rate [ton/h]": p["Fuel [ton/h]"],
            "Fuel rate [kg/h]": p["Fuel [kg/h]"],
            "Fuel rate [L/min]": p["Fuel [L/min]"],
            "Cumulative FOC [ton]": cumulative_foc_ton,
        })

    print(f"{'t[h]':>5}{'R_tot[kN]':>12}{'P_B[kW]':>10}{'RPM':>8}"
          f"{'Fuel[t/h]':>11}{'Cum.FOC[t]':>12}")
    for row in results:
        print(f"{row['time [h]']:>5.0f}{row['R_total [kN]']:>12.1f}"
              f"{row['P_B [kW]']:>10.0f}{row['RPM']:>8.1f}"
              f"{row['Fuel rate [ton/h]']:>11.3f}{row['Cumulative FOC [ton]']:>12.2f}")

    # スライド用の推移グラフを作りやすいようCSVにも出力しておく
    import csv
    with open("voyage_timeseries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# ----------------------------------------------------------------------
# 実装ノート (要対応・要確認事項のサマリ)
# ----------------------------------------------------------------------
# 1. ship_params.py の model1_params / model3_params が実船スケールの値
#    (L=230m級, rho_water=1025, seawater Viscosity)になっているか要確認。
#    水槽試験用の模型スケール値のままだと is_full_scale=True にしても
#    Re や nabla が模型のままで実船抵抗として使えない。
# 2. Added_Resistance の C_aw テーブルは補間範囲(lambda/Lpp=0.65-1.95)外で
#    定数クランプ。実海域スペクトル積分では短波(反射優位)・長波域を
#    物理式(例: Faltinsen 1980短波近似、STAwave-1)で外挿するか、
#    積分範囲をこの校正範囲内に絞ること。
# 3. PropulsionModel の eta_D, eta_S, sfoc は暫定値。実際は
#    (a) Holtrop-Mennenの伴流率・推力減少率回帰式 or 自航試験結果、
#    (b) プロペラ単独性能(Kt-Kq-J)曲線、
#    (c) 主機の負荷率別SFOCマップ
#    に置き換えることを推奨。
# 4. 波向き・風向・船首方位の角度基準(0度の定義、右回り/左回り)を
#    三モジュール間で揃えること。本ファイルは全て「船首を0度、そこからの
#    相対角」で統一しているが、実海域データ(ERA5等)は北を0度とした
#    絶対方位で来るはずなので、真方位 -> 相対角の変換(ship heading控除)
#    を別途追加する必要がある。
# 5. added_resistance_irregular は正面付加抵抗(平均増加抵抗)のみを返す。
#    実際の運航では出会周期 T_e のシフト(遭遇周波数)による共振帯の
#    シフトも考慮したい場合は、応答計算時に omega ではなく omega_e で
#    重み付けするなど拡張が必要(現状は深海波・正面近似で簡略化)。