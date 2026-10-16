# -*- coding: utf-8 -*-
"""
KCS(満載状態, 3600TEU, container_ship.py v3ジオメトリ準拠)の風圧抵抗計算モデル
================================================================================

参考文献:
  Fujiwara, T., Ueno, M., Ikeda, Y. (2005).
  "成分分離型モデルを利用した新しい風圧力推定法"
  日本船舶海洋工学会論文集 第2号, pp.243-255.
  ("A New Estimation Method of Wind Forces and Moments acting on Ships
    on the basis of Physical Component Models")

係数について
------------
Table 3 の回帰係数を直接使用。Ro-Ro旅客船の実測データ(同論文 Table A-1,
Fig.14)で符号・オーダーを検算済み:
  CX(psi=0)   = -0.463  (逆風で負の抵抗 -> 符号OK)
  CX(psi=180) = +0.571  (追い風で正の推力 -> 符号OK)

  ※以前検討していた「ITTC 7.5-04-01-01.2」ベースの係数セットは、0-90度域
    の符号が本実装と全て反転しており(OCR誤りの疑い)、不採用とした。

8パラメータについて
--------------------
container_ship.py (v3) の compute_AL_summary() / compute_AF_components() の
計算結果をそのまま使用。KCS 3600TEU・満載・船尾コンテナ5段のジオメトリ
から直接算出した実値であり、他船型からのスケーリング値ではない。

【重要】C の符号規約の違いに注意
  container_ship.py の C は「正の値 = 重心が船尾側」と定義されている
  (= midship - x_c)。一方、Fujiwara原論文の C は「船首方向が正」
  (= x_c - midship)。このモデルでは Fujiwara の規約に統一するため、
  container_ship.py の出力 C_script をそのまま使わず、符号を反転させた
  値 (C = x_c - midship) を KCS_FULL_LOAD に格納している。

  container_ship.py 側の計算結果: x_c=106.25m (AP基準), midship=115.0m
    -> C_script (船尾側が正) = +8.75 m
    -> C_fujiwara (船首方向が正) = -8.75 m  ← 本ファイルではこちらを使用

A_OD についての注意
  Fujiwara の A_OD は「デッキ上構造物の側面投影面積」であり、ブリッジ
  だけでなく甲板上コンテナも含む。container_ship.py の凡例ラベル
  「ブリッジ A_OD: 270m²」はブリッジ単体の面積であり、Fujiwara式に渡す
  A_OD はコンテナ(船尾671+船首1476.2)+ブリッジ270 の合計 2417.2 m² が
  正しい。

単位系: SI (m, m^2, kg/m^3, m/s, N)
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Fujiwara(2005) 回帰式の係数表 (Table 3, 検算済み)
# ----------------------------------------------------------------------
_ALPHA = {0: 0.404, 1: 0.368, 2: 0.902}                              # C_CF   (eq.12)
_BETA1 = {0: -0.922, 1: 0.507, 2: 1.162}                              # C_LF1  (eq.14, 0-90deg)
_BETA2 = {0: 0.018, 1: -5.091, 2: 10.367, 3: -3.011, 4: -0.341}       # C_LF2  (eq.15, 90-180deg)
_GAMMA1 = {0: 0.116, 1: 3.345}                                        # C_YM1  (eq.17, 0-90deg)
_GAMMA2 = {0: 0.446, 1: 2.192}                                        # C_YM2  (eq.18, 90-180deg)
_DELTA1 = {0: 0.458, 1: 3.245, 2: -2.313}                             # C_XLI1 (eq.19, 0-90deg)
_DELTA2 = {0: -1.901, 1: 12.727, 2: 24.407, 3: -40.310, 4: -5.481}    # C_XLI2 (eq.20, 90-180deg)
_EPS1 = {0: -0.585, 1: -0.906, 2: 3.239}                              # C_ALF1 (eq.21, 0-90deg)
_EPS2 = {0: -0.314, 1: -1.117}                                        # C_ALF2 (eq.22, 90-180deg)


def fujiwara_cx_cy_cn(psi_deg, params):
    """
    Fujiwara(2005)の成分分離型モデルにより CX, CY, CN を計算する。

    Parameters
    ----------
    psi_deg : 相対風向角[deg] (0-180), スカラーまたは配列
    params : dict。必要キー:
        LOA [m]  : 全長
        B   [m]  : 型幅
        AF  [m2] : 正面投影面積
        AL  [m2] : 側面投影面積
        AOD [m2] : デッキ上構造物の側面投影面積(コンテナ+ブリッジ)
        C   [m]  : 側面積重心の船体中央からの前後座標(船首方向が正)
        HC  [m]  : 喫水から側面積重心までの高さ
        HBR [m]  : 喫水からブリッジ等最高構造物までの高さ

    Returns
    -------
    CX, CY, CN (psi_degと同じ形状)
    """
    psi = np.atleast_1d(np.asarray(psi_deg, dtype=float))
    psi_rad = np.deg2rad(psi)

    LOA, B, AF, AL, AOD, C, HC, HBR = (params["LOA"], params["B"], params["AF"],
                                         params["AL"], params["AOD"], params["C"],
                                         params["HC"], params["HBR"])

    C_CF = _ALPHA[0] + _ALPHA[1] * (AF / (B * HBR)) + _ALPHA[2] * (HBR / LOA)

    C_LF1 = _BETA1[0] + _BETA1[1] * (AL / (LOA * B)) + _BETA1[2] * (C / LOA)
    C_LF2 = (_BETA2[0] + _BETA2[1] * (B / LOA) + _BETA2[2] * (HC / LOA)
             + _BETA2[3] * (AOD / LOA ** 2) + _BETA2[4] * (AF / B ** 2))
    C_LF = np.where(psi <= 90, C_LF1, C_LF2)

    C_YM1 = _GAMMA1[0] + _GAMMA1[1] * (AF / (LOA * B))
    C_YM2 = _GAMMA2[0] + _GAMMA2[1] * (AOD / LOA ** 2)
    C_YM = np.where(psi <= 90, C_YM1, C_YM2)
    C_YLI = np.pi * (AL / LOA ** 2) + C_YM

    C_XLI1 = _DELTA1[0] + _DELTA1[1] * (AL / (LOA * HBR)) + _DELTA1[2] * (AF / (B * HBR))
    C_XLI2 = (_DELTA2[0] + _DELTA2[1] * (AL / (LOA * HBR)) + _DELTA2[2] * (AF / AL)
              + _DELTA2[3] * (B / LOA) + _DELTA2[4] * (AF / (B * HBR)))
    C_XLI = np.where(psi <= 90, C_XLI1, C_XLI2)

    C_ALF1 = _EPS1[0] + _EPS1[1] * (AOD / AL) + _EPS1[2] * (B / LOA)
    C_ALF2 = _EPS2[0] + _EPS2[1] * (AOD / AL)
    C_ALF = np.where(psi <= 90, C_ALF1, C_ALF2)

    CX = (C_LF * np.cos(psi_rad)
          + C_XLI * (np.sin(psi_rad) - 0.5 * np.sin(psi_rad) * np.cos(psi_rad) ** 2)
          * np.sin(psi_rad) * np.cos(psi_rad)
          + C_ALF * np.sin(psi_rad) * np.cos(psi_rad) ** 3)

    CY = (C_CF * np.sin(psi_rad) ** 2
          + C_YLI * (np.cos(psi_rad) + 0.5 * np.sin(psi_rad) ** 2 * np.cos(psi_rad))
          * np.sin(psi_rad) * np.cos(psi_rad))

    CN = CY * (0.927 * (C / LOA) - 0.149 * (psi_rad - np.pi / 2))

    return (CX[0], CY[0], CN[0]) if CX.size == 1 else (CX, CY, CN)


def cx_fujiwara(psi_deg, params):
    """縦方向風圧抵抗係数 C_X(psi) のみを返す(fuel_consumption.py 等との互換用)。"""
    cx, _, _ = fujiwara_cx_cy_cn(psi_deg, params)
    return cx


def wind_resistance(psi_deg, U_A, params, rho_air=1.222):
    """
    縦方向風圧抵抗 R_AA [N] を返す。
    U_A [m/s]: 見かけ風速(相対風速)
    ITTC定義: R_AA = 1/2 * rho_air * U_A^2 * A_F * (-C_X)
             (C_X<0は向かい風での抵抗=正のR_AAに対応)
    """
    cx = cx_fujiwara(psi_deg, params)
    AF = params["AF"]
    RAA = 0.5 * rho_air * U_A ** 2 * AF * (-cx)
    return RAA, cx


# ----------------------------------------------------------------------
# 2. KCS(満載状態, 3600TEU)の8パラメータ
#    container_ship.py (v3) compute_AL_summary() / compute_AF_components()
#    の計算結果をそのまま使用(スケーリング値ではない実計算値)。
# ----------------------------------------------------------------------
KCS_FULL_LOAD = {
    "LOA": 230.0,     # [m] 垂線間長
    "B":   32.2,      # [m] 型幅
    "AF":  754.5,     # [m2] 正面投影面積 (船体264.0+コンテナ1-4段348.9+5段目73.8+ブリッジ上部63.8+マスト4.0)
    "AL":  4303.2,    # [m2] 側面投影面積 (船体1886.0+船尾コンテナ671.0+ブリッジ270.0+船首コンテナ1476.2)
    "AOD": 2417.2,    # [m2] デッキ上構造物側面積 (船尾コンテナ671.0+ブリッジ270.0+船首コンテナ1476.2)
    "C":   -24.2,     # [m] 側面積重心の船体中央からの位置(船首方向が正)
                       #     container_ship.py の出力は符号規約が逆(+8.75=船尾側が正)なので反転済み
    "HC":  10.03,     # [m] 側面積重心の喫水線からの高さ
    "HBR": 28.20,     # [m] 喫水線からブリッジ最上部(マスト含む)までの高さ
}


if __name__ == "__main__":
    psi = np.linspace(0, 180, 181)
    cx, cy, cn = fujiwara_cx_cy_cn(psi, KCS_FULL_LOAD)

    U_A = 15.0  # [m/s] 例: 見かけ風速15m/s
    RAA, _ = wind_resistance(psi, U_A, KCS_FULL_LOAD)

    print("KCS 満載状態 風圧抵抗係数・風圧抵抗 (U_A = {:.1f} m/s)".format(U_A))
    print(f"{'psi[deg]':>10}{'C_X[-]':>12}{'C_Y[-]':>12}{'R_AA[kN]':>14}")
    for a in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
        idx = list(psi).index(a)
        print(f"{a:>10}{cx[idx]:>12.4f}{cy[idx]:>12.4f}{RAA[idx]/1e3:>14.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(psi, cx, lw=2, label="$C_X$")
    axes[0].plot(psi, cy, lw=2, label="$C_Y$", linestyle="--")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("relative wind angle psi [deg]")
    axes[0].set_ylabel("Coefficient [-]")
    axes[0].set_title("KCS wind coefficients (Fujiwara 2005, full load)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for u in [10, 15, 20, 25]:
        R, _ = wind_resistance(psi, u, KCS_FULL_LOAD)
        axes[1].plot(psi, R / 1e3, lw=2, label=f"U_A = {u} m/s")
    axes[1].set_xlabel("relative wind angle psi [deg]")
    axes[1].set_ylabel("R_AA [kN]")
    axes[1].set_title("KCS wind resistance R_AA(psi)\n(full load)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("kcs_wind_resistance.png", dpi=150)
    print("\n図を kcs_wind_resistance.png に保存しました。")