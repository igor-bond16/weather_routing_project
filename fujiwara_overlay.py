"""
既存の wind_resistance_validation スクリプトに追加して使うためのモジュール。

使い方:
    from fujiwara_overlay import fujiwara_cx_cy_cn, KCS_PARAMS

    psi_fuji, cx_fuji, cy_fuji, cn_fuji = fujiwara_cx_cy_cn(**KCS_PARAMS)
    plt.plot(psi_fuji, cx_fuji, 'g--', label='Fujiwara (2005) Estimate', linewidth=2)

KCS_PARAMS は側面ジオメトリ(container_ship.py の寸法定義)から
Fujiwara(2005)モデル用の8パラメータを計算したもの。
船尾コンテナは5段の設定を採用。
"""
import numpy as np

# ============================================================
# Fujiwara, Ueno & Ikeda (2005) 回帰係数表 (Table 3)
# 出典: 藤原敏文, 上野道雄, 池田良穂 (2005)
#      "成分分離型モデルを利用した新しい風圧力推定法"
#      日本船舶海洋工学会論文集 第2号, pp.243-255
# Ro-Ro旅客船データ(Table A-1)で符号・オーダーを検算済み
# ============================================================
_ALPHA = {0: 0.404, 1: 0.368, 2: 0.902}
_BETA1 = {0: -0.922, 1: 0.507, 2: 1.162}
_BETA2 = {0: 0.018, 1: -5.091, 2: 10.367, 3: -3.011, 4: -0.341}
_GAMMA1 = {0: 0.116, 1: 3.345}
_GAMMA2 = {0: 0.446, 1: 2.192}
_DELTA1 = {0: 0.458, 1: 3.245, 2: -2.313}
_DELTA2 = {0: -1.901, 1: 12.727, 2: 24.407, 3: -40.310, 4: -5.481}
_EPS1 = {0: -0.585, 1: -0.906, 2: 3.239}
_EPS2 = {0: -0.314, 1: -1.117}


def fujiwara_cx_cy_cn(LOA, B, HBR, AF, AL, AOD, C, HC):
    """
    Fujiwara(2005)の成分分離型モデルにより CX, CY, CN を計算する。

    Parameters (全て実船スケール、SI単位)
    ----------
    LOA : 全長 [m] (KCSでは LPP を代用)
    B   : 船幅 [m]
    HBR : 喫水からブリッジ等最高構造物までの高さ [m]
    AF  : 正面投影面積 [m^2]
    AL  : 側面投影面積 [m^2]
    AOD : デッキ上構造物の側面投影面積 [m^2]
    C   : 船体中央から側面積重心までの前後座標 [m] (船首方向が正)
    HC  : 喫水から側面積重心までの高さ [m]

    Returns
    -------
    psi_deg, CX, CY, CN : ndarray (0〜180度, 1度刻み)
    """
    psi_deg = np.arange(0, 181, 1)
    psi = np.deg2rad(psi_deg)

    C_CF = _ALPHA[0] + _ALPHA[1] * (AF / (B * HBR)) + _ALPHA[2] * (HBR / LOA)

    C_LF1 = _BETA1[0] + _BETA1[1] * (AL / (LOA * B)) + _BETA1[2] * (C / LOA)
    C_LF2 = (_BETA2[0] + _BETA2[1] * (B / LOA) + _BETA2[2] * (HC / LOA)
             + _BETA2[3] * (AOD / LOA ** 2) + _BETA2[4] * (AF / B ** 2))
    C_LF = np.where(psi_deg <= 90, C_LF1, C_LF2)

    C_YM1 = _GAMMA1[0] + _GAMMA1[1] * (AF / (LOA * B))
    C_YM2 = _GAMMA2[0] + _GAMMA2[1] * (AOD / LOA ** 2)
    C_YM = np.where(psi_deg <= 90, C_YM1, C_YM2)
    C_YLI = np.pi * (AL / LOA ** 2) + C_YM

    C_XLI1 = _DELTA1[0] + _DELTA1[1] * (AL / (LOA * HBR)) + _DELTA1[2] * (AF / (B * HBR))
    C_XLI2 = (_DELTA2[0] + _DELTA2[1] * (AL / (LOA * HBR)) + _DELTA2[2] * (AF / AL)
              + _DELTA2[3] * (B / LOA) + _DELTA2[4] * (AF / (B * HBR)))
    C_XLI = np.where(psi_deg <= 90, C_XLI1, C_XLI2)

    C_ALF1 = _EPS1[0] + _EPS1[1] * (AOD / AL) + _EPS1[2] * (B / LOA)
    C_ALF2 = _EPS2[0] + _EPS2[1] * (AOD / AL)
    C_ALF = np.where(psi_deg <= 90, C_ALF1, C_ALF2)

    CX = (C_LF * np.cos(psi)
          + C_XLI * (np.sin(psi) - 0.5 * np.sin(psi) * np.cos(psi) ** 2) * np.sin(psi) * np.cos(psi)
          + C_ALF * np.sin(psi) * np.cos(psi) ** 3)

    CY = (C_CF * np.sin(psi) ** 2
          + C_YLI * (np.cos(psi) + 0.5 * np.sin(psi) ** 2 * np.cos(psi)) * np.sin(psi) * np.cos(psi))

    CN = CY * (0.927 * (C / LOA) - 0.149 * (psi - np.pi / 2))

    return psi_deg, CX, CY, CN


# ------------------------------------------------------------
# KCS 3600TEU の8パラメータ (船尾コンテナ5段の設定)
# container_ship.py の寸法定義から算出
#   LPP=230.0m, B=32.2m, 喫水=10.8m, 乾舷=8.2m
#   船尾5段(4ベイ) + ブリッジ(幅15m,高さ18m) + FORE_BLOCKS(11ベイ)
#   船尾マージン=22m (仮定 - 実際の値が分かり次第要更新)
# ------------------------------------------------------------
KCS_PARAMS = dict(
    LOA=230.0,
    B=32.2,
    HBR=26.20,
    AF=774.0,
    AL=4169.0,
    AOD=2283.0,
    C=-4.65,
    HC=9.81,
)


if __name__ == "__main__":
    psi, cx, cy, cn = fujiwara_cx_cy_cn(**KCS_PARAMS)
    print("psi[deg]   CX       CY       CN")
    for ang in range(0, 181, 10):
        print(f"{ang:6d}   {cx[ang]:+.3f}   {cy[ang]:+.3f}   {cn[ang]:+.3f}")