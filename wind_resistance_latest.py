# -*- coding: utf-8 -*-
"""
KCS(満載状態) 風圧抵抗 Cx 検証スクリプト
==========================================

3本の曲線を比較する:
  1. My Model (KCS Custom)      : ITTC 6750TEU代表値の補間 (KCS_Cx_datasets.ods)
  2. Fujiwara (2005) Estimate   : KCS実船寸法から回帰式で直接推定
  3. Paper Data (Andersen EFD)  : Andersen(2013) 9000+TEU船の風洞実測データ

Fujiwara係数について
---------------------
Fujiwara, Ueno & Ikeda (2005) "成分分離型モデルを利用した新しい風圧力推定法"
日本船舶海洋工学会論文集 第2号, pp.243-255, Table 3 の値を直接使用。
Ro-Ro旅客船の実測データ(同論文 Table A-1, Fig.14)で符号・オーダーを検算済み:
  - CX(psi=0)  = -0.463 (実測: 逆風で負の抵抗 -> 符号OK)
  - CX(psi=180)= +0.571 (実測: 追い風で正の推力 -> 符号OK)

  ※以前検討した「ITTC 7.5-04-01-01.2」ベースの別係数セットは、0-90度域の
    符号が本実装と全て反転しており(OCR誤りの疑い)、90-180度域も出典側が
    自ら精度に懸念を表明していたため、本スクリプトでは不採用とした。

単位系: SI (m, m^2, kg/m^3, m/s, N)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ============================================================
# 1. Fujiwara(2005) 回帰式 (Table 3, 検算済み)
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
    """Fujiwara(2005)の成分分離型モデルにより CX, CY, CN(0-180度, 1度刻み)を返す。"""
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


# KCS 3600TEU の8パラメータ(船尾コンテナ5段の設定。船尾マージン22mは仮定 - 要確認)
KCS_PARAMS = dict(
    LOA=230.0, B=32.2, HBR=26.20,
    AF=774.0, AL=4169.0, AOD=2283.0, C=-4.65, HC=9.81,
)


# ============================================================
# 2. 自作モデル (ITTC 6750TEU代表値の補間)
# ============================================================
df = pd.read_excel("./datasets/KCS_Cx_datasets.ods", engine="odf")
psi_points = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
cx_points = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
cx_interpolator = interp1d(psi_points, cx_points, kind='cubic', fill_value="extrapolate")


def calculate_wind_resistance(psi_deg, U_WR, A_XV=774.0, rho_A=1.222):
    """ITTC公式による風圧抵抗 R_AA [N] の計算"""
    cx = cx_interpolator(psi_deg)
    r_aa = 0.5 * rho_A * (U_WR ** 2) * (-cx) * A_XV
    return cx, r_aa


# ============================================================
# 3. 論文実測データ (Andersen 2013, 9000+TEU EFD)
# ============================================================
df_paper = pd.read_excel("./datasets/Wind_Resistance_EFD_data.ods", engine="odf", header=None)
psi_paper = pd.to_numeric(df_paper.iloc[:, 0], errors='coerce').values
cx_paper = pd.to_numeric(df_paper.iloc[:, 2], errors='coerce').values

# ============================================================
# 4. Fujiwara推定 (KCS実船寸法から直接計算)
# ============================================================
psi_fuji, cx_fuji, cy_fuji, cn_fuji = fujiwara_cx_cy_cn(**KCS_PARAMS)

# ============================================================
# 5. 比較プロット
# ============================================================
psi_fine = np.linspace(0, 180, 200)
cx_fine_model = cx_interpolator(psi_fine)

plt.figure(figsize=(9, 5))
plt.plot(psi_fine, cx_fine_model, 'g-', label='My Model (KCS Custom, ITTC 6750TEU)', linewidth=2)
plt.plot(psi_fuji, cx_fuji, 'b--', label='Fujiwara (2005) Estimate (KCS actual dims)', linewidth=2)
plt.plot(psi_paper, cx_paper, 'ro', label='Paper Data (Andersen 2013, 9000+TEU EFD)', markersize=4, alpha=0.8)

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel(r'Relative Wind Angle $\psi_{WR}$ [deg]', fontsize=12)
plt.ylabel('Wind Resistance Coefficient $C_X$ [-]', fontsize=12)
plt.title('Validation: Comparison of $C_X$ (Model vs Fujiwara vs EFD)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

validation_output = "wind_resistance_validation.png"
plt.savefig(validation_output, dpi=150)
print(f"比較検証グラフを '{validation_output}' に保存しました。")
plt.close()

# ============================================================
# 6. 誤差解析: Fujiwara推定 vs 実測、自作モデル vs 実測 の両方を比較
# ============================================================
cx_model_at_paper_angles = cx_interpolator(psi_paper)
cx_fuji_interp = interp1d(psi_fuji, cx_fuji, kind='cubic', fill_value="extrapolate")
cx_fuji_at_paper_angles = cx_fuji_interp(psi_paper)

diff_model = cx_paper - cx_model_at_paper_angles
diff_fuji = cx_paper - cx_fuji_at_paper_angles

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(psi_paper, diff_model, 'o-', color='blue', label='My Model - EFD', alpha=0.7)
axes[0].plot(psi_paper, diff_fuji, 's-', color='green', label='Fujiwara - EFD', alpha=0.7)
axes[0].axhline(0, color='gray', ls='--')
axes[0].set_ylabel('Cx difference\n(paper - model)', fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].legend()

rmse_model = np.sqrt(np.nanmean(diff_model ** 2))
rmse_fuji = np.sqrt(np.nanmean(diff_fuji ** 2))
axes[1].bar(['My Model\n(ITTC 6750TEU)', 'Fujiwara\n(2005) Estimate'],
            [rmse_model, rmse_fuji], color=['blue', 'green'], alpha=0.7)
axes[1].set_ylabel('RMSE vs EFD data', fontsize=11)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
error_output = "kcs_cx_error_analysis.png"
plt.savefig(error_output, dpi=150)
print(f"誤差解析グラフを '{error_output}' に保存しました。")
print(f"\nRMSE (My Model vs EFD)  = {rmse_model:.4f}")
print(f"RMSE (Fujiwara vs EFD)  = {rmse_fuji:.4f}")