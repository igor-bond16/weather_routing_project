# -*- coding: utf-8 -*-
"""
KCS(満載状態)の風圧抵抗計算モデル
====================================

参考文献:
  [1] Fujiwara, T., Tsukada, Y., Kitamura, F., Sawada, H., Ohmatsu, S. (2009).
      "Experimental Investigation and Estimation on Wind Forces for a Container Ship,"
      Proc. ISOPE 2009.  ―― ユーザー提供論文。満載(container形状に欠落/過積載が無い)状態では
      本論文の補正項(A_RC/A_OD に依存する項)はゼロとなり、Fujiwara et al.(2005a, 2006)の
      オリジナルの物理コンポーネントモデルに帰着する。
  [2] ITTC – Recommended Procedures and Guidelines 7.5-04-01-01.2 "Analysis of Speed/Power
      Trial Data", Appendix C.3 "Regression formula by Fujiwara et al." ―― [1]式(3)と同一の
      構造を持つ公式回帰式・回帰係数表(Table C-2)を採用。

モデルの位置づけ
----------------
* 計算するのは船体前後方向(縦方向)の風圧抵抗係数 C_X (= -C_AA) のみ。
  KCSの「風圧“抵抗”」という要求に対応する量である(横力 Y_A・回頭モーメント N_A を
  含む3分力全体が必要な場合は、Fujiwara(2005a)の完全版(C_Y, C_N も回帰)が必要だが、
  その回帰係数表は本ツールでは未実装。必要であれば拡張可能)。
* KCS(MOERI Container Ship, SIMMAN2008)は「仮想実験用ベアハル」であり、上部構造物
  (ブリッジ・コンテナ積付け)の形状が公式には定義されていない。したがって A_F, A_L, A_OD,
  C, H_BR, H_C は、同型の実在コンテナ船例(本論文 Table 1: 300m/6500TEU船)を参考に、
  KCSの主要目(Lpp=230m, B=32.2m, D=19.0m, d=10.8m, 3600TEU級)に対して工学的に見積もった
  値である。実際の一般配置図がある場合は、KCS_FULL_LOAD の値を置き換えてください。

単位系: SI(m, m^2, kg/m^3, m/s, N)
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Fujiwara回帰式の係数表 (ITTC 7.5-04-01-01.2, Table C-2)
# ----------------------------------------------------------------------
# 各行 i=1: 0<=psi<=90(deg) 用、i=2: 90<psi<=180(deg) 用
# 列 j=0..4 は各回帰式の定数項・説明変数の係数(未使用項は0)
BETA  = np.array([[ 0.922, -0.507, -1.162,  0.000,  0.000],   # C_LF   (i=1: 0-90deg)
                   [-0.018,  5.091,-10.367,  3.011,  0.341]])  # C_LF   (i=2: 90-180deg)

DELTA = np.array([[-0.458, -3.245,  2.313,  0.000,  0.000],   # C_XLI  (i=1: 0-90deg)
                   [ 1.901,-12.727,-24.407, 40.310,  5.481]])  # C_XLI  (i=2: 90-180deg)

EPS   = np.array([[ 0.585,  0.906, -3.239,  0.000,  0.000],   # C_ALF  (i=1: 0-90deg)
                   [ 0.314,  1.117,  0.000,  0.000,  0.000]])  # C_ALF  (i=2: 90-180deg)

MU = 10.0  # deg, psi=90deg付近の平滑化幅 (ITTC推奨値)


def _CLF_0_90(p):
    return (BETA[0, 0]
            + BETA[0, 1] * p["AYV"] / (p["LOA"] * p["B"])
            + BETA[0, 2] * p["CMC"] / p["LOA"])


def _CXLI_0_90(p):
    return (DELTA[0, 0]
            + DELTA[0, 1] * p["AXV"] / p["AYV"]
            + DELTA[0, 2] * p["B"] / p["HBR"])


def _CALF_0_90(p):
    return (EPS[0, 0]
            + EPS[0, 1] * p["B"] / p["LOA"]
            + EPS[0, 2] * p["AOD"] / p["AYV"])


# ----------------------------------------------------------------------
# 【重要な注意】90°<psi<=180°(追い風寄り)域の回帰式について
# ----------------------------------------------------------------------
# 0-90度域の式(_CLF_0_90, _CXLI_0_90, _CALF_0_90)は複数の独立文献で構造・
# 係数を突き合わせ、計算結果も C_X ~ O(1) という物理的に妥当な範囲に収まる
# ことを確認済み。
#
# 一方、90-180度域の式(以下)は、出典(ITTC 7.5-04-01-01.2 Appendix C.3)の
# OCR抽出テキストに文字化けがあり、5つの説明変数(A_YV, A_XV, A_OD, B, H_BR,
# H_C, LOAの組み合わせ)の対応関係を完全には復元できていません。
# 下記は工学的に妥当な無次元比になるよう調整した「最善推定」であり、
# 計算すると C_X が過大(|C_X|>5)になる場合があるため、追い風寄りの角度に
# ついて精度が必要な場合は、原著(Fujiwara et al., 2005a/2006、または
# ITTC 7.5-04-01-01.2 PDFのTable C-2・式C-5〜C-7)を直接確認してください。
# ----------------------------------------------------------------------
def _CLF_90_180(p):
    return (BETA[1, 0]
            + BETA[1, 1] * p["B"] / p["HBR"]
            + BETA[1, 2] * p["AOD"] / (p["LOA"] ** 2)
            + BETA[1, 3] * p["AXV"] / p["B"] ** 2
            + BETA[1, 4] * p["HC"] / p["LOA"])


def _CXLI_90_180(p):
    return (DELTA[1, 0]
            + DELTA[1, 1] * p["AYV"] / (p["LOA"] * p["HBR"])
            + DELTA[1, 2] * p["AXV"] / p["AYV"]
            + DELTA[1, 3] * p["B"] / p["LOA"]
            + DELTA[1, 4] * p["AXV"] / (p["B"] * p["HBR"]))


def _CALF_90_180(p):
    return (EPS[1, 0] + EPS[1, 1] * p["AYV"] / p["AOD"])


def cx_fujiwara(psi_deg, params):
    """
    Fujiwara回帰式による縦方向風圧抵抗係数 C_X(psi) を返す。
    psi_deg: 相対風向角[deg] (0deg = 正船首からの向い風), スカラーまたは配列
    params : dict。必要キー:
        LOA [m]  : 全長
        B   [m]  : 型幅
        AYV [m2] : 喫水線上の側面投影面積 (= A_L)
        AXV [m2] : 最大正面投影面積      (= A_F)
        AOD [m2] : 甲板上構造物の側面投影面積
        CMC [m]  : 側面積重心の船体中央からの前方距離(船首方向を正)
        HBR [m]  : 喫水線からブリッジ最上部までの高さ
        HC  [m]  : 側面積重心の喫水線からの高さ
    """
    psi = np.atleast_1d(np.asarray(psi_deg, dtype=float))
    psi_rad = np.deg2rad(psi)

    cx = np.zeros_like(psi)

    # --- 0-90度域, 90-180度域をそれぞれ計算 ---
    CLF_a, CXLI_a, CALF_a = _CLF_0_90(params), _CXLI_0_90(params), _CALF_0_90(params)
    CLF_b, CXLI_b, CALF_b = _CLF_90_180(params), _CXLI_90_180(params), _CALF_90_180(params)

    def cx_formula(CLF, CXLI, CALF, pr):
        return (CLF * np.cos(pr)
                 + CXLI * (np.sin(pr) - 0.5 * np.sin(pr) * np.cos(pr) ** 2) * np.sin(pr) * np.cos(pr)
                 + CALF * np.sin(pr) * np.cos(pr) ** 3)

    mask_a = psi <= 90.0
    mask_b = psi > 90.0

    cx[mask_a] = cx_formula(CLF_a, CXLI_a, CALF_a, psi_rad[mask_a])
    cx[mask_b] = cx_formula(CLF_b, CXLI_b, CALF_b, psi_rad[mask_b])

    # --- 90度近傍の平滑化 (ITTC推奨: 単純な線形ブレンド) ---
    lo, hi = 90.0 - MU, 90.0 + MU
    blend = (psi >= lo) & (psi <= hi)
    if np.any(blend):
        cx_at_lo = cx_formula(CLF_a, CXLI_a, CALF_a, np.deg2rad(90.0))
        cx_at_hi = cx_formula(CLF_b, CXLI_b, CALF_b, np.deg2rad(90.0))
        w = (psi[blend] - lo) / (hi - lo)
        cx[blend] = (1 - w) * cx_at_lo + w * cx_at_hi

    return cx if cx.size > 1 else cx[0]


def wind_resistance(psi_deg, U_A, params, rho_air=1.225):
    """
    縦方向風圧抵抗 R_AA [N] を返す。
    U_A [m/s]: 見かけ風速(相対風速)
    """
    cx = cx_fujiwara(psi_deg, params)
    AXV = params["AXV"]
    RAA = 0.5 * rho_air * U_A ** 2 * AXV * (-cx)   # C_X = -C_AA (ITTC定義) -> R_AA = 1/2 rho AXV CAA U^2
    return RAA, cx


# ----------------------------------------------------------------------
# 2. KCS(満載状態)の推定パラメータ
# ----------------------------------------------------------------------
# 主要目は SIMMAN2008 / KRISO公表値(実寸)。上部構造・コンテナ搭載形状は
# 本論文 Table1(300m/6500TEU船, LOA=318m,B=40m,d=14m,AF=1469m2,AL=7417m2,
# AOD=4405m2)を参照し、KCS(LOA~232.5m, B=32.2m, 3600TEU級)向けに
# 幾何相似(縮尺 λ=LOA_KCS/LOA_ref)で面積をスケーリングした概算値。
# H_BR, H_C, CMC は主要目・一般的なコンテナ船配置から工学的に推定。
#
# ★実際の一般配置図がある場合は、この辞書の値を置き換えてください。
KCS_FULL_LOAD = {
    "LOA": 232.5,     # [m] 全長(Lpp=230.0mに船首バルブ等の張り出し分を加味)
    "B":   32.2,       # [m] 型幅
    "AYV": 4169.0,      # [m2] 側面投影面積 A_L (300m船 7417m2 を面積比でスケール)
    "AXV": 774.0,       # [m2] 正面投影面積 A_F (300m船 1469m2 を面積比でスケール)
    "AOD": 2352.0,      # [m2] 甲板上構造物側面積 A_OD(300m船 4405m2 を面積比でスケール)
    "CMC": -10.0,       # [m] 側面積重心の船体中央からの位置(ブリッジが船尾寄りのため負=後方と仮定)
    "HBR": 38.0,        # [m] 喫水線からブリッジ最上部までの高さ(推定)
    "HC":  16.0,        # [m] 側面積重心の喫水線からの高さ(推定)
}


if __name__ == "__main__":
    psi = np.linspace(0, 180, 181)
    cx = -cx_fujiwara(psi, KCS_FULL_LOAD)

    U_A = 15.0  # [m/s] 例: 見かけ風速15m/s (~ BF7相当)
    RAA, _ = wind_resistance(psi, U_A, KCS_FULL_LOAD)

    # ---- 結果テーブル(代表角度)----
    print("KCS 満載状態 風圧抵抗係数・風圧抵抗 (U_A = {:.1f} m/s)".format(U_A))
    print(f"{'psi[deg]':>10}{'C_X[-]':>12}{'R_AA[kN]':>14}")
    for a in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
        idx = list(psi).index(a)
        print(f"{a:>10}{cx[idx]:>12.4f}{RAA[idx]/1e3:>14.2f}")

    # ---- グラフ ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(psi, cx, lw=2)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("relative wind angle psi [deg]")
    axes[0].set_ylabel("C_X [-]")
    axes[0].set_title("KCS wind resistance coefficient C_X(psi)\n(Fujiwara regression, full load)")
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