import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ship_params import kcs_full as param

from scipy.special import i1, k1

class KCS:
    def __init__(self, is_full_scale=False, L_bwl=75.0, ratio_zero=0.30):
        self.is_full_scale = is_full_scale

        #watted area
        self.S = param["S_rudder"] + param["S_w/o_rudder"]

        # ------------------------------------------------------------------
        # 真の C_aw テーブル (スケール混同を解消した厳密な再逆算値)
        # 実験データ(L_pp=3.1589m, v=1.34m/s, S=1.8037m2)から再計算
        # ------------------------------------------------------------------
        wlr = np.array([0.650, 0.850, 1.150, 1.370, 1.950])
        caw = np.array([3.20, 7.44, 10.66, 7.81, 1.90])
        
        # extrapolate(外挿)を外し、範囲外は端の値を維持するか0にする安全装置
        # self._caw_f = interp1d(wlr, caw, kind='cubic', bounds_error=False, fill_value=(74.6, 0.0))
        self._caw_f = interp1d(wlr, caw, kind='linear', bounds_error=False,
                       fill_value=(caw[0], 0.0))

        # ------------------------------------------------------------------
        # 角度修正関数 f(chi): EFD の角度依存比率 (chi=0° を 1.0 に正規化)
        # ------------------------------------------------------------------
        ea = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        ef = np.array([9.18, 7.99, 2.76, 3.57, -0.47])
        ef = ef / ef[0]
        self._angle_f = interp1d(ea, ef, kind='cubic', bounds_error=False, fill_value=(1.0, -0.051))
        # ------------------------------------------------------------------
        # 追加：短波長域 (λ/L < 0.65) の推定用パラメータ設定
        # ------------------------------------------------------------------
        self.L_bwl = L_bwl
        self.draft = param["T"]  # ship_params.py の "T" を使用
        self.r0 = ratio_zero
        L = param["L_pp"]
        B = param["B_wl"]
        
        # プラトー値 σ_p
        self._sigma_p = 0.5 * (L / B) * np.sqrt(B / self.L_bwl)
        self._r_edge = 0.65
        self._caw_edge = float(self._caw_f(self._r_edge))
        
        # 境界での反射成分と運動成分
        refl_edge = self._sigma_p * self._alpha_T(self._r_edge * L)
        self._motion_edge = max(self._caw_edge - refl_edge, 0.0)

        # ------------------------------------------------------------------
        # 追加：長波長域 (λ/L > 1.95) の推定用パラメータ設定
        # (ITTC 2014 STAwave-2 の長波長漸近特性に基づく)
        # ------------------------------------------------------------------
        self._r_long_edge = wlr[-1]   # 1.950
        self._caw_long_edge = caw[-1] # 1.90

    def _alpha_T(self, wave_length):
        """喫水減衰係数 α_T の計算"""
        x = 2.0 * np.pi / wave_length * self.draft
        x = 2.0 * x
        pi2i = (np.pi * i1(x)) ** 2
        return float(pi2i / (pi2i + k1(x) ** 2))

    def C_aw(self, wave_length):
        """波長から C_aw を返す（スプライン補間）"""
       # correct_L_pp = 3.1589 
        ratio = wave_length / param["L_pp"]
        
        # ① 短波長域 (λ/L < 0.65): 反射成分と運動成分で推定 (STAwave-1の考え方を拡張)
        if ratio < self._r_edge:
            refl = self._sigma_p * self._alpha_T(wave_length)
            motion = self._motion_edge * max((ratio - self.r0) / (self._r_edge - self.r0), 0.0)
            return float(refl + motion)
            
        # ② 長波長域 (λ/L > 1.95): ITTC 2014 STAwave-2 に準拠した長波長漸近減衰
        if ratio > self._r_long_edge:
            # STAwave-2の ω^11 減衰特性は、波長比 λ/L の -5.5 乗に相当する
            decay = (ratio / self._r_long_edge) ** -5.5
            return float(self._caw_long_edge * decay)
            
        # ③ 表の範囲内 (0.65 <= λ/L <= 1.95): スプライン補間値を使用
        return float(self._caw_f(ratio))

    def angle_factor(self, chi_deg):
        """角度修正関数 f(chi)、左右対称処理込み"""
        chi = chi_deg % 360
        if chi > 180:
            chi = 360 - chi
        return float(self._angle_f(chi))

    def calc_total_R_AW(self, wave_height, wave_length, chi_deg, v_ms=None):
        zeta_a   = wave_height / 2.0
        C_aw_val = self.C_aw(wave_length)
        R_aw_0 = (param["rho_water"] * param["g"] * zeta_a ** 2
                * (param["B_wl"] ** 2 / param["L_pp"]) * C_aw_val)
        R = R_aw_0 * max(self.angle_factor(chi_deg), 0.0)   # 追波の負値を防ぐ
        if v_ms is not None:
            Fr = v_ms / np.sqrt(param["g"] * param["L_pp"])
            R *= float(np.clip((Fr / 0.26) ** 0.8, 0.0, 1.5))
        return R

    # def calc_total_R_AW(self, wave_height, wave_length, chi_deg, v_ms=None):
    #     """波浪中抵抗増加 [N]"""
    #     zeta_a    = wave_height / 2.0
    #     C_aw_val  = self.C_aw(wave_length)
        
    #     #correct_L_pp = 3.1589
    #     #correct_B_wl = 0.442
        
    #     R_aw_0    = (
    #         param["rho_water"] * param["g"]
    #         * zeta_a ** 2
    #         * (param["B_wl"] ** 2 / param["L_pp"])
    #         * C_aw_val
    #     )
    #     return R_aw_0 * self.angle_factor(chi_deg)

  #  def calc_delta_ct(self, wave_height, wave_length, chi_deg, v_ms):
  #      """無次元化された抵抗増加係数 ΔCt × 10³"""
  #      correct_S = 1.8037
  #      R = self.calc_total_R_AW(wave_height, wave_length, chi_deg, v_ms)
  #      return (R / (0.5 * param["rho_water"] * v_ms ** 2 * correct_S)) * 1e3

# ===========================================================================
# 動作確認とグラフ描画 (Case 2.11 検証)
# ===========================================================================
# ===========================================================================
# 動作確認とグラフ描画 (Case 2.11 検証)
# ===========================================================================
#if __name__ == "__main__":
#    v_ms_test    = 1.34
#    wave_length  = 3.1589 * 1.0   # 正しい λ/L = 1.0 の波長
#    wave_height  = 0.045

    # 評価する5つの角度
#    angles   = np.array([0, 45, 90, 135, 180])
    
    # EFDデータ
 #   dct_efd  = np.array([9.18, 7.99, 2.76, 3.57, -0.47])

  #  model = KCS()

    # 5つの角度での計算値を格納するリスト
   # delta_ct_calc = []
    #for angle in angles:
     #   dct = model.calc_delta_ct(wave_height, wave_length, angle, v_ms_test)
      #  delta_ct_calc.append(dct)

    # ==========================================
    # グラフ描画 (画像と同じスタイルに完全再現)
    # ==========================================
    #plt.figure(figsize=(9, 6))

    # EFDデータ (黒い実線と丸マーカー)
    #plt.plot(angles, dct_efd, 'ko-', markersize=9, linewidth=2, zorder=5, label='EFD Data (Stocker 2016)')

    # KCS_v4 計算値 (青い破線とバツマーカー)
    # markeredgewidth=2 でバツ印を少し太くし、画像の見栄えに近づけます
    #plt.plot(angles, delta_ct_calc, 'bx--', markersize=9, linewidth=2, markeredgewidth=2, zorder=4, label='KCS')

    #plt.xlabel('Heading Angle $\chi$ (deg)', fontsize=14)
    #plt.ylabel('Added Resistance Coefficient $\Delta C_T \\times 10^3$', fontsize=14)
   # plt.title('Validation', fontsize=18)
    #plt.xticks(angles)
    
    # 画像と同じ細かい点線のグリッド
    #plt.grid(True, linestyle=':')
    #plt.legend(fontsize=12)

    #plt.tight_layout()
    #plt.savefig("validation_KCS_v4_Discrete.png", dpi=300)
   # print("\nValidation graph successfully saved as validation_KCS_v4_Discrete.png!")
    