import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ship_params import model3_params as param

class KCS:
    def __init__(self, is_full_scale=False):
        self.is_full_scale = is_full_scale

        #watted area
        self.S = param["S_rudder"] + param["S_w/o_rudder"]

        # ------------------------------------------------------------------
        # 真の C_aw テーブル (スケール混同を解消した厳密な再逆算値)
        # 実験データ(L_pp=3.1589m, v=1.34m/s, S=1.8037m2)から再計算
        # ------------------------------------------------------------------
        wlr = np.array([0.650, 0.850, 1.150, 1.370, 1.950])
        caw = np.array([74.6, 49.5, 36.9, 22.8, 7.5])
        
        # extrapolate(外挿)を外し、範囲外は端の値を維持するか0にする安全装置
        self._caw_f = interp1d(wlr, caw, kind='cubic', bounds_error=False, fill_value=(74.6, 0.0))

        # ------------------------------------------------------------------
        # 角度修正関数 f(chi): EFD の角度依存比率 (chi=0° を 1.0 に正規化)
        # ------------------------------------------------------------------
        ea = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        ef = np.array([9.18, 7.99, 2.76, 3.57, -0.47])
        ef = ef / ef[0]
        self._angle_f = interp1d(ea, ef, kind='cubic', bounds_error=False, fill_value=(1.0, -0.051))

    def C_aw(self, wave_length):
        """波長から C_aw を返す（スプライン補間）"""
       # correct_L_pp = 3.1589 
        ratio = wave_length / param["L_pp"]
        return float(self._caw_f(ratio))

    def angle_factor(self, chi_deg):
        """角度修正関数 f(chi)、左右対称処理込み"""
        chi = chi_deg % 360
        if chi > 180:
            chi = 360 - chi
        return float(self._angle_f(chi))

    def calc_total_R_AW(self, wave_height, wave_length, chi_deg, v_ms=None):
        """波浪中抵抗増加 [N]"""
        zeta_a    = wave_height / 2.0
        C_aw_val  = self.C_aw(wave_length)
        
        #correct_L_pp = 3.1589
        #correct_B_wl = 0.442
        
        R_aw_0    = (
            param["rho_water"] * param["g"]
            * zeta_a ** 2
            * (param["B_wl"] ** 2 / param["L_pp"])
            * C_aw_val
        )
        return R_aw_0 * self.angle_factor(chi_deg)

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
    