import numpy as np
import matplotlib.pyplot as plt
from ship_params import model1_params as param
import pandas as pd

class KCS:
    def __init__(self,is_full_scale=False):
        self.is_full_scale = is_full_scale

        #wetted area
        self.S = param["S_rudder"] + param["S_w/o_rudder"]

        # --- 2. 形状影響係数 (1+k1) の計算 (Holtrop 1984) ---
        self.c14 = 1 + 0.011*param["C_stern"]
        self.Cp = param["Cb"] / param["Cm"]

        self.L_R = param["L_wl"]*(1 - self.Cp + 0.06 * self.Cp * param["LCB"] / (4 * self.Cp - 1))
                    
        if self.is_full_scale:
            self.one_plus_k1 = (
                0.93 + 0.487118 * self.c14 * (param["B_wl"]/param["L_wl"])**1.06806*
                (param["T"]/param["L_wl"])**0.46106*
                (param["L_wl"]/self.L_R)**0.121563*
                (param["L_wl"]**3/param["nabla"])**0.36486*
                (1-self.Cp)**(-0.604247)
            )
        else:
            self.one_plus_k1 = 1.103


        self.T_F = param["T"]

        self.is_full_scale = is_full_scale

        if self.T_F / param["L_wl"] <= 0.04:
            c4 = self.T_F / param["L_wl"]
        else:
            c4 = 0.04

        if self.is_full_scale:
            self.Ca = 0.006 * (param["L_wl"] + 100)**(-0.16) - 0.00205 + 0.003 * np.sqrt(param["L_wl"] / 7.5) * (param["Cb"]**4) * c2 * (0.04 - c4)
        else:
            self.Ca = 0.0

    def calc_calm_water_resistance(self, v_ms):
     #   v_ms = v_knots * 0.51444
        Fn = v_ms / np.sqrt(param["g"] * param["L_wl"])
        
        # --- A. 摩擦抵抗 (Rf) ---
        Re = (v_ms * param["L_wl"]) / (param["Viscosity"])
        Cf = 0.075 / ((np.log10(Re) - 2)**2)
        Rf = 0.5 * param["rho_water"] * (v_ms**2) * self.S * Cf
        
        # 粘性抵抗 (Rf * (1+k1))
        R_viscous = Rf * self.one_plus_k1
        
        # --- B. 造波抵抗 (Rw) ---
        if param["B_wl"] / param["L_wl"] < 0.11:
            c7 = 0.229577 * (param["B_wl"] / param["L_wl"])**0.33333
        elif 0.11 <= param["B_wl"] / param["L_wl"] <= 0.25:
            c7 = param["B_wl"] / param["L_wl"]
        else:
            c7 = 0.5 - 0.0625 * (param["L_wl"] / param["B_wl"])

        i_E =( 
                1 + 89 * np.exp(-(param["L_wl"] / param["B_wl"])**0.80856 * (1-param["C_wp"])**0.30484 * 
                (1-self.Cp-0.0225*param["LCB"])**0.6367 * (self.L_R/param["B_wl"])**0.34574 * 
                (100*param["nabla"]/param["L_wl"]**3)**0.16302)
                              )
        
        c1 = 2223105 * c7**3.78613 * (param["T"]/param["B_wl"])**1.07961 * (90-i_E)**(-1.37565)
        
        c3 = 0.56 * param["A_BT"]**1.5 / (param["B_wl"] * param["T"] * (0.31*np.sqrt(param["A_BT"]) + self.T_F - param["h_B"]))
        c2 = np.exp(-1.89 * np.sqrt(c3))
        
        c5 = 1 - 0.8 * param["A_T"] / (param["B_wl"] * param["T"] * param["Cm"])
        
        if self.Cp < 0.80:
            c16 = 8.07981*self.Cp - 13.8673*self.Cp**2 + 6.984388*self.Cp**3
        else:
            c16 = 1.73014 - 0.7067*self.Cp
            
        m1 = (
            0.0140407*(param["L_wl"]/param["T"]) 
            - 1.75254*(param["nabla"]**(1/3)/param["L_wl"]) 
            - 4.79323*(param["B_wl"] / param["L_wl"]) - c16
              )
        
        if param["L_wl"]**3 / param["nabla"] < 512:
            c15 = - 1.69385
        elif 512 <= param["L_wl"]**3 / param["nabla"] <= 1727:
            c15 = -1.69385 + (param["L_wl"] / param["nabla"]**(1/3) - 8.0) / 2.36
        else:
            c15 = 0.0
            
        m2 = c15 * self.Cp**2 * np.exp(-0.1 * Fn**(-2))
        
        if param["L_wl"] / param["B_wl"] < 12:
            lambda_val = 1.446 * self.Cp - 0.03 * param["L_wl"] / param["B_wl"]
        else:
            lambda_val = 1.446 * self.Cp - 0.36
            
        # 造波抵抗本体
        Rw = (
            c1 * c2 * c5 * param["nabla"] * param["rho_water"] * param["g"] * 
             np.exp(m1 * Fn**(-0.9) + m2 * np.cos(lambda_val * Fn**(-2)))
             )
        
        # --- C. 模型-実船相関補正抵抗 (Ra) ---
        Ra = 0.5 * param["rho_water"] * (v_ms**2) * self.S * self.Ca

               
        Rt = R_viscous + Rw + Ra

        Ct = Rt / (0.5 * param["rho_water"] * v_ms**2 * self.S)

        return {"Fn": Fn, "v_knots": v_ms, "Re": Re, "Rt(N)": Rt, "Ct_calc": Ct*1000}

    

ship = KCS()

efd_data = [
    {"v_ms":0.915,"Fn": 0.108, "Ct_efd": 3.796},
    {"v_ms":1.281,"Fn": 0.152, "Ct_efd": 3.641},
    {"v_ms":1.647,"Fn": 0.195, "Ct_efd": 3.475},
    {"v_ms":1.922,"Fn": 0.227, "Ct_efd": 3.467},
    {"v_ms":2.196,"Fn": 0.260, "Ct_efd": 3.711},
    {"v_ms":2.379,"Fn": 0.282, "Ct_efd": 4.501}
]

# 計算と結果の格納
results = []
for data in efd_data:
    res = ship.calc_calm_water_resistance(data["v_ms"])
    res["Ct_efd"] = data["Ct_efd"]
    # 誤差(%)を計算: (計算値 - 実験値) / 実験値 * 100
    res["Error(%)"] = (res["Ct_calc"] - data["Ct_efd"]) / data["Ct_efd"] * 100
    results.append(res)

# Pandasで綺麗な表（データフレーム）を作成
df = pd.DataFrame(results)
# 見やすいように列の順番を整理して表示
df = df[["Fn", "v_knots", "Re", "Ct_efd", "Ct_calc", "Error(%)", "Rt(N)"]]
print("\n=== Validation 比較表 ===")
print(df.to_string(index=False, float_format="%.4f"))

# ==========================================
# Validation グラフの描画
# ==========================================
# EFDのプロット点
fn_efd = [d["Fn"] for d in efd_data]
ct_efd = [d["Ct_efd"] for d in efd_data]

# 計算モデル（Holtrop）の滑らかな曲線を描くための細かい点
fn_calc_range = np.linspace(0.1, 0.3, 50)
ct_calc = [ship.calc_calm_water_resistance(fn*np.sqrt(param["g"]*param["L_wl"]))["Ct_calc"] for fn in fn_calc_range]

plt.figure(figsize=(8, 5))
# 実験値は点（散布図）でプロットするのが流体力学の論文のセオリーです
plt.plot(fn_efd, ct_efd, 'ro', label='EFD Data (Tokyo 2015)')
# 計算値は線でプロットします
plt.plot(fn_calc_range, ct_calc, 'b-', label='Holtrop Calculation')

plt.xlabel('Froude Number (Fn)', fontsize=12)
plt.ylabel(r'$C_T \times 10^3$', fontsize=12)
plt.title('Validation of Total Resistance Coefficient (KCS Case 2.1)', fontsize=14)
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig("validation_KCS_Case2.1.png", dpi=300)
print("\nValidation graph successfully saved as validation_KCS_Case2.1.png!")