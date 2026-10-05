# full scale params
kcs_full = {
    "L_wl":232.5,         #length of waterline m
    "L_pp": 230.0,        #length of ship m
    "B_wl":32.2,          #Maximum beam of waterline m
    "D":19.0,             #Depth m
    "T":10.8,             #Draft m
    "Cb":0.651,           #block coefficient
    "Cm":0.985,           #Midship section coefficient
    "lcb":-1.48,          #
    "nabla":52030,        #Displacement Volume m^3
    "C_wp": 0.74,         #waterplane area coefficient
    "C_stern": 0,        #船尾形状係数
    "h_B": 4,             # バルブバス・バウ中心高さ m
    "A_BT": 20.0,         # バルブバス・バウ横断面積 m^2
    "A_T": 16.0,          # トランサム（船尾）の浸水面積 m^2
    "rho_water":999.5,
    "rho_air":1.225,
    "g":9.81,
    "S_rudder":115.0,
    "S_w/o_rudder":9424
}

#2015 Model1 params
L_pp_model1 = 7.2786
ratio_1 = L_pp_model1 / kcs_full["L_pp"]

model1_params = {        
    "L_pp": L_pp_model1,
    "L_wl": kcs_full["L_wl"]*ratio_1,
    "B_wl":kcs_full["B_wl"]*ratio_1,
    "D":kcs_full["D"]*ratio_1,
    "T":kcs_full["T"]*ratio_1,
    "Cb":kcs_full["Cb"],
    "Cm":kcs_full["Cm"],
    "LCB":kcs_full["lcb"],
    "nabla":kcs_full["nabla"]*(ratio_1**3),
    "C_wp": kcs_full["C_wp"],
    "C_stern": kcs_full["C_stern"],
    "h_B": kcs_full["h_B"] * ratio_1,
    "A_BT": kcs_full["A_BT"] * (ratio_1 ** 2),
    "A_T": kcs_full["A_T"] * (ratio_1 ** 2),
    "rho_water":kcs_full["rho_water"],
    "rho_air":kcs_full["rho_air"],
    "g":kcs_full["g"],
    "S_rudder":kcs_full["S_rudder"]*(ratio_1**2),
    "S_w/o_rudder":kcs_full["S_w/o_rudder"]*(ratio_1**2),
    "Viscosity":1.27e-06
}

#2015 Model3 params
L_pp_model3 = 3.16
ratio_3 = L_pp_model3 / kcs_full["L_pp"]
model3_params = {        
    "L_pp": L_pp_model3,
    "L_wl": kcs_full["L_wl"]*ratio_3,
    "B_wl":kcs_full["B_wl"]*ratio_3,
    "D":kcs_full["D"]*ratio_3,
    "T":kcs_full["T"]*ratio_3,
    "Cb":kcs_full["Cb"],
    "Cm":kcs_full["Cm"],
    "LCB":kcs_full["lcb"],
    "nabla":kcs_full["nabla"]*(ratio_3**3),
    "C_wp": kcs_full["C_wp"],
    "C_stern": kcs_full["C_stern"],
    "h_B": kcs_full["h_B"] * ratio_3,
    "A_BT": kcs_full["A_BT"] * (ratio_3 ** 2),
    "A_T": kcs_full["A_T"] * (ratio_3 ** 2),
    "rho_water":kcs_full["rho_water"],
    "rho_air":kcs_full["rho_air"],
    "g":kcs_full["g"],
    "S_rudder":kcs_full["S_rudder"]*(ratio_3**2),
    "S_w/o_rudder":kcs_full["S_w/o_rudder"]*(ratio_3**2),
    "Viscosity":1.27e-06
}