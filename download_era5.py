import cdsapi

c = cdsapi.Client()

print("船舶用のERA5データ（海面風・波高・海陸マスク）をダウンロードします...")

c.retrieve(
    'reanalysis-era5-single-levels', # 海面・地表面のデータセットに変更
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', # 海面から10m高の東西風
            '10m_v_component_of_wind', # 海面から10m高の南北風
            'significant_height_of_combined_wind_waves_and_swell', # 有義波高（波の高さ）
            'land_sea_mask',           # 海陸マスク（0が海、1が陸。ルーティングに超重要！）
        ],
        'year': '2025',
        'month': '12',
        'day': '01',
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        'area': [
            50, 125, 20, 150, # 日本周辺エリア
        ],
    },
    'era5_ship_japan.nc') # 保存ファイル名を変更

print("ダウンロード完了: era5_ship_japan.nc")