import earthaccess
from datetime import datetime, timedelta

# 論文の航海スケジュール (reproduce_wang2019.py の PAPER 辞書より)
VOYAGE_DATES = [
    "20080117", "20080523", "20081224", # 東航
    "20080129", "20080218", "20080424", "20081214"  # 西航
]

# ダウンロード対象の日数を指定（航海は大体4日なので、+1日余裕を見て5日間）
DAYS_TO_DOWNLOAD = 5

# 1. 認証処理
earthaccess.login(persist=True)

download_dir = "." 
total_downloaded = 0

for date_str in VOYAGE_DATES:
    # 検索用の日付文字列を作成
    start_date = datetime.strptime(date_str, "%Y%m%d")
    end_date = start_date + timedelta(days=DAYS_TO_DOWNLOAD)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"\n--- 航海 {date_str} のOSCARデータを検索 ({start_str} 〜 {end_str}) ---")
    
    results = earthaccess.search_data(
        short_name="OSCAR_L4_OC_FINAL_V2.0",
        temporal=(start_str, end_str),
        bounding_box=(-70.0, 35.0, 5.0, 62.0)
    )
    
    print(f"{len(results)} 件見つかりました。")
    
    if len(results) > 0:
        earthaccess.download(results, local_path=download_dir)
        total_downloaded += len(results)

print(f"\nすべてのダウンロードが完了しました。合計 {total_downloaded} 個のファイルを保存しました。")