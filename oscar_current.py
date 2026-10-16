import datetime as dt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class OSCARCurrent:
    """(lat, lon, t_h) -> (u_current, v_current) を返す補間クラス
    t_h は t0_utc からの経過時間 [h]
    """
    def __init__(self, paths, t0_utc: dt.datetime):
        # 複数ファイルを結合
        ds = xr.concat([xr.open_dataset(p) for p in paths], dim="time").sortby("time")

        # 1. 時間軸の変換（object型対策）
        times = ds["time"].values
        try:
            times_np = times.astype("datetime64[s]")
        except TypeError:
            times_np = np.array([np.datetime64(t.strftime('%Y-%m-%dT%H:%M:%S')) for t in times])
        th = ((times_np - np.datetime64(t0_utc)) / np.timedelta64(1, "h")).astype(float)

        # 2. 変数 "lat", "lon" から本当の座標値を取得
        lat = ds["lat"].values
        lon = ds["lon"].values

        self._lon_360 = lon.max() > 180.0

        # 西航路向けに経度を -180〜180 に補正
        if lon.max() > 180.0:
            lon = np.where(lon > 180.0, lon - 360.0, lon)

        # 3. 補間関数(RegularGridInterpolator)のために必ず昇順ソートする
        lat_idx = np.argsort(lat)
        lon_idx = np.argsort(lon)
        lat = lat[lat_idx]
        lon = lon[lon_idx]

        u_da = ds["u"]
        v_da = ds["v"]

        # 4. time, latitude, longitude 以外の余分な次元（depthなど）があれば削除
        for d in list(u_da.dims):
            if d not in ("time", "latitude", "longitude"):
                u_da = u_da.isel({d: 0})
                v_da = v_da.isel({d: 0})

        # 5. データ配列を抽出（次元名は latitude, longitude）
        u_cur = u_da.transpose("time", "latitude", "longitude").fillna(0.0).values
        v_cur = v_da.transpose("time", "latitude", "longitude").fillna(0.0).values

        # 6. 緯度・経度のソート順を配列データにも適用
        u_cur = u_cur[:, lat_idx, :][:, :, lon_idx]
        v_cur = v_cur[:, lat_idx, :][:, :, lon_idx]

        data = np.stack([u_cur, v_cur], axis=-1)

        print(f"[OSCAR読込完了] 時間範囲: {th.min():.1f}h 〜 {th.max():.1f}h, 経度: {lon.min():.1f} 〜 {lon.max():.1f}, 緯度: {lat.min():.1f} 〜 {lat.max():.1f}")

        # fill_value=None で範囲外の時間は一番近い時間のデータを採用
        self._interp = RegularGridInterpolator(
            (th, lat, lon), data, bounds_error=False, fill_value=None
        )

    def __call__(self, lat, lon, t_h):
            # 内部グリッドに合わせて、問い合わせる経度も必ず -180〜180 に変換する
            lon_q = ((lon + 180.0) % 360.0) - 180.0
            try:
                res = self._interp((t_h, lat, lon_q))
                return res[0], res[1]
            except ValueError:
                return 0.0, 0.0