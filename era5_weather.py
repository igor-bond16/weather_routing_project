# era5_weather.py
import datetime as dt
import math
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import os
import zipfile
import shutil
import tempfile

from dijkstra_3d import WeatherPoint

ERA5_VARS = [
    "significant_height_of_combined_wind_waves_and_swell",  # swh
    "peak_wave_period",                                     # pp1d
    "mean_wave_direction",                                  # mwd（波が来る方向, 真北基準）
    "10m_u_component_of_wind",                              # u10
    "10m_v_component_of_wind",                              # v10
]


def download_era5(t0: dt.datetime, hours: float, area, out_prefix="era5"):
    """area = [N, W, S, E]。月ごとに1リクエストで取得し、ファイル名リストを返す。"""
    import cdsapi
    c = cdsapi.Client()
    t_end = t0 + dt.timedelta(hours=hours)
    months: dict = {}
    t = t0
    while t <= t_end:
        months.setdefault((t.year, t.month), set()).add(t.day)
        t += dt.timedelta(days=1)
    months.setdefault((t_end.year, t_end.month), set()).add(t_end.day)

    paths = []
    for (y, m), days in sorted(months.items()):
        target = f"{out_prefix}_{y}{m:02d}.nc"
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": ERA5_VARS,
            "year": str(y), "month": f"{m:02d}",
            "day": [f"{d:02d}" for d in sorted(days)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": "netcdf", "download_format": "unarchived",
        }, target)
        paths.append(target)
    return paths


def _read_nc(path_ascii):
    """ASCIIのみのパスにあるnetCDFを読み込み、メモリに載せて閉じる。"""
    ds, last_err = None, None

    # 1) 通常の読み込み（h5netcdf → netcdf4 の順）
    for engine in ("h5netcdf", "netcdf4"):
        try:
            with xr.open_dataset(path_ascii, engine=engine) as d:
                ds = d.load()
            break
        except Exception as e:
            last_err = e

    # 2) 全体読みが HDF error で失敗する環境向け：時刻ごとに読んで結合
    if ds is None:
        try:
            with xr.open_dataset(path_ascii, engine="netcdf4") as d:
                tdim = "valid_time" if "valid_time" in d.dims else "time"
                ds = xr.concat([d.isel({tdim: i}).load()
                                for i in range(d.sizes[tdim])], dim=tdim)
        except Exception as e:
            raise RuntimeError(f"{path_ascii} を読めません: {last_err} / {e}")

    if "valid_time" in ds.dims:                      # 新CDSのnetCDFは valid_time
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars([v for v in ("number", "expver") if v in ds.coords])

def _open(path):
    # netCDF4 は日本語を含むパスを開けないため、一時フォルダ（ASCIIパス）を経由する
    tmpdir = tempfile.mkdtemp(prefix="era5_")
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head[:2] == b"PK":                        # zip: 展開して中のnetCDFをすべて読む
            with zipfile.ZipFile(path) as z:
                z.extractall(tmpdir)
            files = [os.path.join(tmpdir, n) for n in sorted(os.listdir(tmpdir))
                     if n.endswith(".nc")]
        elif head == b"GRIB":
            raise ValueError(f"{path} はGRIB形式です。download_era5 の data_format を確認してください")
        else:                                        # 通常のnetCDF
            tmp = os.path.join(tmpdir, "single.nc")
            shutil.copyfile(path, tmp)
            files = [tmp]
        parts = [_read_nc(p) for p in files]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 風(0.25°)と波(0.5°)で格子が違うので、最も粗い格子に揃えてから結合する
    # 粗い格子の点は細かい格子の部分集合なので、補間せず該当点を選ぶだけでよい（軽い）
    ref = min(parts, key=lambda d: d.sizes["latitude"] * d.sizes["longitude"])
    aligned = []
    for d in parts:
        if (d.sizes["latitude"], d.sizes["longitude"]) != (ref.sizes["latitude"], ref.sizes["longitude"]):
            d = d.sel(latitude=ref["latitude"], longitude=ref["longitude"],
                      method="nearest", tolerance=0.01)
        aligned.append(d)
    return xr.merge(aligned, compat="override")

class ERA5Weather:
    """(lat, lon, t_h) -> WeatherPoint。t_h は t0_utc からの経過時間 [h]。"""

    def __init__(self, paths, t0_utc: dt.datetime):
        ds = xr.concat([_open(p) for p in paths], dim="time").sortby("time")
        ds = ds.sortby("latitude").sortby("longitude")

        th = ((ds["time"].values - np.datetime64(t0_utc)) / np.timedelta64(1, "h")).astype(float)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        self._lon_360 = lon.max() > 180.0

        swh = ds["swh"].fillna(0.0).values           # 陸上は NaN → 静穏として扱う
        pp1d = ds["pp1d"].fillna(0.0).values
        mwd = np.deg2rad(ds["mwd"].values)           # 方向は sin/cos で補間
        u10, v10 = ds["u10"].fillna(0.0).values, ds["v10"].fillna(0.0).values
        data = np.stack([swh, pp1d, np.nan_to_num(np.sin(mwd)),
                         np.nan_to_num(np.cos(mwd)), u10, v10], axis=-1)

        self._interp = RegularGridInterpolator((th, lat, lon), data,
                                               bounds_error=False, fill_value=None)

    def __call__(self, lat, lon, t_h):
        lon_q = lon % 360.0 if self._lon_360 else ((lon + 180.0) % 360.0) - 180.0
        hs, tp, s, c, u, v = self._interp([[t_h, lat, lon_q]])[0]
        return WeatherPoint(
            Hs=max(float(hs), 0.0),
            Tp=max(float(tp), 3.0),
            wave_from_deg=math.degrees(math.atan2(s, c)) % 360.0,
            wind_speed=math.hypot(u, v),
            wind_from_deg=(180.0 + math.degrees(math.atan2(u, v))) % 360.0,  # 吹いてくる方向
        )