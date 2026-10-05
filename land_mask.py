# land_mask.py
"""global_land_mask.globe の軽量な代替（Natural Earth の陸地ポリゴンを使用）。

メモリを節約するため、
  - 対象海域のバウンディングボックスと交わるポリゴンだけを読み込む
  - ポリゴンを結合（unary_union）せず、個別に判定する
"""
import numpy as np
import shapely
import cartopy.io.shapereader as shpreader

_RESOLUTION = "50m"                       # '110m'（粗い・軽い）/ '50m' / '10m'
_BBOX = (-100.0, 10.0, 40.0, 80.0)        # (lon_min, lat_min, lon_max, lat_max) 北大西洋周辺。他海域では広げる

_geoms = None
_bounds = None


def _load():
    global _geoms, _bounds
    if _geoms is not None:
        return
    path = shpreader.natural_earth(resolution=_RESOLUTION, category="physical", name="land")
    parts = []
    for g in shpreader.Reader(path).geometries():
        parts.extend(shapely.get_parts(g))            # MultiPolygon を単一ポリゴンに分解
    parts = np.array(parts, dtype=object)

    b = shapely.bounds(parts)                          # (n, 4): minx, miny, maxx, maxy
    lon0, lat0, lon1, lat1 = _BBOX
    keep = (b[:, 2] >= lon0) & (b[:, 0] <= lon1) & (b[:, 3] >= lat0) & (b[:, 1] <= lat1)
    _geoms, _bounds = parts[keep], b[keep]
    for g in _geoms:
        shapely.prepare(g)                             # 点判定を高速化
    print(f"[land_mask] {_RESOLUTION}: {len(_geoms)} polygons loaded")


def _is_land_scalar(lat, lon):
    lon = ((lon + 180.0) % 360.0) - 180.0
    cand = np.where((_bounds[:, 0] <= lon) & (lon <= _bounds[:, 2]) &
                    (_bounds[:, 1] <= lat) & (lat <= _bounds[:, 3]))[0]
    for i in cand:
        if shapely.intersects_xy(_geoms[i], lon, lat):
            return True
    return False


class _Globe:
    def is_land(self, lat, lon):
        _load()
        lat_a, lon_a = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        if lat_a.ndim == 0:
            return _is_land_scalar(float(lat_a), float(lon_a))
        out = np.fromiter((_is_land_scalar(la, lo) for la, lo in zip(lat_a.ravel(), lon_a.ravel())),
                          dtype=bool, count=lat_a.size)
        return out.reshape(lat_a.shape)

    def is_ocean(self, lat, lon):
        res = self.is_land(lat, lon)
        return (not res) if isinstance(res, bool) else ~res


globe = _Globe()