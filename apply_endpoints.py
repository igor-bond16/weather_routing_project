# apply_endpoints.py  端点を固定値に置き換える（元ファイルは .bak に退避）
import re, shutil, sys

FILES = ["reproduce_wang2019.py", "calibrate_gc.py", "make_weather_gif.py", "run_dp_one.py"]

DEFS = '''EAST_PT = (47.6, -10.0)    # 欧州側（論文 Fig.10 から読取）
WEST_PT = (46.0, -52.0)    # 北米側（論文 Fig.10 から読取）


def get_endpoints(direction):
    """direction: 'E'=東航(北米→欧州), 'W'=西航(欧州→北米)。(start, end) を返す。"""
    return (WEST_PT, EAST_PT) if direction == "E" else (EAST_PT, WEST_PT)


'''

# 有効な3行ブロック（行頭が # の行は対象外）。R. 接頭辞とインデントを保持する
BLOCK = re.compile(
    r'^(?P<ind>[ \t]*)east_pt, west_lat = \(49\.3, -10\.0\), 46\.5\n'
    r'[ \t]*west_pt = \(west_lat, (?P<r>\w+\.)?calibrate_west_lon\(east_pt, west_lat, ref\["GC"\]\[2\]\)\)\n'
    r'[ \t]*start, end = \(west_pt, east_pt\) if ref\["dir"\] == "E" else \(east_pt, west_pt\)\n',
    re.M)

for fn in FILES:
    try:
        s = open(fn, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"skip {fn}（なし）"); continue
    new, n = BLOCK.subn(lambda m: f'{m["ind"]}start, end = {m["r"] or ""}get_endpoints(ref["dir"])\n', s)
    if fn == "reproduce_wang2019.py":
        if "def get_endpoints" not in new:
            new = new.replace("def calibrate_west_lon", DEFS + "def calibrate_west_lon", 1)
    if fn == "make_weather_gif.py":
        new = re.sub(r'^(\s*)from reproduce_wang2019 import PAPER, calibrate_west_lon\n',
                     r'\1from reproduce_wang2019 import PAPER, get_endpoints\n', new, flags=re.M)
    if new != s:
        shutil.copyfile(fn, fn + ".bak")
        open(fn, "w", encoding="utf-8").write(new)
    print(f"{fn}: 置換 {n} 箇所")