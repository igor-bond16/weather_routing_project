import pygame
import sys
import math

# ======================================================================
# KCS 3600 TEU 風圧パラメータ定義図 (v3)
#
# Fujiwara et al.(2006)「成分分離型モデルを利用した新しい風圧推定法」
# Fig.2 "Definitions of each parameter on ship form" のスタイルで、
# KCS 3600TEUモデルの実ジオメトリから
#   - A_L (側面積) / A_F (正面投影面積) とその内訳
#   - C, H_C (A_L重心の位置) , H_BR (最高点高さ)
#   - Blind Zone(船首方向の死角)
#   - 積載コンテナ数(甲板+船倉、40ftコンテナ換算)
# を計算し、1枚の図にまとめる。
#
# v2 -> v3 変更点:
#   1. 船倉(Hold)のコンテナも甲板と同様に「範囲」ではなく個々の
#      コンテナ(ベイ×段の1マス)が見えるグリッド表示にした。
#   2. コンテナは個別マス目、甲板上の構造物(ブリッジ=A_OD)は
#      引き続き斜線(ハッチング)で面積を表現(両者を視覚的に区別)。
#   3. 船倉はブリッジ直下(機関室)にはコンテナを積めないため、
#      その区間は空けている。
#   4. 船倉も甲板と同様、船尾・船首に向かうほど積める段数が
#      減るテーパー構成にした。
#   5. A_L の重心(G)は、甲板コンテナ・船体・ブリッジの各要素を
#      個々の矩形(=個々のコンテナ相当)に分解し、面積で重み付けした
#      平均位置として再計算した(船倉コンテナは喫水線より下にあり
#      風圧を受けないため、A_L / 重心計算には含めない)。
#   6. 各計算のロジックをコード内コメントで明示した。
# ======================================================================

# (WIDTH, HEIGHTは側面図/正面図それぞれの render_*_surface() 内で個別に定義)

# ---- 配色 ----
BG = (255, 255, 255)
HULL_GRAY = (150, 155, 160)
HULL_OUTLINE = (60, 65, 70)
DECK_CONTAINER = (214, 111, 43)     # 甲板上コンテナ(A_Lに算入)
HOLD_CONTAINER = (232, 165, 120)    # 船倉内コンテナ(A_Lには算入しない、識別用に淡色)
CONTAINER_OUTLINE = (110, 55, 20)
BRIDGE_WHITE = (245, 245, 245)
TEXT_COLOR = (25, 25, 30)
DIM_COLOR = (60, 60, 65)
AL_HATCH = (30, 90, 200)            # A_L(側面積)を表す青ハッチ
AOD_HATCH = (200, 30, 40)           # A_OD(ブリッジ側面積)を表す赤ハッチ
WATER_COLOR = (0, 110, 200)
CHIP_BG = (255, 255, 255)
CHIP_BORDER = (90, 90, 95)
LOS_COLOR = (220, 30, 30)
OK_COLOR = (0, 130, 60)
NG_COLOR = (200, 0, 0)
GRID_LINE = (90, 90, 90)            # 個々のコンテナを区切るグリッド線

# ======================================================================
# 船体の基本寸法(m) — KCS 3600TEUモデル
# ======================================================================
L = 230.0          # 垂線間長 L_OA
BEAM = 32.2        # 船幅 B
DRAFT = 10.8       # 喫水 d
FREEBOARD = 8.2    # 乾舷（喫水線から甲板までの高さ）
BAY = 12.2         # コンテナ1ベイの長さ(40ftコンテナ1個分)
TIER_H = 2.75      # コンテナ1段の高さ
ROW_W = 2.44       # コンテナ1列の幅(20ft/40ft共通の幅)

# ---- 甲板(Deck)上コンテナの配置 ----
# 船尾側(Aft): ブリッジまで4ベイ、一律5段
# 船首側(Fore): ブリッジから船首まで11ベイ、ブリッジ側5段->船首1段で漸減
AFT_BAYS = 4
AFT_TIERS = 5
FORE_TIERS = [5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 2]  # 船首側の先端を1,2段->2,3段に変更(視認性/積載数向上)
DECK_ROWS = 13      # 甲板の最大列数。船幅32.2m ÷ コンテナ幅2.44m ≈13.2 -> 13列が限界

# ---- ブリッジ(居住区)の寸法 ----
BRIDGE_W = 15.0
BRIDGE_H = 18.0
MAST_H = 2.0
MAST_W = 2.0

# ---- 船倉(Hold)内コンテナの配置 ----
# ブリッジ直下は機関室があるためコンテナを積めない(=ベイ配列に含めない)。
# 船尾・船首に向かうほど船体が細く浅くなるため、積める段数を減らす。
HOLD_ROWS = 11              # 二重船殻(バラストタンク)の分だけ甲板より列数を減らす
# 船倉コンテナは「甲板直下(HOLD_TOP_Z)を基準に上詰め」で積む。
# 船底側の空きスペースが、実際の船体形状(バラスト・燃料タンク)に対応する。
# 配列はAP(船尾)側をindex0とする。ユーザー指定「機関室(ブリッジ)側から船尾に
# 向けて5,5,4,3段」を、この並び順(船尾->ブリッジ)に変換すると [3,4,5,5]。
AFT_HOLD_TIERS = [3, 4, 5, 5]                       # 船尾(AP)->ブリッジ方向: 3,4,5,5段
FORE_HOLD_TIERS = [6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4]  # ブリッジ->船首(FP) に向けて減少
HOLD_TIERS_MAX = 6          # 有効深さ(喫水+乾舷から二重底/ハッチコーミング分を除いた高さ)÷段高 で決定
# 有効深さの根拠: (DRAFT - 二重底1.5m) + (FREEBOARD - ハッチコーミング等1.0m)
#              = (10.8-1.5) + (8.2-1.0) = 9.3 + 7.2 = 16.5m -> 16.5 / 2.75 = 6.0段
HOLD_BOTTOM_Z = -(DRAFT - 1.5)   # 船倉最下段の底面高さ(喫水線=0基準、二重底1.5m分を除く)
HOLD_TOP_Z = FREEBOARD - 1.0     # 船倉最上段の上限(ハッチコーミング分を除いた甲板直下)

BLIND_ZONE_LIMIT_M = 460.0  # 船首方向の死角の一般的な基準値

# ---- 船体長さ方向の配置(x=0が船尾(AP)、x=L=230が船首(FP)) ----
STERN_MARGIN = 9.0                              # AP〜最初のコンテナベイまでの余白(操舵機室等)
AFT_LEN = AFT_BAYS * BAY
BRIDGE_START = STERN_MARGIN + AFT_LEN + 2.0     # 船尾コンテナ群の後に2mの隙間を置いてブリッジ
FORE_START = BRIDGE_START + BRIDGE_W + 2.0      # ブリッジの前に2mの隙間を置いて船首コンテナ群
FORE_LEN = len(FORE_TIERS) * BAY

_FONT_CACHE = {}


def get_font(size, bold=False):
    """日本語グリフに対応したフォントを優先的に探し、無ければ既定にフォールバック。"""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = ["notosanscjkjp", "notosansjp", "msgothic", "meiryo", "yugothic",
                  "hiraginosans", "ipagothic", "ipapgothic", "takao"]
    path = None
    for name in candidates:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            break
    font = pygame.font.Font(path, size) if path else pygame.font.SysFont("Arial", size, bold=bold)
    _FONT_CACHE[key] = font
    return font


# ----------------------------------------------------------------------
# 描画補助関数
# ----------------------------------------------------------------------

def blit_halo_text(surface, text, font, color, pos, center=False, pad=3):
    """白い背景(ハロー)付きでテキストを描画。色付き/ハッチ背景の上でも
    文字が読みやすくなるよう、全ての注記テキストはこの関数を通す。"""
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    bg_rect = rect.inflate(pad * 2, pad * 2)
    bg = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(bg, (255, 255, 255, 230), bg.get_rect(), border_radius=4)
    surface.blit(bg, bg_rect.topleft)
    surface.blit(surf, rect.topleft)
    return rect


def draw_dim_line(surface, p1, p2, label, font, color=DIM_COLOR, label_offset=(0, -10)):
    """両端にティックの付いた寸法線を描き、中央にハロー付きラベルを置く。"""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    perp_x, perp_y = -uy, ux
    tick = 6
    pygame.draw.line(surface, color, p1, p2, 2)
    pygame.draw.line(surface, color,
                      (x1 - perp_x * tick, y1 - perp_y * tick),
                      (x1 + perp_x * tick, y1 + perp_y * tick), 2)
    pygame.draw.line(surface, color,
                      (x2 - perp_x * tick, y2 - perp_y * tick),
                      (x2 + perp_x * tick, y2 + perp_y * tick), 2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if label:
        blit_halo_text(surface, label, font, color, (mx + label_offset[0], my + label_offset[1]), center=True)


def draw_dashed_line(surf, color, start_pos, end_pos, width=2, dash_length=12):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = math.hypot(x2 - x1, y2 - y1)
    if dl == 0:
        return
    dashes = max(1, int(dl / dash_length))
    for i in range(dashes):
        if i % 2 == 0:
            s = (x1 + (x2 - x1) * i / dashes, y1 + (y2 - y1) * i / dashes)
            e = (x1 + (x2 - x1) * (i + 1) / dashes, y1 + (y2 - y1) * (i + 1) / dashes)
            pygame.draw.line(surf, color, s, e, width)


def draw_hatch_in_rect(surface, rect, color, spacing=7, lw=1):
    """矩形内だけに斜線(ハッチング)を描く。A_L / A_OD など、風圧計算の
    対象となる「面積」であることを示すために使う(コンテナには使わない)。"""
    x, y, w, h = rect
    clip_rect = pygame.Rect(x, y, w, h)
    old_clip = surface.get_clip()
    surface.set_clip(clip_rect)
    diag = int(w + h)
    for i in range(-diag, diag, spacing):
        pygame.draw.line(surface, color, (x + i, y), (x + i + h, y + h), lw)
    surface.set_clip(old_clip)


def draw_chip(surface, x, y, lines, accent=None, anchor="topleft"):
    """白背景+角丸枠の情報チップ。lines=[(text,font,color), ...]"""
    pad_x, pad_y, gap = 10, 8, 4
    widths, heights = [], []
    for text, font, color in lines:
        surf = font.render(text, True, color)
        widths.append(surf.get_width())
        heights.append(surf.get_height())
    w = max(widths) + pad_x * 2
    h = sum(heights) + gap * (len(lines) - 1) + pad_y * 2
    if anchor == "topleft":
        bx, by = x, y
    elif anchor == "topright":
        bx, by = x - w, y
    elif anchor == "bottomleft":
        bx, by = x, y - h
    elif anchor == "bottomright":
        bx, by = x - w, y - h
    else:
        bx, by = x, y
    rect = pygame.Rect(bx, by, w, h)
    shadow = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 35), (3, 4, w, h), border_radius=8)
    surface.blit(shadow, (bx - 3, by - 3))
    pygame.draw.rect(surface, CHIP_BG, rect, border_radius=8)
    pygame.draw.rect(surface, accent if accent else CHIP_BORDER, rect, 2, border_radius=8)
    ty = by + pad_y
    for text, font, color in lines:
        surf = font.render(text, True, color)
        surface.blit(surf, (bx + pad_x, ty))
        ty += surf.get_height() + gap
    return rect


def draw_container_grid(surface, rx, ry, rw, rh, tiers, color, outline):
    """1ベイ分の矩形を「段(tier)」の数だけ横線で区切り、個々のコンテナが
    1マスずつ見えるように描画する(領域全体を1つの塗りつぶしにしない)。"""
    tier_h_px = rh / tiers
    for t in range(tiers):
        cell = pygame.Rect(rx, ry + t * tier_h_px, rw, tier_h_px)
        pygame.draw.rect(surface, color, cell)
        pygame.draw.rect(surface, outline, cell, 1)


# ======================================================================
# 面積・重心・コンテナ数の計算
# ======================================================================

def compute_AL_components():
    """A_L(側面積)を構成する各要素の面積と、それぞれの図心(x方向=船体
    中央からの位置、z方向=喫水線からの高さ)を求める。

    重要: 船倉(Hold)内のコンテナは喫水線より下にあり風を受けないため、
    A_L・重心計算には一切含めない(甲板上コンテナ・船体・ブリッジのみ)。

    各要素の図心は「面積で重み付けした平均位置」を個々のコンテナ単位で
    積み上げて求める。1ベイ内の各段は同じ幅・同じ位置なので、
    ベイ単位でまとめて計算しても、1段ずつ積み上げて計算しても
    結果は数学的に完全に一致する(長方形の重心は等分割しても変わらない
    ため)。ここでは検算のしやすさを優先し、まず「甲板コンテナは
    ベイ×段の1マスごと」に面積要素を作り、それを合算する。
    """
    components = []  # (名前, 面積[m2], x_c[m], z_c[m])

    # --- 1. 船体(freeboard分のみ。喫水下は風を受けないので含めない) ---
    hull_area = L * FREEBOARD
    components.append(("船体 Hull side", hull_area, L / 2, FREEBOARD / 2))

    # --- 2. 甲板上・船尾コンテナ (ベイ×段を1マスずつ積み上げ) ---
    for bay_i in range(AFT_BAYS):
        bay_x0 = STERN_MARGIN + bay_i * BAY
        for tier in range(AFT_TIERS):
            area = BAY * TIER_H
            x_c = bay_x0 + BAY / 2
            z_c = FREEBOARD + tier * TIER_H + TIER_H / 2
            components.append((None, area, x_c, z_c))  # 個別マスは凡例に出さないのでNone

    # --- 3. ブリッジ (A_OD) ---
    bridge_area = BRIDGE_W * BRIDGE_H
    components.append(("ブリッジ A_OD", bridge_area,
                        BRIDGE_START + BRIDGE_W / 2, FREEBOARD + BRIDGE_H / 2))

    # --- 4. 甲板上・船首コンテナ (ベイ×段を1マスずつ積み上げ、段数はベイごとに可変) ---
    for bay_i, tiers in enumerate(FORE_TIERS):
        bay_x0 = FORE_START + bay_i * BAY
        for tier in range(tiers):
            area = BAY * TIER_H
            x_c = bay_x0 + BAY / 2
            z_c = FREEBOARD + tier * TIER_H + TIER_H / 2
            components.append((None, area, x_c, z_c))

    # --- 凡例表示用に「船尾コンテナ」「船首コンテナ」は集計してまとめ直す ---
    aft_area = BAY * TIER_H * AFT_BAYS * AFT_TIERS
    aft_xc = STERN_MARGIN + AFT_LEN / 2
    aft_zc = FREEBOARD + (AFT_TIERS * TIER_H) / 2
    fore_area = sum(BAY * TIER_H * t for t in FORE_TIERS)
    fore_xw = sum((BAY * TIER_H * t) * (FORE_START + i * BAY + BAY / 2) for i, t in enumerate(FORE_TIERS))
    fore_zw = sum((BAY * TIER_H * t) * (FREEBOARD + (t * TIER_H) / 2) for i, t in enumerate(FORE_TIERS))
    fore_xc = fore_xw / fore_area
    fore_zc = fore_zw / fore_area

    legend = [
        ("船体 Hull side", hull_area, L / 2, FREEBOARD / 2),
        ("船尾コンテナ Aft", aft_area, aft_xc, aft_zc),
        ("ブリッジ A_OD", bridge_area, BRIDGE_START + BRIDGE_W / 2, FREEBOARD + BRIDGE_H / 2),
        ("船首コンテナ Fore", fore_area, fore_xc, fore_zc),
    ]

    return components, legend


def compute_AL_summary():
    """全要素(個々のコンテナマス単位)を面積で重み付けして A_L 全体の
    重心(x_c, z_c)を求める。

        x_c = Σ(area_i * x_i) / Σ(area_i)
        z_c = Σ(area_i * z_i) / Σ(area_i)

    C  : 船体中央(midship = L/2)から重心までの前後方向の距離
         (正の値 = 重心が船尾側にオフセットしていることを示す)
    H_C: 喫水線から重心までの高さ
    H_BR: 喫水線からブリッジ最上部(マスト含む)までの高さ
    """
    components, legend = compute_AL_components()
    A_L = sum(c[1] for c in components)
    x_c = sum(c[1] * c[2] for c in components) / A_L
    z_c = sum(c[1] * c[3] for c in components) / A_L
    midship = L / 2
    C = midship - x_c
    H_C = z_c
    H_BR = FREEBOARD + BRIDGE_H + MAST_H
    return legend, A_L, C, H_C, H_BR, x_c, z_c


def compute_AF_components():
    """A_F(正面投影面積)を、高さ方向に重なりのないバンドに分けて求める。
    船首から見た輪郭(シルエット)なので、奥行き方向(=どのベイか)は
    関係なく「その高さで最大どこまで幅があるか」だけで決まる。
    """
    return [
        ("船体正面", BEAM, FREEBOARD),
        ("コンテナ1-4段", 13 * ROW_W, 4 * TIER_H),
        ("コンテナ5段目", 11 * ROW_W, 1 * TIER_H),
        ("ブリッジ上部", BRIDGE_W, BRIDGE_H - (4 * TIER_H + 1 * TIER_H)),
        ("マスト", MAST_W, MAST_H),
    ]


def compute_blind_zone():
    """ブリッジ前面(見張り位置)から、船首側コンテナの頂部越しに見える
    水面までの距離(死角の長さ)を求める。

    座標系はz軸を喫水線=0とした上向き正。見張りの目の位置(eye_x, eye_z)
    から船首コンテナの各バイ最前部・最上部(obs_x, obs_z)へ向かう視線の
    傾き dz/dx を全ベイについて求め、その中で最大(=最も水平に近い、
    つまり最も見えにくい)ものが実際の視線を決める。
    """
    eye_x = BRIDGE_START + BRIDGE_W
    eye_z = FREEBOARD + BRIDGE_H - 0.5

    max_slope = -float('inf')
    for i, tiers in enumerate(FORE_TIERS):
        obs_x = FORE_START + i * BAY + BAY
        obs_z = FREEBOARD + tiers * TIER_H
        dx = obs_x - eye_x
        dz = obs_z - eye_z
        if dx > 0:
            slope = dz / dx
            if slope > max_slope:
                max_slope = slope

    if max_slope != -float('inf') and max_slope < 0:
        water_x = eye_x - eye_z / max_slope
    else:
        water_x = eye_x + 9999  # 最も高いコンテナが目線より高く、水面が見えない状態

    blind_zone = max(0.0, water_x - L)
    return blind_zone, eye_x, eye_z


def compute_container_counts():
    """甲板(Deck)・船倉(Hold)それぞれのコンテナ数(40ftコンテナ換算個数)
    を「列数 × 段数 × ベイ数」の積み上げで求める。

    甲板: 列数=DECK_ROWS(=13、船幅32.2mに入る最大列数)
          船尾4ベイ×AFT_TIERS段 + 船首各ベイの段数(FORE_TIERSの合計)

    船倉: 列数=HOLD_ROWS(=11、二重船殻分を差し引き)
          ブリッジ直下は機関室のため除外(=ベイ配列に含めていない)
          船尾側 AFT_HOLD_TIERS、船首側 FORE_HOLD_TIERS の合計
    """
    deck_aft = DECK_ROWS * AFT_TIERS * AFT_BAYS
    deck_fore = DECK_ROWS * sum(FORE_TIERS)
    deck_total = deck_aft + deck_fore

    hold_aft = HOLD_ROWS * sum(AFT_HOLD_TIERS)
    hold_fore = HOLD_ROWS * sum(FORE_HOLD_TIERS)
    hold_total = hold_aft + hold_fore

    return {
        "deck_aft": deck_aft, "deck_fore": deck_fore, "deck_total": deck_total,
        "hold_aft": hold_aft, "hold_fore": hold_fore, "hold_total": hold_total,
        "total": deck_total + hold_total,
    }


# ======================================================================
# 描画
# ======================================================================

def draw_side_view(surface, ox, oy, scale_x, scale_z):
    """側面図。ox,oyは (x=0[船尾AP], z=0[喫水線]) のピクセル位置。"""
    font_s = get_font(14)
    font_m = get_font(16, bold=True)

    def px(x_m):
        return ox + x_m * scale_x

    def pz(z_m):
        return oy - z_m * scale_z

    # --- 喫水線 ---
    pygame.draw.line(surface, WATER_COLOR, (px(-5), pz(0)), (px(L + 5), pz(0)), 2)
    blit_halo_text(surface, "W.L.", font_s, WATER_COLOR, (px(-5) - 32, pz(0) - 9))

    # --- 船体外形(喫水線から甲板まで=freeboard、喫水線から船底まで=draft) ---
    hull_pts = [
        (px(0), pz(FREEBOARD)), (px(L), pz(FREEBOARD)),
        (px(L) - 12, pz(-DRAFT)), (px(0) + 18, pz(-DRAFT)),
    ]
    pygame.draw.polygon(surface, HULL_GRAY, hull_pts)
    pygame.draw.polygon(surface, HULL_OUTLINE, hull_pts, 2)

    # --- 船体側面のハッチング(freeboard部分=A_Lに算入される範囲) ---
    hull_rect = (px(0), pz(FREEBOARD), px(L) - px(0), pz(0) - pz(FREEBOARD))
    draw_hatch_in_rect(surface, hull_rect, (*AL_HATCH, 90), spacing=9)
    pygame.draw.rect(surface, HULL_OUTLINE, hull_rect, 1)

    # --- 甲板上・船尾コンテナ(個々のマスが見えるグリッド表示、A_Lに算入) ---
    for bay_i in range(AFT_BAYS):
        bx0 = STERN_MARGIN + bay_i * BAY
        rx, rw = px(bx0), px(bx0 + BAY) - px(bx0)
        ry, rh = pz(FREEBOARD + AFT_TIERS * TIER_H), pz(FREEBOARD) - pz(FREEBOARD + AFT_TIERS * TIER_H)
        draw_container_grid(surface, rx, ry, rw, rh, AFT_TIERS, DECK_CONTAINER, CONTAINER_OUTLINE)

    # --- 甲板上・船首コンテナ(ベイごとに段数が違う。死角の原因でもある) ---
    for i, tiers in enumerate(FORE_TIERS):
        bx0 = FORE_START + i * BAY
        rx, rw = px(bx0), px(bx0 + BAY) - px(bx0)
        ry, rh = pz(FREEBOARD + tiers * TIER_H), pz(FREEBOARD) - pz(FREEBOARD + tiers * TIER_H)
        draw_container_grid(surface, rx, ry, rw, rh, tiers, DECK_CONTAINER, CONTAINER_OUTLINE)

    # --- 船倉(Hold)コンテナ: ブリッジ下は空け、船尾・船首でテーパー ---
    # 甲板直下(HOLD_TOP_Z)を基準に「上詰め」で積む。船体形状により段数が
    # 少ないベイほど、船底側(HOLD_BOTTOM_Z寄り)に空きスペース(バラスト/
    # 燃料タンク相当)が残る形になる。
    WB_COLOR = (190, 215, 235)

    def draw_hold_bay(bay_i, tiers, start_x):
        bx0 = start_x + bay_i * BAY
        rx, rw = px(bx0), px(bx0 + BAY) - px(bx0)
        z_bot = HOLD_TOP_Z - tiers * TIER_H  # 段数が少ないほど底が高い位置になる(=下に空きが残る)
        ry, rh = pz(HOLD_TOP_Z), pz(z_bot) - pz(HOLD_TOP_Z)
        draw_container_grid(surface, rx, ry, rw, rh, tiers, HOLD_CONTAINER, CONTAINER_OUTLINE)
        # 船底側の空きスペース = バラスト/燃料タンク(WB)として明示
       # if z_bot > HOLD_BOTTOM_Z:
       #     tank_rect = pygame.Rect(rx, pz(z_bot), rw, pz(HOLD_BOTTOM_Z) - pz(z_bot))
       #     pygame.draw.rect(surface, WB_COLOR, tank_rect)
       #     pygame.draw.rect(surface, HULL_OUTLINE, tank_rect, 1)
       #     if tank_rect.height > 14:
       #         blit_halo_text(surface, "WB", get_font(10), TEXT_COLOR, tank_rect.center, center=True)

    for bay_i, tiers in enumerate(AFT_HOLD_TIERS):
        draw_hold_bay(bay_i, tiers, STERN_MARGIN)
    for bay_i, tiers in enumerate(FORE_HOLD_TIERS):
        draw_hold_bay(bay_i, tiers, FORE_START)
    # ブリッジ直下(機関室、コンテナなし)の範囲を明示する薄い枠
    eng_rect = pygame.Rect(px(BRIDGE_START), pz(HOLD_BOTTOM_Z + HOLD_TIERS_MAX * TIER_H),
                            px(BRIDGE_START + BRIDGE_W) - px(BRIDGE_START),
                            pz(HOLD_BOTTOM_Z) - pz(HOLD_BOTTOM_Z + HOLD_TIERS_MAX * TIER_H))
    pygame.draw.rect(surface, (225, 225, 225), eng_rect)
    pygame.draw.rect(surface, HULL_OUTLINE, eng_rect, 1)
    blit_halo_text(surface, "機関室", get_font(11), TEXT_COLOR, eng_rect.center, center=True)

    # --- ブリッジ (A_OD、斜線でハッチング) ---
    bridge_rect = pygame.Rect(px(BRIDGE_START), pz(FREEBOARD + BRIDGE_H),
                               px(BRIDGE_START + BRIDGE_W) - px(BRIDGE_START),
                               pz(FREEBOARD) - pz(FREEBOARD + BRIDGE_H))
    pygame.draw.rect(surface, BRIDGE_WHITE, bridge_rect)
    pygame.draw.rect(surface, HULL_OUTLINE, bridge_rect, 2)
    draw_hatch_in_rect(surface, bridge_rect, (*AOD_HATCH, 130), spacing=6)
    mast_x = px(BRIDGE_START + BRIDGE_W / 2)
    pygame.draw.line(surface, HULL_OUTLINE, (mast_x, pz(FREEBOARD + BRIDGE_H)),
                      (mast_x, pz(FREEBOARD + BRIDGE_H + MAST_H)), 3)

    # --- Blind Zone(死角)の可視化 ---
    blind_zone_m, eye_x_m, eye_z_m = compute_blind_zone()
    eye_px = (px(eye_x_m), pz(eye_z_m))
    water_x_m = L + blind_zone_m
    water_px = (px(water_x_m), pz(0))

    blind_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(blind_surf, (220, 30, 30, 55), [
        eye_px, water_px, (px(L), pz(0)), (px(L), pz(FREEBOARD)), (px(BRIDGE_START + BRIDGE_W), pz(FREEBOARD))
    ])
    surface.blit(blind_surf, (0, 0))
    draw_dashed_line(surface, LOS_COLOR, eye_px, water_px, width=2)
    pygame.draw.circle(surface, (0, 0, 0), (int(eye_px[0]), int(eye_px[1])), 4)

    is_ok = blind_zone_m <= BLIND_ZONE_LIMIT_M
    bz_color = OK_COLOR if is_ok else NG_COLOR
    status = "OK" if is_ok else "NG"
    dim_y = pz(0) + 22
    draw_dim_line(surface, (px(L), dim_y), (water_px[0], dim_y), "", font_s, bz_color)
    pygame.draw.line(surface, bz_color, (px(L), pz(0)), (px(L), dim_y + 4), 1)
    pygame.draw.line(surface, bz_color, (water_px[0], pz(0)), (water_px[0], dim_y + 4), 1)
    draw_chip(surface, (px(L) + water_px[0]) / 2, dim_y + 10,
              [(f"死角 (Blind Zone) {blind_zone_m:.1f} m ", font_s, bz_color),
               (f"基準 {BLIND_ZONE_LIMIT_M:.0f} m 以下", get_font(12), (110, 110, 110))],
              accent=bz_color, anchor="topleft")

    # --- 寸法線: L_OA ---
    draw_dim_line(surface, (px(0), pz(-DRAFT) + 30), (px(L), pz(-DRAFT) + 30),
                  f"L_OA = {L:.0f} m", font_m, DIM_COLOR)

    # --- 重心 G, C, H_C, H_BR ---
    legend, A_L, C, H_C, H_BR, x_c, z_c = compute_AL_summary()
    gx, gy = px(x_c), pz(z_c)
    pygame.draw.circle(surface, (0, 0, 0), (int(gx), int(gy)), 5)
    pygame.draw.circle(surface, (255, 255, 255), (int(gx), int(gy)), 2)
    blit_halo_text(surface, "G", font_s, TEXT_COLOR, (gx + 10, gy - 10))

    mid_x = px(L / 2)
    #draw_dim_line(surface, (mid_x, pz(0) + 55), (gx, pz(0) + 55),
    #              f"C = {C:.1f} m", font_m, (150, 30, 130), (0, -8))
    #pygame.draw.line(surface, (150, 30, 130), (mid_x, pz(0)), (mid_x, pz(0) + 60), 1)
    #pygame.draw.line(surface, (150, 30, 130), (gx, pz(0)), (gx, pz(0) + 60), 1)

   # draw_dim_line(surface, (gx - 45, pz(0)), (gx - 45, gy),
   #               f"H_C={H_C:.1f}m", font_s, (0, 90, 170), (-10, 0))

    br_x = px(BRIDGE_START + BRIDGE_W + 25)
    #draw_dim_line(surface, (br_x, pz(0)), (br_x, pz(H_BR)),
    #              f"H_BR={H_BR:.1f}m", font_m, (150, 60, 0), (52, 0))
    #pygame.draw.line(surface, (150, 60, 0), (px(BRIDGE_START + BRIDGE_W), pz(H_BR)), (br_x, pz(H_BR)), 1)

    blit_halo_text(surface, "側面図", get_font(19, bold=True), TEXT_COLOR,
                   (px(0), pz(H_BR) - 34))
    blit_halo_text(surface, "Aft", font_m, TEXT_COLOR, (px(0) - 8, pz(0) + 28))
    blit_halo_text(surface, "Fore", font_m, TEXT_COLOR, (px(L) - 34, pz(0) + 28))

    # --- 凡例: 甲板コンテナ(濃い色)と船倉コンテナ(淡い色)の違いを明示 ---
    lx, ly = px(0), pz(-DRAFT) + 60
    pygame.draw.rect(surface, DECK_CONTAINER, (lx, ly, 16, 12))
    blit_halo_text(surface, "甲板コンテナ(A_Lに算入)", get_font(12), TEXT_COLOR, (lx + 22, ly - 1))
    pygame.draw.rect(surface, HOLD_CONTAINER, (lx + 220, ly, 16, 12))
    blit_halo_text(surface, "船倉コンテナ(喫水下、A_Lには算入しない)", get_font(12), TEXT_COLOR, (lx + 242, ly - 1))


def draw_AF_box(surface, ox, oy, scale, af_comps, A_F):
    """正面図。ox,oyは (喫水線, 船体中心)のピクセル位置(ベース中央)。"""
    blit_halo_text(surface, "正面図 (Front View) — A_F 内訳", get_font(19, bold=True), TEXT_COLOR, (ox - 150, oy - 320))

    def rect_for(w_m, h_m, z0_m):
        w = w_m * scale
        h = h_m * scale
        x = ox - w / 2
        y = oy - (z0_m + h_m) * scale
        return pygame.Rect(x, y, w, h)

    # A_F内訳のうちコンテナ部分は列×段が見えるグリッドで表示する
    grid_tiers = {1: 4, 2: 1}  # インデックス1=1-4段(4段分), インデックス2=5段目(1段分)
    grid_rows = {1: 13, 2: 11}

    z = 0.0
    colors = [HULL_GRAY, DECK_CONTAINER, DECK_CONTAINER, BRIDGE_WHITE, HULL_GRAY]
    band_labels = []
    for idx, ((name, w_m, h_m), col) in enumerate(zip(af_comps, colors)):
        rect = rect_for(w_m, h_m, z)
        if idx in grid_tiers:
            # コンテナ部分: 列(縦線)×段(横線)のグリッドで個々のコンテナを表現
            cols = grid_rows[idx]
            tiers = grid_tiers[idx]
            cw = rect.width / cols
            ch = rect.height / tiers
            for c in range(cols):
                for t in range(tiers):
                    cell = pygame.Rect(rect.x + c * cw, rect.y + t * ch, cw, ch)
                    pygame.draw.rect(surface, col, cell)
                    pygame.draw.rect(surface, GRID_LINE, cell, 1)
        else:
            # 船体・ブリッジ・マスト部分は面積のみなので斜線ハッチで表現
            pygame.draw.rect(surface, col, rect)
            pygame.draw.rect(surface, HULL_OUTLINE, rect, 1)
            draw_hatch_in_rect(surface, rect, (*AL_HATCH, 70), spacing=8)
        band_labels.append((chr(0x2460 + len(band_labels)), rect))  # ①②③...
        blit_halo_text(surface, band_labels[-1][0], get_font(14, bold=True), TEXT_COLOR,
                       (rect.centerx, rect.centery), center=True)
        z += h_m

    base_y = oy
    draw_dim_line(surface, (ox - BEAM * scale / 2, base_y + 25), (ox + BEAM * scale / 2, base_y + 25),
                  f"B = {BEAM} m", get_font(15, bold=True), DIM_COLOR)
    pygame.draw.line(surface, WATER_COLOR, (ox - BEAM * scale / 2 - 10, base_y), (ox + BEAM * scale / 2 + 10, base_y), 2)

    chip_y = oy - z * scale - 55
    draw_chip(surface, ox - 95, chip_y,
              [("正面投影面積 A_F", get_font(13), TEXT_COLOR), (f"{A_F:,.0f} m²", get_font(20, bold=True), (150, 60, 0))],
              accent=(190, 90, 20))

    font_s = get_font(13)
    lines = [("A_F 正面積 内訳", get_font(14, bold=True), TEXT_COLOR)]
    for (name, w_m, h_m), (num, _) in zip(af_comps, band_labels):
        area = w_m * h_m
        pct = area / A_F * 100
        lines.append((f"{num} {name}: {area:,.0f} m² ({pct:.0f}%)", font_s, TEXT_COLOR))
    lines.append((f"合計 A_F = {A_F:,.0f} m²", get_font(15, bold=True), (150, 60, 0)))
    draw_chip(surface, ox - 95, oy + 55, lines, accent=(190, 90, 20))


def draw_container_count_chip(surface, x, y):
    """積載コンテナ数(甲板+船倉、40ft換算)を計算式つきで表示するチップ。"""
    cc = compute_container_counts()
    lines = [
        ("積載コンテナ数 (40ft換算)", get_font(14, bold=True), TEXT_COLOR),
        (f"甲板 船尾: {DECK_ROWS}列×{AFT_TIERS}段×{AFT_BAYS}ベイ = {cc['deck_aft']:,}", get_font(12), TEXT_COLOR),
        (f"甲板 船首: {DECK_ROWS}列×(段数合計{sum(FORE_TIERS)}) = {cc['deck_fore']:,}", get_font(12), TEXT_COLOR),
        (f"船倉 船尾: {HOLD_ROWS}列×(段数合計{sum(AFT_HOLD_TIERS)}) = {cc['hold_aft']:,}", get_font(12), TEXT_COLOR),
        (f"船倉 船首: {HOLD_ROWS}列×(段数合計{sum(FORE_HOLD_TIERS)}) = {cc['hold_fore']:,}", get_font(12), TEXT_COLOR),
        (f"合計 = {cc['total']:,} 個 (上限1,800以内)", get_font(16, bold=True), (0, 120, 0)),
    ]
    return draw_chip(surface, x, y, lines, accent=(0, 130, 0))


def render_side_view_surface():
    """側面図(A_L内訳・積付・死角チェック)を、それ単体で完結したサイズの
    Surfaceに描画して返す。単独の画像ファイルとして書き出す用。"""
    W, H = 1700, 950
    surface = pygame.Surface((W, H))
    surface.fill(BG)

    title = get_font(23, bold=True).render(
        "KCS 3600 TEU コンテナ船　側面図", True, TEXT_COLOR)
    surface.blit(title, ((W - title.get_width()) // 2, 14))

    draw_side_view(surface, 60, 620, scale_x=3.3, scale_z=7.0)

    legend, A_L, C, H_C, H_BR, x_c, z_c = compute_AL_summary()
    lines = [("A_L 側面積 内訳", get_font(14, bold=True), TEXT_COLOR)]
    for name, area, _, _ in legend:
        pct = area / A_L * 100
        lines.append((f"  {name}: {area:,.0f} m² ({pct:.0f}%)", get_font(13), TEXT_COLOR))
    lines.append((f"合計 A_L = {A_L:,.0f} m²", get_font(15, bold=True), (0, 60, 140)))
    al_rect = draw_chip(surface, 60, 760, lines, accent=(0, 90, 170))

    draw_container_count_chip(surface, al_rect.right + 20, 760)

    return surface


def render_front_view_surface():
    """正面図(A_F内訳)を、それ単体で完結したサイズのSurfaceに描画して返す。
    単独の画像ファイルとして書き出す用。"""
    W, H = 800, 950
    surface = pygame.Surface((W, H))
    surface.fill(BG)

    title = get_font(23, bold=True).render("KCS 3600 TEU　正面図 (A_F内訳)", True, TEXT_COLOR)
    surface.blit(title, ((W - title.get_width()) // 2, 14))

    af_comps = compute_AF_components()
    A_F = sum(w * h for _, w, h in af_comps)
    draw_AF_box(surface, W // 2, 660, scale=4.6, af_comps=af_comps, A_F=A_F)

    return surface


def main():
    """対話的にウィンドウ表示する場合は側面図を表示する。
    正面図は run: `python container_ship.py front` で単独表示できる。"""
    pygame.init()
    show_front = len(sys.argv) > 1 and sys.argv[1] == "front"
    W, H = (800, 950) if show_front else (1700, 950)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("KCS 3600 TEU - 風圧パラメータ定義図")
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        surface = render_front_view_surface() if show_front else render_side_view_surface()
        screen.blit(surface, (0, 0))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()