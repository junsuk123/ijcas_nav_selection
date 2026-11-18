"""Outdoor scenes of the IJCAS rig (Isaac-only builders; geometry also used by the GNSS sky model).

urban        street grid of 4 x 4 blocks (40 m blocks, 16 m streets), 2 x 2 buildings per block with
             seeded heights, two open plazas; textured facades.  Buildings block and reflect GNSS
             signals (ijcas_sim/gnss_sky.py): NLOS, multipath, fewer satellites, worse DOP.
solar_farm   16 rows of identical tilted panel tables (30 m tables, 1 m gaps, 7 m row pitch, 500 m long)
             on a flat field: repetitive structure; the row ends are > 120 m (LiDAR range) from the
             flight track.  Open sky.
tunnel_road  straight road with a 400 m tunnel (11 m x 7 m box section) under a hill, identical light
             fixtures every 12 m inside, guardrails, light poles and roadside objects outside.  GNSS is
             lost inside by geometry; the middle of the tunnel is > 120 m from both portals.

All geometry is axis aligned or tilted cuboids with colliders (FixedCuboid), so the PhysX ray casts of
the GNSS model see what the RTX LiDAR and cameras see.  Textures are generated from the seed.
"""
from __future__ import annotations

import numpy as np

from .scenes import TEXTURE_DIR, _box, _lights, _material, make_texture


def _obox(path, center, size, quat_wxyz, material=None, color=(0.6, 0.6, 0.6)):
    from isaacsim.core.api.objects import FixedCuboid
    cube = FixedCuboid(prim_path=path, position=np.asarray(center, float), scale=np.asarray(size, float),
                       orientation=np.asarray(quat_wxyz, float), color=np.asarray(color, float))
    if material is not None:
        cube.apply_visual_material(material)
    return cube


def _png(name, img):
    import cv2
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEXTURE_DIR / f"{name}.png"
    if not path.exists():
        cv2.imwrite(str(path), img)
    return str(path)


def facade_texture(seed: int, size: int = 1024) -> str:
    """Concrete with a window grid (4 x 4 windows per tile), random window brightness and dirt."""
    import cv2
    rng = np.random.default_rng(seed)
    base = cv2.imread(make_texture("facade_base", seed, size, contrast=0.6))
    img = cv2.addWeighted(base, 0.5, np.full_like(base, int(rng.uniform(120, 190))), 0.5, 0)
    cell = size // 4
    for i in range(4):
        for j in range(4):
            x0, y0 = i * cell + cell // 5, j * cell + cell // 4
            x1, y1 = x0 + int(cell * 0.6), y0 + int(cell * 0.5)
            v = int(rng.uniform(15, 90) if rng.random() < 0.8 else rng.uniform(160, 240))
            cv2.rectangle(img, (x0, y0), (x1, y1), (v + 10, v + 5, v), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), max(2, size // 200))
    return _png(f"facade_{seed}_{size}", img)


def panel_texture(seed: int, size: int = 512) -> str:
    """One PV module: dark blue cells (6 x 10) with light grid lines, slight per-module variation."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (int(rng.uniform(70, 95)), int(rng.uniform(35, 50)), int(rng.uniform(15, 30)))   # BGR
    for k in range(1, 6):
        img[:, k * size // 6 - 1:k * size // 6 + 1] = (150, 150, 150)
    for k in range(1, 10):
        img[k * size // 10 - 1:k * size // 10 + 1, :] = (150, 150, 150)
    img[:3], img[-3:], img[:, :3], img[:, -3:] = 200, 200, 200, 200
    return _png(f"panel_{seed}_{size}", img)


def tile_texture(seed: int, size: int = 512, n: int = 8) -> str:
    """Tunnel wall tiles: light grey tiles with grout lines and mild noise (repetitive, low texture)."""
    rng = np.random.default_rng(seed)
    img = (np.full((size, size), 175.0) + rng.normal(0, 6, (size, size))).clip(0, 255)
    step = size // n
    for k in range(n + 1):
        img[:, max(0, k * step - 2):k * step + 2] = 110
        img[max(0, k * step - 2):k * step + 2, :] = 110
    return _png(f"tile_{seed}_{size}", np.stack([img] * 3, -1).astype(np.uint8))


def road_texture(seed: int, size: int = 1024) -> str:
    """Asphalt with a dashed centre line and edge lines; one tile = 10 m x 14 m of road."""
    import cv2
    base = cv2.imread(make_texture("asphalt", seed, size, contrast=0.35))
    img = cv2.addWeighted(base, 0.35, np.full_like(base, 70), 0.65, 0)
    c = size // 2
    img[c - 6:c + 6, : size // 2] = 225                       # dashed centre line (5 m dash, 5 m gap)
    img[int(size * 0.07) - 4:int(size * 0.07) + 4, :] = 225  # edge lines
    img[int(size * 0.93) - 4:int(size * 0.93) + 4, :] = 225
    return _png(f"road_{seed}_{size}", img)


# ------------------------------------------------------------------------------------------------
def build_urban(stage, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nb, block, street = int(cfg["blocks"]), float(cfg["block_m"]), float(cfg["street_m"])
    hmin, hmax = (float(v) for v in cfg["height_m"])
    pitch = block + street
    half = (nb * pitch) / 2 + street
    ground = _material("/World/Looks/ground", make_texture("asphalt", seed + 7, contrast=0.35),
                       (2 * half / 8.0, 2 * half / 8.0))
    _box("/World/Scene/ground", [0, 0, -0.1], [2 * half + 40, 2 * half + 40, 0.2], ground)
    mats = [_material(f"/World/Looks/facade{k}", facade_texture(seed * 10 + k), (1.0, 1.0)) for k in range(8)]
    plazas = {tuple(p) for p in cfg.get("plazas", [])}
    centres = [(k - (nb - 1) / 2) * pitch for k in range(nb)]
    gap = float(cfg.get("building_gap_m", 4.0))
    bsize = (block - gap) / 2
    buildings = []
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            if (round(cx), round(cy)) in plazas:
                continue
            for a in (-1, 1):
                for b in (-1, 1):
                    h = float(rng.uniform(hmin, hmax))
                    x, y = cx + a * (bsize + gap) / 2, cy + b * (bsize + gap) / 2
                    k = len(buildings)
                    _box(f"/World/Scene/bld{k}", [x, y, h / 2], [bsize, bsize, h], mats[k % len(mats)])
                    buildings.append([round(x, 1), round(y, 1), round(h, 1)])
    _lights(stage, sun_intensity=3000.0, dome_intensity=800.0)
    return {"buildings": buildings, "plazas": sorted(plazas)}


def build_solar_farm(stage, cfg: dict, seed: int) -> dict:
    rows, pitch = int(cfg["rows"]), float(cfg["row_pitch_m"])
    length, tlen, gap = float(cfg["row_length_m"]), float(cfg["table_m"]), float(cfg["table_gap_m"])
    width, tilt, low = float(cfg["table_width_m"]), np.radians(float(cfg["tilt_deg"])), float(cfg["low_edge_m"])
    ground = _material("/World/Looks/ground", make_texture("ground", seed), (length / 12.0, length / 12.0))
    _box("/World/Scene/ground", [0, 0, -0.1], [length + 200, rows * pitch + 200, 0.2], ground)
    panel = _material("/World/Looks/panel", panel_texture(seed), (tlen / 1.0, width / 2.0))
    steel = _material("/World/Looks/steel", make_texture("steel", seed + 3, 256, contrast=0.3), (tlen / 2.0, 1.0))
    q = [np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0]           # about +x: faces south (-y) and up
    zc = low + width / 2 * np.sin(tilt)
    n_t = int(length // (tlen + gap))
    xs = [(-length / 2 + (k + 0.5) * (tlen + gap)) for k in range(n_t)]
    k = 0
    for r in range(rows):
        y = (r - (rows - 1) / 2) * pitch
        for x in xs:
            _obox(f"/World/Scene/panel{k}", [x, y, zc], [tlen, width, 0.05], q, panel)
            _box(f"/World/Scene/beam{k}", [x, y, (zc - 0.05) / 2], [tlen, 0.15, zc - 0.05], steel)
            k += 1
    _lights(stage, sun_intensity=3500.0, dome_intensity=800.0)
    return {"rows": rows, "tables": k, "row_y": [(r - (rows - 1) / 2) * pitch for r in range(rows)]}


def build_tunnel_road(stage, cfg: dict, seed: int, path_xy: np.ndarray | None = None) -> dict:
    from pxr import Gf, UsdGeom, UsdLux
    rng = np.random.default_rng(seed)
    road_half = float(cfg["road_length_m"]) / 2
    t0, t1 = (float(v) for v in cfg["tunnel_x_m"])
    w, h, th = float(cfg["tunnel_width_m"]), float(cfg["tunnel_height_m"]), 1.0
    tl = t1 - t0
    tc = (t0 + t1) / 2
    ground = _material("/World/Looks/ground", make_texture("ground", seed), (road_half / 6.0, 30.0))
    _box("/World/Scene/ground", [0, 0, -0.1], [2 * road_half + 120, 360, 0.2], ground)
    road = _material("/World/Looks/road", road_texture(seed), (2 * road_half / 10.0, 1.0))
    _box("/World/Scene/road", [0, 0, 0.025], [2 * road_half, 14.0, 0.05], road)
    tiles = _material("/World/Looks/tiles", tile_texture(seed), (tl / 2.0, h / 2.0))
    for s in (-1, 1):
        _box(f"/World/Scene/tunnel_wall_{'n' if s > 0 else 's'}", [tc, s * (w / 2 + th / 2), h / 2],
             [tl, th, h], tiles)
    _box("/World/Scene/tunnel_roof", [tc, 0, h + th / 2], [tl, w + 2 * th, th], tiles)
    hill = _material("/World/Looks/hill", make_texture("hill", seed + 5, contrast=0.8), (tl / 10.0, 6.0))
    hh = float(cfg.get("hill_height_m", 18.0))
    side = 60.0
    for s in (-1, 1):
        _box(f"/World/Scene/hill_{'n' if s > 0 else 's'}", [tc, s * (w / 2 + th + side / 2), hh / 2],
             [tl, side, hh], hill)
    _box("/World/Scene/hill_top", [tc, 0, (h + th + hh) / 2], [tl, w + 2 * th, hh - h - th], hill)
    steel = _material("/World/Looks/fixture", make_texture("steel", seed + 3, 256, contrast=0.3), (1.0, 1.0))
    spacing = float(cfg["light_spacing_m"])
    for k, x in enumerate(np.arange(t0 + spacing / 2, t1, spacing)):
        _box(f"/World/Scene/fixture{k}", [x, 0, h - 0.1], [1.2, 0.5, 0.15], steel)
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/t{k}")
        light.CreateIntensityAttr(float(cfg.get("light_intensity", 40000.0)))
        light.CreateRadiusAttr(0.2)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x), 0.0, h - 0.4))
    rail = _material("/World/Looks/rail", make_texture("rail", seed + 4, 256, contrast=0.4), (25.0, 1.0))
    keep = float(cfg.get("outside_furniture_x_m", 340.0))
    for s in (-1, 1):
        for a, b in ((-road_half, t0), (t1, road_half)):
            _box(f"/World/Scene/rail_{s}_{int(a)}", [(a + b) / 2, s * 7.5, 0.4], [b - a, 0.3, 0.8], rail)
        for k, x in enumerate([x for x in np.arange(-keep, keep + 1, 30.0) if x < t0 - 5 or x > t1 + 5]):
            _box(f"/World/Scene/pole_{s}_{k}", [x, s * 9.0, 4.0], [0.25, 0.25, 8.0], rail)
    objs = []
    for i in range(int(cfg.get("roadside_objects", 24))):
        tex = make_texture("object", seed * 100 + i, size=512)
        mat = _material(f"/World/Looks/obj{i}", tex, (1.0, 1.0))
        hgt = float(rng.uniform(3.0, 12.0))
        sx, sy = rng.uniform(4.0, 14.0, 2)
        for _ in range(200):
            x = float(rng.choice([-1, 1]) * rng.uniform(t1 + 20, road_half - 20))
            y = float(rng.choice([-1, 1]) * rng.uniform(16, 70))
            reach = 6.0 + 0.5 * float(np.hypot(sx, sy))
            if path_xy is None or np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) > reach:
                break
        _box(f"/World/Scene/obj{i}", [x, y, hgt / 2], [sx, sy, hgt], mat)
        objs.append([round(x, 1), round(y, 1), round(hgt, 1)])
    _lights(stage, sun_intensity=3500.0, dome_intensity=800.0)
    return {"tunnel_x_m": [t0, t1], "objects": objs}
