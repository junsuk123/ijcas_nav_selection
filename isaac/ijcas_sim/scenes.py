"""Scene builders for the IJCAS rig (Isaac-only).

warehouse  : Isaac "Full Warehouse" (feature-rich, shelving aisles), loaded through Pegasus.
open_field : 200 m textured ground + sparse textured boxes/cylinders (weak LiDAR structure).
corridor   : rectangular ring of 5 m wide corridors, floor + ceiling, walls with a texture tile
             repeated every ``pattern_period_m`` (long straight legs -> LiDAR degeneracy along the
             corridor axis, repetitive visual texture).
office     : Isaac "Office", held out for S09.

Procedural textures are generated from the scene seed (deterministic) into $IJCAS_DATA/textures.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

TEXTURE_DIR = Path(os.environ.get("IJCAS_DATA", str(Path.home() / "ijcas_data"))) / "textures"


def make_texture(name: str, seed: int, size: int = 1024, contrast: float = 1.0) -> str:
    """Random multi-scale blobs + rectangles: many trackable corners at several scales."""
    import cv2
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEXTURE_DIR / f"{name}_{seed}_{size}.png"
    if path.exists():
        return str(path)
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128.0)
    for scale in (64, 16, 4):
        noise = rng.standard_normal((size // scale + 1, size // scale + 1))
        img += 25.0 * cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(int(size * size / 4000)):
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(6, size // 12, 2)
        val = float(rng.uniform(20, 235))
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), val, -1)
        else:
            cv2.circle(img, (int(x), int(y)), int(w // 2), val, -1)
    img = np.clip(128 + contrast * (img - 128), 0, 255).astype(np.uint8)
    tint = rng.uniform(0.75, 1.0, 3)
    rgb = np.stack([img * tint[0], img * tint[1], img * tint[2]], -1).astype(np.uint8)
    cv2.imwrite(str(path), rgb)
    return str(path)


def _material(path: str, texture: str, tiles: tuple[float, float]):
    from isaacsim.core.api.materials import OmniPBR
    return OmniPBR(prim_path=path, texture_path=texture,
                   texture_scale=np.array(tiles, float), color=np.array([1.0, 1.0, 1.0]))


def _box(path, center, size, material=None, color=(0.6, 0.6, 0.6)):
    from isaacsim.core.api.objects import FixedCuboid
    cube = FixedCuboid(prim_path=path, position=np.asarray(center, float),
                       scale=np.asarray(size, float), color=np.asarray(color, float))
    if material is not None:
        cube.apply_visual_material(material)
    return cube


def _lights(stage, sun_intensity=3000.0, dome_intensity=600.0):
    from pxr import Gf, UsdLux
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(sun_intensity)
    sun.CreateAngleAttr(1.0)
    from pxr import UsdGeom
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 30.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(dome_intensity)


def build_open_field(stage, cfg: dict, seed: int, path_xy: np.ndarray | None = None,
                     keep_out_m: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    size = float(cfg["size_m"])
    ground_tex = make_texture("ground", seed)
    _box("/World/Scene/ground", [0, 0, -0.1], [size, size, 0.2],
         _material("/World/Looks/ground", ground_tex, (size / 12.0, size / 12.0)))
    area = float(cfg["object_area_m"])
    objs = []
    for i in range(int(cfg["objects"])):
        tex = make_texture("object", seed * 100 + i, size=512)
        mat = _material(f"/World/Looks/obj{i}", tex, (1.0, 1.0))
        h = float(rng.uniform(1.0, 6.0))
        w, d = rng.uniform(0.6, 4.0, 2)
        for _ in range(200):
            x, y = rng.uniform(-area / 2 + 10, area / 2 + 10, 2)
            reach = keep_out_m + 0.5 * float(np.hypot(w, d))
            if path_xy is None or np.min(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y)) > reach:
                break
        _box(f"/World/Scene/obj{i}", [x, y, h / 2], [w, d, h], mat)
        objs.append([round(float(x), 2), round(float(y), 2), round(h, 2)])
    _lights(stage)
    return {"objects": objs}


def build_corridor(stage, cfg: dict, seed: int) -> dict:
    lx, ly = (float(v) for v in cfg["outer_m"])
    w, h = float(cfg["width_m"]), float(cfg["height_m"])
    period = float(cfg["pattern_period_m"])
    wall_tex = make_texture("wall", seed, size=512)
    floor_tex = make_texture("floor", seed + 1)
    t = 0.3
    floor = _material("/World/Looks/floor", floor_tex, (lx / 6.0, ly / 6.0))
    _box("/World/Scene/floor", [0, 0, -0.1], [lx + 2 * t, ly + 2 * t, 0.2], floor)
    _box("/World/Scene/ceiling", [0, 0, h + 0.1], [lx + 2 * t, ly + 2 * t, 0.2], floor)

    def wall(name, cx, cy, length, along_x):
        size = [length, t, h] if along_x else [t, length, h]
        mat = _material(f"/World/Looks/{name}", wall_tex, (length / period, 1.0))
        _box(f"/World/Scene/{name}", [cx, cy, h / 2], size, mat)

    ix, iy = lx / 2 - w, ly / 2 - w            # inner block half extents
    wall("outer_s", 0, -ly / 2 - t / 2, lx + 2 * t, True)
    wall("outer_n", 0, ly / 2 + t / 2, lx + 2 * t, True)
    wall("outer_w", -lx / 2 - t / 2, 0, ly, False)
    wall("outer_e", lx / 2 + t / 2, 0, ly, False)
    wall("inner_s", 0, -iy + t / 2, 2 * ix, True)
    wall("inner_n", 0, iy - t / 2, 2 * ix, True)
    wall("inner_w", -ix + t / 2, 0, 2 * iy, False)
    wall("inner_e", ix - t / 2, 0, 2 * iy, False)
    from pxr import UsdLux
    for k, x in enumerate(np.arange(-lx / 2 + 4, lx / 2, 8.0)):
        for j, y in enumerate((-ly / 2 + w / 2, ly / 2 - w / 2)):
            light = UsdLux.SphereLight.Define(stage, f"/World/Lights/h{k}_{j}")
            light.CreateIntensityAttr(30000.0)
            light.CreateRadiusAttr(0.15)
            from pxr import Gf, UsdGeom
            UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, h - 0.3))
    for k, y in enumerate(np.arange(-ly / 2 + 4, ly / 2, 8.0)):
        for j, x in enumerate((-lx / 2 + w / 2, lx / 2 - w / 2)):
            light = UsdLux.SphereLight.Define(stage, f"/World/Lights/v{k}_{j}")
            light.CreateIntensityAttr(30000.0)
            light.CreateRadiusAttr(0.15)
            from pxr import Gf, UsdGeom
            UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, h - 0.3))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    dome.CreateIntensityAttr(150.0)
    return {"inner_block_half": [ix, iy]}


def build_scene(pg, app, name: str, cfg: dict, seed: int, path_xy=None) -> dict:
    """Load or build scene ``name`` into the Pegasus world; returns a description."""
    from isaacsim.core.utils.stage import is_stage_loading
    stage = pg.world.stage
    kind = cfg["kind"]
    if kind == "isaac_environment":
        from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
        pg.load_environment(SIMULATION_ENVIRONMENTS[cfg["environment"]])
        while is_stage_loading():
            app.update()
        return {"kind": kind, "environment": cfg["environment"],
                "url": SIMULATION_ENVIRONMENTS[cfg["environment"]]}
    if kind == "procedural_open_field":
        return {"kind": kind, **build_open_field(stage, cfg, seed, path_xy)}
    if kind == "procedural_corridor":
        return {"kind": kind, **build_corridor(stage, cfg, seed)}
    raise ValueError(f"unknown scene kind {kind}")
