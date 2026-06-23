import bpy
import os
import math
from mathutils import Vector
from itertools import product

ROOT = "/root/autodl-tmp/cv_final/final_assets"
PATH_C = os.environ.get("C_PATH", os.path.join(ROOT, "object_C_magic123", "mesh.obj"))
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", os.environ.get("TAG", "C_grid_full_v1"))
os.makedirs(OUT_DIR, exist_ok=True)

RES_X = int(os.environ.get("RES_X", "720"))
RES_Y = int(os.environ.get("RES_Y", "720"))
SAMPLES = int(os.environ.get("SAMPLES", "4"))
TARGET_MAX_DIM = float(os.environ.get("TARGET_MAX_DIM", "1.7"))

USE_CLAY = int(os.environ.get("USE_CLAY", "1"))

def parse_list(name, default):
    return [float(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]

C_RX_LIST = parse_list("C_RX_LIST", "0,180")
C_RY_LIST = parse_list("C_RY_LIST", "-30,0,30")
C_RZ_LIST = parse_list("C_RZ_LIST", "0,30,60,90,120,150,180,210,240,270,300,330")

def fmt(x):
    x = int(round(x))
    return f"neg{abs(x)}" if x < 0 else str(x)

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def bbox_world(obj):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return bmin, bmax

def import_obj(path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.obj(filepath=path)
    after = set(bpy.data.objects.keys())
    objs = [bpy.data.objects[n] for n in sorted(after - before) if bpy.data.objects[n].type == "MESH"]
    if not objs:
        objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError("No mesh imported from " + path)
    print("[IMPORT]", path, [o.name for o in objs])
    return objs

def join_meshes(objs, name):
    meshes = [o for o in objs if o.type == "MESH"]
    if len(meshes) == 1:
        meshes[0].name = name
        return meshes[0]

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    return obj

def set_origin_bounds(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

def normalize_object(obj):
    bpy.context.view_layer.update()
    set_origin_bounds(obj)

    bmin, bmax = bbox_world(obj)
    extent = bmax - bmin
    max_dim = max(extent.x, extent.y, extent.z, 1e-6)
    scale = TARGET_MAX_DIM / max_dim
    obj.scale = (obj.scale.x * scale, obj.scale.y * scale, obj.scale.z * scale)
    bpy.context.view_layer.update()

    bmin, bmax = bbox_world(obj)
    center = (bmin + bmax) * 0.5
    obj.location.x -= center.x
    obj.location.y -= center.y
    obj.location.z -= bmin.z
    bpy.context.view_layer.update()

def apply_clay(obj):
    mat = bpy.data.materials.new("C_clay_visible")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.50, 0.43, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.15

    obj.data.materials.clear()
    obj.data.materials.append(mat)

def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES

    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 1.5

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = -0.15
    scene.view_settings.gamma = 1.0

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.62, 0.62, 0.62, 1.0)
        bg.inputs[1].default_value = 0.8

def setup_lights():
    bpy.ops.object.light_add(type="SUN", location=(4, -4, 6))
    sun = bpy.context.object
    sun.data.energy = 2.2
    sun.rotation_euler = (math.radians(45), 0, math.radians(35))

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -3.5, 3.0))
    area = bpy.context.object
    area.data.energy = 1700
    area.data.size = 4.0
    look_at(area, Vector((0, 0, 0.5)))

def setup_camera():
    bpy.ops.object.camera_add(location=(2.6, -4.8, 2.0))
    cam = bpy.context.object
    cam.data.type = "ORTHO"
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    look_at(cam, Vector((0, 0, 0.65)))
    bpy.context.scene.camera = cam
    return cam

def center_after_rotation(obj):
    bmin, bmax = bbox_world(obj)
    center = (bmin + bmax) * 0.5
    obj.location.x -= center.x
    obj.location.y -= center.y
    obj.location.z -= bmin.z
    bpy.context.view_layer.update()

def render_pose(obj, cam, rx, ry, rz):
    obj.location = Vector((0, 0, 0))
    obj.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    bpy.context.view_layer.update()

    center_after_rotation(obj)

    bmin, bmax = bbox_world(obj)
    extent = bmax - bmin
    max_dim = max(extent.x, extent.y, extent.z, 0.2)
    cam.data.ortho_scale = max_dim * 1.25

    name = f"C_rx{fmt(rx)}_ry{fmt(ry)}_rz{fmt(rz)}.png"
    out = os.path.join(OUT_DIR, name)
    bpy.context.scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("[SAVED]", out)

clear_scene()
setup_scene()
setup_lights()
cam = setup_camera()

print("[INFO] PATH_C:", PATH_C)
objs = import_obj(PATH_C)
obj_c = join_meshes(objs, "Object_C")
normalize_object(obj_c)

if USE_CLAY:
    apply_clay(obj_c)

print("[INFO] RX:", C_RX_LIST)
print("[INFO] RY:", C_RY_LIST)
print("[INFO] RZ:", C_RZ_LIST)

count = 0
for rx, ry, rz in product(C_RX_LIST, C_RY_LIST, C_RZ_LIST):
    render_pose(obj_c, cam, rx, ry, rz)
    count += 1

print("[DONE] total renders:", count)
print("[OUT_DIR]", OUT_DIR)
