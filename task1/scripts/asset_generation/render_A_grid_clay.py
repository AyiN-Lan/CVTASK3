import bpy
import os
import math
from mathutils import Vector

ROOT = "/root/autodl-tmp/cv_final/final_assets"
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", os.environ.get("TAG", "A_grid_clay"))
os.makedirs(OUT_DIR, exist_ok=True)

A_PATH = os.environ.get(
    "A_PATH",
    os.path.join(ROOT, "fusion_scene_strict", "A_clean", "A_clean_top1.ply")
)

RES_X = int(os.environ.get("RES_X", "960"))
RES_Y = int(os.environ.get("RES_Y", "540"))
SAMPLES = int(os.environ.get("SAMPLES", "4"))

def parse_list(name, default):
    return [float(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]

A_RX_LIST = parse_list("A_RX_LIST", "0,180")
A_RY_LIST = parse_list("A_RY_LIST", "-30,0,30")
A_RZ_LIST = parse_list("A_RZ_LIST", "60,90,120,240,270,300")

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def import_ply(path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_mesh.ply(filepath=path)
    after = set(bpy.data.objects.keys())
    objs = [bpy.data.objects[n] for n in sorted(after - before) if bpy.data.objects[n].type == "MESH"]
    if not objs:
        raise RuntimeError("No mesh imported from " + path)
    print("[IMPORT]", path, [o.name for o in objs])
    return objs

def join_objects(objs, name):
    if len(objs) == 1:
        objs[0].name = name
        return objs[0]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    return obj

def bbox(obj):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return bmin, bmax

def center_obj(obj):
    bmin, bmax = bbox(obj)
    c = (bmin + bmax) * 0.5
    obj.location -= c
    bpy.context.view_layer.update()
    return bbox(obj)

def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def make_clay_mat():
    mat = bpy.data.materials.new("A_clay_visible")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.45, 0.34, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.82
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.15
    return mat

def apply_mat(obj):
    mat = make_clay_mat()
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

    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = -0.25
    scene.view_settings.gamma = 1.0

    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.62, 0.62, 0.62, 1)
    bg.inputs[1].default_value = 0.8

def setup_lights():
    bpy.ops.object.light_add(type="SUN", location=(3, -4, 5))
    sun = bpy.context.object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(35))

    bpy.ops.object.light_add(type="AREA", location=(-2.5, -3.5, 3.0))
    area = bpy.context.object
    area.data.energy = 1800
    area.data.size = 4.0
    look_at(area, Vector((0, 0, 0)))

def setup_camera():
    bpy.ops.object.camera_add(location=(0.0, -5.0, 1.8))
    cam = bpy.context.object
    cam.data.type = "ORTHO"
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100
    look_at(cam, Vector((0, 0, 0)))
    bpy.context.scene.camera = cam
    return cam

def render_pose(obj, cam, rx, ry, rz):
    obj.location = Vector((0, 0, 0))
    obj.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    bpy.context.view_layer.update()

    bmin, bmax = center_obj(obj)
    extent = bmax - bmin
    max_dim = max(extent.x, extent.y, extent.z, 0.2)
    cam.data.ortho_scale = max_dim * 1.25

    name = f"A_rx{int(rx)}_ry{int(ry)}_rz{int(rz)}.png"
    out = os.path.join(OUT_DIR, name)
    bpy.context.scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print("[SAVED]", out)

clear_scene()
setup_scene()
setup_lights()
cam = setup_camera()

print("[INFO] A_PATH:", A_PATH)
obj = join_objects(import_ply(A_PATH), "Object_A")
apply_mat(obj)

for rx in A_RX_LIST:
    for ry in A_RY_LIST:
        for rz in A_RZ_LIST:
            render_pose(obj, cam, rx, ry, rz)

blend_path = os.path.join(OUT_DIR, "A_grid_clay.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("[BLEND]", blend_path)
print("[DONE]")
