import bpy
import os
import math
from mathutils import Vector

ROOT = "/root/autodl-tmp/cv_final/final_assets"
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", os.environ.get("TAG", "bg_only_tune"))
os.makedirs(OUT_DIR, exist_ok=True)

PATH_GARDEN = os.path.join(ROOT, "background_garden_2dgs", "train", "ours_30000", "fuse_post.ply")
PATH_A = os.path.join(ROOT, "fusion_scene_strict", "A_clean", "A_clean_top1.ply")
PATH_B = os.path.join(ROOT, "object_B_threestudio", "model.obj")
PATH_C = os.path.join(ROOT, "object_C_magic123", "mesh.obj")

RES_X = int(os.environ.get("RES_X", 1280))
RES_Y = int(os.environ.get("RES_Y", 720))
SAMPLES = int(os.environ.get("SAMPLES", 4))

CAM_DIST = float(os.environ.get("CAM_DIST", 5.4))
CAM_PITCH_DEG = float(os.environ.get("CAM_PITCH_DEG", 20.0))   
CAM_LOOK_Z = float(os.environ.get("CAM_LOOK_Z", 0.90))
LENS = float(os.environ.get("LENS", 48.0))

A_RX = float(os.environ.get("A_RX", 180))
A_RY = float(os.environ.get("A_RY", -30))
A_RZ = float(os.environ.get("A_RZ", 270))
A_SCALE = float(os.environ.get("A_SCALE", 2.0))

B_RX = float(os.environ.get("B_RX", 0))
B_RY = float(os.environ.get("B_RY", 0))
B_RZ = float(os.environ.get("B_RZ", 0))
B_SCALE = float(os.environ.get("B_SCALE", 1.0))

C_RX = float(os.environ.get("C_RX", 90))
C_RY = float(os.environ.get("C_RY", -90))
C_RZ = float(os.environ.get("C_RZ", 195))
C_SCALE = float(os.environ.get("C_SCALE", 1.0))

OBJ_Y = float(os.environ.get("OBJ_Y", 3.0))
OBJ_Z = float(os.environ.get("OBJ_Z", 0.75))

A_X = float(os.environ.get("A_X", -1.25))
B_X = float(os.environ.get("B_X", 0.00))
C_X = float(os.environ.get("C_X", 1.05))

BG_RIGHT = float(os.environ.get("BG_RIGHT", 0.0))
BG_UP = float(os.environ.get("BG_UP", -1.0))
BG_DEPTH = float(os.environ.get("BG_DEPTH", 2.2))
BG_TILT_X = float(os.environ.get("BG_TILT_X", -12.0)) 
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False

def add_lights():
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 8))
    sun = bpy.context.object
    sun.data.energy = 2.0

    bpy.ops.object.light_add(type='AREA', location=(0, 5, 6))
    area = bpy.context.object
    area.data.energy = 3000
    area.data.shape = 'RECTANGLE'
    area.data.size = 8
    area.data.size_y = 8

def import_ply(path):
    bpy.ops.import_mesh.ply(filepath=path)
    return bpy.context.selected_objects[:]

def import_obj(path):
    bpy.ops.import_scene.obj(filepath=path)
    return bpy.context.selected_objects[:]

def join_objects(objs, name):
    if len(objs) == 1:
        obj = objs[0]
        obj.name = name
        return obj
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    return obj

def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def bbox_world(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    mx = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return mn, mx

def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


clear_scene()
setup_render()
add_lights()

bg = join_objects(import_ply(PATH_GARDEN), "Garden_BG")

obj_a = join_objects(import_ply(PATH_A), "Object_A")
obj_b = join_objects(import_obj(PATH_B), "Object_B")
obj_c = join_objects(import_obj(PATH_C), "Object_C")

obj_a.location = Vector((A_X, OBJ_Y, OBJ_Z))
obj_b.location = Vector((B_X, OBJ_Y, OBJ_Z))
obj_c.location = Vector((C_X, OBJ_Y, OBJ_Z))

obj_a.rotation_euler = (math.radians(A_RX), math.radians(A_RY), math.radians(A_RZ))
obj_b.rotation_euler = (math.radians(B_RX), math.radians(B_RY), math.radians(B_RZ))
obj_c.rotation_euler = (math.radians(C_RX), math.radians(C_RY), math.radians(C_RZ))

obj_a.scale = (A_SCALE, A_SCALE, A_SCALE)
obj_b.scale = (B_SCALE, B_SCALE, B_SCALE)
obj_c.scale = (C_SCALE, C_SCALE, C_SCALE)

mn_a, mx_a = bbox_world(obj_a)
mn_b, mx_b = bbox_world(obj_b)
mn_c, mx_c = bbox_world(obj_c)

mn = Vector((min(mn_a.x, mn_b.x, mn_c.x), min(mn_a.y, mn_b.y, mn_c.y), min(mn_a.z, mn_b.z, mn_c.z)))
mx = Vector((max(mx_a.x, mx_b.x, mx_c.x), max(mx_a.y, mx_b.y, mx_c.y), max(mx_a.z, mx_b.z, mx_c.z)))

focus = (mn + mx) * 0.5
look_target = Vector((focus.x, focus.y, CAM_LOOK_Z))

pitch = math.radians(CAM_PITCH_DEG)
cam_x = focus.x
cam_y = focus.y + CAM_DIST * math.cos(pitch)
cam_z = look_target.z + CAM_DIST * math.sin(pitch)

bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
cam = bpy.context.object
cam.data.lens = LENS
bpy.context.scene.camera = cam
look_at(cam, look_target)

cam_dir = (look_target - cam.location).normalized()
cam_right = cam.matrix_world.to_quaternion() @ Vector((1, 0, 0))
cam_up = cam.matrix_world.to_quaternion() @ Vector((0, 1, 0))

bg.location = Vector((focus.x, focus.y, look_target.z))
bg.location += cam_right * BG_RIGHT
bg.location += cam_up * BG_UP
bg.location += cam_dir * BG_DEPTH
bg.rotation_euler = (math.radians(BG_TILT_X), 0.0, 0.0)

print("=== FIXED ABC + FIXED CAMERA + BG ONLY TUNE ===")
print(f"A pose: rx={A_RX}, ry={A_RY}, rz={A_RZ}, scale={A_SCALE}")
print(f"B pose: rx={B_RX}, ry={B_RY}, rz={B_RZ}, scale={B_SCALE}")
print(f"C pose: rx={C_RX}, ry={C_RY}, rz={C_RZ}, scale={C_SCALE}")
print(f"camera: dist={CAM_DIST}, pitch={CAM_PITCH_DEG}, look_z={CAM_LOOK_Z}, lens={LENS}")
print(f"background: right={BG_RIGHT}, up={BG_UP}, depth={BG_DEPTH}, tilt_x={BG_TILT_X}")
print("focus:", focus)
print("look_target:", look_target)
print("cam loc:", cam.location)
print("bg loc:", bg.location)
print("bg rot:", bg.rotation_euler)

def hide_all():
    for o in [bg, obj_a, obj_b, obj_c]:
        o.hide_render = True

hide_all()
obj_a.hide_render = False
obj_b.hide_render = False
obj_c.hide_render = False
render_to(os.path.join(OUT_DIR, "00_ABC_only_no_bg.png"))

hide_all()
bg.hide_render = False
render_to(os.path.join(OUT_DIR, "01_bg_only.png"))

hide_all()
bg.hide_render = False
obj_a.hide_render = False
render_to(os.path.join(OUT_DIR, "02_garden_A.png"))

hide_all()
bg.hide_render = False
obj_b.hide_render = False
render_to(os.path.join(OUT_DIR, "03_garden_B.png"))

hide_all()
bg.hide_render = False
obj_c.hide_render = False
render_to(os.path.join(OUT_DIR, "04_garden_C.png"))

hide_all()
bg.hide_render = False
obj_a.hide_render = False
obj_b.hide_render = False
obj_c.hide_render = False
render_to(os.path.join(OUT_DIR, "05_garden_ABC_all.png"))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "bg_only_tune.blend"))
print("[DONE]")
print("[OUT_DIR]", OUT_DIR)
