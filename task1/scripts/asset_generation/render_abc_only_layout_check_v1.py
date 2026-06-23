import bpy
import os
import math
from mathutils import Vector

TAG = os.environ.get("TAG", "abc_only_layout_check_v1")
RES_X = int(os.environ.get("RES_X", "1280"))
RES_Y = int(os.environ.get("RES_Y", "720"))
SAMPLES = int(os.environ.get("SAMPLES", "4"))

CAM_DIST = float(os.environ.get("CAM_DIST", "6.2"))
CAM_PITCH_DEG = float(os.environ.get("CAM_PITCH_DEG", "18"))
CAM_LOOK_Z = float(os.environ.get("CAM_LOOK_Z", "0.95"))
LENS = float(os.environ.get("LENS", "52"))

A_SCALE = float(os.environ.get("A_SCALE", "1.45"))

A_RIGHT = float(os.environ.get("A_RIGHT", "-1.35"))
A_UP    = float(os.environ.get("A_UP", "0.00"))
A_DEPTH = float(os.environ.get("A_DEPTH", "0.25"))

B_RIGHT = float(os.environ.get("B_RIGHT", "-0.05"))
B_UP    = float(os.environ.get("B_UP", "0.00"))
B_DEPTH = float(os.environ.get("B_DEPTH", "0.18"))

C_RIGHT = float(os.environ.get("C_RIGHT", "1.25"))
C_UP    = float(os.environ.get("C_UP", "0.00"))
C_DEPTH = float(os.environ.get("C_DEPTH", "0.18"))

C_RX = float(os.environ.get("C_RX", "90"))
C_RY = float(os.environ.get("C_RY", "-90"))
C_RZ = float(os.environ.get("C_RZ", "195"))

ROOT = "/root/autodl-tmp/cv_final/final_assets"
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", TAG)
os.makedirs(OUT_DIR, exist_ok=True)

scene = bpy.context.scene
cam = scene.camera
if cam is None:
    raise RuntimeError("No active camera")

cam.data.lens = LENS
cam.data.clip_start = 0.01
cam.data.clip_end = 200.0

bg = bpy.data.objects.get("Garden_BG")
obj_a = bpy.data.objects.get("Object_A")
obj_b = bpy.data.objects.get("Object_B")
obj_c = bpy.data.objects.get("Object_C")

for name, obj in [("Garden_BG", bg), ("Object_A", obj_a), ("Object_B", obj_b), ("Object_C", obj_c)]:
    if obj is None:
        raise RuntimeError(f"Missing object: {name}")

def look_at(obj, target):
    direction = (target - obj.location).normalized()
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def bbox_center(obj):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(pts, Vector((0, 0, 0))) / 8.0

def move_bbox_center_to(obj, target):
    current = bbox_center(obj)
    obj.location += (target - current)
    bpy.context.view_layer.update()

def render_still(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("Saved:", path)

obj_a.rotation_euler = (math.radians(180), math.radians(-30), math.radians(270))
obj_a.scale = (A_SCALE, A_SCALE, A_SCALE)

obj_b.rotation_euler = (0, 0, 0)
obj_b.scale = (1, 1, 1)

obj_c.rotation_euler = (math.radians(C_RX), math.radians(C_RY), math.radians(C_RZ))
obj_c.scale = (1, 1, 1)

bpy.context.view_layer.update()

bc_center = (bbox_center(obj_b) + bbox_center(obj_c)) / 2.0
focus = Vector((bc_center.x, bc_center.y, CAM_LOOK_Z))

cam_height = CAM_DIST * math.tan(math.radians(CAM_PITCH_DEG))
cam.location = Vector((focus.x, focus.y - CAM_DIST, focus.z + cam_height))
look_at(cam, focus)

bpy.context.view_layer.update()

R = cam.matrix_world.to_3x3()
cam_right = (R @ Vector((1, 0, 0))).normalized()
cam_up = (R @ Vector((0, 1, 0))).normalized()
cam_forward = (R @ Vector((0, 0, -1))).normalized()

anchor = focus

def target_from_camera(right, up, depth):
    return anchor + right * cam_right + up * cam_up + depth * cam_forward

move_bbox_center_to(obj_a, target_from_camera(A_RIGHT, A_UP, A_DEPTH))
move_bbox_center_to(obj_b, target_from_camera(B_RIGHT, B_UP, B_DEPTH))
move_bbox_center_to(obj_c, target_from_camera(C_RIGHT, C_UP, C_DEPTH))

scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world.color = (0.72, 0.72, 0.72)

if scene.render.engine == "CYCLES":
    scene.cycles.samples = SAMPLES

light_data = bpy.data.lights.new("ABC_CHECK_SUN", type="SUN")
light_data.energy = 3.0
light = bpy.data.objects.new("ABC_CHECK_SUN", light_data)
bpy.context.collection.objects.link(light)
light.rotation_euler = (math.radians(45), 0, math.radians(35))

mesh_objs = [o for o in scene.objects if o.type == "MESH"]

def show_only(names):
    for o in mesh_objs:
        o.hide_render = (o.name not in names)
        o.hide_viewport = (o.name not in names)
    bpy.context.view_layer.update()

print("=== ABC ONLY CHECK ===")
print("A scale/right/up/depth:", A_SCALE, A_RIGHT, A_UP, A_DEPTH)
print("B right/up/depth:", B_RIGHT, B_UP, B_DEPTH)
print("C rot/right/up/depth:", C_RX, C_RY, C_RZ, C_RIGHT, C_UP, C_DEPTH)
print("camera:", CAM_DIST, CAM_PITCH_DEG, CAM_LOOK_Z, LENS)

show_only({"Object_A"})
render_still(os.path.join(OUT_DIR, "00_A_only.png"))

show_only({"Object_B"})
render_still(os.path.join(OUT_DIR, "01_B_only.png"))

show_only({"Object_C"})
render_still(os.path.join(OUT_DIR, "02_C_only.png"))

show_only({"Object_A", "Object_B", "Object_C"})
render_still(os.path.join(OUT_DIR, "03_ABC_only_no_bg.png"))

print("[DONE]", OUT_DIR)
