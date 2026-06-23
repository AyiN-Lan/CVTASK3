import bpy, os, math
from mathutils import Vector

TAG = os.environ.get("TAG", "preview10_manual_drop")
ROOT = "/root/autodl-tmp/cv_final/final_assets"
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", TAG)
FRAME_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

RES_X = int(os.environ.get("RES_X", "1280"))
RES_Y = int(os.environ.get("RES_Y", "720"))
SAMPLES = int(os.environ.get("SAMPLES", "32"))
FRAMES = int(os.environ.get("FRAMES", "10"))

CAM_DIST = 6.2
CAM_PITCH_DEG = 18.0
CAM_LOOK_Z = 0.95
LENS = 52.0
CAM_SWAY_X = float(os.environ.get("CAM_SWAY_X", "0.28"))
CAM_BREATH = float(os.environ.get("CAM_BREATH", "0.22"))
CAM_BOB_Z = float(os.environ.get("CAM_BOB_Z", "0.04"))

A_RX, A_RY, A_RZ = 180.0, -30.0, 270.0
A_SCALE = float(os.environ.get("A_SCALE", "0.95"))
A_RIGHT = float(os.environ.get("A_RIGHT", "-2.15"))
A_UP = float(os.environ.get("A_UP", "0.50"))
A_DEPTH = float(os.environ.get("A_DEPTH", "0.25"))

B_RX, B_RY, B_RZ = 0.0, 0.0, 0.0
B_SCALE = float(os.environ.get("B_SCALE", "0.66"))
B_RIGHT = float(os.environ.get("B_RIGHT", "-0.05"))
B_UP = float(os.environ.get("B_UP", "0.90"))
B_DEPTH = float(os.environ.get("B_DEPTH", "0.18"))

C_RX, C_RY, C_RZ = 90.0, -100.0, 165.0
C_SCALE = float(os.environ.get("C_SCALE", "0.92"))
C_RIGHT = float(os.environ.get("C_RIGHT", "1.78"))
C_UP = float(os.environ.get("C_UP", "0.36"))
C_DEPTH = float(os.environ.get("C_DEPTH", "0.18"))

A_DROP_Z = float(os.environ.get("A_DROP_Z", "0.10"))
B_DROP_Z = float(os.environ.get("B_DROP_Z", "0.18"))
C_DROP_Z = float(os.environ.get("C_DROP_Z", "0.08"))

BG_UP = float(os.environ.get("BG_UP", "-1.10"))
BG_DEPTH = float(os.environ.get("BG_DEPTH", "3.40"))
BG_TILT_X = float(os.environ.get("BG_TILT_X", "-100"))
BG_SCALE = float(os.environ.get("BG_SCALE", "1.8"))

BACKDROP_DEPTH = float(os.environ.get("BACKDROP_DEPTH", "8.0"))
BACKDROP_UP = float(os.environ.get("BACKDROP_UP", "0.10"))
BACKDROP_WIDTH = float(os.environ.get("BACKDROP_WIDTH", "18.0"))
BACKDROP_HEIGHT = float(os.environ.get("BACKDROP_HEIGHT", "10.0"))
BACKDROP_R = float(os.environ.get("BACKDROP_R", "0.40"))
BACKDROP_G = float(os.environ.get("BACKDROP_G", "0.56"))
BACKDROP_B = float(os.environ.get("BACKDROP_B", "0.30"))

scene = bpy.context.scene

scene.render.engine = "CYCLES"
scene.cycles.samples = SAMPLES
scene.cycles.preview_samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transparent_max_bounces = 4

try:
    bpy.context.view_layer.cycles.use_denoising = False
except Exception as e:
    print("[WARN] view_layer denoising off failed:", e)

prefs = bpy.context.preferences
prefs.addons["cycles"].preferences.compute_device_type = "CUDA"
prefs.addons["cycles"].preferences.get_devices()
for d in prefs.addons["cycles"].preferences.devices:
    d.use = (d.type == "CUDA")
    print("[CYCLES DEVICE]", d.name, d.type, d.use)
scene.cycles.device = "GPU"

scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"

try:
    scene.view_settings.view_transform = "Standard"
except Exception as e:
    print("[WARN] view_transform:", e)
try:
    scene.view_settings.look = "None"
except Exception as e:
    print("[WARN] look:", e)

scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.color = (BACKDROP_R, BACKDROP_G, BACKDROP_B)
scene.world.use_nodes = True
nodes = scene.world.node_tree.nodes
bg_node = nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (BACKDROP_R, BACKDROP_G, BACKDROP_B, 1.0)
    bg_node.inputs[1].default_value = 0.7

cam = scene.camera
if cam is None:
    cams = [o for o in scene.objects if o.type == "CAMERA"]
    if not cams:
        raise RuntimeError("No camera found")
    cam = cams[0]
    scene.camera = cam

cam.data.lens = LENS
cam.data.clip_start = 0.01
cam.data.clip_end = 300.0

obj_a = bpy.data.objects["Object_A"]
obj_b = bpy.data.objects["Object_B"]
obj_c = bpy.data.objects["Object_C"]
bg = bpy.data.objects["Garden_BG"]

orig = {}
for o in [obj_a, obj_b, obj_c, bg]:
    orig[o.name] = (
        o.location.copy(),
        o.rotation_euler.copy(),
        o.scale.copy()
    )

def restore_obj(o):
    loc, rot, scl = orig[o.name]
    o.location = loc.copy()
    o.rotation_euler = rot.copy()
    o.scale = scl.copy()

for o in scene.objects:
    o.hide_render = False
    o.hide_viewport = False

for old in list(bpy.data.objects):
    if old.name.startswith("CONFIRM_BACKDROP_") or old.name == "BACKDROP_EXT":
        bpy.data.objects.remove(old, do_unlink=True)

if "BACKDROP_EXT_MAT" in bpy.data.materials:
    bpy.data.materials.remove(bpy.data.materials["BACKDROP_EXT_MAT"], do_unlink=True)

sun_data = bpy.data.lights.new("CONFIRM_BACKDROP_SUN", type="SUN")
sun_data.energy = 2.5
sun = bpy.data.objects.new("CONFIRM_BACKDROP_SUN", sun_data)
bpy.context.collection.objects.link(sun)
sun.rotation_euler = (math.radians(45), 0, math.radians(35))

area_data = bpy.data.lights.new("CONFIRM_BACKDROP_AREA", type="AREA")
area_data.energy = 600
area_data.size = 6
area = bpy.data.objects.new("CONFIRM_BACKDROP_AREA", area_data)
bpy.context.collection.objects.link(area)
area.location = (0, -3, 5)

bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
backdrop = bpy.context.active_object
backdrop.name = "BACKDROP_EXT"

mat = bpy.data.materials.new("BACKDROP_EXT_MAT")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (BACKDROP_R, BACKDROP_G, BACKDROP_B, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05
backdrop.data.materials.clear()
backdrop.data.materials.append(mat)

def rot(rx, ry, rz):
    return (math.radians(rx), math.radians(ry), math.radians(rz))

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

def visible_meshes_only(names):
    for o in scene.objects:
        if o.type == "MESH":
            o.hide_render = (o.name not in names)
            o.hide_viewport = o.hide_render
        else:
            o.hide_render = False
            o.hide_viewport = False
    bpy.context.view_layer.update()

def layout_scene(frame_idx, total_frames):
    for o in [obj_a, obj_b, obj_c, bg]:
        restore_obj(o)

    obj_a.rotation_euler = rot(A_RX, A_RY, A_RZ)
    obj_a.scale = (A_SCALE, A_SCALE, A_SCALE)

    obj_b.rotation_euler = rot(B_RX, B_RY, B_RZ)
    obj_b.scale = (B_SCALE, B_SCALE, B_SCALE)

    obj_c.rotation_euler = rot(C_RX, C_RY, C_RZ)
    obj_c.scale = (C_SCALE, C_SCALE, C_SCALE)

    bpy.context.view_layer.update()

    bc_center = (bbox_center(obj_b) + bbox_center(obj_c)) / 2.0
    focus = Vector((bc_center.x, bc_center.y, CAM_LOOK_Z))

    cam_height = CAM_DIST * math.tan(math.radians(CAM_PITCH_DEG))
    t = 0.0 if total_frames <= 1 else frame_idx / (total_frames - 1)
    phase = 2.0 * math.pi * t
    sway_x = CAM_SWAY_X * math.sin(phase)
    breath = CAM_BREATH * math.sin(phase - math.pi / 2.0)
    bob_z = CAM_BOB_Z * math.sin(phase + math.pi / 2.0)

    cam.location = Vector((focus.x + sway_x, focus.y - CAM_DIST + breath, focus.z + cam_height + bob_z))
    look_at(cam, focus)
    bpy.context.view_layer.update()

    R = cam.matrix_world.to_3x3()
    cam_right = (R @ Vector((1, 0, 0))).normalized()
    cam_up = (R @ Vector((0, 1, 0))).normalized()
    cam_forward = (R @ Vector((0, 0, -1))).normalized()

    def target_from_camera(right, up, depth):
        return focus + right * cam_right + up * cam_up + depth * cam_forward

    move_bbox_center_to(obj_a, target_from_camera(A_RIGHT, A_UP, A_DEPTH))
    move_bbox_center_to(obj_b, target_from_camera(B_RIGHT, B_UP, B_DEPTH))
    move_bbox_center_to(obj_c, target_from_camera(C_RIGHT, C_UP, C_DEPTH))

    obj_a.location.z -= A_DROP_Z
    obj_b.location.z -= B_DROP_Z
    obj_c.location.z -= C_DROP_Z
    bpy.context.view_layer.update()

    bg.scale = orig["Garden_BG"][2].copy() * BG_SCALE
    bg.rotation_euler = orig["Garden_BG"][1].copy()
    bg.rotation_euler.x += math.radians(BG_TILT_X)
    bpy.context.view_layer.update()
    move_bbox_center_to(bg, target_from_camera(0.0, BG_UP, BG_DEPTH))

    backdrop.location = target_from_camera(0.0, BACKDROP_UP, BACKDROP_DEPTH)
    backdrop.rotation_euler = cam.rotation_euler
    backdrop.scale = (BACKDROP_WIDTH, BACKDROP_HEIGHT, 1.0)

    visible_meshes_only({"Object_A", "Object_B", "Object_C", "Garden_BG", "BACKDROP_EXT"})

for i in range(FRAMES):
    scene.frame_set(i + 1)
    layout_scene(i, FRAMES)
    out_path = os.path.join(FRAME_DIR, f"frame_{i+1:04d}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print("[SAVED]", out_path)

print("[OUT_DIR]", OUT_DIR)
print("[FRAME_DIR]", FRAME_DIR)
print("[DONE]")
