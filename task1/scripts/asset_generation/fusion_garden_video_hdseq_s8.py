import bpy
import os
import math
import addon_utils
from mathutils import Vector

BG_PATH = "/root/autodl-tmp/cv_final/final_assets/background_garden_2dgs/train/ours_30000/fuse_post.ply"
A_PATH  = "/root/autodl-tmp/cv_final/final_assets/object_A_2dgs/train/ours_30000/fuse_post.ply"
B_PATH  = "/root/autodl-tmp/cv_final/final_assets/object_B_threestudio/model.obj"
C_PATH  = "/root/autodl-tmp/cv_final/final_assets/object_C_magic123/mesh.obj"

OUT_DIR   = "/root/autodl-tmp/cv_final/final_assets/fusion_scene_final"
OUT_MP4   = os.path.join(OUT_DIR, "fusion_garden_roam_hd_720p.mp4")
OUT_PNG   = os.path.join(OUT_DIR, "fusion_cover_hd_720p_s8.png")
OUT_BLEND = os.path.join(OUT_DIR, "fusion_scene_hd_720p_s8.blend")

os.makedirs(OUT_DIR, exist_ok=True)

RENDER_X = 1280
RENDER_Y = 720
FPS = 24
FRAME_START = 1
FRAME_END = 144  
ROT_BG = (math.radians(90), 0.0, 0.0)
ROT_A  = (math.radians(90), 0.0, 0.0)
ROT_B  = (0.0, 0.0, 0.0)
ROT_C  = (0.0, 0.0, 0.0)

POS_A = (-2.5,  0.8, 0.0)
POS_B = ( 0.0,  0.4, 0.0)
POS_C = ( 2.5,  0.8, 0.0)

SIZE_A = 1.20
SIZE_B = 1.10
SIZE_C = 1.10

CAM_RADIUS = 7.5
CAM_HEIGHT = 2.8
TARGET_HEIGHT = 0.85

addon_utils.enable("io_mesh_ply", default_set=True)
addon_utils.enable("io_scene_obj", default_set=True)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for datablock in list(bpy.data.meshes):
    if datablock.users == 0:
        bpy.data.meshes.remove(datablock)
for datablock in list(bpy.data.materials):
    if datablock.users == 0:
        bpy.data.materials.remove(datablock)
for datablock in list(bpy.data.images):
    if datablock.users == 0:
        bpy.data.images.remove(datablock)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 8
scene.eevee.use_gtao = False
scene.eevee.gtao_factor = 1.2
scene.eevee.use_bloom = False
scene.eevee.bloom_intensity = 0.02
scene.render.resolution_x = RENDER_X
scene.render.resolution_y = RENDER_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = FRAME_START
scene.frame_end = FRAME_END

def imported_meshes(before_names):
    new_objs = [o for o in bpy.data.objects if o.name not in before_names]
    return [o for o in new_objs if o.type == 'MESH']

def import_ply(path):
    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_mesh.ply(filepath=path)
    return imported_meshes(before)

def import_obj(path):
    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=path)
    return imported_meshes(before)

def select_only(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    if objs:
        bpy.context.view_layer.objects.active = objs[0]

def join_objects(objs, name):
    if len(objs) == 1:
        objs[0].name = name
        return objs[0]
    select_only(objs)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    return obj

def bounds_world(obj):
    pts = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return min_v, max_v

def bounds_multi(objs):
    pts = []
    for obj in objs:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return min_v, max_v

def normalize_and_place(objs, target_size, target_loc):
    bpy.context.view_layer.update()
    min_v, max_v = bounds_multi(objs)
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 1e-8:
        s = target_size / max_dim
        for o in objs:
            o.scale = (o.scale.x * s, o.scale.y * s, o.scale.z * s)

    bpy.context.view_layer.update()
    min_v, max_v = bounds_multi(objs)
    center = (min_v + max_v) / 2.0
    bottom = min_v.z

    dx = target_loc[0] - center.x
    dy = target_loc[1] - center.y
    dz = target_loc[2] - bottom

    for o in objs:
        o.location.x += dx
        o.location.y += dy
        o.location.z += dz

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def make_principled(name, base=(0.7, 0.7, 0.7, 1.0), rough=0.55, spec=0.25):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular"].default_value = spec
    return mat

def assign_material_if_empty(objs, mat):
    for o in objs:
        if o.type != 'MESH':
            continue
        if len(o.data.materials) == 0:
            o.data.materials.append(mat)

def add_text_label(text, location, size=0.22):
    bpy.ops.object.text_add(location=location)
    t = bpy.context.active_object
    t.data.body = text
    t.data.size = size
    t.rotation_euler = (math.radians(90), 0, 0)
    mat = make_principled("LabelMat_" + text, base=(0.05, 0.05, 0.05, 1.0), rough=0.7, spec=0.1)
    if len(t.data.materials) == 0:
        t.data.materials.append(mat)
    else:
        t.data.materials[0] = mat
    return t

world = bpy.data.worlds["World"]
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs[1].default_value = 0.55

bpy.ops.object.light_add(type='SUN', location=(0, 0, 8))
sun = bpy.context.active_object
sun.data.energy = 2.3
sun.rotation_euler = (math.radians(35), math.radians(0), math.radians(25))

bpy.ops.object.light_add(type='AREA', location=(-5.0, -4.0, 4.5))
key = bpy.context.active_object
key.data.energy = 2600
key.data.size = 5.0
look_at(key, (0, 0, 0.8))

bpy.ops.object.light_add(type='AREA', location=(4.0, -3.0, 3.5))
fill = bpy.context.active_object
fill.data.energy = 1700
fill.data.size = 4.0
look_at(fill, (0, 0, 0.8))

bg_objs = import_ply(BG_PATH)
for o in bg_objs:
    o.rotation_euler = ROT_BG

assign_material_if_empty(bg_objs, make_principled("GardenFallback", base=(0.66, 0.74, 0.63, 1.0), rough=0.95, spec=0.08))
bg_obj = join_objects(bg_objs, "GardenBG")

bpy.context.view_layer.update()
bg_min, bg_max = bounds_world(bg_obj)
bg_center = (bg_min + bg_max) / 2.0
bg_ground = bg_min.z

objs_a = import_ply(A_PATH)
for o in objs_a:
    o.rotation_euler = ROT_A
assign_material_if_empty(objs_a, make_principled("MatA", base=(0.72, 0.72, 0.76, 1.0), rough=0.6, spec=0.2))
normalize_and_place(
    objs_a,
    target_size=SIZE_A,
    target_loc=(bg_center.x + POS_A[0], bg_center.y + POS_A[1], bg_ground + POS_A[2] + 0.03)
)
objA = join_objects(objs_a, "ObjectA")
objA.rotation_euler.z += math.radians(8)

objs_b = import_obj(B_PATH)
for o in objs_b:
    o.rotation_euler = ROT_B
assign_material_if_empty(objs_b, make_principled("MatB", base=(0.63, 0.46, 0.27, 1.0), rough=0.62, spec=0.22))
normalize_and_place(
    objs_b,
    target_size=SIZE_B,
    target_loc=(bg_center.x + POS_B[0], bg_center.y + POS_B[1], bg_ground + POS_B[2] + 0.03)
)
objB = join_objects(objs_b, "ObjectB")
objB.rotation_euler.z += math.radians(-10)

objs_c = import_obj(C_PATH)
for o in objs_c:
    o.rotation_euler = ROT_C
normalize_and_place(
    objs_c,
    target_size=SIZE_C,
    target_loc=(bg_center.x + POS_C[0], bg_center.y + POS_C[1], bg_ground + POS_C[2] + 0.03)
)
objC = join_objects(objs_c, "ObjectC")
objC.rotation_euler.z += math.radians(8)

add_text_label("A: 2DGS (video)",    (bg_center.x - 3.3, bg_center.y - 1.1, bg_ground + 0.02), size=0.18)
add_text_label("B: threestudio",     (bg_center.x - 0.9, bg_center.y - 1.1, bg_ground + 0.02), size=0.18)
add_text_label("C: Magic123",        (bg_center.x + 1.8, bg_center.y - 1.1, bg_ground + 0.02), size=0.18)

target = bpy.data.objects.new("CameraTarget", None)
target.location = (
    bg_center.x,
    bg_center.y + 0.35,
    bg_ground + TARGET_HEIGHT
)
bpy.context.collection.objects.link(target)

bpy.ops.curve.primitive_bezier_circle_add(radius=CAM_RADIUS, location=target.location)
cam_path = bpy.context.active_object
cam_path.name = "CameraPath"
cam_path.scale = (1.15, 0.85, 1.0)

bpy.ops.object.camera_add(location=(target.location.x, target.location.y - CAM_RADIUS, target.location.z + CAM_HEIGHT))
cam = bpy.context.active_object
cam.name = "RoamCamera"
cam.data.lens = 38
scene.camera = cam

trk = cam.constraints.new(type='TRACK_TO')
trk.target = target
trk.track_axis = 'TRACK_NEGATIVE_Z'
trk.up_axis = 'UP_Y'

fp = cam.constraints.new(type='FOLLOW_PATH')
fp.target = cam_path
fp.use_curve_follow = False
fp.use_fixed_location = True
fp.offset_factor = 0.0

scene.frame_set(FRAME_START)
fp.offset_factor = 0.0
fp.keyframe_insert(data_path="offset_factor", frame=FRAME_START)

scene.frame_set(FRAME_END)
fp.offset_factor = 1.0
fp.keyframe_insert(data_path="offset_factor", frame=FRAME_END)

scene.frame_set(FRAME_START)
target.location.z = bg_ground + TARGET_HEIGHT
target.keyframe_insert(data_path="location", frame=FRAME_START)

scene.frame_set((FRAME_START + FRAME_END)//2)
target.location.z = bg_ground + TARGET_HEIGHT + 0.25
target.keyframe_insert(data_path="location", frame=(FRAME_START + FRAME_END)//2)

scene.frame_set(FRAME_END)
target.location.z = bg_ground + TARGET_HEIGHT
target.keyframe_insert(data_path="location", frame=FRAME_END)

for fcurve in cam.animation_data.action.fcurves if cam.animation_data and cam.animation_data.action else []:
    for kp in fcurve.keyframe_points:
        kp.interpolation = 'BEZIER'

if target.animation_data and target.animation_data.action:
    for fcurve in target.animation_data.action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'BEZIER'
scene.frame_set(60)
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUT_PNG
bpy.ops.render.render(write_still=True)


FRAMES_DIR = os.path.join(OUT_DIR, "frames_hd_720p_s8")
os.makedirs(FRAMES_DIR, exist_ok=True)

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = os.path.join(FRAMES_DIR, "frame_")
bpy.ops.render.render(animation=True)

bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

print("Saved cover:", OUT_PNG)
print("Saved frames:", os.path.join(OUT_DIR, "frames_hd_720p"))
print("Saved blend:", OUT_BLEND)
