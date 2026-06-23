import bpy
import os
import math
from mathutils import Vector

ROOT = "/root/autodl-tmp/cv_final/final_assets"
OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", os.environ.get("TAG", "A_grid_preview"))
os.makedirs(OUT_DIR, exist_ok=True)

PATH_A = os.environ.get(
    "A_PATH",
    os.path.join(ROOT, "fusion_scene_strict", "A_clean", "A_clean_top1.ply")
)

RES_X = int(os.environ.get("RES_X", "1280"))
RES_Y = int(os.environ.get("RES_Y", "720"))
SAMPLES = int(os.environ.get("SAMPLES", "16"))

def parse_list(name, default):
    s = os.environ.get(name, default)
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals

A_RX_LIST = parse_list("A_RX_LIST", "0,180")
A_RY_LIST = parse_list("A_RY_LIST", "-20,0,20")
A_RZ_LIST = parse_list("A_RZ_LIST", "210,240,270,300")

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

def import_ply(filepath):
    before = set(bpy.data.objects.keys())
    ok = False

    try:
        bpy.ops.wm.ply_import(filepath=filepath)
        ok = True
    except Exception:
        pass

    if not ok:
        try:
            bpy.ops.import_mesh.ply(filepath=filepath)
            ok = True
        except Exception:
            pass

    if not ok:
        raise RuntimeError(f"Failed to import ply: {filepath}")

    after = set(bpy.data.objects.keys())
    new_names = sorted(list(after - before))
    objs = [bpy.data.objects[n] for n in new_names if bpy.data.objects[n].type == 'MESH']
    print("[IMPORTED]", filepath, [o.name for o in objs])
    return objs

def join_objects(objs, name):
    if len(objs) == 0:
        raise RuntimeError("No objects to join")
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

def set_origin_to_geometry(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

def bbox_world_obj(o):
    pts = [o.matrix_world @ Vector(c) for c in o.bound_box]
    bmin = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return bmin, bmax

def look_at(obj, target):
    direction = target - obj.location
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()

def setup_vertex_color_material(obj):
    if obj.type != 'MESH':
        return

    mesh = obj.data

    color_attr_name = None

    if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
        color_attr_name = mesh.color_attributes[0].name
    elif hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        color_attr_name = mesh.vertex_colors[0].name

    mat = bpy.data.materials.new(name="A_VCol_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in nodes:
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 0.75
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.2

    if color_attr_name is not None:
        attr = nodes.new(type="ShaderNodeAttribute")
        attr.location = (-250, 0)
        attr.attribute_name = color_attr_name
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.65, 0.65, 0.65, 1.0)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

def setup_scene_render():
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100

    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        scene.render.engine = 'BLENDER_EEVEE'

    if scene.render.engine in {'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'}:
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = SAMPLES

    scene.view_settings.view_transform = 'Standard'
    try:
        scene.view_settings.look = 'None'
    except Exception:
        pass
    scene.view_settings.exposure = -0.5
    scene.view_settings.gamma = 1.0

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.72, 0.72, 0.72, 1.0)
        bg.inputs[1].default_value = 0.6

def setup_lights():
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 6))
    sun = bpy.context.object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(35), 0, math.radians(25))

    bpy.ops.object.light_add(type='AREA', location=(-2, -4, 3))
    area = bpy.context.object
    area.data.energy = 2500
    area.data.shape = 'RECTANGLE'
    area.data.size = 4
    area.data.size_y = 4
    look_at(area, Vector((0, 0, 0.5)))

def setup_ground():
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.02))
    plane = bpy.context.object
    plane.name = "Ground"

    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.86, 0.86, 0.86, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.95
    plane.data.materials.append(mat)

def setup_camera_for_obj(obj):
    bmin, bmax = bbox_world_obj(obj)
    focus = (bmin + bmax) * 0.5
    extent = bmax - bmin
    max_dim = max(extent.x, extent.y, extent.z, 0.5)

    dist = max(3.2, max_dim * 2.2)

    cam_loc = Vector((
        focus.x + 0.8,
        focus.y - dist,
        focus.z + dist * 0.35,
    ))
    look_target = focus + Vector((0, 0, max(0.15, extent.z * 0.08)))

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    cam.data.lens = 45
    cam.data.clip_start = 0.01
    cam.data.clip_end = 200.0
    look_at(cam, look_target)
    bpy.context.scene.camera = cam

    print("cam loc:", cam_loc)
    print("look target:", look_target)

def render_one(obj, rx, ry, rz):
    obj.rotation_euler = (
        math.radians(rx),
        math.radians(ry),
        math.radians(rz),
    )

    name = f"A_rx{int(rx)}_ry{int(ry)}_rz{int(rz)}.png"
    path = os.path.join(OUT_DIR, name)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[SAVED]", path)
clear_scene()
setup_scene_render()
setup_lights()
setup_ground()

print("[INFO] importing:", PATH_A)
obj_a = join_objects(import_ply(PATH_A), "Object_A")
set_origin_to_geometry(obj_a)

obj_a.location = (0, 0, 0)

setup_vertex_color_material(obj_a)
setup_camera_for_obj(obj_a)

render_one(obj_a, 0, 0, 0)

for rx in A_RX_LIST:
    for ry in A_RY_LIST:
        for rz in A_RZ_LIST:
            render_one(obj_a, rx, ry, rz)

blend_path = os.path.join(OUT_DIR, "A_grid_preview.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("[BLEND]", blend_path)
print("[DONE]")
