import os
import math
import bpy
from mathutils import Vector

ROOT = "/root/autodl-tmp/cv_final/final_assets"
A_PATH = os.path.join(ROOT, "object_A_2dgs", "train", "ours_30000", "fuse_post.ply")

OUT_DIR = os.path.join(ROOT, "fusion_scene_strict", "A_clean")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = int(os.environ.get("SAMPLES", "4"))
RENDER_X = int(os.environ.get("RENDER_X", "1280"))
RENDER_Y = int(os.environ.get("RENDER_Y", "720"))
KEEP_TOP_LIST = [1, 3, 6, 10]
CUSTOM_KEEP_RANKS = os.environ.get("KEEP_RANKS", "").strip()

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES
    scene.eevee.use_gtao = True
    scene.eevee.gtao_factor = 1.0
    scene.eevee.use_bloom = False

    scene.render.resolution_x = RENDER_X
    scene.render.resolution_y = RENDER_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.86, 0.87, 0.89, 1)
        bg.inputs[1].default_value = 0.7
    return scene

def import_ply(path):
    before = set(o.name for o in bpy.data.objects)
    print("[IMPORT]", path)
    bpy.ops.import_mesh.ply(filepath=path)
    objs = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
    print("[IMPORTED]", [o.name for o in objs])
    if not objs:
        raise RuntimeError("No mesh imported")
    return objs[0]

def make_vcol_mat(name, layer="Col"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (450, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (180, 0)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular"].default_value = 0.04

    vcol = nt.nodes.new("ShaderNodeVertexColor")
    vcol.location = (-100, 0)
    vcol.layer_name = layer

    nt.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def make_plain_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Specular"].default_value = 0.05
    return mat

def assign_vcol_or_plain(obj):
    layer = None
    if hasattr(obj.data, "vertex_colors") and len(obj.data.vertex_colors) > 0:
        layer = obj.data.vertex_colors[0].name
    if hasattr(obj.data, "color_attributes") and len(obj.data.color_attributes) > 0:
        layer = obj.data.color_attributes[0].name

    obj.data.materials.clear()
    if layer:
        obj.data.materials.append(make_vcol_mat("A_vcol_" + obj.name, layer))
    else:
        obj.data.materials.append(make_plain_mat("A_plain_" + obj.name, (0.72, 0.65, 0.55, 1)))

def bbox(obj):
    bpy.context.view_layer.update()
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

def combined_bbox(objs):
    boxes = [bbox(o) for o in objs]
    return (
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        min(b[4] for b in boxes),
        max(b[5] for b in boxes),
    )

def center_and_scale_objects(objs, target_size=2.4):
    xmin, xmax, ymin, ymax, zmin, zmax = combined_bbox(objs)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    maxdim = max(xmax - xmin, ymax - ymin, zmax - zmin)

    if maxdim < 1e-8:
        return

    s = target_size / maxdim

    for o in objs:
        o.location.x -= cx
        o.location.y -= cy
        o.location.z -= zmin
        o.scale = (o.scale.x * s, o.scale.y * s, o.scale.z * s)

    bpy.context.view_layer.update()

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

def setup_camera_for(objs):
    xmin, xmax, ymin, ymax, zmin, zmax = combined_bbox(objs)
    center = Vector(((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2))
    extent = Vector((xmax-xmin, ymax-ymin, zmax-zmin))
    maxdim = max(extent.x, extent.y, extent.z, 0.5)

    dist = max(3.0, maxdim * 2.4)
    cam_loc = center + Vector((1.0, -dist, dist * 0.45))

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    cam.data.lens = 45
    cam.data.clip_start = 0.01
    cam.data.clip_end = 200
    look_at(cam, center + Vector((0, 0, max(0.1, extent.z * 0.15))))
    bpy.context.scene.camera = cam

def add_lights():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 8))
    sun = bpy.context.object
    sun.data.energy = 2.2
    sun.rotation_euler = (math.radians(45), 0, math.radians(35))

    bpy.ops.object.light_add(type="AREA", location=(0, -4, 4))
    area = bpy.context.object
    area.data.energy = 1800
    area.data.size = 5

def set_only_visible(keep):
    keep_set = set(keep)
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_viewport = o not in keep_set
            o.hide_render = o not in keep_set

def render_candidate(name, keep):
    set_only_visible(keep)
    setup_camera_for(keep)
    path = os.path.join(OUT_DIR, name + ".png")
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[PREVIEW]", path)

def export_candidate(name, keep):
    set_only_visible(keep)
    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.select_set(True)
    bpy.context.view_layer.objects.active = keep[0]

    out = os.path.join(OUT_DIR, name + ".ply")
    try:
        bpy.ops.export_mesh.ply(filepath=out, use_selection=True, use_colors=True)
    except TypeError:
        bpy.ops.export_mesh.ply(filepath=out, use_selection=True)
    print("[EXPORT]", out)

def write_report(comps):
    report = os.path.join(OUT_DIR, "A_components_report.txt")
    with open(report, "w") as f:
        f.write("rank\tname\tverts\tfaces\tbbox_dims\tcenter\n")
        for i, o in enumerate(comps):
            xmin, xmax, ymin, ymax, zmin, zmax = bbox(o)
            dims = (xmax-xmin, ymax-ymin, zmax-zmin)
            center = ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)
            f.write(
                f"{i}\t{o.name}\t{len(o.data.vertices)}\t{len(o.data.polygons)}"
                f"\t({dims[0]:.5f},{dims[1]:.5f},{dims[2]:.5f})"
                f"\t({center[0]:.5f},{center[1]:.5f},{center[2]:.5f})\n"
            )
    print("[REPORT]", report)

clear_scene()
scene = setup_scene()
add_lights()

obj = import_ply(A_PATH)
assign_vcol_or_plain(obj)

print("[INFO] original verts:", len(obj.data.vertices), "faces:", len(obj.data.polygons))
print("[INFO] separating by loose parts... this may take a long time")

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.separate(type="LOOSE")
bpy.ops.object.mode_set(mode="OBJECT")

components = [o for o in bpy.data.objects if o.type == "MESH"]
for o in components:
    assign_vcol_or_plain(o)

components.sort(key=lambda o: (len(o.data.polygons), len(o.data.vertices)), reverse=True)

print("[INFO] num components:", len(components))
write_report(components)

center_and_scale_objects(components, target_size=2.4)

for n in KEEP_TOP_LIST:
    keep = components[:min(n, len(components))]
    name = f"A_clean_top{n}"
    export_candidate(name, keep)
    render_candidate(name + "_preview", keep)

if CUSTOM_KEEP_RANKS:
    ranks = []
    for x in CUSTOM_KEEP_RANKS.split(","):
        x = x.strip()
        if x:
            ranks.append(int(x))
    keep = [components[i] for i in ranks if 0 <= i < len(components)]
    if keep:
        tag = "A_clean_ranks_" + "_".join(str(i) for i in ranks)
        export_candidate(tag, keep)
        render_candidate(tag + "_preview", keep)

blend_path = os.path.join(OUT_DIR, "A_clean_components.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("[BLEND]", blend_path)
print("[DONE] A cleaning candidates generated.")
