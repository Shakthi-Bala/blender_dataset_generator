# ----------------------------------------------------
# Dataset generator (Blender 4.x) — no compositor, no window.view_layer
#  - RGB pass: Emission (sRGB) → true color
#  - MASK pass: toggled collections → BW mask
#  - Keypoints: 4 plane corners per window (pixels)
#  - 1..4 non-overlapping windows, multiple cameras
#  - Output: images/, masks/, keypoints/
#  - Resolution: 640 x 648
# ----------------------------------------------------
import bpy, bmesh, os, random, math, json
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ===== USER SETTINGS =====
WINDOW_PNG_PATH = r"C:\Blender_dataset\assets\window1.png"  # <-- CHANGE
OUTPUT_DIR      = r"C:\Blender_dataset\images"              # <-- CHANGE
NUM_IMAGES      = 6
IMG_RES         = (640, 648)
SEED            = 42

ENGINE          = "BLENDER_EEVEE_NEXT"  # or "CYCLES"
N_CAMERAS       = 3
MAX_WINDOWS     = 4
WHITE_THRESH    = 0.97
# =========================

random.seed(SEED)

# ---------- Folders ----------
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
MSK_DIR = os.path.join(OUTPUT_DIR, "masks")
KPT_DIR = os.path.join(OUTPUT_DIR, "keypoints")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MSK_DIR, exist_ok=True)
os.makedirs(KPT_DIR, exist_ok=True)

# ---------- Scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = ENGINE if ENGINE in {"BLENDER_EEVEE_NEXT","BLENDER_WORKBENCH","CYCLES"} else "BLENDER_EEVEE_NEXT"
scene.render.resolution_x, scene.render.resolution_y = IMG_RES
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.use_nodes = False  # no compositor

# Also enforce RGB output by default
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode  = 'RGB'
scene.render.image_settings.color_depth = '8'

# ---------- Collections ----------
root = scene.collection
def mk_coll(name):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name)
        root.children.link(c)
    return c

windows_RGB   = mk_coll("Windows_RGB")   # window in true color
windows_MASK  = mk_coll("Windows_MASK")  # window mask twins
occluders     = mk_coll("Occluders")     # distractors (RGB only)
backdrop_coll = mk_coll("Backdrop")      # background plane
misc          = mk_coll("Misc")          # light, etc.

# Helper: find a LayerCollection (per-view-layer view onto a Collection)
vl = scene.view_layers[0]  # single view layer strategy
def find_layer_collection(layer_collection, name):
    if layer_collection.collection.name == name:
        return layer_collection
    for child in layer_collection.children:
        r = find_layer_collection(child, name)
        if r: return r
    return None

lc_backdrop = find_layer_collection(vl.layer_collection, "Backdrop")
lc_windows_rgb = find_layer_collection(vl.layer_collection, "Windows_RGB")
lc_windows_msk = find_layer_collection(vl.layer_collection, "Windows_MASK")
lc_occluders   = find_layer_collection(vl.layer_collection, "Occluders")
lc_misc        = find_layer_collection(vl.layer_collection, "Misc")

def set_excludes_for_rgb():
    # RGB wants: backdrop + occluders + windows_RGB + misc
    lc_backdrop.exclude   = False
    lc_occluders.exclude  = False
    lc_windows_rgb.exclude= False
    lc_windows_msk.exclude= True   # hide mask twins
    lc_misc.exclude       = False

def set_excludes_for_mask():
    # MASK wants: backdrop + windows_MASK (no occluders, no rgb windows, no misc)
    lc_backdrop.exclude   = False
    lc_occluders.exclude  = True
    lc_windows_rgb.exclude= True
    lc_windows_msk.exclude= False
    lc_misc.exclude       = True

# ---------- Light ----------
ld = bpy.data.lights.new("KeyLight", 'AREA')
light = bpy.data.objects.new("KeyLight", ld)
root.objects.link(light); root.objects.unlink(light); misc.objects.link(light)

# ---------- Cameras ----------
cams = []
for i in range(N_CAMERAS):
    cdata = bpy.data.cameras.new(f"Camera{i+1}")
    cob   = bpy.data.objects.new(f"Camera{i+1}", cdata)
    root.objects.link(cob)
    cams.append(cob)

# ---------- Mesh helpers ----------
def create_plane(size=5.0, name="Plane"):
    me = bpy.data.meshes.new(name + "_Mesh")
    ob = bpy.data.objects.new(name, me)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size)
    bm.to_mesh(me); bm.free()
    return ob

def add_uv_square(me):
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers.active
    uvs = [(0,0),(1,0),(1,1),(0,1)]
    idx = 0
    for p in me.polygons:
        for li in p.loop_indices:
            uv.data[li].uv = uvs[idx % 4]; idx += 1

# ---------- Background ----------
def bg_material():
    m = bpy.data.materials.new("BG_Mat"); m.use_nodes = True
    N,L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    out  = N.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (200, 0)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    noise = N.new("ShaderNodeTexNoise"); noise.location = (-400, 0)
    noise.inputs["Scale"].default_value = random.uniform(4, 18)
    mix = N.new("ShaderNodeMixRGB"); mix.location = (0, 0)
    mix.inputs["Color1"].default_value = (random.random(), random.random(), random.random(), 1)
    mix.inputs["Color2"].default_value = (random.random(), random.random(), random.random(), 1)
    L.new(noise.outputs["Fac"], mix.inputs["Fac"])
    L.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = random.uniform(0.6, 1.0)
    return m

backdrop = create_plane(5.0, "Backdrop")
backdrop.location = (0,0,-1.0)
backdrop_coll.objects.link(backdrop)
backdrop.data.materials.append(bg_material())
def refresh_backdrop():
    backdrop.data.materials.clear()
    backdrop.data.materials.append(bg_material())

# ---------- Load TWO image datablocks ----------
img_rgb  = bpy.data.images.load(WINDOW_PNG_PATH)
img_mask = bpy.data.images.load(WINDOW_PNG_PATH)
try:
    img_rgb.colorspace_settings.name  = 'sRGB'
    img_mask.colorspace_settings.name = 'Non-Color'
except Exception:
    pass

print(f"[INFO] window.png: size={img_rgb.size[0]}x{img_rgb.size[1]}, channels={getattr(img_rgb,'channels','?')}, colorspace={getattr(img_rgb.colorspace_settings,'name','?')}")

# ---------- Materials ----------
def window_rgb_material(img):
    """True-color (no shading)"""
    m = bpy.data.materials.new("Window_RGB_Mat"); m.use_nodes = True
    N,L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    out  = N.new("ShaderNodeOutputMaterial"); out.location = (420, 0)
    emi  = N.new("ShaderNodeEmission");       emi.location = (220, 0); emi.inputs["Strength"].default_value = 1.0
    tex  = N.new("ShaderNodeTexImage");       tex.location = (-200, 0); tex.image = img; tex.interpolation = 'Smart'
    if hasattr(tex, "alpha_mode"): tex.alpha_mode = 'STRAIGHT'
    uv   = N.new("ShaderNodeTexCoord");       uv.location  = (-400, 0)
    L.new(uv.outputs["UV"], tex.inputs["Vector"])
    L.new(tex.outputs["Color"], emi.inputs["Color"])
    L.new(emi.outputs["Emission"], out.inputs["Surface"])
    if hasattr(m, "blend_method"): m.blend_method = 'OPAQUE'
    return m

def window_mask_material(img, thr=WHITE_THRESH):
    """Emit 1 where NOT white, else 0."""
    m = bpy.data.materials.new("Window_MASK_Mat"); m.use_nodes = True
    N,L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    out  = N.new("ShaderNodeOutputMaterial"); out.location = (420, 0)
    emi  = N.new("ShaderNodeEmission");       emi.location = (220, 0); emi.inputs["Strength"].default_value = 1.0
    tex  = N.new("ShaderNodeTexImage");       tex.location = (-220, 40); tex.image = img
    uv   = N.new("ShaderNodeTexCoord");       uv.location  = (-420, 0)
    bw   = N.new("ShaderNodeRGBToBW");        bw.location  = (-40,  40)
    lt   = N.new("ShaderNodeMath");           lt.location  = (120, 140); lt.operation='LESS_THAN'; lt.inputs[1].default_value = thr
    L.new(uv.outputs["UV"], tex.inputs["Vector"])
    L.new(tex.outputs["Color"], bw.inputs["Color"])
    L.new(bw.outputs["Val"], lt.inputs[0])
    L.new(lt.outputs["Value"], emi.inputs["Color"])
    L.new(emi.outputs["Emission"], out.inputs["Surface"])
    return m

mat_rgb  = window_rgb_material(img_rgb)
mat_mask = window_mask_material(img_mask, WHITE_THRESH)

# ---------- Window pairs ----------
def make_window_pair(name):
    # RGB object
    me_r = bpy.data.meshes.new(name + "_RGB_Mesh")
    ob_r = bpy.data.objects.new(name + "_RGB", me_r)
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5); bm.to_mesh(me_r); bm.free()
    add_uv_square(me_r); ob_r.data.materials.append(mat_rgb); windows_RGB.objects.link(ob_r)
    # MASK twin
    me_m = bpy.data.meshes.new(name + "_MSK_Mesh")
    ob_m = bpy.data.objects.new(name + "_MSK", me_m)
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5); bm.to_mesh(me_m); bm.free()
    add_uv_square(me_m); ob_m.data.materials.append(mat_mask); windows_MASK.objects.link(ob_m)
    return ob_r, ob_m

pairs = [make_window_pair(f"window_{i+1}") for i in range(MAX_WINDOWS)]
def sync_twins():
    for r, m in pairs:
        m.location = r.location.copy()
        m.rotation_euler = r.rotation_euler.copy()
        m.scale = r.scale.copy()

# ---------- Occluders ----------
def cleanup_occluders():
    for o in list(occluders.objects):
        if o.name.startswith("occ_"):
            bpy.data.objects.remove(o, do_unlink=True)

def spawn_occluder():
    name = f"occ_{random.randint(0, 1_000_000)}"
    me = bpy.data.meshes.new(name + "_Mesh")
    ob = bpy.data.objects.new(name, me)
    occluders.objects.link(ob)
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0); bm.to_mesh(me); bm.free()
    ob.scale = (random.uniform(0.05,0.30),)*3
    ob.location = (random.uniform(-0.9,0.9), random.uniform(-0.6,0.6), random.uniform(-0.2,0.5))
    mat = bpy.data.materials.new(name + "Mat"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (random.random(), random.random(), random.random(), 1)
        bsdf.inputs["Roughness"].default_value = 1.0
    ob.data.materials.append(mat)

# ---------- Randomization ----------
def refresh_backdrop_material(): refresh_backdrop()
def randomize_light():
    light.location = (random.uniform(-2.5,2.5), random.uniform(-2.5,2.5), random.uniform(1.5,3.5))
    light.data.energy = random.uniform(600, 1600)
def set_cam_orbit(cam, dist, az, el):
    az = math.radians(az); el = math.radians(el)
    pos = Vector((dist*math.cos(el)*math.cos(az), dist*math.cos(el)*math.sin(az), dist*math.sin(el)))
    cam.location = pos
    cam.rotation_euler = (Vector((0,0,0)) - cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.lens = random.uniform(35, 60)
def randomize_cameras():
    bd = random.uniform(1.6, 2.8); be = random.uniform(20, 55); ba = random.uniform(-180, 180); step = 360.0 / N_CAMERAS
    for i, cam in enumerate(cams):
        set_cam_orbit(cam, bd + random.uniform(-0.2,0.2), ba + i*step + random.uniform(-12,12), be + random.uniform(-6,6))
def sample_nonoverlap(n, d=0.38):
    pts = []; tries = 0
    while len(pts) < n and tries < 250:
        tries += 1
        c = (random.uniform(-0.9,0.9), random.uniform(-0.6,0.6))
        if all(((c[0]-x)**2 + (c[1]-y)**2)**0.5 >= d for x,y in pts): pts.append(c)
    return pts
def place_windows(n):
    centers = sample_nonoverlap(n)
    for idx, (r, m) in enumerate(pairs):
        vis = idx < len(centers)
        for o in (r, m): o.hide_viewport = not vis; o.hide_render = not vis
        if vis:
            rx = math.radians(random.uniform(-25, 25))
            ry = math.radians(random.uniform(-25, 25))
            rz = math.radians(random.uniform(0, 360))
            s  = random.uniform(0.7, 1.25)
            r.rotation_euler = (rx, ry, rz); r.scale = (s, s, s)
            r.location = (centers[idx][0], centers[idx][1], random.uniform(-0.02, 0.02))
    sync_twins()

# ---------- Keypoints ----------
def plane_world_corners(ob):
    me = ob.data
    return [ob.matrix_world @ v.co for v in me.vertices[:4]]

def project_pixels(cam, wpts, W, H):
    out = []
    for wp in wpts:
        uvw = world_to_camera_view(scene, cam, wp)  # x,y in [0,1], y=0 bottom
        x = max(0, min(W-1, round(uvw.x * (W-1))))
        y = max(0, min(H-1, round(uvw.y * (H-1))))
        out.append((int(x), int(y)))
    return out

def order_tl_tr_br_bl(pxpts):
    s = sorted(pxpts, key=lambda p: (-p[1], p[0]))  # top first, left before right
    top = sorted(s[:2], key=lambda p: p[0])
    bot = sorted(s[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]

# ---------- Main ----------
print(f"[INFO] Generating {NUM_IMAGES} samples × {N_CAMERAS} cameras @ {IMG_RES[0]}x{IMG_RES[1]}")
scene.frame_start, scene.frame_end = 1, NUM_IMAGES

for f in range(1, NUM_IMAGES+1):
    scene.frame_set(f)
    refresh_backdrop_material()
    randomize_light()
    randomize_cameras()
    place_windows(random.randint(1, MAX_WINDOWS))

    cleanup_occluders()
    for _ in range(random.randint(0, 2)):
        spawn_occluder()

    for ci, cam in enumerate(cams, start=1):
        scene.camera = cam
        stem = f"c{ci}_{f:05d}"

        # --- RGB PASS ---
        scene.render.image_settings.color_mode = 'RGB'
        set_excludes_for_rgb()
        scene.render.filepath = os.path.join(IMG_DIR, f"img_{stem}.png")
        bpy.ops.render.render(write_still=True)

        # --- MASK PASS ---
        scene.render.image_settings.color_mode = 'BW'
        set_excludes_for_mask()
        scene.render.filepath = os.path.join(MSK_DIR, f"mask_{stem}.png")
        bpy.ops.render.render(write_still=True)

        # --- KEYPOINTS ---
        rec = []
        for r, _m in pairs:
            if r.hide_render: continue
            px = project_pixels(cam, plane_world_corners(r), IMG_RES[0], IMG_RES[1])
            px = order_tl_tr_br_bl(px)
            rec.append({"object_name": r.name, "keypoints": [{"x": x, "y": y} for x, y in px]})
        with open(os.path.join(KPT_DIR, f"img_{stem}.json"), "w") as jf:
            json.dump({
                "image": f"img_{stem}.png",
                "mask":  f"mask_{stem}.png",
                "resolution": {"width": IMG_RES[0], "height": IMG_RES[1]},
                "camera": f"Camera{ci}",
                "objects": rec
            }, jf, indent=2)

print("[DONE] ✅")
