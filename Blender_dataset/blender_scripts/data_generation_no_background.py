# ----------------------------------------------------
# Window Dataset — Multi-Window, Skew, In-FOV, Random Occlusions
# + RGB (images/), UNION masks (masks/), Keypoints (keypoints/)
#  - Masks do NOT cancel when windows overlap
#  - Occluder cubes placed at random locations in the image (board area)
#  - 4 keypoints per visible window (TL, TR, BR, BL) in pixel coords
# ----------------------------------------------------
import bpy, bmesh, os, math, random, json
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ===== EDIT THESE PATHS =====
WINDOW_IMAGE_DIR   = r"C:\Users\ShakthiBala\Downloads"
WINDOW_IMAGE_FILE  = "Window_texture.png"
OUTPUT_DIR         = r"C:\Blender_dataset\window_dataset"
# ============================

# ===== DATASET / SCENE SETTINGS =====
SEED               = 11
IMG_RES            = (640, 648)
NUM_IMAGES         = 2000

# Camera (perpendicular to a board at Z=0)
CAMERA_DIST_MIN    = 3.8
CAMERA_DIST_MAX    = 6.2
CAMERA_LENS_MM     = 45.0

# Windows
MAX_WINDOWS        = 5
WINDOW_SIZE_RANGE  = (0.50, 1.10)     # meters (square side)
MIN_SKEW_DEG       = 6.0
MAX_SKEW_DEG       = 15.0
BOARD_HALF_EXTENTS = (1.6, 1.2)       # placement bounds on the board (x,y) in meters

# Occluders (cubes) placed at random spots in the image (board area)
NUM_OCCLUDERS_MIN  = 1
NUM_OCCLUDERS_MAX  = 4
OCCLUDER_SIZE_RANGE= (0.12, 0.40)
OCCLUDER_Z_OFFSET  = (0.3, 2.0)       # distance in front of board toward camera

# Random lights per frame
NUM_RANDOM_LIGHTS  = 4
LIGHT_POS_RANGE    = { "x": (-3.5, 3.5), "y": (-3.5, 3.5), "z": (0.6, 3.8) }
LIGHT_ENERGY_RANGE = (500.0, 2200.0)
LIGHT_SIZE_RANGE   = (0.18, 0.70)

# If source image has no alpha, use luminance key for the hollow:
WHITE_THRESH       = 0.97
# ====================================

random.seed(SEED)

# ---------- Paths ----------
IMG_DIR  = os.path.join(OUTPUT_DIR, "images")
MSK_DIR  = os.path.join(OUTPUT_DIR, "masks")
KPT_DIR  = os.path.join(OUTPUT_DIR, "keypoints")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MSK_DIR, exist_ok=True)
os.makedirs(KPT_DIR, exist_ok=True)

# ---------- Fresh Scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x, scene.render.resolution_y = IMG_RES
scene.render.resolution_percentage = 100
scene.use_nodes = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_depth = '8'
# Neutral color management
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# ---------- Collections ----------
root = scene.collection
def mk_coll(name):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name)
        root.children.link(c)
    return c

coll_board    = mk_coll("Board")       # board & windows
coll_lights   = mk_coll("RndLights")   # random lights
coll_occluder = mk_coll("Occluders")   # random cubes

# ---------- Camera ----------
cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
root.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.data.lens = CAMERA_LENS_MM
cam_obj.data.clip_start = 0.01
cam_obj.data.clip_end   = 100.0

# ---------- Board (reference plane at Z=0) ----------
def create_board(size=(3.8, 3.0), name="BoardPlane"):
    me = bpy.data.meshes.new(name + "_Mesh")
    ob = bpy.data.objects.new(name, me)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    bm.to_mesh(me); bm.free()
    ob.scale = (size[0]*0.5, size[1]*0.5, 1.0)
    ob.location = (0.0, 0.0, 0.0)
    coll_board.objects.link(ob)
    ob.hide_render = True  # reference only
    return ob

board = create_board(size=(BOARD_HALF_EXTENTS[0]*2.4, BOARD_HALF_EXTENTS[1]*2.4))

# ---------- Window Mesh ----------
def create_window_plane(size=0.8, name="Window"):
    me = bpy.data.meshes.new(name + "_Mesh")
    ob = bpy.data.objects.new(name, me)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    bm.to_mesh(me); bm.free()

    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv = me.uv_layers.active
    uvs = [(0,0),(1,0),(1,1),(0,1)]
    idx = 0
    for p in me.polygons:
        for li in p.loop_indices:
            uv.data[li].uv = uvs[idx % 4]; idx += 1

    ob.scale = (size*0.5, size*0.5, 1.0)
    coll_board.objects.link(ob)
    return ob

# ---------- Materials ----------
def set_transparency_flags(mat):
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND'    # enable alpha blending
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    if hasattr(mat, "shadow_method"):
        try: mat.shadow_method = 'NONE'
        except Exception: pass

def build_window_rgb_material(img_path, thr=WHITE_THRESH):
    """Emission+Transparent: frame visible, center true alpha hole."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image missing: {img_path}")

    mat = bpy.data.materials.new("Window_RGB_EmitTransp")
    mat.use_nodes = True
    set_transparency_flags(mat)

    nt = mat.node_tree
    n  = nt.nodes
    l  = nt.links
    n.clear()

    out  = n.new("ShaderNodeOutputMaterial"); out.location = (820, 0)
    mix  = n.new("ShaderNodeMixShader");       mix.location = (580, 0)
    trn  = n.new("ShaderNodeBsdfTransparent"); trn.location = (340, 150)
    emi  = n.new("ShaderNodeEmission");        emi.location = (340, -120)
    emi.inputs["Strength"].default_value = 1.0

    tex  = n.new("ShaderNodeTexImage");        tex.location = (-420, 0)
    tex.image = bpy.data.images.load(img_path, check_existing=True)
    if hasattr(tex, "alpha_mode"): tex.alpha_mode = 'STRAIGHT'

    # Use alpha if present; else luminance threshold
    try:
        has_alpha = getattr(tex.image, "channels", 4) >= 4
    except Exception:
        has_alpha = True

    if has_alpha and "Alpha" in tex.outputs.keys():
        fac_socket = tex.outputs["Alpha"]
    else:
        rgb2bw = n.new("ShaderNodeRGBToBW");   rgb2bw.location = (-180, 0)
        gt     = n.new("ShaderNodeMath");      gt.location     = (  20, 0); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = thr
        inv    = n.new("ShaderNodeMath");      inv.location    = ( 170, 0); inv.operation = 'SUBTRACT';     inv.inputs[0].default_value = 1.0
        l.new(tex.outputs["Color"], rgb2bw.inputs["Color"])
        l.new(rgb2bw.outputs["Val"], gt.inputs[0])
        l.new(gt.outputs["Value"],   inv.inputs[1])   # center->0, frame->1
        fac_socket = inv.outputs["Value"]

    l.new(tex.outputs["Color"], emi.inputs["Color"])
    l.new(fac_socket,           mix.inputs["Fac"])
    l.new(trn.outputs["BSDF"],  mix.inputs[1])
    l.new(emi.outputs["Emission"], mix.inputs[2])
    l.new(mix.outputs["Shader"], out.inputs["Surface"])

    return mat

def build_window_mask_material(img_path, thr=WHITE_THRESH):
    """UNION mask material: Transparent center (alpha=0), White frame (alpha=1).
       Using alpha blend so overlaps add (no cancellation)."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image missing: {img_path}")

    mat = bpy.data.materials.new("Window_MASK_Union")
    mat.use_nodes = True
    set_transparency_flags(mat)

    nt = mat.node_tree
    n  = nt.nodes
    l  = nt.links
    n.clear()

    out  = n.new("ShaderNodeOutputMaterial"); out.location = (820, 0)
    mix  = n.new("ShaderNodeMixShader");       mix.location = (580, 0)
    trn  = n.new("ShaderNodeBsdfTransparent"); trn.location = (340, 150)
    emi  = n.new("ShaderNodeEmission");        emi.location = (340, -120)
    emi.inputs["Color"].default_value = (1,1,1,1)
    emi.inputs["Strength"].default_value = 1.0

    tex  = n.new("ShaderNodeTexImage");        tex.location = (-420, 0)
    tex.image = bpy.data.images.load(img_path, check_existing=True)
    if hasattr(tex, "alpha_mode"): tex.alpha_mode = 'STRAIGHT'

    try:
        has_alpha = getattr(tex.image, "channels", 4) >= 4
    except Exception:
        has_alpha = True

    if has_alpha and "Alpha" in tex.outputs.keys():
        fac_socket = tex.outputs["Alpha"]
    else:
        rgb2bw = n.new("ShaderNodeRGBToBW");   rgb2bw.location = (-180, 0)
        gt     = n.new("ShaderNodeMath");      gt.location     = (  20, 0); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = thr
        inv    = n.new("ShaderNodeMath");      inv.location    = ( 170, 0); inv.operation = 'SUBTRACT';     inv.inputs[0].default_value = 1.0
        l.new(tex.outputs["Color"], rgb2bw.inputs["Color"])
        l.new(rgb2bw.outputs["Val"], gt.inputs[0])
        l.new(gt.outputs["Value"],   inv.inputs[1])  # 1 on frame, 0 on center
        fac_socket = inv.outputs["Value"]

    l.new(fac_socket, mix.inputs["Fac"])
    l.new(trn.outputs["BSDF"],  mix.inputs[1])
    l.new(emi.outputs["Emission"], mix.inputs[2])
    l.new(mix.outputs["Shader"], out.inputs["Surface"])

    return mat

def black_emission_material():
    m = bpy.data.materials.new("Black_Emission")
    m.use_nodes = True
    n = m.node_tree.nodes; l = m.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial"); out.location = (300,0)
    emi = n.new("ShaderNodeEmission");       emi.location = (100,0)
    emi.inputs["Color"].default_value = (0,0,0,1)
    emi.inputs["Strength"].default_value = 1.0
    l.new(emi.outputs["Emission"], out.inputs["Surface"])
    return m

img_path = os.path.join(WINDOW_IMAGE_DIR, WINDOW_IMAGE_FILE)
win_rgb_mat  = build_window_rgb_material(img_path, WHITE_THRESH)
win_mask_mat = build_window_mask_material(img_path, WHITE_THRESH)
black_mat    = black_emission_material()

# ---------- Geometry helpers ----------
def plane_world_corners(ob):
    return [ob.matrix_world @ v.co for v in ob.data.vertices[:4]]

def all_corners_in_frame(cam, corners):
    for wp in corners:
        uvw = world_to_camera_view(scene, cam, wp)
        if not (0.0 <= uvw.x <= 1.0 and 0.0 <= uvw.y <= 1.0 and uvw.z > 0.0):
            return False
    return True

# reorder to TL, TR, BR, BL in pixel space
def order_tl_tr_br_bl(pxpts):
    # pxpts: list of (x,y)
    # blender y increases up, image pixels y increases down, but we clamp later
    # we can sort by y desc (top first) and x asc
    s = sorted(pxpts, key=lambda p: (-p[1], p[0]))
    top = sorted(s[:2], key=lambda p: p[0])
    bot = sorted(s[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]

def project_to_pixels(cam, world_points, W, H):
    out = []
    for wp in world_points:
        uvw = world_to_camera_view(scene, cam, wp)
        x = int(round(uvw.x * (W-1)))
        y = int(round((1.0 - uvw.y) * (H-1)))  # flip y to image space
        x = max(0, min(W-1, x))
        y = max(0, min(H-1, y))
        out.append((x, y))
    return out

# ---------- Board / Camera ----------
def board_normal_world(board_obj):
    return (board_obj.matrix_world.to_3x3() @ Vector((0,0,1))).normalized()

def place_camera_perpendicular(cam, board_obj, dist):
    target = board_obj.matrix_world.translation.copy()
    n = board_normal_world(board_obj)
    cam.location = target + n * dist
    forward = (target - cam.location).normalized()
    cam.rotation_euler = forward.to_track_quat('-Z', 'Y').to_euler()

# ---------- Lights ----------
def clear_random_lights():
    for o in list(coll_lights.objects):
        bpy.data.objects.remove(o, do_unlink=True)

def spawn_random_lights(n=NUM_RANDOM_LIGHTS):
    clear_random_lights()
    for i in range(n):
        ltype = random.choice(['POINT', 'AREA'])
        ldata = bpy.data.lights.new(f"RndLightData_{i}", ltype)
        lobj  = bpy.data.objects.new(f"RndLight_{i}", ldata)
        lobj.location = (
            random.uniform(*LIGHT_POS_RANGE["x"]),
            random.uniform(*LIGHT_POS_RANGE["y"]),
            random.uniform(*LIGHT_POS_RANGE["z"]),
        )
        ldata.energy = random.uniform(*LIGHT_ENERGY_RANGE)
        if ltype == 'AREA':
            ldata.shape = 'SQUARE'
            ldata.size  = random.uniform(*LIGHT_SIZE_RANGE)
        ldata.color = (
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
        )
        coll_lights.objects.link(lobj)

# ---------- Occluders (random in image/board area) ----------
def clear_occluders():
    for o in list(coll_occluder.objects):
        bpy.data.objects.remove(o, do_unlink=True)

def spawn_occluders_random(board_obj, n_min=NUM_OCCLUDERS_MIN, n_max=NUM_OCCLUDERS_MAX):
    """Place cubes at random (x,y) on the board area, pulled toward camera."""
    count = random.randint(n_min, n_max)
    n = board_normal_world(board_obj)
    board_center = board_obj.matrix_world.translation.copy()
    for i in range(count):
        me = bpy.data.meshes.new(f"Occ_{i}_Mesh")
        ob = bpy.data.objects.new(f"Occ_{i}", me)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(me); bm.free()

        s = random.uniform(*OCCLUDER_SIZE_RANGE)
        ob.scale = (s, s, s)

        depth = random.uniform(*OCCLUDER_Z_OFFSET)
        base_pos = board_center - n * depth  # in front of board toward camera

        # random XY on board area
        px = random.uniform(-BOARD_HALF_EXTENTS[0], BOARD_HALF_EXTENTS[0])
        py = random.uniform(-BOARD_HALF_EXTENTS[1], BOARD_HALF_EXTENTS[1])
        pos = Vector((px, py, base_pos.z))

        ob.location = pos
        coll_occluder.objects.link(ob)

# ---------- Window placement with in-FOV ----------
def sample_window_pose_and_size():
    size = random.uniform(*WINDOW_SIZE_RANGE)
    def random_skew():
        rx = random.uniform(-MAX_SKEW_DEG, MAX_SKEW_DEG)
        ry = random.uniform(-MAX_SKEW_DEG, MAX_SKEW_DEG)
        rz = random.uniform(-MAX_SKEW_DEG, MAX_SKEW_DEG)
        if max(abs(rx), abs(ry), abs(rz)) < MIN_SKEW_DEG:
            axis = random.choice([0,1,2])
            val  = MIN_SKEW_DEG * (1 if random.random()>0.5 else -1)
            if axis==0: rx = val
            elif axis==1: ry = val
            else: rz = val
        return (rx, ry, rz)
    rx, ry, rz = random_skew()
    cx = random.uniform(-BOARD_HALF_EXTENTS[0], BOARD_HALF_EXTENTS[0])
    cy = random.uniform(-BOARD_HALF_EXTENTS[1], BOARD_HALF_EXTENTS[1])
    cz = 0.0
    return size, (cx, cy, cz), (rx, ry, rz)

def ensure_in_fov(cam, win_obj, max_tries=100):
    for _ in range(max_tries):
        size, loc, rdeg = sample_window_pose_and_size()
        win_obj.scale = (size*0.5, size*0.5, 1.0)
        win_obj.location = Vector(loc)
        win_obj.rotation_euler = tuple(math.radians(a) for a in rdeg)
        if all_corners_in_frame(cam, plane_world_corners(win_obj)):
            return True, (loc[0], loc[1])
    return False, (0.0, 0.0)

# ---------- Window pool ----------
def get_or_make_windows(n_needed):
    existing = [o for o in coll_board.objects if o.name.startswith("Window_")]
    while len(existing) < n_needed:
        ob = create_window_plane(size=random.uniform(*WINDOW_SIZE_RANGE), name=f"Window_{len(existing)+1}")
        ob.data.materials.clear()
        ob.data.materials.append(win_rgb_mat)
        existing.append(ob)
    for o in existing:
        o.hide_render = True
        o.hide_viewport = True
    return existing

# ---------- Render Loop ----------
for f in range(1, NUM_IMAGES+1):
    # Camera perpendicular at random distance
    dist = random.uniform(CAMERA_DIST_MIN, CAMERA_DIST_MAX)
    place_camera_perpendicular(cam_obj, board, dist)

    # Lights
    spawn_random_lights(NUM_RANDOM_LIGHTS)

    # Windows
    n_windows = random.randint(1, MAX_WINDOWS)
    windows = get_or_make_windows(n_windows)

    visible_windows = []
    for w in windows[:n_windows]:
        w.hide_render = False
        w.hide_viewport = False
        ok, _center = ensure_in_fov(cam_obj, w, max_tries=140)
        if not ok:
            w.scale = (0.35, 0.35, 1.0)
            w.location = Vector((0.0, 0.0, 0.0))
            w.rotation_euler = (0.0, 0.0, 0.0)
        visible_windows.append(w)

    # Occluders: random spots in image
    clear_occluders()
    spawn_occluders_random(board)

    # --------- RGB pass ---------
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    for w in visible_windows:
        w.data.materials.clear()
        w.data.materials.append(win_rgb_mat)
    scene.render.filepath = os.path.join(IMG_DIR, f"img_{f:04d}.png")
    bpy.ops.render.render(write_still=True)

    # --------- MASK pass ---------
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'BW'
    # world black
    if not bpy.data.worlds:
        bpy.data.worlds.new("MaskWorld")
    scene.world = bpy.data.worlds[0]
    scene.world.use_nodes = True
    wn = scene.world.node_tree.nodes; wl = scene.world.node_tree.links
    wn.clear()
    w_out = wn.new("ShaderNodeOutputWorld")
    w_bg  = wn.new("ShaderNodeBackground")
    w_bg.inputs["Color"].default_value = (0,0,0,1)
    wl.new(w_bg.outputs["Background"], w_out.inputs["Surface"])
    # swap materials for mask
    for w in visible_windows:
        w.data.materials.clear()
        w.data.materials.append(win_mask_mat)
    for o in coll_occluder.objects:
        if o.type == 'MESH':
            o.data.materials.clear()
            o.data.materials.append(black_mat)

    scene.render.filepath = os.path.join(MSK_DIR, f"mask_{f:04d}.png")
    bpy.ops.render.render(write_still=True)

    # --------- KEYPOINTS pass (just compute, no render) ---------
    kpt_records = []
    for w in visible_windows:
        # project 4 corners
        corners_world = plane_world_corners(w)
        pxpts = project_to_pixels(cam_obj, corners_world, IMG_RES[0], IMG_RES[1])
        pxpts = order_tl_tr_br_bl(pxpts)  # TL, TR, BR, BL
        kpt_records.append({
            "object_name": w.name,
            "keypoints": [{"x": x, "y": y} for (x, y) in pxpts]
        })
    kpt_payload = {
        "frame": f,
        "image": f"img_{f:04d}.png",
        "mask":  f"mask_{f:04d}.png",
        "resolution": {"width": IMG_RES[0], "height": IMG_RES[1]},
        "objects": kpt_records
    }
    with open(os.path.join(KPT_DIR, f"kpt_{f:04d}.json"), "w") as jf:
        json.dump(kpt_payload, jf, indent=2)

print("[DONE] ✅ images/, masks/, keypoints/ saved.")
