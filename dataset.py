import os
import sys
import time
import bpy
import random
import math
import json
import argparse
from enum import Enum
from dataclasses import dataclass
from mathutils import Vector, Quaternion
from bl_math import lerp


SATELLITE_NAME = "Body"
CAMERA_NAME = "Camera" # f = 17.5131 mm     w = 11.2512 mm
LIGHT_NAME = "Sun"

NEAR = 1
FAR = 10

OUTPUT_DIRECTORY = "images"

EPSILON = 1e-3

@dataclass
class Frustum :
    left: float
    right: float
    near: float
    far: float
    bottom: float
    top: float

@dataclass
class BBox :
    vm: Vector # Minimum point
    vp: Vector # Maximum point

@dataclass
class BBoxFrustum :
    near: float
    far: float
    left_near: float
    right_near: float
    bottom_near: float
    top_near: float
    left_far: float
    right_far: float
    bottom_far: float
    top_far: float

def get_sensor_size(camera: bpy.types.Camera) -> tuple[float, float] :
    """Return the sensor width and height of the given camera (in millimeters)."""

    # Width / height
    aspect_ratio = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y

    if camera.data.sensor_fit == "AUTO" :

        if aspect_ratio < 1 :
            height = camera.data.sensor_width
            width = height * aspect_ratio
        else :
            width = camera.data.sensor_width
            height = width / aspect_ratio

    elif camera.data.sensor_fit == "HORIZONTAL" :
        width = camera.data.sensor_width
        height = width / aspect_ratio

    elif camera.data.sensor_fit == "VERTICAL" :
        height = camera.data.sensor_height
        width = height * aspect_ratio
    
    return width, height

def get_frustum(camera: bpy.types.Camera, near: float | None = None, far: float | None = None) -> Frustum :
    """Return the frustum parameters of the given perspective camera (in meters).
    If near or far is ommited the camera clipping parameter are used."""

    if near is None :
        near = camera.data.clip_start # in meters

    if far is None :
        far = camera.data.clip_end # in meters

    # Sensor width and height 
    width, height = get_sensor_size(camera) # in millimeters

    # Camera lens is in millimeters
    left  = width * (near / camera.data.lens) * (camera.data.shift_x - 0.5)
    right = width * (near / camera.data.lens) * (camera.data.shift_x + 0.5)
    bottom = height * (near / camera.data.lens) * (camera.data.shift_y - 0.5)
    top    = height * (near / camera.data.lens) * (camera.data.shift_y + 0.5) 

    return Frustum(left, right, near, far, bottom, top)

def get_bbox_frustum(frustum: Frustum, bbox: BBox = BBox(Vector(), Vector())) -> BBoxFrustum :
    """Return the bounding box frustum for the given frustum and bounding box.
    Throw a ValueError if the output bounding box frustum is empty."""

    # Left plane: x = left/near * y + left_shift
    # Right plane: x = right/near * y - right_shift
    # Bottom plane: z = bottom/near * y + bottom_shift
    # Top plane: z = top/near * y - top_shift

    # Shift to respect bbox along y axis
    y_left_shift   =  bbox.vm.y * frustum.left   / frustum.near 
    y_right_shift  = -bbox.vm.y * frustum.right  / frustum.near 
    y_bottom_shift =  bbox.vm.y * frustum.bottom / frustum.near 
    y_top_shift    = -bbox.vm.y * frustum.top    / frustum.near 

    # Take into account x and z bbox size
    left_shift   =   y_left_shift - bbox.vm.x
    right_shift  =  y_right_shift + bbox.vp.x
    bottom_shift = y_bottom_shift - bbox.vm.z
    top_shift    =    y_top_shift + bbox.vp.z

    # Minimum y value according to planes along x or z
    px_min_y = (  left_shift + right_shift) * frustum.near / (frustum.right - frustum.left)
    pz_min_y = (bottom_shift +   top_shift) * frustum.near / (frustum.top   - frustum.bottom)

    # New near and far values
    far = frustum.far - bbox.vp.y
    near = max(px_min_y, pz_min_y, frustum.near - bbox.vm.y)

    # Inject y = near in planes equations
    left_near   = frustum.left / frustum.near * near + left_shift
    right_near  = frustum.right / frustum.near * near - right_shift
    bottom_near = frustum.bottom / frustum.near * near + bottom_shift
    top_near    = frustum.top / frustum.near * near - top_shift

    # Inject y = far in planes equations
    left_far = frustum.left / frustum.near * far + left_shift
    right_far = frustum.right / frustum.near * far - right_shift
    bottom_far = frustum.bottom / frustum.near * far + bottom_shift
    top_far = frustum.top / frustum.near * far - top_shift

    try :
        assert near < far + EPSILON
        assert left_near < right_near + EPSILON
        assert left_far < right_far + EPSILON
        assert bottom_near < top_near + EPSILON
        assert bottom_far < top_far + EPSILON
    except AssertionError as e:
        raise ValueError("empty bounding box frustum") from e        

    return BBoxFrustum(near, far, left_near, right_near, bottom_near, top_near, left_far, right_far, bottom_far, top_far)


def get_random_point(frustum: Frustum | BBoxFrustum, a: float | None = None, b: float | None = None, c: float | None = None) -> Vector :
    """Return a random point in the given frustum."""

    if a is None :
        a = random.random()
    if b is None :
        b = random.random()
    if c is None :
        c = random.random()
    
    y = lerp(frustum.near, frustum.far, b)

    if isinstance(frustum, Frustum) :
        x = lerp(frustum.left, frustum.right, a) * lerp(1, frustum.far / frustum.near, b)
        z = lerp(frustum.bottom, frustum.top, c) * lerp(1, frustum.far / frustum.near, b)

    else :
        x = lerp(lerp(frustum.left_near, frustum.right_near, a), lerp(frustum.left_far, frustum.right_far, a), b)
        z = lerp(lerp(frustum.bottom_near, frustum.top_near, c), lerp(frustum.bottom_far, frustum.top_far, c), b)

    return Vector((x,y,z))

def delete_object(object_name: str) -> None :
    """Delete the given object from Blender."""

    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    if object_name in bpy.data.objects :
        obj = bpy.data.objects[object_name]

        # Select object and delete it
        obj.select_set(True) 
        bpy.ops.object.delete()



def create_box(pos: Vector, size: Vector, name: str = "Box") -> None :
    """Create a box at the given position."""

    bpy.ops.mesh.primitive_cube_add(size=1)
    
    box = bpy.context.active_object

    # Set object and mesh names
    box.name = name
    box.data.name = name

    box.location = pos
    box.scale = size

def create_pyramid(apex: Vector, points: list[Vector], name: str = "Pyramid") -> None :
    """Create a pyramid with the given points."""

    center = (apex + sum(points, start=Vector())) / (len(points) + 1)

    vertices = [apex - center] + [p - center for p in points]

    faces = [(0, i, i+1) for i in range(1, len(vertices) - 1)]
    faces.append((0, len(points), 1)) # Append last side
    faces.append(tuple(range(1, len(points) + 1))) # Append base

    mesh: bpy.types.Mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)

    obj: bpy.types.Object = bpy.data.objects.new(name, mesh) # Create an object with the mesh
    obj.location = center
    bpy.context.scene.collection.objects.link(obj) # Link obj to the scene



def get_random_quaternion() -> Quaternion :
    """Return a random quaternion representing a rotation."""
    q = Quaternion(random.gauss(0,1) for i in range(4))
    q.normalize()
    return q


def set_object_quaternion(obj: bpy.types.Object, q: Quaternion) -> None :
    """Set the object rotation using a quaternion."""
    mode = obj.rotation_mode
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = q
    obj.rotation_mode = mode

def get_bbox(obj: bpy.types.Object, recursive=False) -> BBox :
    """Return the axis aligned bounding box coordinates based on the object aligned bounding box.
    Children objects can be recursively included in the bounding box."""

    vm = Vector((math.inf, math.inf, math.inf))
    vp = -vm

    objs = [obj]

    while len(objs) != 0 :
        current_obj = objs.pop()

        for corner in map(Vector, current_obj.bound_box) :
            p = current_obj.matrix_world @ corner

            vm.x = min(vm.x, p.x)
            vm.y = min(vm.y, p.y)
            vm.z = min(vm.z, p.z)
            vp.x = max(vp.x, p.x)
            vp.y = max(vp.y, p.y)
            vp.z = max(vp.z, p.z)

        # If recursive: add children to objects list
        if recursive:
            objs += current_obj.children
    
    return BBox(vm, vp)


def create_camera_file(camera: bpy.types.Camera, frustum: Frustum, file: str) -> None :
    """Create the camera file with the given camera settings."""

    captor_x, captor_y = get_sensor_size(camera)
    captor_x /= 1000  # Convertion from millimeters to meters
    captor_y /= 1000 # Same

    f = camera.data.lens / 1000 # Focal in meters


    world_camera_matrix = camera.matrix_world.inverted()

    projection_matrix = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=bpy.context.scene.render.resolution_x,
        y=bpy.context.scene.render.resolution_y,
        scale_x=bpy.context.scene.render.pixel_aspect_x,
        scale_y=bpy.context.scene.render.pixel_aspect_y,
    )
    
    M = projection_matrix @ world_camera_matrix

    data = {
        "cx": captor_x, 
        "cy": captor_y,
        "pxx": bpy.context.scene.render.resolution_x,
        "pxy": bpy.context.scene.render.resolution_y,
        "f": f,
        "near": frustum.near,
        "far": frustum.far,
        "M": list(map(list,M))
    }
    
    with open(file, "w") as f:
        json.dump(data, f)




    


def render_image(path: str) -> None :
    """Render an image to the given path."""
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still = True)


def generate_dataset(satellite: bpy.types.Object, camera: bpy.types.Camera, light: bpy.types.Light, near: float, far: float, size: int, dir: str | None = None) -> None :
    """Generate a dataset with the given objects of the given size.
    The images are saved in the given directory
    If no output directory is given, the process is done without outputing anything."""

    assert size > 0
    assert near > 0
    assert far > 0
    assert near < far

    format = bpy.context.scene.render.image_settings.file_format
    extension = format.lower()

    # Compute camera frustum
    frustum = get_frustum(camera, near, far)

    output_enabled = dir is not None

    if output_enabled:
        nb_digits = len("%d" % (size-1))
        num_pattern = f"%0{nb_digits}d"
        image_filename_pattern = f"img{num_pattern}.{extension}"
        json_filename_pattern = f"img{num_pattern}.json"

        # Create camera json file
        create_camera_file(camera, frustum, os.path.join(dir, "camera.json"))

    for i in range(size) :
        # Randomly rotate the light
        set_object_quaternion(light, get_random_quaternion())

        # Randmoly rotate the satellite
        q = get_random_quaternion()
        set_object_quaternion(satellite, q)

        # Force to update satellite rotation 
        bpy.context.view_layer.update()

        # Compute satellite bounding box
        bbox = get_bbox(satellite, True)

        # Center the bbox
        bbox = BBox(bbox.vm - satellite.location, bbox.vp - satellite.location)

        # Compute bbox frustum
        bbox_frustum = get_bbox_frustum(frustum, bbox)

        # Place the satellite in the bbox frustum
        p = get_random_point(bbox_frustum)
        satellite.location = p


        if output_enabled :
            image_filename = image_filename_pattern % i
            image_full_path = os.path.join(dir, image_filename)

            render_image(image_full_path)

            json_full_path = os.path.join(dir, json_filename_pattern % i)

            # Output the labels
            label = {"img": image_filename, "t": [p.x, p.y, p.z], "q": [q.w, q.x, q.y, q.z]}
            with open(json_full_path, "w") as f:
                json.dump(label, f)




if __name__ == "__main__":
    
    time_start = time.time()

    # Python args are after --
    try :
        args = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        args = []
    
    parser = argparse.ArgumentParser(
        prog = "dataset",
        description = "generate a dataset of satellite images")

    parser.add_argument("-f", "--force", help="force the output", action="store_true")
    parser.add_argument("-c", "--count", help="number of images", default=1, type=int)
    parser.add_argument("-o", "--output", help="output directory", default=None)

    parsed_args = parser.parse_args(args)

    # Get count and output arguments
    count: int = parsed_args.count
    force_output = parsed_args.force
    output_enabled = parsed_args.output is not None

    output_directory: str | None
    if output_enabled :
        output_directory: str = os.path.abspath(parsed_args.output)
    else :
        output_directory = None


    satellite = bpy.data.objects[SATELLITE_NAME]
    camera = bpy.data.objects[CAMERA_NAME]
    light = bpy.data.objects[LIGHT_NAME]

    if output_enabled :
        if os.path.exists(output_directory) :
            if not force_output and len(os.listdir(output_directory)) != 0 :
                raise FileExistsError(f"output directory '{output_directory}' is not empty")
        else :
            os.makedirs(output_directory)


    generate_dataset(satellite, camera, light, NEAR, FAR, count, output_directory)

    execution_time = time.time() - time_start
    print(f"Generation done in {execution_time:.3f} seconds.")