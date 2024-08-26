import json
import numpy as np
import sys

NEAR = 1
FAR = 10

focal_length_mm = 30
sensor_width_mm = 35
fov_radians = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
fov_degrees = np.degrees(fov_radians)
aspect_ratio = 512 / 512
 

def quaternion_to_rotation_matrix(q):
    # convert a quaternion into a 3x3 rotation matrix
    q = np.array(q, dtype=np.float64)
    q /= np.linalg.norm(q) 
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*(q1*q2 - q3*q0), 2*(q1*q3 + q2*q0)],
        [2*(q1*q2 + q3*q0), 1 - 2*q1**2 - 2*q3**2, 2*(q2*q3 - q1*q0)],
        [2*(q1*q3 - q2*q0), 2*(q2*q3 + q1*q0), 1 - 2*q1**2 - 2*q2**2]
    ])

def orthographic_matrix(left, right, bottom, top):
    # create 4x4 orthographic projection matrix
    tx = -(right + left) / (right - left)
    ty = -(top + bottom) / (top - bottom)
    tz = -(FAR + NEAR) / (FAR - NEAR)
    return np.array([
        [2 / (right - left), 0, 0, tx],
        [0, 2 / (top - bottom), 0, ty],
        [0, 0, -2 / (FAR - NEAR), tz],
        [0, 0, 0, 1]
    ])


def perspective_projection_matrix():
    # create 4x4 perspective projection matrix with a given focal length f
    f = 1 / np.tan(fov_radians / 2)
    A = (NEAR + FAR) / (NEAR - FAR) 
    B = (2 * NEAR * FAR) / (NEAR - FAR)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, A, B],
        [0, 0,-1, 0]  # homogeneous coordinate
    ])

def project_to_2d(point, ortho_matrix, persp_matrix):
    # orthographic projection followed by perspective projection
    point_homogeneous = np.append(point, 1)
    point_ortho = ortho_matrix.dot(point_homogeneous)
    projected_point = persp_matrix.dot(point_ortho)
    if projected_point[3] == 0:
        return None
    x = projected_point[0] / projected_point[3] 
    y = projected_point[1] / projected_point[3] 
    return [x, y]

def process_images_and_keypoints(image_data, keypoint_data):
    ortho_matrix = orthographic_matrix(left=-256, right=256, bottom=-256, top=256)
    persp_matrix = perspective_projection_matrix()
    results = []
    for image_info in image_data:

        img_name = image_info["img"]
        t = np.array(image_info['t'])
        q = image_info['q']
        rotation_matrix = quaternion_to_rotation_matrix(q)

        transformed_keypoints = []
        for keypoint in keypoint_data['key_points']:
            keypoint_name = keypoint['name']
            position = np.array(keypoint['position']) + t
            position = rotation_matrix.dot(position)
            position_2d = project_to_2d(position, ortho_matrix, persp_matrix)

            if position_2d:
                transformed_keypoints.append({
                    'name': keypoint_name,
                    'position_2d': position_2d
                })

        results.append({
            'img': img_name,
            't': t.tolist(),
            'q': q,
            'transformed_keypoints': transformed_keypoints
        })

    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python transfo.py <path_to_image_data.json> <path_to_keypoint_data.json>")
        sys.exit(1)

    image_file = sys.argv[1]
    keypoint_file = sys.argv[2]

    with open(image_file, 'r') as f:
        image_data = json.load(f)

    with open(keypoint_file, 'r') as f:
        keypoint_data = json.load(f)

    output_data = process_images_and_keypoints(image_data, keypoint_data)

    print(output_data)

    with open('output_transformed_keypoints.json', 'w') as f:
        json.dump(output_data, f, indent=4)
