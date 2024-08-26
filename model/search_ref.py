import os
import sys
import json
import numpy as np
from pyquaternion import Quaternion
from PIL import Image

# Exit codes
WRONG_USAGE = 1
FILE_NOT_FOUND = 2

HELP_MSG = f"""\
usage: {os.path.basename(__file__)} input [output]
     input: speedplusv2 directory
    output: output directory"""

IMAGES_PATH = "synthetic/images/"
LABELS_PATH = "synthetic/train.json"
CAMERA_PATH = "camera.json"

NB_SIDES = 6
FRONT, RIGHT, UP, BACK, LEFT, DOWN = range(NB_SIDES)
SIDES = (FRONT, RIGHT, UP, BACK, LEFT, DOWN)
SIDE_NAMES = ("front", "right", "up", "back", "left", "down")
TOP_SIDES = (UP, UP, BACK, UP, UP, FRONT) # Top side for the reference image

VECTORS = [None] * NB_SIDES
VECTORS[FRONT] = np.array([0,-1,0]) # Weird referential
VECTORS[RIGHT] = np.array([1,0,0])  # To place 
VECTORS[UP]    = np.array([0,0,1])  # Front vector along the main antenna
VECTORS[BACK] = -VECTORS[FRONT]
VECTORS[LEFT] = -VECTORS[RIGHT]
VECTORS[DOWN] = -VECTORS[UP]

OUTPUT_SIZE = 1000

def getTMatrix(q, t) :
    """Return the 4x4 transformation matrix of the given rotation q and translation t."""
    M = q.transformation_matrix
    M[:3,3] = t
    return M

def getTiMatrix(q, t) :
    """Return the 4x4 inverse transformation matrix of the given rotation q and translation t."""
    Ti = np.zeros((4,4))
    Ti[:3,3] = -t
    Ri = q.inverse.transformation_matrix
    M = np.dot(Ri, Ti)
    return M

class Camera:

    FX_KEY = "fx"
    FY_KEY = "fy"
    PPX_KEY = "ppx"
    PPY_KEY = "ppy"
    W_KEY = "Nu"
    H_KEY = "Nv"

    def __init__(self, fx, fy, ppx, ppy, w, h) :
        """Create a camera with the given parameters."""

        self.fx = fx # Horizontal focal length [m]
        self.fy = fy # Vertical focal length [m]
        self.ppx = ppx # Horizontal pixel pitch [m / px]
        self.ppy = ppy # Vertical pixel pitch [m / px]
        self.w = w # Width [px]
        self.h = h # Height [px]
        self.fpx = fx / ppx # Horizontal focal length [px]
        self.fpy = fy / ppy # Vertical focal length [px]

        # Projection matrix (x,y,z) [m] -> (x,y,w) [px]
        self.M = np.array([[self.fpx,        0, self.w / 2],
                           [       0, self.fpy, self.h / 2],
                           [       0,        0,          1]])
    
    def fromJson(file) :
        """Return a camera read from the given file."""

        data = json.load(file)

        fx = data[Camera.FX_KEY]
        fy = data[Camera.FY_KEY]
        ppx = data[Camera.PPX_KEY]
        ppy = data[Camera.PPY_KEY]
        w = data[Camera.W_KEY]
        h = data[Camera.H_KEY]

        return Camera(fx, fy, ppx, ppy, w, h)

class Label:

    T_KEY = "r_Vo2To_vbs_true" # Translation
    Q_KEY = "q_vbs2tango_true" # Quaternion
    FILE_KEY = "filename"

    def __init__(self, filename, q, t):
        """Create a label with the given parameters."""

        self.filename = filename # Image file name
        self.q = q # Quaternion
        self.t = t # Translation

        self._M = None # 4x4 transformation matrix
    
    @property
    def M(self) :
        if self._M == None :
            self._M = getTMatrix(self.q, self.t)
        return self._M
    
    def fromJson(file) :
        """Return a list of labels read from the given file."""

        data = json.load(file)

        labels = []

        for item in data :
            filename = item[Label.FILE_KEY]
            q = Quaternion(item[Label.Q_KEY]).normalised
            t = np.array(item[Label.T_KEY])
            
            label = Label(filename, q, t)
            labels.append(label)
        
        return labels


def getReferences(labels) :
    """Return the indexes of the best reference images and the dot product values."""

    indexes = [None] * NB_SIDES
    dots = [-2] * NB_SIDES

    for i in range(len(labels)) :
        label = labels[i]

        for side in SIDES :
            v = label.q.rotate(VECTORS[side]) # Apply rotation to side vector
            dot = np.dot(v, VECTORS[DOWN]) # No rotation is taken from downside

            if dot > dots[side] :
                dots[side] = dot
                indexes[side] = i
    
    indexes = np.array(indexes)
    dots = np.array(dots)

    return indexes, dots

def getCorrection(label, side, camera, size) :
    """Return the correction (t,r,s) of the given reference image."""

    T = label.M # Transformation matrix

    P = np.zeros((3,4)) # Projection matrix
    P[:,:3] = camera.M

    origin = np.array([0,0,0,1])
    originT = np.dot(T, origin) # Transformed origin
    originP = np.dot(P, originT) # Projected origin
    originP /= originP[2] # Divide by z
    originP = originP[:2] # Only keep (x,y)

    top = np.ones(4)
    top[:3] = VECTORS[TOP_SIDES[side]]
    topT = np.dot(T, top) # Transform top
    topP = np.dot(P, topT) # Projected top
    topP /= topP[2] # Divide by z
    topP = topP[:2] # Only keep (x,y)

    deltaTopP = topP - originP

    center = np.array([camera.w, camera.h]) / 2

    t = center - originP
    r = np.arctan2(deltaTopP[1], deltaTopP[0]) + np.pi / 2
    s = size / np.linalg.norm(deltaTopP)

    return t, r, s


def translateImage(image, t) :
    """Return the given image translated by t."""
    return image.rotate(0, translate=tuple(t))


def translateRotateImage(image, t, r) :
    """Return the given image translated by t and rotated by r.
    Intermediate cut out is avoided."""
    rotPoint = np.array(image.size) / 2 - t # Rotation center
    return image.rotate(r, Image.BICUBIC, center=tuple(rotPoint), translate=tuple(t))

def zoomImage(image, s) :
    """Return the given image zoomed by a factor s."""
    box = np.array(image.size) / s
    center = np.array(image.size) / 2
    
    tl = center - box / 2 # Top left corner
    br = tl + box         # Bottom right corner

    return image.crop((tl[0], tl[1], br[0], br[1])).resize(image.size, Image.BICUBIC)


def correctImage(image, t, r, s) :
    """Return the image with translation, rotation (deg) and scale applied."""
    return zoomImage(translateRotateImage(image, t, r), s)

    
if __name__ == "__main__" :

    # usage: search_ref input [output]

    if len(sys.argv) == 2 :
        speedDir = sys.argv[1]
        outputDir = os.getcwd()

    elif len(sys.argv) == 3 :
        speedDir = sys.argv[1]
        outputDir = sys.argv[2]

    else :
        print(HELP_MSG, file=sys.stderr)
        sys.exit(WRONG_USAGE)
    
    # Check speed directory exists
    if not os.path.exists(speedDir) :
        print(f"Error while opening input directory: {speedDir} not found.", file=sys.stderr)
        sys.exit(FILE_NOT_FOUND)

    
    cameraPath = os.path.join(speedDir, CAMERA_PATH)
    labelsPath = os.path.join(speedDir, LABELS_PATH)

    try :
        with open(cameraPath) as f :
            camera = Camera.fromJson(f)
        
        with open(labelsPath) as f :
            labels = Label.fromJson(f)

    except FileNotFoundError as e :
        print(f"Error while opening input directory: {e.filename} not found.", file=sys.stderr)
        sys.exit(FILE_NOT_FOUND)
    
    # Create output directory if needed
    if not os.path.exists(outputDir) :
        os.makedirs(outputDir)

    indexes, _ = getReferences(labels)

    # Print the header
    print("side filename translation_x translation_y rotation")

    for side in SIDES :
        sidename = SIDE_NAMES[side]
        i = indexes[side]
        label = labels[i]
        
        filename = label.filename

        t, r, s = getCorrection(label, side, camera, OUTPUT_SIZE)
        r = np.rad2deg(r)

        with Image.open(os.path.join(speedDir, IMAGES_PATH, filename)) as image :
            referenceImage = correctImage(image, t, r, s)
            referencePath = os.path.join(outputDir, sidename + ".png")
            referenceImage.save(referencePath)

        # Print the entry
        print(f"{sidename} {filename} {t[0]:.0f} {t[1]:.0f} {r:.0f}")