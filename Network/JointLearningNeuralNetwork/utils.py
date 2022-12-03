import numpy as np
import cv2
import pickle
# from transforms3d.axangles import axangle2mat


bone_colors = [
  (255, 0, 0), (255, 32, 32), (255, 64, 64), (255, 128, 128),
  (0, 255, 0), (32, 255, 32), (64, 255, 64), (128, 255, 128),
  (0, 0, 255), (32, 32, 255), (64, 64, 255), (128, 128, 255),
  (255, 255, 0), (255, 255, 32), (255, 255, 64), (255, 255, 128),
  (255, 0, 255), (255, 32, 255), (255, 64, 255), (255, 128, 255)
]


def hmap_to_uv(hmap):
  shape = hmap.shape
  hmap = np.reshape(hmap, [-1, shape[-1]])
  v, h = np.unravel_index(np.argmax(hmap, 0), shape[:-1])
  coord = np.stack([v, h], 1)
  return coord


def render_armature_from_uv(uv, canvas, armature,
                             conf=None, conf_thres=0, thickness=None):
  if canvas.dtype != np.uint8:
    print('canvas must be uint8 type')
    exit(0)
  if thickness is None:
    thickness = max(canvas.shape[0] // 128, 1)
  for i, bone in enumerate(armature.bones):
    color = bone_colors[i]
    start = (int(uv[bone[0]][1]), int(uv[bone[0]][0]))
    end = (int(uv[bone[1]][1]), int(uv[bone[1]][0]))

    anyzero = lambda x: x[0] * x[1] == 0
    unsure = lambda x: conf is not None and conf[x] < conf_thres
    if anyzero(start) or unsure(bone[0]) or anyzero(end) or unsure(bone[1]):
      continue

    cv2.line(canvas, start, end, color, thickness)
  return canvas


def render_armature_from_hmap(hmap, canvas, armature,
                              conf=None, conf_thres=0, thickness=None):
  coords = hmap_to_uv(hmap)
  bones = render_armature_from_uv(
    coords, canvas, armature, conf, conf_thres, thickness
  )
  return bones


def render_gaussian_hmap(centers, shape, sigma=1):
  x = [i for i in range(shape[1])]
  y = [i for i in range(shape[0])]
  xx, yy = np.meshgrid(x, y)
  xx = np.reshape(xx.astype(np.float32), [shape[0], shape[1], 1])
  yy = np.reshape(yy.astype(np.float32), [shape[0], shape[1], 1])
  x = np.reshape(centers[:,1], [1, 1, -1])
  y = np.reshape(centers[:,0], [1, 1, -1])
  distance = np.square(xx - x) + np.square(yy - y)
  hmap = np.exp(-distance / (2 * sigma**2 )) / np.sqrt(2 * np.pi * sigma**2)
  hmap /= (
    np.max(hmap, axis=(0, 1), keepdims=True) + np.finfo(np.float32).eps
  )
  return hmap


def xyz_to_delta(xyz, armature):
  delta = []
  for c, p in enumerate(armature.parents):
    if p is not None:
      delta.append(xyz[c] - xyz[p])
    else:
      delta.append(np.zeros_like(delta[-1]))
  delta = np.stack(delta)
  return delta

def xyz_to_delta_norm(xyz, joints_def):
  """
  Compute bone orientations from joint coordinates (child joint - parent joint).
  The returned vectors are normalized.
  For the root joint, it will be a zero vector.

  Parameters
  ----------
  xyz : np.ndarray, shape [J, 3]
    Joint coordinates.
  joints_def : object
    An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

  Returns
  -------
  np.ndarray, shape [J, 3]
    The **unit** vectors from each child joint to its parent joint.
    For the root joint, it's are zero vector.
  np.ndarray, shape [J, 1]
    The length of each bone (from child joint to parent joint).
    For the root joint, it's zero.
  """
  delta = []
  for j in range(joints_def.n_joints):
    p = joints_def.parents[j]
    if p is None:
      delta.append(np.zeros(3))
    else:
      delta.append(xyz[j] - xyz[p])
  delta = np.stack(delta, 0)
  lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
  delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
  return delta, lengths

# def xyz_to_mesh(xyz, armature, thickness=0.2):
#   n_bones = len(armature.bones)
#   faces = np.empty([n_bones * 8, 3], dtype=np.int32)
#   verts = np.empty([n_bones * 6, 3], dtype=np.float32)
#   for i, b in enumerate(armature.bones):
#     a = xyz[b[1]]
#     b = xyz[b[0]]
#     ab = b - a
#     f = a + thickness * ab
#
#     if ab[0] == 0:
#       ax = [0, 1, 0]
#     else:
#       ax = [-ab[1]/ab[0], 1, 0]
#
#     fd = np.transpose(axangle2mat(ax, -np.pi/2).dot(np.transpose(ab))) \
#          * thickness / 1.2
#     d = fd + f
#     c = np.transpose(axangle2mat(ab, -np.pi/2).dot(np.transpose(fd))) + f
#     e = np.transpose(axangle2mat(ab, np.pi/2).dot(np.transpose(fd))) + f
#     g = np.transpose(axangle2mat(ab, np.pi).dot(np.transpose(fd))) + f
#
#     verts[i*6+0] = a
#     verts[i*6+1] = b
#     verts[i*6+2] = c
#     verts[i*6+3] = d
#     verts[i*6+4] = e
#     verts[i*6+5] = g
#
#     faces[i*8+0] = np.flip(np.array([0, 2, 3], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+1] = np.flip(np.array([0, 3, 4], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+2] = np.flip(np.array([0, 4, 5], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+3] = np.flip(np.array([0, 5, 2], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+4] = np.flip(np.array([1, 4, 3], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+5] = np.flip(np.array([1, 3, 2], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+6] = np.flip(np.array([1, 5, 4], dtype=np.int32), axis=0) + i * 6
#     faces[i*8+7] = np.flip(np.array([1, 2, 5], dtype=np.int32), axis=0) + i * 6
#
#   return verts, faces

def load_pkl(path):
  """
  Load pickle data.

  Parameter
  ---------
  path: Path to pickle file.

  Return
  ------
  Data in pickle file.

  """
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data

def imresize(img, size):
  """
  Resize an image with cv2.INTER_LINEAR.

  Parameters
  ----------
  size: (width, height)

  """
  return cv2.resize(img, size, cv2.INTER_LINEAR)