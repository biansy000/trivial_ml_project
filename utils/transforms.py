import random
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.core.fromnumeric import size
matplotlib.use('Agg') 

path = '/DATA1/siyuan/ML_proj_dataset/'


def read_img(idx, is_train=True):
    if is_train:
        img_path = os.path.join(path, 'train_img', f'{idx}.png')
        label_path = os.path.join(path, 'train_label', f'{idx}.png')
    else:
        img_path = os.path.join(path, 'test_img', f'{idx}.png')
        label_path = os.path.join(path, 'test_label', f'{idx}.png')

    img = Image.open(img_path)
    img = img.convert('RGB')

    label = Image.open(label_path)
    label = label.convert('RGB')

    return img, label



def read_data(idx, target_size=(512, 512), is_train=True, data_augment=True):
    img, raw_label = read_img(idx, is_train=is_train)

    inp_w = target_size[1]
    inp_h = target_size[0]

    if (not is_train) and idx == 0:
        raw_label = raw_label.transpose(Image.ROTATE_180)
        raw_label.save('vis/0.png')

    if not data_augment:
        img = np.array(img).astype(np.float32)
        raw_label = 255 - np.array(raw_label).astype(np.float32)
        center, scale = box_to_center_scale(0, 0, img.shape[1], img.shape[0], aspect_ratio=inp_w/inp_h)
        r = 0
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
    else:
        if random.random() < 0.5:
            # FLIP_LEFT_RIGHT
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            raw_label = raw_label.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            # FLIP_TOP_BOTTOM
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            raw_label = raw_label.transpose(Image.FLIP_TOP_BOTTOM)

        img = np.array(img).astype(np.float32)
        raw_label = 255 - np.array(raw_label).astype(np.float32)

        # pixel jittor
        if random.random() < 0.5:
            img = img + np.random.normal(0, 0.02, img.shape)
            img = img + np.random.normal(0, 0.1, size=1)

        center, scale = box_to_center_scale(0, 0, img.shape[1], img.shape[0], aspect_ratio=inp_w/inp_h)

        # scale and rotation
        sf = 0.15
        rf = 15
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.5 else 0
        scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        
        if random.random() < 0.5:
            trans[0, 2] += (random.random()*40-20)
            trans[1, 2] += (random.random()*40-20)

    img = cv2.warpAffine(img, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    label = cv2.warpAffine(raw_label, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_NEAREST)

    # if want to visualize the result of data aug, you may uncomment following codes
    # if (is_train) and idx == 0:
    #     img1 = Image.fromarray(img.astype(np.uint8))
    #     img1.save('vis/train_try_img.png')
    #     img2 = Image.fromarray((label).astype(np.uint8))
    #     img2.save('vis/train_try_label.png')

    img = img / 255 - 0.45
    img = img / 0.2

    label = label / 255
    label[label>0.5] = 1
    label[label<=0.5] = 0

    return img, label[:,:,0]



def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """
    get the affine transformation for scaling, ratation and shift
    use the coordinate of three points to get the affine tranform: 
      1. point a: the center point , 2. point b: the center point on the left side of the original figure
          3. the point c, which satisfies that (c-a) is perpendicular to (b-a)
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt


def box_to_center_scale(x, y, w, h, aspect_ratio=1.0):
    """size format from x,y,w,h -> center, scale"""
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def PIL_shift_img(img, raw_label):
    """shift the figure using functions provided by PIL, in fact not used in the final version"""
    a = 1
    b = 0
    c = random.random() * 20 - 10
    d = 0
    e = 1
    f = random.random() * 20 - 10
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    raw_label = raw_label.transform(raw_label.size, Image.AFFINE, (a, b, c, d, e, f))
    return img, raw_label


