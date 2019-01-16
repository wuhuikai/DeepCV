import os
import uuid
import numpy as np

from chainer import serializers

from skimage import img_as_float
from skimage.io import imread, imsave

from .model import EncoderDecoder

from .gp_gan import gp_gan, ndarray_resize


G = EncoderDecoder(64, 64, 3, 4000, image_size=64)
serializers.load_npz('models/blending/blending_gan.npz', G)
"""
    Note: source image, destination image and mask image have the same size.
"""
def blending(args):
    # load image
    obj  = img_as_float(imread(args.src))
    bg   = img_as_float(imread(args.dst))
    mask = img_as_float(imread(args.mask))
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    src_h, src_w, _ = obj.shape
    src_h, src_w = int(src_h*args.ratio), int(src_w*args.ratio)
    obj = ndarray_resize(obj, (src_h, src_w))
    mask = ndarray_resize(mask, (src_h, src_w), order=0)

    x, y = args.x, args.y
    dst_h, dst_w, _ = bg.shape

    left, top = max(0, -x), max(0, -y)
    right, bottom = min(dst_w, x + src_w) - x, min(dst_h, y + src_h) - y
    x, y = max(0, x), max(0, y)

    new_obj = np.zeros_like(bg)
    new_obj[y:y+bottom-top, x:x+right-left] = obj[top:bottom, left:right]

    new_mask = np.zeros((dst_h, dst_w), bg.dtype)
    new_mask[y:y+bottom-top, x:x+right-left] = mask[top:bottom, left:right]

    blended_im = gp_gan(new_obj, bg, new_mask, G, 64, color_weight=args.color_weight)

    path = os.path.join('static/images', '{}.png'.format(uuid.uuid4()))
    imsave(path, blended_im)

    return {'path': path, 'status': 'success'}
