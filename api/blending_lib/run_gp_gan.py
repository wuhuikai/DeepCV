import os
import uuid

from chainer import serializers

from skimage import img_as_float
from skimage.io import imread, imsave

from .model import EncoderDecoder

from .gp_gan import gp_gan


G = EncoderDecoder(64, 64, 3, 4000, image_size=64)
serializers.load_npz('models/blending/blending_gan.npz', G)
"""
    Note: source image, destination image and mask image have the same size.
"""
def blending(args):
    # load image
    obj = img_as_float(imread(args.src))
    bg  = img_as_float(imread(args.dst))
    mask = imread(args.mask).astype(obj.dtype)

    blended_im = gp_gan(obj, bg, mask, G, 64, color_weight=args.color_weight)

    path = os.path.join('static/images', '{}.png'.format(uuid.uuid4()))
    imsave(path, blended_im)

    return {'path': path, 'status': 'success'}
